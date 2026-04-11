/**
 * Planner — three planning arms for A/B experimentation.
 *
 * Arm A (evo_plan):       Evo generates a plan via single LLM call before spawning the worker.
 * Arm B (worker_plan):    Worker explores read-only, writes plan, Evo reviews, then worker executes.
 * Arm C (multi_explore):  Evo spawns parallel read-only workers, synthesizes findings, writes plan.
 *
 * Stripped from roundtable — uses local runner instead of the full harness.
 */

import fs from "fs"
import path from "path"
import type { OpenAIToolDef, SendWithToolsFn } from "./client"
import { createSendWithTools } from "./client"
import { createToolExecutor, getBaseToolSchemas } from "./tools"
import { runAgent } from "./runner"

// ── Types ─────────────────────────────────────────────────────────────

export type PlanningMode =
  | "none"
  | "evo_plan"
  | "worker_plan"
  | "multi_explore"
  | "explore_then_checklist"
  | "explore_then_checklist_state"
  | "mini_checklist"
  | "mini_checklist_progress"
  | "mini_checklist_progress_finish_gate"
  | "mini_checklist_progress_modify_gate"
  | "mini_checklist_progress_strict"
  | "conveyor_planner"
  | "scaffold_plan"

export interface PlanArtifact {
  mode: PlanningMode
  plan: string
  filePath: string
  style?: "guidance" | "checklist" | "conveyor"
  plannedFiles?: string[]
  checklistSteps?: string[]
  expectedStates?: string[]
  verification?: string
  trackProgress?: boolean
  requireCompletedStepForFinish?: boolean
  requireInProgressStepForModify?: boolean
  conveyorPhase?: boolean
  explorationSummaries?: string[]
  durationMs: number
  tokenCost: number
  inferenceCalls: number
}

export interface PlanningConfig {
  mode: PlanningMode
  task: string
  workDir: string
  /** Model for Evo's planning inference */
  model: string
  /** Endpoint for Evo's planning inference */
  endpoint: string
  /** Model for exploration workers (arms B, C) */
  workerModel?: string
  /** Endpoint for exploration workers (arms B, C) */
  workerEndpoint?: string
  /** Max turns for exploration workers (default: 15) */
  explorerMaxTurns?: number
  /** Number of parallel explorers for arm C (default: 2) */
  explorerCount?: number
  /** Temperature for planning inference (default: 0) */
  temperature?: number
}

// ── Constants ─────────────────────────────────────────────────────────

const PLAN_DIR = "/tmp/plan-lab-plans"

function ensurePlanDir(): void {
  fs.mkdirSync(PLAN_DIR, { recursive: true })
}

function writePlanFile(planId: string, content: string): string {
  ensurePlanDir()
  const filePath = path.join(PLAN_DIR, `${planId}.md`)
  fs.writeFileSync(filePath, content, "utf8")
  return filePath
}

// ── Plan tool (used by Evo to structure its plan output) ──────────────

const WRITE_PLAN_TOOL: OpenAIToolDef = {
  type: "function",
  function: {
    name: "write_plan",
    description: "Write the implementation plan",
    parameters: {
      type: "object",
      properties: {
        context: {
          type: "string",
          description: "One-sentence context: what is being changed and why",
        },
        files_to_modify: {
          type: "string",
          description: 'JSON array of {path, change} objects listing files to modify',
        },
        steps: {
          type: "string",
          description: "JSON array of step strings",
        },
        verification: {
          type: "string",
          description: "Single command or check to verify the change",
        },
      },
      required: ["context", "steps", "verification"],
    },
  },
}

// ── Plan formatting ───────────────────────────────────────────────────

interface StructuredPlan {
  context: string
  files_to_modify?: Array<{ path: string; change: string }> | string
  steps: string[] | string
  verification: string
}

interface ChecklistPlan {
  goal: string
  files: string[] | string
  steps: string[] | string
  expected_states?: string[] | string
  verification: string
}

type ChecklistMode =
  | "explore_then_checklist"
  | "explore_then_checklist_state"
  | "mini_checklist"
  | "mini_checklist_progress"
  | "mini_checklist_progress_finish_gate"
  | "mini_checklist_progress_modify_gate"
  | "mini_checklist_progress_strict"

function formatStructuredPlan(args: StructuredPlan): string {
  const lines: string[] = []
  lines.push("## Context")
  lines.push(args.context)
  lines.push("")

  // Handle files_to_modify
  let files: Array<{ path: string; change: string }> = []
  if (typeof args.files_to_modify === "string") {
    try { files = JSON.parse(args.files_to_modify) } catch { /* skip */ }
  } else if (Array.isArray(args.files_to_modify)) {
    files = args.files_to_modify
  }
  if (files.length > 0) {
    lines.push("## Files to Modify")
    for (const f of files) {
      // No backticks — small models copy them literally into tool calls
      const cleanPath = f.path.replace(/`/g, "").trim()
      lines.push(`- ${cleanPath} -- ${f.change}`)
    }
    lines.push("")
  }

  // Handle steps
  let steps: string[] = []
  if (typeof args.steps === "string") {
    try { steps = JSON.parse(args.steps) } catch { steps = [args.steps] }
  } else if (Array.isArray(args.steps)) {
    steps = args.steps
  }
  lines.push("## Implementation Steps")
  for (let i = 0; i < steps.length; i++) lines.push(`${i + 1}. ${steps[i]}`)
  lines.push("")

  lines.push("## Verification")
  lines.push(args.verification)
  return lines.join("\n")
}

function normalizeStringArray(input: string[] | string | undefined, maxItems = 5): string[] {
  if (!input) return []
  let values: string[] = []

  if (typeof input === "string") {
    try {
      const parsed = JSON.parse(input)
      if (Array.isArray(parsed)) values = parsed.map(v => String(v))
      else values = [input]
    } catch {
      values = input.split("\n").map(v => v.replace(/^-+\s*/, "").trim()).filter(Boolean)
    }
  } else if (Array.isArray(input)) {
    values = input.map(v => String(v))
  }

  return values
    .map(v => v.replace(/`/g, "").trim())
    .filter(Boolean)
    .slice(0, maxItems)
}

function formatChecklistPlan(args: ChecklistPlan): { markdown: string; plannedFiles: string[]; steps: string[]; expectedStates: string[]; verification: string } {
  const plannedFiles = normalizeStringArray(args.files, 3)
  const steps = normalizeStringArray(args.steps, 4).map(toFirstPersonChecklistStep)
  const expectedStates = normalizeStringArray(args.expected_states, 4).map(toFirstPersonExpectedState)
  const verification = toFirstPersonVerification(args.verification.trim())
  const goal = toFirstPersonGoal(args.goal.trim())

  const lines: string[] = []
  lines.push("## My Goal")
  lines.push(goal)
  lines.push("")

  if (plannedFiles.length > 0) {
    lines.push("## Files I Start With")
    for (const file of plannedFiles) lines.push(`- ${file}`)
    lines.push("")
  }

  lines.push("## My Checklist")
  for (let i = 0; i < steps.length; i++) {
    lines.push(`${i + 1}. ${steps[i]}`)
    if (expectedStates[i]) lines.push(`   Expected: ${expectedStates[i]}`)
  }
  lines.push("")
  lines.push("## How I Verify")
  lines.push(verification)

  return {
    markdown: lines.join("\n"),
    plannedFiles,
    steps,
    expectedStates,
    verification,
  }
}

function sentenceCase(text: string): string {
  if (!text) return text
  return text.charAt(0).toLowerCase() + text.slice(1)
}

function toFirstPersonGoal(goal: string): string {
  const clean = goal.replace(/^I\s+/i, "").trim()
  if (!clean) return "I will complete the task correctly."
  return /^will\b/i.test(clean)
    ? `I ${clean}`
    : `I will ${sentenceCase(clean)}`
}

function toFirstPersonChecklistStep(step: string): string {
  const clean = step.replace(/^\d+\.\s*/, "").trim()
  if (!clean) return "I continue with the next concrete step."
  return /^I\b/i.test(clean) ? clean : `I ${sentenceCase(clean)}`
}

function toFirstPersonVerification(verification: string): string {
  const clean = verification.trim()
  if (!clean) return "I run the provided test command."
  return /^I\b/i.test(clean) ? clean : `I run: ${clean}`
}

function toFirstPersonExpectedState(expected: string): string {
  const clean = expected.trim()
  if (!clean) return ""
  return /^I\b/i.test(clean) ? clean : `I expect ${sentenceCase(clean)}`
}

// ── Codebase snapshot (for Arm A context) ─────────────────────────────

function buildCodebaseSnapshot(workDir: string): string {
  const lines: string[] = []

  // List top-level files
  try {
    const entries = fs.readdirSync(workDir, { withFileTypes: true })
    const listing = entries
      .filter(e => !e.name.startsWith(".") && e.name !== "node_modules")
      .slice(0, 30)
      .map(e => `${e.isDirectory() ? "dir/" : ""} ${e.name}`)
      .join("\n")
    lines.push("### Directory Structure")
    lines.push(listing)
    lines.push("")
  } catch { /* skip */ }

  // Read small source files
  const MAX_FILE_SIZE = 2000
  let filesRead = 0
  const readCandidates = collectFiles(workDir, 3)

  for (const relPath of readCandidates) {
    if (filesRead >= 6) break
    const fullPath = path.join(workDir, relPath)
    try {
      const stat = fs.statSync(fullPath)
      if (!stat.isFile() || stat.size > 10_000) continue
      const content = fs.readFileSync(fullPath, "utf8").slice(0, MAX_FILE_SIZE)
      lines.push(`### ${relPath}`)
      lines.push("```")
      lines.push(content)
      lines.push("```")
      lines.push("")
      filesRead++
    } catch { /* skip */ }
  }

  return lines.join("\n") || "No codebase context available."
}

function collectFiles(dir: string, maxDepth: number, prefix = ""): string[] {
  if (maxDepth <= 0) return []
  const results: string[] = []
  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true })
    for (const e of entries) {
      if (e.name.startsWith(".") || e.name === "node_modules") continue
      const relPath = prefix ? `${prefix}/${e.name}` : e.name
      if (e.isDirectory()) {
        results.push(...collectFiles(path.join(dir, e.name), maxDepth - 1, relPath))
      } else if (/\.(js|ts|py|rs|go|rb|java)$/.test(e.name)) {
        results.push(relPath)
      }
    }
  } catch { /* skip */ }
  return results
}

// ── Arm A: Evo-driven plan ────────────────────────────────────────────

const EVO_PLAN_SYSTEM_PROMPT = `I am a software architect. Given a task and a codebase, I produce a concrete implementation plan.`
const MINI_CHECKLIST_SYSTEM_PROMPT = `I am writing a tiny execution checklist that I will follow during implementation.

I keep it concrete and short.
- I name at most 3 files
- I write 2-4 steps as my own intentions
- I add one short expected outcome per step when useful
- I prefer exact file paths
- No explanations or trade-off discussion
- The checklist helps me START in the right place and VERIFY at the end
- All steps are my own thoughts ("I inspect...", "I change...", "I verify...")`

export async function generateEvoPlan(config: PlanningConfig): Promise<PlanArtifact> {
  const start = Date.now()
  const planId = `evo-plan-${Date.now()}`
  let tokenCost = 0
  let inferenceCalls = 0

  const send = createSendWithTools({
    baseUrl: config.endpoint,
    model: config.model,
    maxTokens: 2048,
    timeoutMs: 120_000,
    temperature: config.temperature ?? 0,
  })

  const codebaseContext = buildCodebaseSnapshot(config.workDir)

  const result = await send(
    "Evo Planner",
    [
      { role: "system", content: EVO_PLAN_SYSTEM_PROMPT },
      {
        role: "user",
        content: `## Task\n${config.task}\n\n## Codebase Context\n${codebaseContext}\n\nWrite a concrete implementation plan.`,
      },
    ],
    [WRITE_PLAN_TOOL],
    config.model,
  )

  inferenceCalls++
  tokenCost += result.usage?.total_tokens ?? 0

  let planContent: string
  if (result.toolCalls?.length) {
    try {
      const args = JSON.parse(result.toolCalls[0].function.arguments) as StructuredPlan
      planContent = formatStructuredPlan(args)
    } catch {
      planContent = result.text || "No plan generated."
    }
  } else {
    planContent = result.text || "No plan generated."
  }

  const filePath = writePlanFile(planId, planContent)
  console.log(`  [planner] Arm A: Evo plan generated (${inferenceCalls} calls, ${tokenCost} tokens)`)

  return {
    mode: "evo_plan",
    plan: planContent,
    filePath,
    style: "guidance",
    durationMs: Date.now() - start,
    tokenCost,
    inferenceCalls,
  }
}

const WRITE_CHECKLIST_TOOL: OpenAIToolDef = {
  type: "function",
  function: {
    name: "write_checklist",
    description: "Write a tiny execution checklist",
    parameters: {
      type: "object",
      properties: {
        goal: {
          type: "string",
          description: "One-sentence goal for the task",
        },
        files: {
          type: "string",
          description: "JSON array of up to 3 file paths to investigate or modify",
        },
        steps: {
          type: "string",
          description: "JSON array of 2-4 short imperative execution steps",
        },
        expected_states: {
          type: "string",
          description: "JSON array of 2-4 short expected outcomes, one per step",
        },
        verification: {
          type: "string",
          description: "Single verification command",
        },
      },
      required: ["goal", "steps", "verification"],
    },
  },
}

async function generateChecklistFromContext(
  config: PlanningConfig,
  plannerLabel: string,
  planId: string,
  checklistContext: string,
  mode: ChecklistMode,
): Promise<PlanArtifact> {
  const start = Date.now()
  let tokenCost = 0
  let inferenceCalls = 0

  const send = createSendWithTools({
    baseUrl: config.endpoint,
    model: config.model,
    maxTokens: 1024,
    timeoutMs: 120_000,
    temperature: config.temperature ?? 0,
  })

  const result = await send(
    plannerLabel,
    [
      { role: "system", content: MINI_CHECKLIST_SYSTEM_PROMPT },
      {
        role: "user",
        content: checklistContext,
      },
    ],
    [WRITE_CHECKLIST_TOOL],
    config.model,
  )

  inferenceCalls++
  tokenCost += result.usage?.total_tokens ?? 0

  let artifact: { markdown: string; plannedFiles: string[]; steps: string[]; expectedStates: string[]; verification: string }
  if (result.toolCalls?.length) {
    try {
      const args = JSON.parse(result.toolCalls[0].function.arguments) as ChecklistPlan
      artifact = formatChecklistPlan(args)
    } catch {
      artifact = formatChecklistPlan({
        goal: "Follow the task carefully.",
        files: [],
        steps: [result.text || "Investigate the relevant files, make the fix, and verify it."],
        expected_states: [],
        verification: "Run the provided test command",
      })
    }
  } else {
    artifact = formatChecklistPlan({
      goal: "Follow the task carefully.",
      files: [],
      steps: [result.text || "Investigate the relevant files, make the fix, and verify it."],
      expected_states: [],
      verification: "Run the provided test command",
    })
  }

  const filePath = writePlanFile(planId, artifact.markdown)
  console.log(`  [planner] ${plannerLabel} generated (${inferenceCalls} calls, ${tokenCost} tokens)`)

  const trackProgress = mode !== "mini_checklist" && mode !== "explore_then_checklist"
  const requireCompletedStepForFinish =
    mode === "mini_checklist_progress_finish_gate" || mode === "mini_checklist_progress_strict"
  const requireInProgressStepForModify =
    mode === "mini_checklist_progress_modify_gate" || mode === "mini_checklist_progress_strict"

  return {
    mode,
    plan: artifact.markdown,
    filePath,
    style: "checklist",
    plannedFiles: artifact.plannedFiles,
    checklistSteps: artifact.steps,
    expectedStates: artifact.expectedStates,
    verification: artifact.verification,
    trackProgress,
    requireCompletedStepForFinish,
    requireInProgressStepForModify,
    durationMs: Date.now() - start,
    tokenCost,
    inferenceCalls,
  }
}

export async function generateMiniChecklistPlan(
  config: PlanningConfig,
  mode: ChecklistMode = "mini_checklist",
): Promise<PlanArtifact> {
  const planId = `mini-checklist-${Date.now()}`
  const codebaseContext = buildCodebaseSnapshot(config.workDir)
  return generateChecklistFromContext(
    config,
    "Mini Checklist Planner",
    planId,
    `## Task\n${config.task}\n\n## Codebase Context\n${codebaseContext}\n\nWrite a tiny checklist the execution model can follow.`,
    mode,
  )
}

// ── Arm B: Worker plan-then-execute ───────────────────────────────────

const WORKER_PLAN_SYSTEM_PROMPT = `I am in READ-ONLY planning mode. I must not modify any files — I can only read, search, and explore.

My job:
1. I thoroughly explore the codebase to understand existing patterns
2. I identify files, functions, and utilities relevant to the task
3. I design a concrete implementation approach

When done, I call complete with my full plan as the summary.

My rules:
- I am specific: file paths, function names, code patterns I found
- I include the verification command
- I keep the plan under 50 lines`

const EXPLORE_CHECKLIST_SYSTEM_PROMPT = `I am in READ-ONLY exploration mode.
I must not modify files.
I must not run commands.
I may only read files, search text, list directories, and find matching paths.

My job:
1. I find the smallest set of relevant files
2. I identify the existing implementation pattern
3. I infer the likely verification target from the codebase structure

When done, I call complete with a SHORT findings summary in exactly this shape:

FILES:
- file path
- file path

PATTERN:
- one existing pattern or function to copy or adapt

VERIFY:
- one likely test file, test command, or verification location

CONSTRAINTS:
- one or two constraints

My rules:
- I keep the whole summary under 12 lines
- I prefer exact file paths
- I do not explain my reasoning
- I do not include prose outside those headings`

async function runExploreChecklistExplorer(config: PlanningConfig): Promise<{ findings: string; tokenCost: number; inferenceCalls: number }> {
  const workerModel = config.workerModel || config.model
  const workerEndpoint = config.workerEndpoint || config.endpoint

  console.log(`  [planner] Explore->checklist: spawning read-only explorer...`)

  const explorerSend = createSendWithTools({
    baseUrl: workerEndpoint,
    model: workerModel,
    temperature: 0,
    timeoutMs: 120_000,
  })

  const explorerResult = await runAgent({
    agentName: "Checklist Explorer",
    systemPrompt: EXPLORE_CHECKLIST_SYSTEM_PROMPT,
    userMessage: `## Task\n${config.task}\n\nExplore the codebase and report only the findings most useful for a tiny execution checklist.`,
    model: workerModel,
    endpoint: workerEndpoint,
    sendWithTools: explorerSend,
    executor: createToolExecutor(),
    tools: getBaseToolSchemas().filter(t => ["read", "grep", "list", "glob", "complete"].includes(t.function.name)),
    workDir: config.workDir,
    maxTurns: config.explorerMaxTurns ?? 6,
    allowedTools: ["read", "grep", "list", "glob", "complete"],
    logLabel: "explore-checklist",
    evoObserve: { send: explorerSend, model: workerModel, intervalTurns: 2 },
    recoverySend: explorerSend,
    recoveryModel: workerModel,
  })

  return {
    findings: explorerResult.finalText || "Explorer produced no findings.",
    tokenCost: explorerResult.totalTokens,
    inferenceCalls: explorerResult.turnsWithToolCalls,
  }
}

export async function generateExploreThenChecklistPlan(config: PlanningConfig): Promise<PlanArtifact> {
  const start = Date.now()
  const planId = `explore-checklist-${Date.now()}`
  const exploration = await runExploreChecklistExplorer(config)

  const artifact = await generateChecklistFromContext(
    config,
    "Explore Checklist Planner",
    planId,
    `## Task\n${config.task}\n\n## Explorer Findings\n${exploration.findings}\n\nWrite a tiny checklist the execution model can follow. Use only the files and verification clues from the findings unless the findings are clearly incomplete.`,
    "explore_then_checklist",
  )

  return {
    ...artifact,
    durationMs: Date.now() - start,
    tokenCost: artifact.tokenCost + exploration.tokenCost,
    inferenceCalls: artifact.inferenceCalls + exploration.inferenceCalls,
    explorationSummaries: [exploration.findings],
  }
}

export async function generateExploreThenChecklistStatePlan(config: PlanningConfig): Promise<PlanArtifact> {
  const start = Date.now()
  const exploration = await runExploreChecklistExplorer(config)
  const withStateContext = `## Task\n${config.task}\n\n## Explorer Findings\n${exploration.findings}\n\nWrite a tiny checklist the execution model can follow. Include one short expected outcome for each step. Use only the files and verification clues from the findings unless the findings are clearly incomplete.`

  const rebuilt = await generateChecklistFromContext(
    config,
    "Explore Checklist State Planner",
    `explore-checklist-state-${Date.now()}`,
    withStateContext,
    "explore_then_checklist_state",
  )

  return {
    ...rebuilt,
    durationMs: Date.now() - start,
    tokenCost: exploration.tokenCost + rebuilt.tokenCost,
    inferenceCalls: exploration.inferenceCalls + rebuilt.inferenceCalls,
    explorationSummaries: [exploration.findings],
  }
}

export async function generateWorkerPlan(config: PlanningConfig): Promise<PlanArtifact> {
  const start = Date.now()
  const planId = `worker-plan-${Date.now()}`
  let tokenCost = 0
  let inferenceCalls = 0

  const workerModel = config.workerModel || config.model
  const workerEndpoint = config.workerEndpoint || config.endpoint

  // Phase 1: Worker explores read-only
  console.log(`  [planner] Arm B: Spawning read-only explorer...`)

  const explorerSend = createSendWithTools({
    baseUrl: workerEndpoint,
    model: workerModel,
    temperature: 0,
    timeoutMs: 120_000,
  })

  const explorerResult = await runAgent({
    agentName: "Plan Explorer",
    systemPrompt: WORKER_PLAN_SYSTEM_PROMPT,
    userMessage: `## Task\n${config.task}\n\nExplore the codebase and write a concrete implementation plan. Call complete when your plan is ready.`,
    model: workerModel,
    endpoint: workerEndpoint,
    sendWithTools: explorerSend,
    executor: createToolExecutor(),
    tools: getBaseToolSchemas().filter(t => ["read", "grep", "list", "glob", "bash", "complete"].includes(t.function.name)),
    workDir: config.workDir,
    maxTurns: config.explorerMaxTurns ?? 15,
    allowedTools: ["read", "grep", "list", "glob", "bash", "complete"],
    logLabel: "plan-explorer",
    evoObserve: { send: explorerSend, model: workerModel, intervalTurns: 2 },
    recoverySend: explorerSend,
    recoveryModel: workerModel,
  })

  tokenCost += explorerResult.totalTokens
  inferenceCalls += explorerResult.turnsWithToolCalls

  const rawPlan = explorerResult.finalText || "Worker did not produce a plan."

  // Phase 2: Evo reviews and refines
  const send = createSendWithTools({
    baseUrl: config.endpoint,
    model: config.model,
    maxTokens: 2048,
    timeoutMs: 120_000,
    temperature: config.temperature ?? 0,
  })

  const reviewResult = await send(
    "Evo Plan Reviewer",
    [
      {
        role: "system",
        content: "I am reviewing an implementation plan from my earlier exploration. I refine it into a concrete, actionable plan — fixing vague steps, adding missing file paths, removing prose. I output ONLY the refined plan.",
      },
      {
        role: "user",
        content: `## Original Task\n${config.task}\n\n## Worker's Plan\n${rawPlan}\n\nRefine this into a concrete, actionable plan.`,
      },
    ],
    [WRITE_PLAN_TOOL],
    config.model,
  )

  inferenceCalls++
  tokenCost += reviewResult.usage?.total_tokens ?? 0

  let planContent: string
  if (reviewResult.toolCalls?.length) {
    try {
      const args = JSON.parse(reviewResult.toolCalls[0].function.arguments) as StructuredPlan
      planContent = formatStructuredPlan(args)
    } catch {
      planContent = reviewResult.text || rawPlan
    }
  } else {
    planContent = reviewResult.text || rawPlan
  }

  const filePath = writePlanFile(planId, planContent)
  console.log(`  [planner] Arm B: Worker plan generated and reviewed (${inferenceCalls} calls, ${tokenCost} tokens)`)

  return {
    mode: "worker_plan",
    plan: planContent,
    filePath,
    style: "guidance",
    explorationSummaries: [rawPlan],
    durationMs: Date.now() - start,
    tokenCost,
    inferenceCalls,
  }
}

// ── Arm C: Multi-agent exploration + Evo synthesis ────────────────────

const EXPLORER_SYSTEM_PROMPT = `I am exploring this codebase in READ-ONLY mode. I must not modify any files — I can only read, search, and explore.

My specific focus area is provided below. I explore thoroughly, then call complete with my findings.

My report format:
- File paths I found relevant (with line numbers)
- Functions and utilities that could be reused
- Patterns and conventions I observed
- Concerns or dependencies I noticed`

function deriveExplorerSpecs(count: number): Array<{ name: string; focus: string }> {
  return [
    {
      name: "Architecture Explorer",
      focus: "Explore overall project structure, entry points, and patterns. Find how similar features are implemented. Report file paths and modules.",
    },
    {
      name: "Implementation Explorer",
      focus: "Find specific files, functions, and utilities relevant to the task. Look at import chains and types. Report exact file paths and function signatures.",
    },
    {
      name: "Test Explorer",
      focus: "Find test files, verification patterns, and integration points. Report test file paths and patterns.",
    },
  ].slice(0, count)
}

export async function generateMultiExplorePlan(config: PlanningConfig): Promise<PlanArtifact> {
  const start = Date.now()
  const planId = `multi-explore-${Date.now()}`
  let tokenCost = 0
  let inferenceCalls = 0

  const workerModel = config.workerModel || config.model
  const workerEndpoint = config.workerEndpoint || config.endpoint
  const explorerCount = config.explorerCount ?? 2
  const specs = deriveExplorerSpecs(explorerCount)

  console.log(`  [planner] Arm C: Spawning ${specs.length} parallel explorers...`)

  // Launch all explorers in parallel
  const explorerPromises = specs.map(spec => {
    const explorerSend = createSendWithTools({
      baseUrl: workerEndpoint,
      model: workerModel,
      temperature: 0,
      timeoutMs: 120_000,
    })

    return runAgent({
      agentName: spec.name,
      systemPrompt: EXPLORER_SYSTEM_PROMPT,
      userMessage: `## Task\n${config.task}\n\n## Your Focus\n${spec.focus}\n\nExplore the codebase from this angle. Call complete when done.`,
      model: workerModel,
      endpoint: workerEndpoint,
      sendWithTools: explorerSend,
      executor: createToolExecutor(),
      tools: getBaseToolSchemas().filter(t => ["read", "grep", "list", "glob", "bash", "complete"].includes(t.function.name)),
      workDir: config.workDir,
      maxTurns: config.explorerMaxTurns ?? 12,
      allowedTools: ["read", "grep", "list", "glob", "bash", "complete"],
      logLabel: `explorer-${spec.name.toLowerCase().replace(/\s+/g, "-")}`,
      evoObserve: { send: explorerSend, model: workerModel, intervalTurns: 2 },
      recoverySend: explorerSend,
      recoveryModel: workerModel,
    })
  })

  const explorerResults = await Promise.all(explorerPromises)
  const explorationSummaries: string[] = []

  for (let i = 0; i < explorerResults.length; i++) {
    const result = explorerResults[i]
    tokenCost += result.totalTokens
    inferenceCalls += result.turnsWithToolCalls
    const summary = result.finalText || `Explorer ${specs[i].name} produced no output.`
    explorationSummaries.push(`### ${specs[i].name}\n${summary}`)
    console.log(`  [planner] Explorer "${specs[i].name}": ${result.turns} turns, ${result.totalTokens} tokens`)
  }

  // Evo synthesizes exploration results into a plan
  const send = createSendWithTools({
    baseUrl: config.endpoint,
    model: config.model,
    maxTokens: 2048,
    timeoutMs: 120_000,
    temperature: config.temperature ?? 0,
  })

  const synthesisResult = await send(
    "Evo Synthesizer",
    [
      {
        role: "system",
        content: `I am a software architect. My explorers investigated this codebase from multiple angles. I now synthesize their findings into a single, concrete implementation plan.

I include specific file paths, function names, and line numbers from the findings.

My output:
1. Context: One sentence
2. Files to modify: Each with change description
3. Implementation steps: Numbered, referencing specifics from my explorers
4. Verification: Single command`,
      },
      {
        role: "user",
        content: `## Task\n${config.task}\n\n## Exploration Results\n${explorationSummaries.join("\n\n")}\n\nSynthesize into a concrete plan.`,
      },
    ],
    [WRITE_PLAN_TOOL],
    config.model,
  )

  inferenceCalls++
  tokenCost += synthesisResult.usage?.total_tokens ?? 0

  let planContent: string
  if (synthesisResult.toolCalls?.length) {
    try {
      const args = JSON.parse(synthesisResult.toolCalls[0].function.arguments) as StructuredPlan
      planContent = formatStructuredPlan(args)
    } catch {
      planContent = synthesisResult.text || "No plan generated."
    }
  } else {
    planContent = synthesisResult.text || "No plan generated."
  }

  const filePath = writePlanFile(planId, planContent)
  console.log(`  [planner] Arm C: Multi-explore plan synthesized (${inferenceCalls} calls, ${tokenCost} tokens)`)

  return {
    mode: "multi_explore",
    plan: planContent,
    filePath,
    style: "guidance",
    explorationSummaries,
    durationMs: Date.now() - start,
    tokenCost,
    inferenceCalls,
  }
}

// ── Scaffold Plan: structural guidance without knowledge seeding ──────

const SCAFFOLD_EXPLORER_SYSTEM_PROMPT = `I am in READ-ONLY exploration mode. I cannot modify files or run commands.

My job is to describe the STRUCTURE and INTERFACE CONTRACT of this codebase.

I report FIVE things:
1. FILE LAYOUT: What files exist, what needs to be created
2. EXPORT PATTERN: How modules export (destructured named exports vs default). I note the EXACT pattern.
3. REGISTRATION PATTERN: How modules wire together (imports, factory functions, dependency injection)
4. TEST INTERFACE: What the test requires — exact function names, constructor signatures, factory functions, and return shapes. I note how the test creates and uses each component.
5. DEPENDENCY ORDER: What depends on what — which file must be created first

I do NOT include code snippets. I describe interfaces and patterns in plain English.
I keep the whole report under 25 lines.
I focus especially on:
- What the TEST EXPECTS — this is the contract I must satisfy
- How components CONNECT — what is shared between them (e.g. a shared instance passed to multiple constructors)
- The WIRING PATTERN — how factory functions compose components together

When done, I call complete with my structural report.`

const SCAFFOLD_PLAN_SYSTEM_PROMPT = `I am writing a structural execution plan — a PROCESS guide, not a solution.

My rules:
- I NEVER include code snippets or implementation details
- I DO include interface contracts: what function names the test expects, what export patterns to use, what factory functions must return
- I DO describe WIRING PATTERNS: how components connect (e.g. "factory creates shared instance X, passes it to both A and B so A's output is visible to B")
- I describe the SEQUENCE of operations (create file, then edit file, then verify)
- I tell the worker to READ THE TEST FILE FIRST to understand the exact interface contract
- I tell the worker to STUDY existing patterns before writing new code
- I warn about common pitfalls (e.g. "use named exports if the test destructures", "verify each file creation before moving on")
- I keep the plan under 15 lines
- All steps are my own intentions in first person ("I create...", "I study...", "I verify...")

The plan helps the worker know WHAT INTERFACE to satisfy, WHERE to start, and HOW to verify.`

const WRITE_SCAFFOLD_TOOL: OpenAIToolDef = {
  type: "function",
  function: {
    name: "write_scaffold",
    description: "Write a structural execution plan with process guidance and interface contracts — no code",
    parameters: {
      type: "object",
      properties: {
        approach: {
          type: "string",
          description: "One sentence: what kind of task is this (create new module, modify existing, multi-file coordination)?",
        },
        interface_contract: {
          type: "string",
          description: "What the test expects and how components connect: export patterns, factory signatures, return shapes, and WIRING (how a shared instance connects components). Example: 'createPlayer creates shared EventEmitter, passes it to both Player and EventLogger, so Player events are recorded by Logger'",
        },
        steps: {
          type: "string",
          description: "JSON array of 3-6 process steps. Each step says WHAT to do and WHERE, never HOW in code. Example: 'I study the existing command files to learn the export pattern before writing my own'",
        },
        pitfalls: {
          type: "string",
          description: "JSON array of 1-3 common mistakes to avoid. Process warnings, not code corrections.",
        },
        verification: {
          type: "string",
          description: "How to verify the work is done correctly",
        },
      },
      required: ["approach", "interface_contract", "steps", "verification"],
    },
  },
}

function formatScaffoldPlan(args: { approach: string; interface_contract?: string; steps: string[] | string; pitfalls?: string[] | string; verification: string }): {
  markdown: string
  steps: string[]
} {
  const steps = normalizeStringArray(args.steps, 6).map(toFirstPersonChecklistStep)
  const pitfalls = normalizeStringArray(args.pitfalls, 3)
  const verification = toFirstPersonVerification(args.verification.trim())

  const lines: string[] = []
  lines.push("## My Approach")
  lines.push(toFirstPersonGoal(args.approach))
  lines.push("")
  if (args.interface_contract) {
    lines.push("## Interface Contract (what the test expects)")
    lines.push(args.interface_contract)
    lines.push("")
  }
  lines.push("## My Steps")
  for (let i = 0; i < steps.length; i++) {
    lines.push(`${i + 1}. ${steps[i]}`)
  }
  lines.push("")
  if (pitfalls.length > 0) {
    lines.push("## Pitfalls I Avoid")
    for (const p of pitfalls) lines.push(`- ${p}`)
    lines.push("")
  }
  lines.push("## How I Verify")
  lines.push(verification)

  return { markdown: lines.join("\n"), steps }
}

export async function generateScaffoldPlan(config: PlanningConfig): Promise<PlanArtifact> {
  const start = Date.now()
  const planId = `scaffold-${Date.now()}`
  let tokenCost = 0
  let inferenceCalls = 0

  const workerModel = config.workerModel || config.model
  const workerEndpoint = config.workerEndpoint || config.endpoint

  // Phase 1: Explorer reads the codebase structure (read-only, no commands)
  console.log(`  [planner] Scaffold: spawning structural explorer...`)

  const explorerSend = createSendWithTools({
    baseUrl: workerEndpoint,
    model: workerModel,
    temperature: 0,
    timeoutMs: 120_000,
  })

  const explorerResult = await runAgent({
    agentName: "Scaffold Explorer",
    systemPrompt: SCAFFOLD_EXPLORER_SYSTEM_PROMPT,
    userMessage: `## Task\n${config.task}\n\nExplore the codebase STRUCTURE and the TEST INTERFACE. Read the test file carefully to understand EXACTLY what it requires (function names, export patterns, factory signatures, return shapes). Report file layout, export patterns, and what the test expects. Do NOT include code snippets. Call complete when done.`,
    model: workerModel,
    endpoint: workerEndpoint,
    sendWithTools: explorerSend,
    executor: createToolExecutor(),
    tools: getBaseToolSchemas().filter(t => ["read", "grep", "list", "glob", "complete"].includes(t.function.name)),
    workDir: config.workDir,
    maxTurns: config.explorerMaxTurns ?? 6,
    allowedTools: ["read", "grep", "list", "glob", "complete"],
    logLabel: "scaffold-explorer",
    verbose: !!process.env.VERBOSE && process.env.VERBOSE === "1",
    // Evo observation steers the explorer — critical for models that drift (20B text-only, modify attempts)
    evoObserve: { send: explorerSend, model: workerModel, intervalTurns: 2 },
    // Adaptive recovery on tool failures
    recoverySend: explorerSend,
    recoveryModel: workerModel,
  })

  tokenCost += explorerResult.totalTokens
  inferenceCalls += explorerResult.turnsWithToolCalls

  const findings = explorerResult.finalText || "Explorer produced no findings."
  console.log(`  [planner] Scaffold explorer: ${explorerResult.turns} turns, ${explorerResult.totalTokens} tokens`)

  // Phase 2: Generate structural plan from findings (no codebase snapshot — just findings)
  const planSend = createSendWithTools({
    baseUrl: config.endpoint,
    model: config.model,
    maxTokens: 1024,
    timeoutMs: 120_000,
    temperature: config.temperature ?? 0,
  })

  const planResult = await planSend(
    "Scaffold Planner",
    [
      { role: "system", content: SCAFFOLD_PLAN_SYSTEM_PROMPT },
      {
        role: "user",
        content: `## Task\n${config.task}\n\n## Codebase Structure (from exploration)\n${findings}\n\nWrite a structural execution plan. Remember: NO code snippets, only process guidance.`,
      },
    ],
    [WRITE_SCAFFOLD_TOOL],
    config.model,
  )

  inferenceCalls++
  tokenCost += planResult.usage?.total_tokens ?? 0

  let artifact: { markdown: string; steps: string[] }
  if (planResult.toolCalls?.length) {
    try {
      const args = JSON.parse(planResult.toolCalls[0].function.arguments)
      artifact = formatScaffoldPlan(args)
    } catch {
      artifact = formatScaffoldPlan({
        approach: "Complete the task by following existing patterns.",
        steps: ["I study existing files to learn the pattern", "I create new files following that pattern", "I wire them together", "I verify with the test command"],
        verification: "Run the provided test command",
      })
    }
  } else {
    artifact = formatScaffoldPlan({
      approach: "Complete the task by following existing patterns.",
      steps: ["I study existing files to learn the pattern", "I create new files following that pattern", "I wire them together", "I verify with the test command"],
      verification: "Run the provided test command",
    })
  }

  const filePath = writePlanFile(planId, artifact.markdown)
  console.log(`  [planner] Scaffold plan generated (${inferenceCalls} calls, ${tokenCost} tokens)`)

  return {
    mode: "scaffold_plan",
    plan: artifact.markdown,
    filePath,
    style: "checklist",
    checklistSteps: artifact.steps,
    verification: "Run the provided test command",
    durationMs: Date.now() - start,
    tokenCost,
    inferenceCalls,
    explorationSummaries: [findings],
  }
}

// ── Dispatch ──────────────────────────────────────────────────────────

export async function generatePlan(config: PlanningConfig): Promise<PlanArtifact | null> {
  switch (config.mode) {
    case "none":        return null
    case "evo_plan":    return generateEvoPlan(config)
    case "worker_plan": return generateWorkerPlan(config)
    case "explore_then_checklist": return generateExploreThenChecklistPlan(config)
    case "explore_then_checklist_state": return generateExploreThenChecklistStatePlan(config)
    case "mini_checklist": return generateMiniChecklistPlan(config)
    case "mini_checklist_progress": return generateMiniChecklistPlan(config, "mini_checklist_progress")
    case "mini_checklist_progress_finish_gate": return generateMiniChecklistPlan(config, "mini_checklist_progress_finish_gate")
    case "mini_checklist_progress_modify_gate": return generateMiniChecklistPlan(config, "mini_checklist_progress_modify_gate")
    case "mini_checklist_progress_strict": return generateMiniChecklistPlan(config, "mini_checklist_progress_strict")
    case "multi_explore": return generateMultiExplorePlan(config)
    case "conveyor_planner": return generateConveyorPlannerConfig(config)
    case "scaffold_plan": return generateScaffoldPlan(config)
    default:            return null
  }
}

// ── Conveyor planner (no pre-generated plan — runtime phase gating only) ──

function generateConveyorPlannerConfig(config: PlanningConfig): PlanArtifact {
  const planId = `conveyor-${Date.now()}`
  const filePath = writePlanFile(planId, "(conveyor planner — no plan content, runtime phase gating only)")
  return {
    mode: "conveyor_planner",
    plan: "",
    filePath,
    style: "conveyor",
    conveyorPhase: true,
    durationMs: 0,
    tokenCost: 0,
    inferenceCalls: 0,
  }
}

// ── Plan injection into worker system prompt ──────────────────────────

const PERSPECTIVE = process.env.PERSPECTIVE || "1p"

export function injectPlanIntoPrompt(basePrompt: string, artifact: PlanArtifact, workDir?: string): string {
  let planText = artifact.plan

  // Validate and clean plan file paths against actual workspace
  if (workDir) {
    planText = validatePlanPaths(planText, workDir)
  }

  if (artifact.style === "conveyor") {
    return `${basePrompt}

## My Phases

I work in phases. I explore first, then register what I plan to do, then execute, then verify.

### Explore
I investigate the code to understand the problem. I read files, search for patterns, take notes.
I cannot modify anything yet — I must understand before I act.

### Register + Execute
When I know what needs to change, I call register(intent, file, reason) for each file I will modify.
I state what I intend to do and why. Then I modify only the registered files.
If I discover I need to modify another file, I call register first.

### Verify
I run the tests. If they pass, I finish. If they fail, I go back to exploring.
I do not just retry the same fix — I re-investigate to understand what went wrong.

### My Rules
- I cannot skip phases
- I must register before modifying
- I can only modify files I registered
- If verification fails, I loop back to explore with fresh eyes`
  }

  if (artifact.style === "checklist") {
    const plannedFiles = (artifact.plannedFiles || []).map(f => `- ${f}`).join("\n")
    const checklist = (artifact.checklistSteps || []).map((step, i) => `${i + 1}. ${step}`).join("\n")
    const expectedStates = (artifact.expectedStates || [])
      .map((state, i) => state ? `${i + 1}. ${state}` : "")
      .filter(Boolean)
      .join("\n")
    const progressRules: string[] = []
    if (artifact.trackProgress) {
      progressRules.push(`- Before I start a checklist step, I call checkoff(step="...", status="in_progress")`)
      progressRules.push(`- When I complete a checklist step, I call checkoff(step="...", status="done")`)
      progressRules.push(`- If a step is blocked, I call checkoff(step="...", status="blocked", note="...") before changing approach`)
    }
    if (artifact.requireInProgressStepForModify) {
      progressRules.push(`- I must mark a checklist step in progress before I modify code`)
    }
    if (artifact.requireCompletedStepForFinish) {
      progressRules.push(`- I cannot finish until I have marked at least one checklist step done`)
    }
    const progressRulesText = progressRules.length > 0 ? `\n${progressRules.join("\n")}` : ""

    // For scaffold plans, inject the full plan text which includes the interface contract
    if (artifact.mode === "scaffold_plan" && planText) {
      return `${basePrompt}

## My Execution Plan

${planText}

## My Execution Rules
- I read the test file FIRST to understand the exact interface before writing any code
- I create files in dependency order — foundation modules first
- I verify each file creation with the test before moving on
- If a test fails, I read my code AND the test carefully to understand the exact mismatch
- When an edit attempt fails, I use mode="rewrite" to replace the entire file instead
- I do not skip ahead after a failed test; I investigate the failure first
- If I am stuck on the same file for 3+ turns, I re-read the test to check what interface it actually expects${progressRulesText}`
    }

    return `${basePrompt}

## My Execution Checklist
I use this checklist as my working thread. I keep moving in this order unless the code clearly proves a step wrong.

### I Start Here
${plannedFiles || "- Investigate the most relevant files first"}

### I Do One Thing At A Time
${checklist || "1. Investigate the relevant files\n2. Make the change\n3. Verify it"}

### I Expect To See
${expectedStates || "1. I expect to learn what to change\n2. I expect the code behavior to improve\n3. I expect verification to pass"}

### I Verify
${artifact.verification || "Run the provided test command"}

## My Checklist Rules
- I start with the planned files before searching broadly
- I do not skip ahead after a failed test; I investigate the failure first
- After each step, I compare what I observe to the expected state before I move on
- After changing code, I re-run verification before finishing
- I keep the checklist short in my head: investigate, change, verify${progressRulesText}`
  }

  if (PERSPECTIVE === "2p") {
    return `${basePrompt}

## Implementation Plan (guidance — adapt as needed)
A planning phase produced the following plan. Use it as a starting guide, but trust what you observe over what the plan says.

${planText}

## How to Use This Plan
- The plan gives you a head start — use it to know WHERE to look and WHAT to fix
- If a file path in the plan doesn't work, use investigate to find the correct path
- If a step doesn't make sense after reading the code, skip it and do what's right
- Always verify by running the test command before finishing
- The plan is a suggestion, not a contract — what matters is passing the tests`
  }

  return `${basePrompt}

## My Implementation Plan (guidance — I adapt as needed)
I sketched this plan earlier. I use it as a starting guide, but I trust what I observe over what the plan says.

${planText}

## How I Use This Plan
- This plan gives me a head start — I use it to know WHERE to look and WHAT to fix
- If a file path in the plan doesn't work, I use investigate to find the correct path
- If a step doesn't make sense after reading the code, I skip it and do what's right
- I always verify by running the test command before finishing
- The plan is my starting sketch, not a contract — what matters is passing the tests`
}

// ── Plan path validation ──────────────────────────────────────────────

function validatePlanPaths(planText: string, workDir: string): string {
  const fs = require("fs")
  const path = require("path")

  // Find file paths in the plan (lines starting with "- " followed by a path-like string)
  return planText.replace(/^- (.+?\.\w+)(.*)/gm, (match, filePath, rest) => {
    const cleanPath = filePath.replace(/`/g, "").trim()
    const resolved = path.resolve(workDir, cleanPath)

    if (fs.existsSync(resolved)) {
      return `- ${cleanPath}${rest}`
    }

    // Try to find the file anywhere in the workspace
    const basename = path.basename(cleanPath)
    try {
      const { execSync } = require("child_process")
      const found = execSync(
        `find ${JSON.stringify(workDir)} -name ${JSON.stringify(basename)} -not -path '*/node_modules/*' 2>/dev/null | head -1`,
        { encoding: "utf-8" },
      ).trim()

      if (found) {
        const relativePath = path.relative(workDir, found)
        return `- ${relativePath}${rest} [corrected from: ${cleanPath}]`
      }
    } catch { /* skip */ }

    return `- ${cleanPath}${rest} [WARNING: file not found — use investigate to locate]`
  })
}
