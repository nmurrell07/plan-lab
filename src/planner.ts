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

export type PlanningMode = "none" | "evo_plan" | "worker_plan" | "multi_explore"

export interface PlanArtifact {
  mode: PlanningMode
  plan: string
  filePath: string
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
      if (e.name.startsWith(".") || e.name === "node_modules" || e.name === "test") continue
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

const EVO_PLAN_SYSTEM_PROMPT = `You are a software architect planning an implementation task. You will be given a task description and a codebase snapshot.

Your plan must include:
1. **Context**: One sentence on what is being changed and why
2. **Files to modify**: Each file path with a one-line description
3. **Implementation steps**: Numbered steps referencing specific files and functions
4. **Verification**: The single command that confirms the change works

Rules:
- Be concrete: file paths, function names, what to change
- Follow existing patterns in the codebase
- Keep the plan under 50 lines`

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
    durationMs: Date.now() - start,
    tokenCost,
    inferenceCalls,
  }
}

// ── Arm B: Worker plan-then-execute ───────────────────────────────────

const WORKER_PLAN_SYSTEM_PROMPT = `You are a development agent in READ-ONLY planning mode.

You MUST NOT modify any files. You can only read, search, and explore.

Your job:
1. Thoroughly explore the codebase to understand existing patterns
2. Identify files, functions, and utilities relevant to the task
3. Design a concrete implementation approach

When done, call complete with your full plan as the summary.

Rules:
- Be specific: file paths, function names, code patterns you found
- Include the verification command
- Keep the plan under 50 lines`

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
        content: "You review implementation plans. Given a worker's exploration findings, produce a refined, concrete plan. Fix vague steps, add missing file paths, remove prose. Output ONLY the refined plan.",
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
    explorationSummaries: [rawPlan],
    durationMs: Date.now() - start,
    tokenCost,
    inferenceCalls,
  }
}

// ── Arm C: Multi-agent exploration + Evo synthesis ────────────────────

const EXPLORER_SYSTEM_PROMPT = `You are a codebase exploration agent in READ-ONLY mode.

You MUST NOT modify any files. You can only read, search, and explore.

Your specific focus area will be provided. Explore thoroughly, then call complete with your findings.

Report format:
- File paths you found relevant (with line numbers)
- Functions and utilities that could be reused
- Patterns and conventions you observed
- Concerns or dependencies you noticed`

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
        content: `You are a software architect. Multiple exploration agents investigated a codebase. Synthesize their findings into a single, concrete implementation plan.

Include specific file paths, function names, and line numbers from the findings.

Output:
1. Context: One sentence
2. Files to modify: Each with change description
3. Implementation steps: Numbered, referencing specifics from explorers
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
    explorationSummaries,
    durationMs: Date.now() - start,
    tokenCost,
    inferenceCalls,
  }
}

// ── Dispatch ──────────────────────────────────────────────────────────

export async function generatePlan(config: PlanningConfig): Promise<PlanArtifact | null> {
  switch (config.mode) {
    case "none":        return null
    case "evo_plan":    return generateEvoPlan(config)
    case "worker_plan": return generateWorkerPlan(config)
    case "multi_explore": return generateMultiExplorePlan(config)
    default:            return null
  }
}

// ── Plan injection into worker system prompt ──────────────────────────

export function injectPlanIntoPrompt(basePrompt: string, artifact: PlanArtifact, workDir?: string): string {
  let planText = artifact.plan

  // Validate and clean plan file paths against actual workspace
  if (workDir) {
    planText = validatePlanPaths(planText, workDir)
  }

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
