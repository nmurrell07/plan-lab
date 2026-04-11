#!/usr/bin/env bun
/**
 * PR Pipeline — Read a GitHub PR, scaffold a plan from review feedback,
 * execute via local model, run specialist review, refine, and compare.
 *
 * Usage:
 *   bun run src/pr-pipeline.ts <owner/repo> <pr-number>
 *   bun run src/pr-pipeline.ts nmurrell07/aiohttp 12
 *
 * Or with a full URL:
 *   bun run src/pr-pipeline.ts https://github.com/nmurrell07/aiohttp/pull/12
 *
 * Environment:
 *   MODEL         Model ID (default: qwen3.5-27b)
 *   ENDPOINT      LM Studio endpoint (default: http://192.168.50.117:1234)
 *   MAX_TURNS     Max agent turns per execution pass (default: 25)
 *   REVIEW_LOOPS  Max specialist review/refine loops (default: 2)
 *   VERBOSE       Set to 1 for full transcript
 *   PLAN_MODE     scaffold_plan | evo_plan | none (default: scaffold_plan)
 *   ITERATE_UNTIL_IMPROVEMENT  Set to 1 for performance iteration mode
 *   MAX_ITERATIONS  Max performance iterations (default: 5)
 */

import { execSync } from "child_process"
import fs from "fs/promises"
import path from "path"
import { createSendWithTools } from "./client"
import { createToolExecutor, createEnhancedToolExecutor, getBaseToolSchemas } from "./tools"
import { runAgent } from "./runner"
import { generateScaffoldPlan, generateEvoPlan, injectPlanIntoPrompt } from "./planner"
import type { PlanArtifact } from "./planner"
import { createEmptyLedger } from "./ledger"

// ── Config ──────────────────────────────────────────────────────────────

const MODEL = process.env.MODEL || "qwen3.5-27b"
const ENDPOINT = process.env.ENDPOINT || "http://192.168.50.117:1234"
const MAX_TURNS = parseInt(process.env.MAX_TURNS || "25", 10)
const REVIEW_LOOPS = parseInt(process.env.REVIEW_LOOPS || "2", 10)

// Context budget — how much of the PR content to include based on model size
// Small models (9B/16K ctx) need aggressive truncation; large models can see more
const CONTEXT_BUDGET = parseInt(process.env.CONTEXT_BUDGET || "0", 10) || guessContextBudget(MODEL)

function guessContextBudget(model: string): number {
  const m = model.toLowerCase()
  if (m.includes("9b") || m.includes("8b") || m.includes("7b") || m.includes("4b")) return 2000  // ~16K ctx: tiny budget
  if (m.includes("14b") || m.includes("20b")) return 4000  // ~32K ctx: moderate
  return 6000  // 27B+: full budget
}
const VERBOSE = process.env.VERBOSE === "1"
const PLAN_MODE = process.env.PLAN_MODE || "scaffold_plan"
const ITERATE_UNTIL_IMPROVEMENT = process.env.ITERATE_UNTIL_IMPROVEMENT === "1"
const MAX_ITERATIONS = parseInt(process.env.MAX_ITERATIONS || "5", 10)

// ── Types ───────────────────────────────────────────────────────────────

interface PRIntake {
  owner: string
  repo: string
  number: number
  title: string
  body: string
  baseRef: string
  headRef: string
  state: string
  diff: string
  filesChanged: Array<{ path: string; additions: number; deletions: number }>
  reviewComments: Array<{ author: string; body: string; path?: string; line?: number }>
  issueComments: Array<{ author: string; body: string }>
}

interface SpecialistFeedback {
  role: string
  findings: string[]
  severity: "info" | "warning" | "critical"
}

interface PipelineResult {
  pr: { owner: string; repo: string; number: number; title: string }
  workDir: string
  phases: {
    intake: { ok: boolean; error?: string }
    workspace: { ok: boolean; branch?: string; error?: string }
    plan: { ok: boolean; plan?: string; tokenCost?: number; error?: string }
    execution: { ok: boolean; turns?: number; tokens?: number; filesModified?: boolean; error?: string }
    review: { loops: number; feedback: SpecialistFeedback[]; error?: string }
    comparison: { originalDiff?: string; newDiff?: string; error?: string }
  }
  iterations?: number
  durationSec: number
}

// ── Phase 1: Intake ─────────────────────────────────────────────────────

function shellExec(cmd: string, opts?: { cwd?: string }): string {
  try {
    return execSync(cmd, {
      encoding: "utf-8",
      timeout: 30000,
      cwd: opts?.cwd,
      stdio: ["pipe", "pipe", "pipe"],
    }).trim()
  } catch (err: any) {
    const output = (err.stdout || "") + (err.stderr || "")
    throw new Error(`Command failed: ${cmd.slice(0, 80)}\n${output.slice(0, 500)}`)
  }
}

function parsePRInput(args: string[]): { owner: string; repo: string; number: number } {
  // Handle: https://github.com/owner/repo/pull/123
  if (args.length === 1 && args[0].includes("github.com")) {
    const match = args[0].match(/github\.com\/([^/]+)\/([^/]+)\/pull\/(\d+)/)
    if (match) return { owner: match[1], repo: match[2], number: parseInt(match[3], 10) }
    throw new Error(`Could not parse GitHub PR URL: ${args[0]}`)
  }

  // Handle: owner/repo 123
  if (args.length === 2) {
    const parts = args[0].split("/")
    if (parts.length === 2) {
      return { owner: parts[0], repo: parts[1], number: parseInt(args[1], 10) }
    }
  }

  // Handle: owner repo 123
  if (args.length === 3) {
    return { owner: args[0], repo: args[1], number: parseInt(args[2], 10) }
  }

  throw new Error(
    "Usage: bun run src/pr-pipeline.ts <owner/repo> <pr-number>\n" +
    "   or: bun run src/pr-pipeline.ts <github-pr-url>"
  )
}

async function intakePR(owner: string, repo: string, number: number): Promise<PRIntake> {
  const fullRepo = `${owner}/${repo}`
  console.log(`  [intake] Reading PR #${number} from ${fullRepo}...`)

  // Fetch PR metadata
  const prJson = shellExec(
    `gh pr view ${number} -R ${fullRepo} --json title,body,baseRefName,headRefName,state,files,reviews,comments`
  )
  const pr = JSON.parse(prJson)

  // Fetch the diff
  let diff = ""
  try {
    diff = shellExec(`gh pr diff ${number} -R ${fullRepo}`)
  } catch (err: any) {
    console.log(`  [intake] Warning: could not fetch diff: ${err.message?.slice(0, 80)}`)
  }

  // Fetch review comments (inline code review comments)
  let reviewComments: PRIntake["reviewComments"] = []
  try {
    const reviewJson = shellExec(
      `gh api repos/${fullRepo}/pulls/${number}/comments --jq '[.[] | {author: .user.login, body: .body, path: .path, line: .line}]'`
    )
    reviewComments = JSON.parse(reviewJson || "[]")
  } catch {
    // No inline review comments
  }

  // Also grab review-level comments (approve/request changes with body)
  try {
    const reviews = pr.reviews || []
    for (const review of reviews) {
      if (review.body?.trim()) {
        reviewComments.push({
          author: review.author?.login || "reviewer",
          body: review.body,
        })
      }
    }
  } catch {
    // Skip
  }

  // Issue-level comments
  const issueComments: PRIntake["issueComments"] = (pr.comments || []).map((c: any) => ({
    author: c.author?.login || "unknown",
    body: c.body || "",
  }))

  const filesChanged = (pr.files || []).map((f: any) => ({
    path: f.path,
    additions: f.additions || 0,
    deletions: f.deletions || 0,
  }))

  const intake: PRIntake = {
    owner,
    repo,
    number,
    title: pr.title || "",
    body: pr.body || "",
    baseRef: pr.baseRefName || "main",
    headRef: pr.headRefName || "",
    state: pr.state || "unknown",
    diff,
    filesChanged,
    reviewComments,
    issueComments,
  }

  console.log(`  [intake] Title: ${intake.title}`)
  console.log(`  [intake] State: ${intake.state}`)
  console.log(`  [intake] Files changed: ${filesChanged.length}`)
  console.log(`  [intake] Review comments: ${reviewComments.length}`)
  console.log(`  [intake] Issue comments: ${issueComments.length}`)
  console.log(`  [intake] Diff size: ${diff.length} chars`)

  return intake
}

// ── Phase 2: Workspace Setup ────────────────────────────────────────────

async function setupWorkspace(intake: PRIntake): Promise<string> {
  const workDir = `/tmp/pr-pipeline-${intake.repo}-${intake.number}-${Date.now()}`
  console.log(`  [workspace] Creating workspace: ${workDir}`)

  // Clone the repo (shallow, from the base ref)
  const fullRepo = `${intake.owner}/${intake.repo}`
  try {
    shellExec(
      `gh repo clone ${fullRepo} ${workDir} -- --depth=50 --branch=${intake.baseRef}`,
    )
    console.log(`  [workspace] Cloned ${fullRepo} at ${intake.baseRef}`)
  } catch (err: any) {
    // If the repo is already available locally, try to use it
    const localPath = `/Users/virtualmachine/${intake.repo}`
    try {
      const stat = await fs.stat(localPath)
      if (stat.isDirectory()) {
        console.log(`  [workspace] Using local repo at ${localPath}`)
        shellExec(`git -C ${JSON.stringify(localPath)} checkout ${intake.baseRef} 2>/dev/null || true`)
        shellExec(`git -C ${JSON.stringify(localPath)} clean -fd 2>/dev/null || true`)
        shellExec(`rsync -a --exclude=node_modules --exclude=.git/objects/pack ${JSON.stringify(localPath + "/")} ${JSON.stringify(workDir + "/")}`)
        // Initialize git in the workspace
        shellExec(`cd ${JSON.stringify(workDir)} && git init && git add -A && git commit -m "baseline from ${intake.baseRef}" --allow-empty`)
      }
    } catch {
      throw new Error(`Could not clone ${fullRepo} or find local copy: ${err.message?.slice(0, 200)}`)
    }
  }

  // Create a working branch
  const branchName = `pr-pipeline/${intake.number}-reimpl-${Date.now()}`
  try {
    shellExec(`git -C ${JSON.stringify(workDir)} checkout -b ${branchName}`)
    console.log(`  [workspace] Branch: ${branchName}`)
  } catch (err: any) {
    console.log(`  [workspace] Warning: could not create branch: ${err.message?.slice(0, 80)}`)
  }

  return workDir
}

// ── Phase 3: Plan Generation ────────────────────────────────────────────

function buildTaskFromPR(intake: PRIntake): string {
  const lines: string[] = []

  lines.push(`I need to re-implement the changes from PR #${intake.number}: "${intake.title}".`)
  lines.push("")

  // What the PR was about
  if (intake.body) {
    lines.push("## Original PR Description")
    lines.push(intake.body.slice(0, Math.min(2000, CONTEXT_BUDGET)))
    lines.push("")
  }

  // What files were changed
  if (intake.filesChanged.length > 0) {
    lines.push("## Files That Need Changes")
    for (const f of intake.filesChanged.slice(0, 20)) {
      lines.push(`- ${f.path} (+${f.additions}/-${f.deletions})`)
    }
    lines.push("")
  }

  // Reviewer feedback as constraints
  const allFeedback = [
    ...intake.reviewComments.map(c => ({
      ...c,
      source: "review" as const,
    })),
    ...intake.issueComments.map(c => ({
      ...c,
      source: "comment" as const,
    })),
  ]

  if (allFeedback.length > 0) {
    const maxFeedback = CONTEXT_BUDGET < 3000 ? 3 : 10
    const maxFeedbackLen = CONTEXT_BUDGET < 3000 ? 200 : 500
    lines.push("## Reviewer Feedback (these are constraints I must address)")
    for (const fb of allFeedback.slice(0, maxFeedback)) {
      const prefix = fb.source === "review" && "path" in fb && fb.path
        ? `[${fb.path}${("line" in fb && fb.line) ? `:${fb.line}` : ""}] `
        : ""
      lines.push(`- ${prefix}${fb.author}: ${fb.body.slice(0, maxFeedbackLen)}`)
    }
    lines.push("")
  }

  // Original diff as reference (truncated based on context budget)
  if (intake.diff) {
    const diffBudget = CONTEXT_BUDGET < 3000 ? 1500 : CONTEXT_BUDGET
    lines.push("## Reference Diff (what the original PR changed)")
    lines.push("I should understand the intent, not copy it verbatim.")
    lines.push("```diff")
    lines.push(intake.diff.slice(0, diffBudget))
    if (intake.diff.length > diffBudget) lines.push("... (truncated)")
    lines.push("```")
    lines.push("")
  }

  lines.push("## My Approach")
  lines.push("I will:")
  lines.push("1. Read the relevant files to understand the current state")
  lines.push("2. Understand the intent of the changes from the PR description and diff")
  lines.push("3. Incorporate all reviewer feedback as requirements")
  lines.push("4. Implement the changes, addressing reviewer concerns")
  lines.push("5. Verify the changes compile and make sense")
  lines.push("6. Call complete when done")

  return lines.join("\n")
}

async function generatePlan(
  task: string,
  workDir: string,
): Promise<PlanArtifact | null> {
  if (PLAN_MODE === "none") return null

  console.log(`\n  [planning] Generating ${PLAN_MODE} plan...`)

  const config = {
    mode: PLAN_MODE as "scaffold_plan" | "evo_plan",
    task,
    workDir,
    model: MODEL,
    endpoint: ENDPOINT,
    workerModel: MODEL,
    workerEndpoint: ENDPOINT,
    temperature: 0,
  }

  if (PLAN_MODE === "scaffold_plan") {
    return generateScaffoldPlan(config)
  } else if (PLAN_MODE === "evo_plan") {
    return generateEvoPlan(config)
  }

  return null
}

// ── Phase 4: Execution ──────────────────────────────────────────────────

const BASE_SYSTEM_PROMPT = `I am a senior developer re-implementing a PR. I have the original PR's intent and reviewer feedback as constraints.

I have these tools:
- read(path) -- read a file, returns numbered lines
- write(path, content) -- create a new file with full content
- edit(path, old_string, new_string) -- find exact text in a file and replace it
- grep(pattern) -- search for a regex pattern across the codebase
- bash(command) -- run a shell command (already in the project directory)
- list(path) -- list directory contents
- glob(pattern) -- find files by name pattern
- complete(summary) -- signal I am done

My rules:
- All paths are RELATIVE to the project root (e.g. "src/index.ts"), never absolute
- For edit: old_string must be an EXACT copy of text currently in the file
- For write: I provide the COMPLETE file content
- Shell commands run in the project directory. I do NOT prefix with cd.
- I address ALL reviewer feedback points as requirements
- I verify my changes work before calling complete
- I work step by step: read first, understand the pattern, then make changes`

async function executePass(
  task: string,
  workDir: string,
  plan: PlanArtifact | null,
  additionalConstraints?: string,
): Promise<{ result: import("./runner").RunResult; diff: string }> {
  const send = createSendWithTools({
    baseUrl: ENDPOINT,
    model: MODEL,
    temperature: 0,
    timeoutMs: 300000,
  })

  let systemPrompt = BASE_SYSTEM_PROMPT
  if (plan) {
    systemPrompt = injectPlanIntoPrompt(systemPrompt, plan, workDir)
  }

  let userMessage = `Working directory: ${workDir}\n\n${task}`
  if (additionalConstraints) {
    userMessage += `\n\n## Additional Constraints from Specialist Review\n${additionalConstraints}`
  }

  const ledger = createEmptyLedger(`pr-pipeline`)
  const executor = createEnhancedToolExecutor({ sendWithTools: send, model: MODEL })
  const tools = getBaseToolSchemas()

  const result = await runAgent({
    agentName: "PR Dev",
    systemPrompt,
    userMessage,
    model: MODEL,
    endpoint: ENDPOINT,
    sendWithTools: send,
    executor,
    tools,
    workDir,
    maxTurns: MAX_TURNS,
    allowedTools: ["read", "write", "edit", "grep", "list", "glob", "bash", "complete"],
    logLabel: `pr-pipeline/exec`,
    notes: [],
    interceptors: false,
    failureLedger: ledger,
    compaction: true,
    verbose: VERBOSE,
    recoverySend: PLAN_MODE !== "none" ? send : undefined,
    recoveryModel: PLAN_MODE !== "none" ? MODEL : undefined,
    evoObserve: PLAN_MODE !== "none" ? { send, model: MODEL, intervalTurns: 3 } : undefined,
  })

  // Capture diff
  let diff = ""
  try {
    diff = shellExec(`git -C ${JSON.stringify(workDir)} diff HEAD`, { cwd: workDir })
    if (!diff) {
      // Maybe changes are staged or committed
      diff = shellExec(`git -C ${JSON.stringify(workDir)} diff --cached HEAD`, { cwd: workDir })
    }
    if (!diff) {
      // Try diff against the initial commit
      diff = shellExec(`git -C ${JSON.stringify(workDir)} diff $(git -C ${JSON.stringify(workDir)} rev-list --max-parents=0 HEAD) HEAD`, { cwd: workDir })
    }
  } catch {
    // Fallback: show file changes
    try {
      diff = shellExec(`git -C ${JSON.stringify(workDir)} status --short`)
    } catch {
      diff = "(could not capture diff)"
    }
  }

  return { result, diff }
}

// ── Phase 5: Specialist Review ──────────────────────────────────────────

const SPECIALIST_PERSONAS: Record<string, string> = {
  architect: `I am a Software Architect reviewing a diff. I focus on:
- Contracts between modules: are interfaces respected?
- Data flow: is data passed correctly between components?
- Interface design: are APIs clean and consistent?
- Dependency management: are imports correct and minimal?
- Pattern consistency: does the code follow existing patterns in the codebase?

I produce CONCRETE, ACTIONABLE findings. Each finding includes the file and what needs to change.
I do NOT give generic advice. I only flag real problems I see in the diff.
If the code looks good, I say so briefly.`,

  qa: `I am a QA Engineer reviewing a diff. I focus on:
- Edge cases: what inputs could break this?
- Error handling: are errors caught and reported?
- Test coverage: are the changes tested? What tests are missing?
- Boundary conditions: off-by-one, null/undefined, empty arrays
- Regression risk: could this change break existing functionality?

I produce CONCRETE, ACTIONABLE findings. Each finding includes what could go wrong and how to fix it.
I do NOT give generic advice. I only flag real risks I see in the diff.
If the code handles edge cases well, I say so briefly.`,
}

async function runSpecialistReview(
  diff: string,
  task: string,
  workDir: string,
): Promise<SpecialistFeedback[]> {
  if (!diff || diff.length < 10) {
    return [{ role: "all", findings: ["No meaningful diff to review."], severity: "warning" }]
  }

  const send = createSendWithTools({
    baseUrl: ENDPOINT,
    model: MODEL,
    temperature: 0,
    timeoutMs: 300000,
  })

  const feedback: SpecialistFeedback[] = []

  for (const [role, persona] of Object.entries(SPECIALIST_PERSONAS)) {
    console.log(`  [review] Running ${role} review...`)

    try {
      const result = await send(
        `${role} reviewer`,
        [
          { role: "system", content: persona },
          {
            role: "user",
            content: `## Task Context\n${task.slice(0, 1500)}\n\n## Diff to Review\n\`\`\`diff\n${diff.slice(0, 8000)}\n\`\`\`\n\nReview this diff. List concrete findings as bullet points. If it looks good, say "No issues found."`,
          },
        ],
        [],
        MODEL,
      )

      const text = result.content || result.text || ""
      const findings = text
        .split("\n")
        .filter((line: string) => line.trim().startsWith("-") || line.trim().startsWith("*"))
        .map((line: string) => line.replace(/^[-*]\s*/, "").trim())
        .filter((line: string) => line.length > 5)

      const noIssues = /no\s+(issues?|problems?|concerns?)\s+found/i.test(text)
      const severity = noIssues ? "info" as const
        : findings.some((f: string) => /critical|security|vulnerability|crash|data loss/i.test(f)) ? "critical" as const
        : "warning" as const

      feedback.push({
        role,
        findings: findings.length > 0 ? findings : [noIssues ? "No issues found." : text.slice(0, 500)],
        severity,
      })

      console.log(`  [review] ${role}: ${findings.length} findings (${severity})`)
    } catch (err: any) {
      const errMsg = err?.message || String(err) || "unknown error"
      console.log(`  [review] ${role} review failed: ${errMsg.slice(0, 80)}`)
      feedback.push({ role, findings: [`Review failed: ${errMsg.slice(0, 200)}`], severity: "info" })
    }
  }

  return feedback
}

function formatFeedbackAsConstraints(feedback: SpecialistFeedback[]): string | null {
  const actionable = feedback.filter(f => f.severity !== "info")
  if (actionable.length === 0) return null

  const lines: string[] = []
  for (const fb of actionable) {
    lines.push(`### ${fb.role.charAt(0).toUpperCase() + fb.role.slice(1)} feedback (${fb.severity}):`)
    for (const finding of fb.findings) {
      lines.push(`- ${finding}`)
    }
    lines.push("")
  }
  return lines.join("\n")
}

// ── Phase 6: Comparison ─────────────────────────────────────────────────

function printComparison(originalDiff: string, newDiff: string): void {
  console.log("\n" + "=".repeat(70))
  console.log("  COMPARISON: Original PR vs Pipeline Output")
  console.log("=".repeat(70))

  const origStats = diffStats(originalDiff)
  const newStats = diffStats(newDiff)

  console.log(`\n  Original PR diff:`)
  console.log(`    Files: ${origStats.files}`)
  console.log(`    Additions: +${origStats.additions}`)
  console.log(`    Deletions: -${origStats.deletions}`)
  console.log(`    Size: ${originalDiff.length} chars`)

  console.log(`\n  Pipeline output diff:`)
  console.log(`    Files: ${newStats.files}`)
  console.log(`    Additions: +${newStats.additions}`)
  console.log(`    Deletions: -${newStats.deletions}`)
  console.log(`    Size: ${newDiff.length} chars`)

  // Show common files
  const origFiles = extractDiffFiles(originalDiff)
  const newFiles = extractDiffFiles(newDiff)
  const common = origFiles.filter(f => newFiles.includes(f))
  const onlyOrig = origFiles.filter(f => !newFiles.includes(f))
  const onlyNew = newFiles.filter(f => !origFiles.includes(f))

  if (common.length > 0) {
    console.log(`\n  Common files (${common.length}):`)
    for (const f of common) console.log(`    - ${f}`)
  }
  if (onlyOrig.length > 0) {
    console.log(`\n  Only in original (${onlyOrig.length}):`)
    for (const f of onlyOrig) console.log(`    - ${f}`)
  }
  if (onlyNew.length > 0) {
    console.log(`\n  Only in pipeline output (${onlyNew.length}):`)
    for (const f of onlyNew) console.log(`    - ${f}`)
  }

  console.log("\n" + "=".repeat(70))
}

function diffStats(diff: string): { files: number; additions: number; deletions: number } {
  const lines = diff.split("\n")
  const files = new Set(lines.filter(l => l.startsWith("diff --git")).map(l => l.split(" b/")[1])).size
  const additions = lines.filter(l => l.startsWith("+") && !l.startsWith("+++")).length
  const deletions = lines.filter(l => l.startsWith("-") && !l.startsWith("---")).length
  return { files, additions, deletions }
}

function extractDiffFiles(diff: string): string[] {
  return diff.split("\n")
    .filter(l => l.startsWith("diff --git"))
    .map(l => l.split(" b/")[1])
    .filter(Boolean)
}

// ── Performance Iteration Mode ──────────────────────────────────────────

async function iterateUntilImprovement(
  task: string,
  workDir: string,
  plan: PlanArtifact | null,
): Promise<{ iterations: number; bestDiff: string; improved: boolean }> {
  console.log(`\n  [iterate] Performance iteration mode: max ${MAX_ITERATIONS} attempts`)

  let bestDiff = ""
  let iterations = 0
  let previousFindings: string[] = []

  for (let i = 0; i < MAX_ITERATIONS; i++) {
    iterations = i + 1
    console.log(`\n  [iterate] === Iteration ${iterations} ===`)

    // Reset workspace for each iteration (keep .git)
    if (i > 0) {
      try {
        shellExec(`git -C ${JSON.stringify(workDir)} checkout .`)
        shellExec(`git -C ${JSON.stringify(workDir)} clean -fd`)
      } catch {
        // Continue anyway
      }
    }

    // Build constraints from previous learnings
    let constraints: string | undefined
    if (previousFindings.length > 0) {
      constraints = "## Learnings from Previous Attempts\n" +
        previousFindings.map((f, i) => `${i + 1}. ${f}`).join("\n") +
        "\n\nI must address these points in this attempt."
    }

    // Execute
    const { result, diff } = await executePass(task, workDir, plan, constraints)
    console.log(`  [iterate] Execution: ${result.turns} turns, ${result.filesModified ? "files modified" : "no changes"}`)

    if (!result.filesModified) {
      console.log(`  [iterate] No files modified, trying again...`)
      previousFindings.push("I failed to modify any files -- I need to actually make changes.")
      continue
    }

    // Review
    const feedback = await runSpecialistReview(diff, task, workDir)
    const criticals = feedback.filter(f => f.severity === "critical")
    const warnings = feedback.filter(f => f.severity === "warning")

    if (criticals.length === 0 && warnings.length === 0) {
      console.log(`  [iterate] Clean review on iteration ${iterations}`)
      bestDiff = diff
      return { iterations, bestDiff, improved: true }
    }

    // Collect findings for next iteration
    const newFindings = feedback
      .filter(f => f.severity !== "info")
      .flatMap(f => f.findings.map(finding => `[${f.role}] ${finding}`))
    previousFindings.push(...newFindings)

    bestDiff = diff
    console.log(`  [iterate] ${criticals.length} critical, ${warnings.length} warning findings. Iterating...`)
  }

  console.log(`  [iterate] Reached max iterations (${MAX_ITERATIONS})`)
  return { iterations, bestDiff, improved: false }
}

// ── Main ────────────────────────────────────────────────────────────────

async function main() {
  const startTime = Date.now()
  const args = process.argv.slice(2)

  if (args.length === 0) {
    console.error(
      "Usage: bun run src/pr-pipeline.ts <owner/repo> <pr-number>\n" +
      "   or: bun run src/pr-pipeline.ts <github-pr-url>"
    )
    process.exit(1)
  }

  const { owner, repo, number } = parsePRInput(args)

  console.log("+====================================================================+")
  console.log("|  PR Pipeline                                                       |")
  console.log("+====================================================================+")
  console.log(`  PR: ${owner}/${repo}#${number}`)
  console.log(`  Model: ${MODEL}`)
  console.log(`  Endpoint: ${ENDPOINT}`)
  console.log(`  Plan mode: ${PLAN_MODE}`)
  console.log(`  Max turns: ${MAX_TURNS}`)
  console.log(`  Review loops: ${REVIEW_LOOPS}`)
  console.log(`  Context budget: ${CONTEXT_BUDGET} chars`)
  if (ITERATE_UNTIL_IMPROVEMENT) {
    console.log(`  Performance mode: iterate up to ${MAX_ITERATIONS}x`)
  }

  const pipelineResult: PipelineResult = {
    pr: { owner, repo, number, title: "" },
    workDir: "",
    phases: {
      intake: { ok: false },
      workspace: { ok: false },
      plan: { ok: false },
      execution: { ok: false },
      review: { loops: 0, feedback: [] },
      comparison: {},
    },
    durationSec: 0,
  }

  // ── Phase 1: Intake ──
  console.log("\n--- Phase 1: Intake ---")
  let intake: PRIntake
  try {
    intake = await intakePR(owner, repo, number)
    pipelineResult.pr.title = intake.title
    pipelineResult.phases.intake = { ok: true }
  } catch (err: any) {
    console.error(`  [intake] FAILED: ${err.message}`)
    pipelineResult.phases.intake = { ok: false, error: err.message }
    await saveResult(pipelineResult, startTime)
    process.exit(1)
  }

  // ── Phase 2: Workspace Setup ──
  console.log("\n--- Phase 2: Workspace Setup ---")
  let workDir: string
  try {
    workDir = await setupWorkspace(intake)
    pipelineResult.workDir = workDir
    pipelineResult.phases.workspace = { ok: true, branch: `pr-pipeline/${number}` }
  } catch (err: any) {
    console.error(`  [workspace] FAILED: ${err.message}`)
    pipelineResult.phases.workspace = { ok: false, error: err.message }
    await saveResult(pipelineResult, startTime)
    process.exit(1)
  }

  // ── Phase 3: Plan Generation ──
  console.log("\n--- Phase 3: Plan Generation ---")
  const task = buildTaskFromPR(intake)
  let plan: PlanArtifact | null = null
  try {
    plan = await generatePlan(task, workDir)
    if (plan) {
      console.log(`  [planning] Plan generated (${plan.inferenceCalls} calls, ${plan.tokenCost} tokens)`)
      if (VERBOSE) console.log(`  [planning] Content:\n${plan.plan}`)
      pipelineResult.phases.plan = { ok: true, plan: plan.plan, tokenCost: plan.tokenCost }
    } else {
      console.log(`  [planning] No plan (mode: ${PLAN_MODE})`)
      pipelineResult.phases.plan = { ok: true }
    }
  } catch (err: any) {
    console.error(`  [planning] FAILED: ${err.message?.slice(0, 200)}`)
    pipelineResult.phases.plan = { ok: false, error: err.message }
    // Continue without plan
  }

  // ── Performance iteration mode ──
  if (ITERATE_UNTIL_IMPROVEMENT) {
    console.log("\n--- Performance Iteration Mode ---")
    const iterResult = await iterateUntilImprovement(task, workDir, plan)
    pipelineResult.iterations = iterResult.iterations

    console.log("\n--- Phase 6: Comparison ---")
    if (intake.diff && iterResult.bestDiff) {
      printComparison(intake.diff, iterResult.bestDiff)
      pipelineResult.phases.comparison = {
        originalDiff: intake.diff.slice(0, 5000),
        newDiff: iterResult.bestDiff.slice(0, 5000),
      }
    }

    pipelineResult.phases.execution = { ok: iterResult.improved, turns: 0, filesModified: true }
    await saveResult(pipelineResult, startTime)
    return
  }

  // ── Phase 4: Execution ──
  console.log("\n--- Phase 4: Execution ---")
  let execDiff = ""
  try {
    const { result: execResult, diff } = await executePass(task, workDir, plan)
    execDiff = diff
    console.log(`  [execution] Turns: ${execResult.turns} (${execResult.turnsWithToolCalls} with tools)`)
    console.log(`  [execution] Tokens: ${execResult.totalTokens}`)
    console.log(`  [execution] Files modified: ${execResult.filesModified}`)
    pipelineResult.phases.execution = {
      ok: true,
      turns: execResult.turns,
      tokens: execResult.totalTokens,
      filesModified: execResult.filesModified,
    }
  } catch (err: any) {
    console.error(`  [execution] FAILED: ${err.message?.slice(0, 200)}`)
    pipelineResult.phases.execution = { ok: false, error: err.message }
    await saveResult(pipelineResult, startTime)
    process.exit(1)
  }

  // ── Phase 5: Specialist Review + Refinement Loop ──
  console.log("\n--- Phase 5: Specialist Review ---")
  let currentDiff = execDiff
  let allFeedback: SpecialistFeedback[] = []

  for (let loop = 0; loop < REVIEW_LOOPS; loop++) {
    console.log(`\n  [review] Loop ${loop + 1}/${REVIEW_LOOPS}`)

    const feedback = await runSpecialistReview(currentDiff, task, workDir)
    allFeedback.push(...feedback)
    pipelineResult.phases.review.loops = loop + 1

    const constraints = formatFeedbackAsConstraints(feedback)
    if (!constraints) {
      console.log(`  [review] No actionable feedback -- review passed.`)
      break
    }

    if (loop < REVIEW_LOOPS - 1) {
      // Refine: reset and re-execute with feedback
      console.log(`  [review] Refining with specialist feedback...`)
      try {
        shellExec(`git -C ${JSON.stringify(workDir)} checkout .`)
        shellExec(`git -C ${JSON.stringify(workDir)} clean -fd`)
      } catch {
        // Continue
      }

      try {
        const { diff } = await executePass(task, workDir, plan, constraints)
        currentDiff = diff
      } catch (err: any) {
        console.log(`  [review] Refinement execution failed: ${err.message?.slice(0, 100)}`)
        break
      }
    }
  }

  pipelineResult.phases.review.feedback = allFeedback

  // ── Phase 6: Comparison ──
  console.log("\n--- Phase 6: Comparison ---")
  if (intake.diff && currentDiff) {
    printComparison(intake.diff, currentDiff)
    pipelineResult.phases.comparison = {
      originalDiff: intake.diff.slice(0, 5000),
      newDiff: currentDiff.slice(0, 5000),
    }
  } else {
    console.log("  [comparison] Cannot compare -- missing diff(s)")
    pipelineResult.phases.comparison = { error: "Missing diff data" }
  }

  await saveResult(pipelineResult, startTime)

  // Print feedback summary
  if (allFeedback.length > 0) {
    console.log("\n--- Specialist Feedback Summary ---")
    for (const fb of allFeedback) {
      console.log(`  [${fb.role}] (${fb.severity}):`)
      for (const finding of fb.findings) {
        console.log(`    - ${finding.slice(0, 120)}`)
      }
    }
  }

  console.log(`\n  [pipeline] Workspace preserved at: ${workDir}`)
  console.log(`  [pipeline] Duration: ${Math.round((Date.now() - startTime) / 1000)}s`)
}

async function saveResult(result: PipelineResult, startTime: number): Promise<void> {
  result.durationSec = Math.round((Date.now() - startTime) / 1000)
  const resultFile = path.join(
    "/Users/virtualmachine/plan-lab/results",
    `pr-pipeline-${result.pr.repo}-${result.pr.number}-${Date.now()}.json`,
  )
  try {
    await fs.writeFile(resultFile, JSON.stringify(result, null, 2))
    console.log(`  [pipeline] Result saved to: ${resultFile}`)
  } catch (err: any) {
    console.error(`  [pipeline] Failed to save result: ${err.message?.slice(0, 100)}`)
  }
}

await main()
