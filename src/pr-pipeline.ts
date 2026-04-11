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
 *   BLIND_MODE    1 (default) = model sees ONLY the original issue description (not PR body, diff, or comments)
 *                 0 = include the diff (original behavior)
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

// Blind mode: model sees ONLY the original issue description — no PR body, no diff,
// no reviewer comments. The specialist review is the ONLY feedback mechanism.
// The original diff is fetched post-hoc for comparison only.
const BLIND_MODE = process.env.BLIND_MODE !== "0"  // ON by default

// ── Types ───────────────────────────────────────────────────────────────

interface LinkedIssue {
  number: number
  title: string
  body: string
}

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
  linkedIssues: LinkedIssue[]
}

interface SpecialistFeedback {
  role: string
  findings: string[]
  severity: "info" | "warning" | "critical"
}

interface PipelineResult {
  pr: { owner: string; repo: string; number: number; title: string }
  workDir: string
  blindMode: boolean
  phases: {
    intake: { ok: boolean; error?: string }
    workspace: { ok: boolean; branch?: string; error?: string }
    plan: { ok: boolean; plan?: string; tokenCost?: number; error?: string }
    execution: { ok: boolean; turns?: number; tokens?: number; filesModified?: boolean; error?: string }
    review: { loops: number; feedback: SpecialistFeedback[]; error?: string }
    comparison: {
      originalDiff?: string
      newDiff?: string
      sameFiles?: boolean
      sameFix?: boolean
      reviewerFeedbackCaught?: boolean
      error?: string
    }
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

/** Extract #NNNN references from text, excluding the PR's own number */
function extractIssueRefs(text: string, excludeNumber: number): number[] {
  const refs = new Set<number>()
  const re = /#(\d+)/g
  let match
  while ((match = re.exec(text)) !== null) {
    const num = parseInt(match[1], 10)
    if (num !== excludeNumber && num > 0) refs.add(num)
  }
  return Array.from(refs)
}

/** Fetch original issue descriptions from GitHub */
async function fetchLinkedIssues(owner: string, repo: string, issueNumbers: number[]): Promise<LinkedIssue[]> {
  const fullRepo = `${owner}/${repo}`
  const issues: LinkedIssue[] = []

  for (const num of issueNumbers) {
    try {
      const json = shellExec(`gh issue view ${num} -R ${fullRepo} --json body,title`)
      const parsed = JSON.parse(json)
      if (parsed.title || parsed.body) {
        issues.push({ number: num, title: parsed.title || "", body: parsed.body || "" })
        console.log(`  [intake] Fetched linked issue #${num}: ${(parsed.title || "").slice(0, 80)}`)
      }
    } catch (err: any) {
      console.log(`  [intake] Could not fetch issue #${num}: ${err.message?.slice(0, 80)}`)
    }
  }

  return issues
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

  // Fetch linked issues (from PR body and title)
  const issueRefs = extractIssueRefs((pr.body || "") + " " + (pr.title || ""), number)
  let linkedIssues: LinkedIssue[] = []
  if (issueRefs.length > 0) {
    console.log(`  [intake] Found issue references: ${issueRefs.map(n => `#${n}`).join(", ")}`)
    linkedIssues = await fetchLinkedIssues(owner, repo, issueRefs)
  }

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
    linkedIssues,
  }

  console.log(`  [intake] Title: ${intake.title}`)
  console.log(`  [intake] State: ${intake.state}`)
  console.log(`  [intake] Files changed: ${filesChanged.length}`)
  console.log(`  [intake] Review comments: ${reviewComments.length}`)
  console.log(`  [intake] Issue comments: ${issueComments.length}`)
  console.log(`  [intake] Linked issues: ${linkedIssues.length}`)
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

// ── Blind-mode helpers ─────────────────────────────────────────────────

/** Strip solution hints from PR description — remove diff blocks, code blocks showing fixes,
 * and inline text that reveals the exact code change. The model should only know WHAT the
 * problem is, not HOW it was fixed. */
function sanitizeDescription(body: string, budget: number): string {
  let text = body

  // Remove diff code blocks (```diff ... ```)
  text = text.replace(/```diff[\s\S]*?```/g, "[diff removed for blind evaluation]")

  // Remove code blocks that look like they contain the fix
  // Keep code blocks that describe the problem (e.g., error output, repro steps)
  text = text.replace(/```[\w]*\n[\s\S]*?```/g, (match) => {
    // Keep blocks that look like error output / stack traces
    if (/error|traceback|exception|stack|panic|segfault/i.test(match)) return match
    // Strip blocks that look like they show the fix
    if (/object\.create|__proto__|parsed\s*=|changed.*to|replaced.*with/i.test(match)) {
      return "[code block removed for blind evaluation]"
    }
    // Keep short blocks (likely examples), strip long ones (likely solutions)
    return match.length < 200 ? match : "[code block removed for blind evaluation]"
  })

  // Remove inline sentences that describe the exact fix (e.g., "by changing X to Y")
  // These patterns reveal the solution:
  text = text.replace(/by\s+chang(ing|ed?)\s+`[^`]+`\s+to\s+`[^`]+`/gi, "[specific fix details removed]")
  text = text.replace(/chang(ing|ed?)\s+`[^`]+`\s+to\s+`[^`]+`/gi, "[specific fix details removed]")
  text = text.replace(/replac(ing|ed?)\s+`[^`]+`\s+with\s+`[^`]+`/gi, "[specific fix details removed]")
  text = text.replace(/from\s+`[^`]+`\s+to\s+`[^`]+`/gi, "[specific change removed]")

  // Remove "Summary of changes" sections that describe exactly what was changed
  text = text.replace(/[-*]\s*Use\s+`[^`]+`\s+(?:for|instead\s+of|in\s+place\s+of)\s+[^\n]*/gi, "[change detail removed]")

  // Remove auto-generated "Description" sections from bots that spell out the fix
  text = text.replace(/##\s*Description[\s\S]*?(?=##|\z)/gi, (match) => {
    // Keep it only if it describes the problem, not the solution
    if (/summary\s+of\s+changes|reasoning/i.test(match)) {
      return "[auto-generated description removed for blind evaluation]"
    }
    return match
  })

  // Truncate to budget
  if (text.length > budget) {
    text = text.slice(0, budget) + "\n... (truncated)"
  }

  return text
}

/** Collect all feedback from reviews and issue comments */
function collectFeedback(intake: PRIntake) {
  return [
    ...intake.reviewComments.map(c => ({
      ...c,
      source: "review" as const,
    })),
    ...intake.issueComments.map(c => ({
      ...c,
      source: "comment" as const,
    })),
  ]
}

function buildTaskFromPR(intake: PRIntake): string {
  const lines: string[] = []

  if (BLIND_MODE) {
    // ── Truly blind mode: ONLY the original issue description ──
    // No PR body, no diff, no reviewer comments, no file hints.
    // The model discovers everything through exploration.
    // The specialist review (post-execution) is the ONLY feedback mechanism.
    lines.push(`I need to fix a reported issue in ${intake.owner}/${intake.repo}.`)
    lines.push("")

    // Use the linked issue body if available — this is the original bug report
    // with no solution information. Falls back to PR title only if no issues found.
    if (intake.linkedIssues.length > 0) {
      for (const issue of intake.linkedIssues) {
        lines.push(`## Issue #${issue.number}: ${issue.title}`)
        lines.push("")
        // The original issue body — no sanitization needed because this is
        // the reporter's description, not the PR author's solution.
        // However, some issues (like #7538) may contain the fix — strip those.
        const issueBody = sanitizeDescription(issue.body, CONTEXT_BUDGET)
        lines.push(issueBody)
        lines.push("")
      }
    } else {
      // No linked issues found — fall back to just the PR title (no body)
      lines.push("## Issue Description")
      lines.push(intake.title)
      lines.push("")
    }

    lines.push("## My Approach")
    lines.push("I will:")
    lines.push("1. Explore the codebase to understand the project structure")
    lines.push("2. Find the code related to the reported issue")
    lines.push("3. Investigate the root cause by reading the relevant source files")
    lines.push("4. Develop a fix that addresses the issue without introducing side effects")
    lines.push("5. Verify the changes make sense")
    lines.push("6. Call complete when done")

    return lines.join("\n")
  }

  // ── Non-blind mode: original behavior with diff ──
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
  const allFeedback = collectFeedback(intake)

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

const BLIND_SYSTEM_PROMPT = `I am a senior developer fixing a reported issue. I have only the original issue description. I need to explore the codebase, find the root cause, and implement a clean fix entirely on my own.

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
- I start by exploring the project structure to understand what I am working with
- I investigate thoroughly before making changes -- read the code, understand the patterns
- I make minimal, targeted changes -- I do not refactor unrelated code
- I verify my changes make sense before calling complete
- I work step by step: explore, understand, then fix`

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

  let systemPrompt = BLIND_MODE ? BLIND_SYSTEM_PROMPT : BASE_SYSTEM_PROMPT
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

  // Capture diff — commit model's changes and diff against the pre-execution state
  let diff = ""
  try {
    // Stage and commit model's changes so we have a clean diff
    const wd = JSON.stringify(workDir)
    try {
      shellExec(`git -C ${wd} add -A`)
      shellExec(`git -C ${wd} diff --cached --stat`)
      shellExec(`git -C ${wd} commit -m "pipeline execution" --allow-empty`)
    } catch {
      // May fail if nothing changed
    }
    // Diff between the branch creation point and now
    diff = shellExec(`git -C ${wd} diff HEAD~1..HEAD 2>/dev/null || git -C ${wd} diff HEAD`)
    if (!diff) {
      diff = shellExec(`git -C ${wd} status --short`)
    }
  } catch {
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
    maxTokens: 1024,
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

      let text = result.content || result.text || ""
      // Strip thinking blocks from qwen models
      text = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim()

      const findings = text
        .split("\n")
        .filter((line: string) => line.trim().startsWith("-") || line.trim().startsWith("*"))
        .map((line: string) => line.replace(/^[-*]\s*/, "").trim())
        .filter((line: string) => line.length > 5)

      const noIssues = /no\s+(issues?|problems?|concerns?)\s+found/i.test(text) ||
        /looks?\s+good/i.test(text) || /no\s+actionable/i.test(text) ||
        (findings.length === 0 && text.length < 200)
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

interface ComparisonResult {
  sameFiles: boolean
  sameFix: boolean
  origFiles: string[]
  newFiles: string[]
  commonFiles: string[]
}

function printComparison(originalDiff: string, newDiff: string): ComparisonResult {
  console.log("\n" + "=".repeat(70))
  console.log(BLIND_MODE
    ? "  BLIND COMPARISON: Original PR vs Model's Independent Fix"
    : "  COMPARISON: Original PR vs Pipeline Output")
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

  const sameFiles = common.length > 0 && common.length === origFiles.length && origFiles.length === newFiles.length

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

  // Blind mode: check if the fix is structurally similar
  // A simple heuristic: extract the actual changed lines (not file headers)
  const origChangedLines = extractChangedLines(originalDiff)
  const newChangedLines = extractChangedLines(newDiff)
  const sameFix = origChangedLines.length > 0 && newChangedLines.length > 0 &&
    origChangedLines.some(ol => newChangedLines.some(nl =>
      // Fuzzy match: same key tokens appear
      ol.split(/\s+/).filter(t => t.length > 3).some(token =>
        nl.includes(token)
      )
    ))

  if (BLIND_MODE) {
    console.log(`\n  Blind evaluation:`)
    console.log(`    Same files modified: ${sameFiles ? "YES" : "NO"}`)
    console.log(`    Structurally similar fix: ${sameFix ? "LIKELY" : "DIFFERENT APPROACH"}`)
    if (sameFix) {
      console.log(`    >> Model arrived at a similar fix independently`)
    } else if (common.length > 0) {
      console.log(`    >> Model touched the right files but took a different approach`)
    } else {
      console.log(`    >> Model fixed different files entirely`)
    }
  }

  console.log("\n" + "=".repeat(70))
  return { sameFiles, sameFix, origFiles, newFiles, commonFiles: common }
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

/** Extract actual changed lines (additions) from a diff, excluding headers and lock files */
function extractChangedLines(diff: string): string[] {
  const lines = diff.split("\n")
  const changed: string[] = []
  let currentFile = ""

  for (const line of lines) {
    if (line.startsWith("diff --git")) {
      currentFile = line.split(" b/")[1] || ""
    }
    // Skip lock files and package-lock changes
    if (currentFile.includes("lock.json") || currentFile.includes("lock.yaml")) continue
    // Collect additions (not headers)
    if (line.startsWith("+") && !line.startsWith("+++") && line.trim().length > 3) {
      changed.push(line.slice(1).trim())
    }
  }
  return changed
}

// ── Post-hoc blind comparison with LLM ─────────────────────────────────

async function runBlindComparisonReview(
  originalDiff: string,
  newDiff: string,
  task: string,
  specialistFeedback: SpecialistFeedback[],
  originalReviewComments: PRIntake["reviewComments"],
): Promise<{ reviewerFeedbackCaught: boolean; analysis: string }> {
  const send = createSendWithTools({
    baseUrl: ENDPOINT,
    model: MODEL,
    temperature: 0,
    maxTokens: 1500,
    timeoutMs: 300000,
  })

  console.log(`  [blind-compare] Running post-hoc analysis...`)

  // Format original reviewer concerns
  const originalConcerns = originalReviewComments
    .map(c => `- ${c.author}: ${c.body.slice(0, 300)}`)
    .join("\n")

  // Format specialist findings
  const specialistFindings = specialistFeedback
    .filter(f => f.severity !== "info")
    .map(f => `[${f.role}]: ${f.findings.join("; ")}`)
    .join("\n")

  try {
    const result = await send(
      "blind comparator",
      [
        {
          role: "system",
          content: `I am analyzing a blind re-implementation experiment. A model was given an issue description WITHOUT seeing the original PR's diff, and asked to independently fix the bug. I need to compare the original fix with the model's fix and check whether the specialist review caught the same issues that real reviewers caught.`,
        },
        {
          role: "user",
          content: `## Task Description\n${task.slice(0, 1000)}\n\n## Original PR Diff\n\`\`\`diff\n${originalDiff.slice(0, 3000)}\n\`\`\`\n\n## Model's Blind Fix Diff\n\`\`\`diff\n${newDiff.slice(0, 3000)}\n\`\`\`\n\n## Original Reviewer Comments\n${originalConcerns || "(none)"}\n\n## Specialist Review Findings\n${specialistFindings || "(none)"}\n\nAnalyze:\n1. Did the model find and fix the SAME root cause as the original PR?\n2. Did the specialist review catch the SAME issues that original reviewers flagged?\n3. How does the model's approach compare? (same, different-but-valid, wrong)\n\nBe concrete and specific. Reference actual code changes.`,
        },
      ],
      [],
      MODEL,
    )

    let text = result.content || result.text || ""
    // Strip thinking blocks from qwen models
    text = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim()
    const reviewerFeedbackCaught = /caught|identified|flagged|found.*same|similar.*issue/i.test(text)

    console.log(`  [blind-compare] Reviewer feedback caught: ${reviewerFeedbackCaught ? "YES" : "NO"}`)
    return { reviewerFeedbackCaught, analysis: text }
  } catch (err: any) {
    console.log(`  [blind-compare] Analysis failed: ${err.message?.slice(0, 80)}`)
    return { reviewerFeedbackCaught: false, analysis: `Failed: ${err.message?.slice(0, 200)}` }
  }
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

    // Reset workspace to pre-execution baseline for each iteration
    if (i > 0) {
      try {
        shellExec(`git -C ${JSON.stringify(workDir)} reset --hard HEAD~1 2>/dev/null || git -C ${JSON.stringify(workDir)} checkout . && git -C ${JSON.stringify(workDir)} clean -fd`)
      } catch {
        try {
          shellExec(`git -C ${JSON.stringify(workDir)} checkout .`)
          shellExec(`git -C ${JSON.stringify(workDir)} clean -fd`)
        } catch {
          // Continue anyway
        }
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
  console.log(`  Blind mode: ${BLIND_MODE ? "ON (no diff, model must find the bug)" : "OFF"}`)
  if (ITERATE_UNTIL_IMPROVEMENT) {
    console.log(`  Performance mode: iterate up to ${MAX_ITERATIONS}x`)
  }

  const pipelineResult: PipelineResult = {
    pr: { owner, repo, number, title: "" },
    workDir: "",
    blindMode: BLIND_MODE,
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
  const initialDiff = execDiff  // Keep the first execution diff for comparison
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
      // Refine: reset workspace to pre-execution baseline and re-execute with feedback
      console.log(`  [review] Refining with specialist feedback...`)
      try {
        // Reset to the baseline commit (before any pipeline execution commits)
        // The branch was created from the base ref, so HEAD~N may be the execution commit
        shellExec(`git -C ${JSON.stringify(workDir)} reset --hard HEAD~1 2>/dev/null || git -C ${JSON.stringify(workDir)} checkout . && git -C ${JSON.stringify(workDir)} clean -fd`)
      } catch {
        // Fallback to simple checkout
        try {
          shellExec(`git -C ${JSON.stringify(workDir)} checkout .`)
          shellExec(`git -C ${JSON.stringify(workDir)} clean -fd`)
        } catch {
          // Continue anyway
        }
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
  // Use the initial execution diff for comparison (has the actual code fix),
  // not the refinement diff which may be polluted by npm install artifacts
  const bestDiffForComparison = initialDiff || currentDiff
  if (intake.diff && bestDiffForComparison) {
    const cmpResult = printComparison(intake.diff, bestDiffForComparison)
    pipelineResult.phases.comparison = {
      originalDiff: intake.diff.slice(0, 5000),
      newDiff: bestDiffForComparison.slice(0, 5000),
      sameFiles: cmpResult.sameFiles,
      sameFix: cmpResult.sameFix,
    }

    // In blind mode, run post-hoc LLM comparison
    if (BLIND_MODE && bestDiffForComparison.length > 10) {
      try {
        const blindResult = await runBlindComparisonReview(
          intake.diff,
          bestDiffForComparison,
          task,
          allFeedback,
          intake.reviewComments,
        )
        pipelineResult.phases.comparison.reviewerFeedbackCaught = blindResult.reviewerFeedbackCaught
        console.log(`\n--- Blind Comparison Analysis ---`)
        console.log(blindResult.analysis)
      } catch (err: any) {
        console.log(`  [blind-compare] Failed: ${err.message?.slice(0, 100)}`)
      }
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
