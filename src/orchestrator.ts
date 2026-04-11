#!/usr/bin/env bun
/**
 * Orchestrator — LLM-driven workflow coordinator.
 *
 * Replaces the hardcoded pr-pipeline with a ReAct agent that has meta-tools
 * for managing workers, specialists, and the overall workflow.
 *
 * The orchestrator is a capable model (27B/397B) that makes decisions about:
 * - Which issues to fetch
 * - When to spawn coding workers (smaller models: 9B/20B)
 * - When to run specialist reviews
 * - Whether to refine based on feedback
 * - When the work is complete
 *
 * Usage:
 *   bun run src/orchestrator.ts <owner/repo> <issue-number>
 *
 * Environment:
 *   ORCH_MODEL      Orchestrator model (default: qwen3.5-27b)
 *   ORCH_ENDPOINT   Orchestrator endpoint (default: http://192.168.50.117:1234)
 *   WORKER_MODEL    Worker model (default: qwen/qwen3.5-9b)
 *   WORKER_ENDPOINT Worker endpoint (default: http://192.168.50.206:1234)
 *   MAX_TURNS       Max orchestrator turns (default: 30)
 *   WORKER_TURNS    Max worker turns per spawn (default: 25)
 *   VERBOSE         Set to 1 for full transcript
 */

import { execSync } from "child_process"
import fs from "fs/promises"
import path from "path"
import { createSendWithTools } from "./client"
import type { SendWithToolsFn, OpenAIToolDef, ChatMessage } from "./client"
import { createToolExecutor, createEnhancedToolExecutor, getBaseToolSchemas } from "./tools"
import type { ToolCall, ToolCallResult, ToolExecutor, ExecutionContext } from "./tools"
import { runAgent } from "./runner"
import type { RunResult } from "./runner"
import { generateScaffoldPlan, injectPlanIntoPrompt } from "./planner"
import { createEmptyLedger } from "./ledger"

// ── Config ──────────────────────────────────────────────────────────────

const ORCH_MODEL = process.env.ORCH_MODEL || process.env.MODEL || "qwen3.5-27b"
const ORCH_ENDPOINT = process.env.ORCH_ENDPOINT || process.env.ENDPOINT || "http://192.168.50.117:1234"
const WORKER_MODEL = process.env.WORKER_MODEL || "qwen/qwen3.5-9b"
const WORKER_ENDPOINT = process.env.WORKER_ENDPOINT || "http://192.168.50.206:1234"
const MAX_TURNS = parseInt(process.env.MAX_TURNS || "30", 10)
const WORKER_TURNS = parseInt(process.env.WORKER_TURNS || "25", 10)
const VERBOSE = process.env.VERBOSE === "1"

// ── Orchestrator State ──────────────────────────────────────────────────

interface OrchestratorState {
  workDir: string | null
  repo: string | null
  issueBody: string | null
  workerRuns: Array<{
    model: string
    turns: number
    filesModified: boolean
    diff: string
    summary: string
  }>
  specialistReviews: Array<{
    role: string
    findings: string[]
    severity: string
  }>
  finalSummary: string | null
}

const state: OrchestratorState = {
  workDir: null,
  repo: null,
  issueBody: null,
  workerRuns: [],
  specialistReviews: [],
  finalSummary: null,
}

// ── Shell helper ────────────────────────────────────────────────────────

function shellExec(cmd: string, opts?: { cwd?: string }): string {
  try {
    return execSync(cmd, {
      encoding: "utf-8",
      timeout: 60000,
      cwd: opts?.cwd,
      stdio: ["pipe", "pipe", "pipe"],
    }).trim()
  } catch (err: any) {
    const output = (err.stdout || "") + (err.stderr || "")
    throw new Error(`Command failed: ${cmd.slice(0, 100)}\n${output.slice(0, 500)}`)
  }
}

// ── Orchestrator Tool Implementations ───────────────────────────────────

async function toolReadIssue(args: Record<string, string>): Promise<ToolCallResult> {
  const repo = args.repo || ""
  const num = args.number || ""
  if (!repo || !num) {
    return { tool: "read_issue", success: false, output: "", error: "Missing repo or number. Usage: read_issue(repo='owner/repo', number='123')" }
  }

  try {
    const json = shellExec(`gh issue view ${num} -R ${repo} --json body,title,state,labels`)
    const parsed = JSON.parse(json)
    state.issueBody = parsed.body || ""
    state.repo = repo

    const labels = (parsed.labels || []).map((l: any) => l.name).join(", ")
    const output = `## Issue #${num}: ${parsed.title || "(no title)"}\n\nState: ${parsed.state || "unknown"}\nLabels: ${labels || "none"}\n\n${parsed.body || "(no body)"}`
    return { tool: "read_issue", success: true, output: output.slice(0, 6000) }
  } catch (err: any) {
    return { tool: "read_issue", success: false, output: "", error: err.message?.slice(0, 300) }
  }
}

async function toolSetupWorkspace(args: Record<string, string>): Promise<ToolCallResult> {
  const repo = args.repo || state.repo || ""
  const branch = args.branch || "main"
  if (!repo) {
    return { tool: "setup_workspace", success: false, output: "", error: "Missing repo. Usage: setup_workspace(repo='owner/repo', branch='main')" }
  }

  const workDir = `/tmp/orch-${repo.replace("/", "-")}-${Date.now()}`

  try {
    shellExec(`gh repo clone ${repo} ${workDir} -- --depth=50 --branch=${branch}`)
    const branchName = `orch/fix-${Date.now()}`
    shellExec(`git -C ${JSON.stringify(workDir)} checkout -b ${branchName}`)
    state.workDir = workDir
    state.repo = repo

    // Get a directory listing for context
    const listing = shellExec(`find ${JSON.stringify(workDir)} -maxdepth 2 -not -path '*/node_modules/*' -not -path '*/.git/*' -type f | head -40 | sort`)
    const relativeListing = listing.replace(new RegExp(workDir + "/?", "g"), "")

    return {
      tool: "setup_workspace",
      success: true,
      output: `Workspace ready at: ${workDir}\nBranch: ${branchName}\nBase: ${branch}\n\nTop-level files:\n${relativeListing}`,
    }
  } catch (err: any) {
    return { tool: "setup_workspace", success: false, output: "", error: err.message?.slice(0, 300) }
  }
}

async function toolSpawnWorker(args: Record<string, string>): Promise<ToolCallResult> {
  const task = args.task || ""
  const workDir = args.work_dir || state.workDir || ""
  if (!task) {
    return { tool: "spawn_worker", success: false, output: "", error: "Missing task. Usage: spawn_worker(task='description of what to fix')" }
  }
  if (!workDir) {
    return { tool: "spawn_worker", success: false, output: "", error: "No workspace set up. Call setup_workspace first." }
  }

  // Reset workspace to baseline before each worker run
  if (state.workerRuns.length > 0) {
    try {
      shellExec(`git -C ${JSON.stringify(workDir)} checkout . && git -C ${JSON.stringify(workDir)} clean -fd`)
    } catch {
      // Continue anyway
    }
  }

  console.log(`\n  [orchestrator] Spawning worker: ${WORKER_MODEL}`)
  console.log(`  [orchestrator] Task: ${task.slice(0, 120)}...`)

  const workerSend = createSendWithTools({
    baseUrl: WORKER_ENDPOINT,
    model: WORKER_MODEL,
    temperature: 0,
    timeoutMs: 300000,
  })

  const workerSystemPrompt = `I am a developer fixing a bug. I have only the issue description and must explore the codebase to understand the problem and implement a fix.

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
- Shell commands run in the project directory. I do NOT prefix with cd.
- I explore first, then understand, then fix
- I make minimal, targeted changes
- I call complete when done`

  // Optionally generate a scaffold plan for the worker
  let systemPrompt = workerSystemPrompt
  try {
    const plan = await generateScaffoldPlan({
      mode: "scaffold_plan",
      task,
      workDir,
      model: WORKER_MODEL,
      endpoint: WORKER_ENDPOINT,
      workerModel: WORKER_MODEL,
      workerEndpoint: WORKER_ENDPOINT,
      temperature: 0,
    })
    if (plan) {
      systemPrompt = injectPlanIntoPrompt(workerSystemPrompt, plan, workDir)
      console.log(`  [orchestrator] Worker plan generated (${plan.tokenCost} tokens)`)
    }
  } catch (err: any) {
    console.log(`  [orchestrator] Worker planning failed (continuing without plan): ${err.message?.slice(0, 80)}`)
  }

  const ledger = createEmptyLedger("orch-worker")
  const executor = createEnhancedToolExecutor({ sendWithTools: workerSend, model: WORKER_MODEL })
  const tools = getBaseToolSchemas()

  try {
    const result = await runAgent({
      agentName: "Worker",
      systemPrompt,
      userMessage: `Working directory: ${workDir}\n\n${task}`,
      model: WORKER_MODEL,
      endpoint: WORKER_ENDPOINT,
      sendWithTools: workerSend,
      executor,
      tools,
      workDir,
      maxTurns: WORKER_TURNS,
      allowedTools: ["read", "write", "edit", "grep", "list", "glob", "bash", "complete"],
      logLabel: `orch-worker/${WORKER_MODEL.split("/").pop()}`,
      notes: [],
      interceptors: false,
      failureLedger: ledger,
      compaction: true,
      verbose: VERBOSE,
      recoverySend: workerSend,
      recoveryModel: WORKER_MODEL,
      evoObserve: { send: workerSend, model: WORKER_MODEL, intervalTurns: 3 },
    })

    // Capture diff
    let diff = ""
    try {
      const wd = JSON.stringify(workDir)
      shellExec(`git -C ${wd} add -A`)
      shellExec(`git -C ${wd} commit -m "worker execution" --allow-empty`)
      diff = shellExec(`git -C ${wd} diff HEAD~1..HEAD 2>/dev/null || git -C ${wd} diff HEAD`)
    } catch {
      try {
        diff = shellExec(`git -C ${JSON.stringify(workDir)} status --short`)
      } catch {
        diff = "(could not capture diff)"
      }
    }

    const runSummary = {
      model: WORKER_MODEL,
      turns: result.turns,
      filesModified: result.filesModified,
      diff,
      summary: result.finalText || `Completed in ${result.turns} turns, ${result.turnsWithToolCalls} with tools`,
    }
    state.workerRuns.push(runSummary)

    const output = [
      `## Worker Execution Complete`,
      `Model: ${WORKER_MODEL}`,
      `Turns: ${result.turns} (${result.turnsWithToolCalls} with tools)`,
      `Tokens: ${result.totalTokens}`,
      `Files modified: ${result.filesModified}`,
      `Files touched: ${result.touchedFiles.join(", ") || "none"}`,
      `Loops detected: ${result.loopsDetected}`,
      `Adaptive recoveries: ${result.adaptiveRecoveries}`,
      ``,
      `## Diff`,
      diff.length > 4000 ? diff.slice(0, 4000) + "\n...(truncated)" : diff,
    ].join("\n")

    return { tool: "spawn_worker", success: result.filesModified, output }
  } catch (err: any) {
    return { tool: "spawn_worker", success: false, output: "", error: `Worker failed: ${err.message?.slice(0, 300)}` }
  }
}

async function toolRunSpecialist(args: Record<string, string>): Promise<ToolCallResult> {
  const diff = args.diff || state.workerRuns[state.workerRuns.length - 1]?.diff || ""
  const role = args.role || "architect"

  if (!diff || diff.length < 10) {
    return { tool: "run_specialist", success: false, output: "", error: "No diff to review. Run spawn_worker first." }
  }

  const personas: Record<string, string> = {
    architect: `I am a Software Architect reviewing a diff. I focus on:
- Contracts between modules: are interfaces respected?
- Data flow: is data passed correctly between components?
- Interface design: are APIs clean and consistent?
- Pattern consistency: does the code follow existing patterns?

I produce CONCRETE, ACTIONABLE findings. Each finding includes the file and what needs to change.
I do NOT give generic advice. I only flag real problems I see in the diff.
If the code looks good, I say so briefly.`,

    qa: `I am a QA Engineer reviewing a diff. I focus on:
- Edge cases: what inputs could break this?
- Error handling: are errors caught and reported?
- Boundary conditions: off-by-one, null/undefined, empty arrays
- Regression risk: could this change break existing functionality?

I produce CONCRETE, ACTIONABLE findings. Each finding includes what could go wrong and how to fix it.
I do NOT give generic advice. I only flag real risks I see in the diff.
If the code handles edge cases well, I say so briefly.`,
  }

  const persona = personas[role] || personas.architect

  const send = createSendWithTools({
    baseUrl: ORCH_ENDPOINT,
    model: ORCH_MODEL,
    temperature: 0,
    maxTokens: 1024,
    timeoutMs: 300000,
  })

  try {
    const result = await send(
      `${role} reviewer`,
      [
        { role: "system", content: persona },
        {
          role: "user",
          content: `## Diff to Review\n\`\`\`diff\n${diff.slice(0, 6000)}\n\`\`\`\n\nReview this diff. List concrete findings as bullet points. If it looks good, say "No issues found."`,
        },
      ],
      [],
      ORCH_MODEL,
    )

    let text = result.content || result.text || ""
    // Strip thinking blocks
    text = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim()

    const findings = text
      .split("\n")
      .filter((line: string) => line.trim().startsWith("-") || line.trim().startsWith("*"))
      .map((line: string) => line.replace(/^[-*]\s*/, "").trim())
      .filter((line: string) => line.length > 5)

    const noIssues = /no\s+(issues?|problems?|concerns?)\s+found/i.test(text) ||
      /looks?\s+good/i.test(text) || (findings.length === 0 && text.length < 200)
    const severity = noIssues ? "info"
      : findings.some((f: string) => /critical|security|vulnerability|crash/i.test(f)) ? "critical"
      : "warning"

    state.specialistReviews.push({
      role,
      findings: findings.length > 0 ? findings : [noIssues ? "No issues found." : text.slice(0, 500)],
      severity,
    })

    return {
      tool: "run_specialist",
      success: true,
      output: `## ${role.charAt(0).toUpperCase() + role.slice(1)} Review (${severity})\n\n${findings.length > 0 ? findings.map(f => `- ${f}`).join("\n") : text.slice(0, 1000)}`,
    }
  } catch (err: any) {
    return { tool: "run_specialist", success: false, output: "", error: `Review failed: ${err.message?.slice(0, 200)}` }
  }
}

async function toolGetDiff(args: Record<string, string>): Promise<ToolCallResult> {
  const workDir = args.work_dir || state.workDir || ""
  if (!workDir) {
    return { tool: "get_diff", success: false, output: "", error: "No workspace. Call setup_workspace first." }
  }

  try {
    // Try committed diff first
    let diff = ""
    try {
      diff = shellExec(`git -C ${JSON.stringify(workDir)} diff HEAD~1..HEAD 2>/dev/null`)
    } catch {}
    if (!diff) {
      // Try uncommitted changes
      diff = shellExec(`git -C ${JSON.stringify(workDir)} diff`)
      if (!diff) diff = shellExec(`git -C ${JSON.stringify(workDir)} status --short`)
    }

    return {
      tool: "get_diff",
      success: diff.length > 0,
      output: diff.length > 0 ? diff.slice(0, 6000) : "No changes detected.",
    }
  } catch (err: any) {
    return { tool: "get_diff", success: false, output: "", error: err.message?.slice(0, 200) }
  }
}

function toolSubmitResult(args: Record<string, string>): ToolCallResult {
  const summary = args.summary || "Workflow complete."
  state.finalSummary = summary

  const output = [
    `## Orchestrator Result`,
    ``,
    `Summary: ${summary}`,
    `Worker runs: ${state.workerRuns.length}`,
    `Specialist reviews: ${state.specialistReviews.length}`,
    ``,
    state.workerRuns.map((r, i) => `Worker ${i + 1} (${r.model}): ${r.turns} turns, modified=${r.filesModified}`).join("\n"),
    ``,
    state.specialistReviews.map(r => `${r.role}: ${r.severity} - ${r.findings.length} findings`).join("\n"),
  ].join("\n")

  return { tool: "submit_result", success: true, output }
}

// ── Orchestrator Tool Executor ──────────────────────────────────────────

function createOrchestratorExecutor(): ToolExecutor {
  return {
    async executeAll(calls, _context) {
      const results: ToolCallResult[] = []

      for (const call of calls) {
        switch (call.tool) {
          case "read_issue":
            results.push(await toolReadIssue(call.args))
            break
          case "setup_workspace":
            results.push(await toolSetupWorkspace(call.args))
            break
          case "spawn_worker":
            results.push(await toolSpawnWorker(call.args))
            break
          case "run_specialist":
            results.push(await toolRunSpecialist(call.args))
            break
          case "get_diff":
            results.push(await toolGetDiff(call.args))
            break
          case "submit_result":
            results.push(toolSubmitResult(call.args))
            break
          default:
            results.push({ tool: call.tool, success: false, output: "", error: `Unknown orchestrator tool: ${call.tool}` })
        }
      }

      return results
    },
  }
}

// ── Orchestrator Tool Schemas ───────────────────────────────────────────

function getOrchestratorToolSchemas(): OpenAIToolDef[] {
  return [
    {
      type: "function",
      function: {
        name: "read_issue",
        description: "Fetch a GitHub issue description. Returns the issue title, body, state, and labels.",
        parameters: {
          type: "object",
          properties: {
            repo: { type: "string", description: "Repository in owner/repo format (e.g. 'axios/axios')" },
            number: { type: "string", description: "Issue number" },
          },
          required: ["repo", "number"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "setup_workspace",
        description: "Clone a repository and create a working branch. Must be called before spawn_worker.",
        parameters: {
          type: "object",
          properties: {
            repo: { type: "string", description: "Repository in owner/repo format (e.g. 'axios/axios')" },
            branch: { type: "string", description: "Base branch to start from (default: main)" },
          },
          required: ["repo"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "spawn_worker",
        description: "Launch a coding worker agent to implement a fix. The worker is a separate LLM that explores the codebase and makes changes. Returns the worker's execution summary and diff.",
        parameters: {
          type: "object",
          properties: {
            task: { type: "string", description: "Full task description for the worker — what to investigate and fix" },
            work_dir: { type: "string", description: "Working directory (optional, uses workspace from setup_workspace)" },
          },
          required: ["task"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "run_specialist",
        description: "Run a specialist review on the latest worker's diff. Returns concrete findings.",
        parameters: {
          type: "object",
          properties: {
            role: { type: "string", description: "Specialist role: 'architect' or 'qa'", enum: ["architect", "qa"] },
            diff: { type: "string", description: "Diff to review (optional, uses latest worker diff)" },
          },
          required: ["role"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "get_diff",
        description: "Get the current diff in the workspace.",
        parameters: {
          type: "object",
          properties: {
            work_dir: { type: "string", description: "Working directory (optional)" },
          },
        },
      },
    },
    {
      type: "function",
      function: {
        name: "submit_result",
        description: "Finalize the workflow and submit the result. Call this when the specialist review is clean or after max refinement iterations.",
        parameters: {
          type: "object",
          properties: {
            summary: { type: "string", description: "Summary of what was accomplished" },
          },
          required: ["summary"],
        },
      },
    },
  ]
}

// ── Orchestrator System Prompt ──────────────────────────────────────────

const ORCHESTRATOR_SYSTEM_PROMPT = `I am an orchestrator managing a code improvement workflow. I have specialized tools to coordinate workers and reviewers.

My tools:
- read_issue(repo, number) -- fetch a GitHub issue description
- setup_workspace(repo, branch) -- clone a repository and create a working branch
- spawn_worker(task) -- launch a coding worker agent to implement a fix
- run_specialist(role) -- run a specialist review (architect or qa) on the worker's diff
- get_diff() -- see the current diff in the workspace
- submit_result(summary) -- finalize the workflow

My process:
1. I read the issue to understand what needs fixing
2. I set up a clean workspace from the appropriate branch
3. I spawn a worker with a clear task description based on the issue
4. I review the worker's diff with specialists (both architect and qa)
5. If specialists find issues, I spawn another worker with the original task PLUS the specialist feedback
6. I iterate until the specialist review is clean or I have made 3 attempts
7. I submit the final result

Key principles:
- I give the worker ONLY the issue description — no solution hints
- I pass specialist feedback verbatim to the next worker as constraints
- I do not edit code myself — I orchestrate workers and reviewers
- I am decisive: if 2 iterations produce no improvement, I submit what I have`

// ── Input Parsing ───────────────────────────────────────────────────────

function parseInput(args: string[]): { repo: string; issue: number } {
  // Handle: https://github.com/owner/repo/issues/123
  if (args.length === 1 && args[0].includes("github.com")) {
    const issueMatch = args[0].match(/github\.com\/([^/]+\/[^/]+)\/issues\/(\d+)/)
    if (issueMatch) return { repo: issueMatch[1], issue: parseInt(issueMatch[2], 10) }
    // Also handle PR URLs
    const prMatch = args[0].match(/github\.com\/([^/]+\/[^/]+)\/pull\/(\d+)/)
    if (prMatch) return { repo: prMatch[1], issue: parseInt(prMatch[2], 10) }
    throw new Error(`Could not parse URL: ${args[0]}`)
  }

  // Handle: owner/repo 123
  if (args.length === 2) {
    return { repo: args[0], issue: parseInt(args[1], 10) }
  }

  throw new Error(
    "Usage: bun run src/orchestrator.ts <owner/repo> <issue-number>\n" +
    "   or: bun run src/orchestrator.ts <github-issue-url>"
  )
}

// ── Main ────────────────────────────────────────────────────────────────

async function main() {
  const startTime = Date.now()
  const args = process.argv.slice(2)

  if (args.length === 0) {
    console.error(
      "Usage: bun run src/orchestrator.ts <owner/repo> <issue-number>\n" +
      "   or: bun run src/orchestrator.ts <github-issue-url>"
    )
    process.exit(1)
  }

  const { repo, issue } = parseInput(args)

  console.log("+====================================================================+")
  console.log("|  Orchestrator Pipeline                                             |")
  console.log("+====================================================================+")
  console.log(`  Issue: ${repo}#${issue}`)
  console.log(`  Orchestrator: ${ORCH_MODEL} @ ${ORCH_ENDPOINT}`)
  console.log(`  Worker: ${WORKER_MODEL} @ ${WORKER_ENDPOINT}`)
  console.log(`  Max orchestrator turns: ${MAX_TURNS}`)
  console.log(`  Max worker turns: ${WORKER_TURNS}`)

  // The orchestrator is itself a ReAct agent
  const orchSend = createSendWithTools({
    baseUrl: ORCH_ENDPOINT,
    model: ORCH_MODEL,
    temperature: 0,
    timeoutMs: 300000,
  })

  const orchExecutor = createOrchestratorExecutor()
  const orchTools = getOrchestratorToolSchemas()

  // Build the initial user message — just the target info, no solution details
  const userMessage = `I need to fix issue #${issue} in the ${repo} repository. I should start by reading the issue, then set up a workspace, spawn a worker, review the result, and iterate until the fix is clean.`

  const result = await runAgent({
    agentName: "Orchestrator",
    systemPrompt: ORCHESTRATOR_SYSTEM_PROMPT,
    userMessage,
    model: ORCH_MODEL,
    endpoint: ORCH_ENDPOINT,
    sendWithTools: orchSend,
    executor: orchExecutor,
    tools: orchTools,
    workDir: "/tmp",  // Orchestrator doesn't write files directly
    maxTurns: MAX_TURNS,
    allowedTools: ["read_issue", "setup_workspace", "spawn_worker", "run_specialist", "get_diff", "submit_result"],
    logLabel: `orchestrator/${ORCH_MODEL.split("/").pop()}`,
    notes: [],
    interceptors: false,
    compaction: true,
    verbose: VERBOSE,
  })

  // ── Report ──
  const durationSec = Math.round((Date.now() - startTime) / 1000)

  console.log("\n" + "=".repeat(70))
  console.log("  ORCHESTRATOR REPORT")
  console.log("=".repeat(70))
  console.log(`  Issue: ${repo}#${issue}`)
  console.log(`  Duration: ${durationSec}s`)
  console.log(`  Orchestrator turns: ${result.turns} (${result.turnsWithToolCalls} with tools)`)
  console.log(`  Orchestrator tokens: ${result.totalTokens}`)
  console.log(`  Worker runs: ${state.workerRuns.length}`)
  for (let i = 0; i < state.workerRuns.length; i++) {
    const w = state.workerRuns[i]
    console.log(`    Worker ${i + 1}: ${w.model}, ${w.turns} turns, modified=${w.filesModified}`)
  }
  console.log(`  Specialist reviews: ${state.specialistReviews.length}`)
  for (const r of state.specialistReviews) {
    console.log(`    ${r.role}: ${r.severity} - ${r.findings.length} findings`)
  }
  if (state.finalSummary) {
    console.log(`  Final summary: ${state.finalSummary.slice(0, 200)}`)
  }
  if (state.workDir) {
    console.log(`  Workspace: ${state.workDir}`)
  }
  console.log("=".repeat(70))

  // Save result
  const resultFile = path.join(
    "/Users/virtualmachine/plan-lab/results",
    `orchestrator-${repo.replace("/", "-")}-${issue}-${Date.now()}.json`,
  )
  try {
    await fs.writeFile(resultFile, JSON.stringify({
      repo,
      issue,
      orchModel: ORCH_MODEL,
      workerModel: WORKER_MODEL,
      orchestratorTurns: result.turns,
      orchestratorTokens: result.totalTokens,
      workerRuns: state.workerRuns.map(w => ({
        model: w.model,
        turns: w.turns,
        filesModified: w.filesModified,
        diffSize: w.diff.length,
      })),
      specialistReviews: state.specialistReviews,
      finalSummary: state.finalSummary,
      workDir: state.workDir,
      durationSec,
    }, null, 2))
    console.log(`  Result saved to: ${resultFile}`)
  } catch (err: any) {
    console.error(`  Failed to save result: ${err.message?.slice(0, 100)}`)
  }
}

await main()
