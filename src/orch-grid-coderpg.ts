#!/usr/bin/env bun
/**
 * Orchestrator Grid Test — CodeRPG Desktop Plugin Creation
 *
 * Tests whether the orchestrator pattern can create and register plugins
 * in the CodeRPG Desktop plugin system across model combinations.
 *
 * Grid:
 *   Orchestrators: 27B, 35B
 *   Workers: 9B, 27B, 35B
 *   = 6 combinations
 *
 * Task: Fix the plugin registry so both clock plugins compile.
 *
 * Usage:
 *   bun run src/orch-grid-coderpg.ts
 *
 * Environment:
 *   COMBO       Run a single combo (e.g. "27B-orch/9B-worker")
 *   MAX_TURNS   Max orchestrator turns (default: 30)
 *   WORKER_TURNS Max worker turns per spawn (default: 50)
 *   VERBOSE     Set to 1 for full transcript
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

const SOURCE_DIR = "/Users/virtualmachine/coderpg-desktop"
const MAX_TURNS = parseInt(process.env.MAX_TURNS || "30", 10)
const WORKER_TURNS = parseInt(process.env.WORKER_TURNS || "50", 10)
const VERBOSE = process.env.VERBOSE === "1"
const COMBO_FILTER = process.env.COMBO || ""

const ENDPOINT_A = "http://192.168.50.117:1234"
const ENDPOINT_B = "http://192.168.50.206:1234"

// ── Model Grid ──────────────────────────────────────────────────────────

interface ModelDef {
  label: string
  model: string
  endpoint: string
}

const MODELS: Record<string, ModelDef> = {
  "9B":   { label: "9B",   model: "qwen/qwen3.5-9b",       endpoint: ENDPOINT_A },
  "27B":  { label: "27B",  model: "qwen3.5-27b",            endpoint: ENDPOINT_A },
  "35B":  { label: "35B",  model: "qwen/qwen3.5-35b-a3b",  endpoint: ENDPOINT_B },
  "397B": { label: "397B", model: "qwen3.5-397b-a17b",      endpoint: ENDPOINT_B },
}

interface GridCombo {
  name: string
  orch: ModelDef
  worker: ModelDef
}

const GRID: GridCombo[] = [
  { name: "27B-orch/9B-worker",  orch: MODELS["27B"], worker: MODELS["9B"] },
  { name: "27B-orch/27B-worker", orch: MODELS["27B"], worker: MODELS["27B"] },
  { name: "27B-orch/35B-worker", orch: MODELS["27B"], worker: MODELS["35B"] },
  { name: "35B-orch/9B-worker",  orch: MODELS["35B"], worker: MODELS["9B"] },
  { name: "35B-orch/27B-worker", orch: MODELS["35B"], worker: MODELS["27B"] },
  { name: "35B-orch/35B-worker", orch: MODELS["35B"], worker: MODELS["35B"] },
  { name: "9B-orch/9B-worker",    orch: MODELS["9B"],   worker: MODELS["9B"] },
  { name: "9B-orch/27B-worker",   orch: MODELS["9B"],   worker: MODELS["27B"] },
  { name: "9B-orch/35B-worker",   orch: MODELS["9B"],   worker: MODELS["35B"] },
  { name: "397B-orch/9B-worker",  orch: MODELS["397B"], worker: MODELS["9B"] },
  { name: "397B-orch/27B-worker", orch: MODELS["397B"], worker: MODELS["27B"] },
  { name: "397B-orch/35B-worker", orch: MODELS["397B"], worker: MODELS["35B"] },
]

// ── Task Definition ─────────────────────────────────────────────────────

const TASK_DESCRIPTION = `Fix the plugin registry in CodeRPG Desktop so that both clock plugins (analog and digital) are properly registered and the plugin system compiles without errors in src/plugins/ and src/windows/types.ts.

The plugin system is broken. Run the verification command to see the errors, then fix them.

Verification command: npx tsc --noEmit 2>&1 | grep -E 'src/plugins|src/windows/types' | grep 'error TS' || echo "PASS: no plugin errors"

IMPORTANT:
- Study the existing code first to understand the architecture before making changes
- The analog clock panel at src/panels/analog-clock.tsx exists but is NOT registered in the plugin registry
- Only modify or create files in src/plugins/. Do NOT touch other directories.
- All shell commands run in the working directory already -- do NOT use cd.
- There may be unrelated type errors in other files. Ignore those. Only care about errors in src/plugins/ and src/windows/types.ts.
- When verification shows "PASS: no plugin errors", call complete.`

const VERIFY_CMD = "npx tsc --noEmit 2>&1 | grep -E 'src/plugins|src/windows/types' | grep 'error TS' || echo 'PASS: no plugin errors'"

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

// ── Workspace Setup ─────────────────────────────────────────────────────

async function createWorkspace(): Promise<string> {
  const runDir = `/tmp/coderpg-orch-${Date.now()}`
  console.log(`  [setup] Creating workspace: ${runDir}`)

  // Reset source
  try {
    execSync(`git -C ${JSON.stringify(SOURCE_DIR)} checkout . 2>/dev/null`, { stdio: "pipe" })
    execSync(`git -C ${JSON.stringify(SOURCE_DIR)} clean -fd 2>/dev/null`, { stdio: "pipe" })
  } catch { /* ok */ }

  // Copy to isolated dir
  execSync(`rsync -a --exclude=node_modules --exclude=.git --exclude=dist ${JSON.stringify(SOURCE_DIR + "/")} ${JSON.stringify(runDir + "/")}`, { stdio: "pipe", timeout: 60000 })

  // Initialize git for diff tracking
  execSync(`git init ${JSON.stringify(runDir)}`, { stdio: "pipe" })
  execSync(`git -C ${JSON.stringify(runDir)} add -A`, { stdio: "pipe" })
  execSync(`git -C ${JSON.stringify(runDir)} commit -m "baseline" --allow-empty`, { stdio: "pipe" })

  // Symlink node_modules from source (faster than copying)
  try {
    execSync(`ln -s ${JSON.stringify(SOURCE_DIR + "/node_modules")} ${JSON.stringify(runDir + "/node_modules")}`, { stdio: "pipe" })
  } catch { /* may already exist */ }

  // Write the correct broken plugins/index.ts — the source file may have been corrupted
  // by a previous experiment, so we write the canonical broken state directly.
  try {
    const pluginsIndex = path.join(runDir, "src/plugins/index.ts")
    const brokenContent = [
      `// Plugin registry for CodeRPG Desktop panels and extensions`,
      `import NetworkHealthPanel from '../panels/network-health'`,
      `import ClockPanel from '../panels/clock'`,
      `import CharacterPanel from '../panels/character'`,
      `import InferencePanel from '../panels/inference'`,
      `import InventoryPanel from '../panels/inventory'`,
      `import ProgressionPanel from '../panels/progression'`,
      `import SpellbookPanel from '../panels/spellbook'`,
      `import WorkflowPanel from '../panels/workflow'`,
      ``,
      `// Re-export for other consumers`,
      `export { NetworkHealthPanel, ClockPanel, CharacterPanel, InferencePanel }`,
      `export { InventoryPanel, ProgressionPanel, SpellbookPanel, WorkflowPanel }`,
      ``,
      `// Desktop plugin registry for window system`,
      `import type { DesktopPlugin, DesktopPluginId } from './types'`,
      ``,
      `const PANELS: DesktopPlugin[] = [`,
      `  { id: 'network-health', name: 'Network Health', component: NetworkHealthPanel },`,
      `  { id: 'clock', name: 'Clock', component: ClockPanel },`,
      `  { id: 'character', name: 'Character', component: CharacterPanel },`,
      `  { id: 'inference', name: 'Inference', component: InferencePanel },`,
      `  { id: 'inventory', name: 'Inventory', component: InventoryPanel },`,
      `  { id: 'progression', name: 'Progression', component: ProgressionPanel },`,
      `  { id: 'spellbook', name: 'Spellbook', component: SpellbookPanel },`,
      `  { id: 'workflow', name: 'Workflow', component: WorkflowPanel },`,
      `]`,
      ``,
      `export const DESKTOP_PLUGINS = PANELS`,
      `export const DESKTOP_PLUGIN_MAP = Object.fromEntries(`,
      `  PANELS.map((p) => [p.id, p])`,
      `)`,
      ``,
    ].join("\n")
    await fs.writeFile(pluginsIndex, brokenContent, "utf-8")

    // Delete types.ts if it exists from a previous run
    try {
      await fs.unlink(path.join(runDir, "src/plugins/types.ts"))
    } catch { /* doesn't exist, that's fine */ }
  } catch (err: any) {
    console.log(`  [setup] Warning: could not inject broken state: ${err.message?.slice(0, 100)}`)
  }

  // Re-commit with the broken state as the baseline
  execSync(`git -C ${JSON.stringify(runDir)} add -A`, { stdio: "pipe" })
  execSync(`git -C ${JSON.stringify(runDir)} commit -m "baseline with broken plugin types" --allow-empty`, { stdio: "pipe" })

  return runDir
}

// ── Verification ────────────────────────────────────────────────────────

function verify(workDir: string): { pass: boolean; output: string } {
  try {
    let tscOutput = ""
    try {
      tscOutput = execSync("npx tsc --noEmit 2>&1", {
        cwd: workDir,
        encoding: "utf-8",
        timeout: 120000,
        shell: "/bin/bash",
      })
    } catch (err: any) {
      tscOutput = err.stdout || err.stderr || ""
    }

    const pluginErrors = tscOutput
      .split("\n")
      .filter(line => /src\/plugins|src\/windows\/types/.test(line) && /error TS/.test(line))

    if (pluginErrors.length === 0) {
      return { pass: true, output: "No plugin-related type errors" }
    } else {
      return { pass: false, output: pluginErrors.join("\n") }
    }
  } catch (err: any) {
    return { pass: false, output: `Verify error: ${err.message?.slice(0, 500)}` }
  }
}

// ── Orchestrator State (per-run) ────────────────────────────────────────

interface OrchestratorState {
  workDir: string | null
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

// ── Orchestrator Tool Implementations ───────────────────────────────────

function createOrchTools(
  state: OrchestratorState,
  workerModel: string,
  workerEndpoint: string,
  orchSend: SendWithToolsFn,
  orchModel: string,
): { executor: ToolExecutor; schemas: OpenAIToolDef[] } {

  async function toolSetupWorkspace(_args: Record<string, string>): Promise<ToolCallResult> {
    try {
      const workDir = await createWorkspace()
      state.workDir = workDir

      const listing = shellExec(`find ${JSON.stringify(workDir)} -maxdepth 3 -not -path '*/node_modules/*' -not -path '*/.git/*' -type f -name '*.ts' -o -name '*.tsx' | head -40 | sort`)
      const relativeListing = listing.replace(new RegExp(workDir + "/?", "g"), "")

      // Verify baseline fails
      const baseline = verify(workDir)

      return {
        tool: "setup_workspace",
        success: true,
        output: `Workspace ready at: ${workDir}\nBaseline compilation: ${baseline.pass ? "PASS (nothing to fix)" : "FAIL (expected)"}\nBaseline errors: ${baseline.output.slice(0, 300)}\n\nRelevant files:\n${relativeListing}`,
      }
    } catch (err: any) {
      return { tool: "setup_workspace", success: false, output: "", error: err.message?.slice(0, 300) }
    }
  }

  async function toolSpawnWorker(args: Record<string, string>): Promise<ToolCallResult> {
    const task = args.task || ""
    const workDir = state.workDir || ""
    if (!task) {
      return { tool: "spawn_worker", success: false, output: "", error: "Missing task. Usage: spawn_worker(task='description of what to fix')" }
    }
    if (!workDir) {
      return { tool: "spawn_worker", success: false, output: "", error: "No workspace set up. Call setup_workspace first." }
    }

    // Reset workspace to baseline before each worker run (except first)
    if (state.workerRuns.length > 0) {
      try {
        shellExec(`git -C ${JSON.stringify(workDir)} checkout . && git -C ${JSON.stringify(workDir)} clean -fd`)
      } catch { /* continue */ }
    }

    console.log(`\n  [orchestrator] Spawning worker: ${workerModel}`)
    console.log(`  [orchestrator] Task: ${task.slice(0, 120)}...`)

    const workerSend = createSendWithTools({
      baseUrl: workerEndpoint,
      model: workerModel,
      temperature: 0,
      timeoutMs: 300000,
    })

    const workerSystemPrompt = `I am a developer fixing a plugin registry issue. I have only the task description and must explore the codebase to understand the problem and implement a fix.

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
- All paths are RELATIVE to the project root (e.g. "src/plugins/index.ts"), never absolute
- For edit: old_string must be an EXACT copy of text currently in the file
- Shell commands run in the project directory. I do NOT prefix with cd.
- I explore first, then understand, then fix
- I make minimal, targeted changes
- I call complete when done`

    // Generate scaffold plan for the worker
    let systemPrompt = workerSystemPrompt
    try {
      const plan = await generateScaffoldPlan({
        mode: "scaffold_plan",
        task,
        workDir,
        model: workerModel,
        endpoint: workerEndpoint,
        workerModel: workerModel,
        workerEndpoint: workerEndpoint,
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
    const executor = createEnhancedToolExecutor({ sendWithTools: workerSend, model: workerModel })
    const tools = getBaseToolSchemas()

    try {
      const result = await runAgent({
        agentName: "Worker",
        systemPrompt,
        userMessage: `Working directory: ${workDir}\n\n${task}`,
        model: workerModel,
        endpoint: workerEndpoint,
        sendWithTools: workerSend,
        executor,
        tools,
        workDir,
        maxTurns: WORKER_TURNS,
        testCmd: VERIFY_CMD,
        allowedTools: ["read", "write", "edit", "grep", "list", "glob", "bash", "complete"],
        logLabel: `orch-worker/${workerModel.split("/").pop()}`,
        notes: [],
        interceptors: false,
        failureLedger: ledger,
        compaction: true,
        verbose: VERBOSE,
        recoverySend: workerSend,
        recoveryModel: workerModel,
        evoObserve: { send: workerSend, model: workerModel, intervalTurns: 3 },
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

      // Run verification
      const verifyResult = verify(workDir)

      const runSummary = {
        model: workerModel,
        turns: result.turns,
        filesModified: result.filesModified,
        diff,
        summary: result.finalText || `Completed in ${result.turns} turns, ${result.turnsWithToolCalls} with tools`,
      }
      state.workerRuns.push(runSummary)

      const output = [
        `## Worker Execution Complete`,
        `Model: ${workerModel}`,
        `Turns: ${result.turns} (${result.turnsWithToolCalls} with tools)`,
        `Tokens: ${result.totalTokens}`,
        `Files modified: ${result.filesModified}`,
        `Files touched: ${result.touchedFiles.join(", ") || "none"}`,
        `Loops detected: ${result.loopsDetected}`,
        `Adaptive recoveries: ${result.adaptiveRecoveries}`,
        ``,
        `## Verification`,
        verifyResult.pass ? "PASS: No plugin-related type errors" : `FAIL: ${verifyResult.output.slice(0, 500)}`,
        ``,
        `## Diff`,
        diff.length > 4000 ? diff.slice(0, 4000) + "\n...(truncated)" : diff,
      ].join("\n")

      return { tool: "spawn_worker", success: verifyResult.pass, output }
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

    try {
      const result = await orchSend(
        `${role} reviewer`,
        [
          { role: "system", content: persona },
          {
            role: "user",
            content: `## Diff to Review\n\`\`\`diff\n${diff.slice(0, 6000)}\n\`\`\`\n\nReview this diff. List concrete findings as bullet points. If it looks good, say "No issues found."`,
          },
        ],
        [],
        orchModel,
      )

      let text = result.content || result.text || ""
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

  async function toolGetDiff(_args: Record<string, string>): Promise<ToolCallResult> {
    const workDir = state.workDir || ""
    if (!workDir) {
      return { tool: "get_diff", success: false, output: "", error: "No workspace. Call setup_workspace first." }
    }

    try {
      let diff = ""
      try {
        diff = shellExec(`git -C ${JSON.stringify(workDir)} diff HEAD~1..HEAD 2>/dev/null`)
      } catch {}
      if (!diff) {
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

  function toolVerify(_args: Record<string, string>): ToolCallResult {
    const workDir = state.workDir || ""
    if (!workDir) {
      return { tool: "verify", success: false, output: "", error: "No workspace. Call setup_workspace first." }
    }
    const result = verify(workDir)
    return { tool: "verify", success: result.pass, output: result.pass ? "PASS: No plugin errors" : `FAIL: ${result.output}` }
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

  // ── Executor ──
  const executor: ToolExecutor = {
    async executeAll(calls, _context) {
      const results: ToolCallResult[] = []
      for (const call of calls) {
        switch (call.tool) {
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
          case "verify":
            results.push(toolVerify(call.args))
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

  // ── Schemas ──
  const schemas: OpenAIToolDef[] = [
    {
      type: "function",
      function: {
        name: "setup_workspace",
        description: "Create an isolated workspace for the task. Creates a copy of the CodeRPG Desktop codebase. Must be called before spawn_worker.",
        parameters: { type: "object", properties: {} },
      },
    },
    {
      type: "function",
      function: {
        name: "spawn_worker",
        description: "Launch a coding worker agent to implement changes. The worker is a separate LLM that explores the codebase and makes changes. Returns the worker's execution summary, verification result, and diff.",
        parameters: {
          type: "object",
          properties: {
            task: { type: "string", description: "Full task description for the worker — what to investigate and fix" },
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
        parameters: { type: "object", properties: {} },
      },
    },
    {
      type: "function",
      function: {
        name: "verify",
        description: "Run the TypeScript compiler to verify plugin-related errors are fixed.",
        parameters: { type: "object", properties: {} },
      },
    },
    {
      type: "function",
      function: {
        name: "submit_result",
        description: "Finalize the workflow and submit the result. Call this when verification passes or after max refinement iterations.",
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

  return { executor, schemas }
}

// ── Orchestrator System Prompt ──────────────────────────────────────────

const ORCHESTRATOR_SYSTEM_PROMPT = `I am an orchestrator managing a code improvement workflow for the CodeRPG Desktop plugin system.

My tools:
- setup_workspace() -- create an isolated workspace with the CodeRPG Desktop codebase
- spawn_worker(task) -- launch a coding worker to implement changes
- run_specialist(role) -- run a specialist review (architect or qa) on the worker's diff
- get_diff() -- see the current diff in the workspace
- verify() -- run the TypeScript compiler to check if plugin errors are fixed
- submit_result(summary) -- finalize the workflow

My process:
1. I set up a clean workspace
2. I spawn a worker with a clear task description
3. If the worker's verification passes, I submit the result
4. If it fails, I run specialist reviews to understand what went wrong
5. I spawn another worker with the original task PLUS the specialist feedback as constraints
6. I iterate until verification passes or I have made 3 attempts
7. I submit the final result

Key principles:
- I give the worker ONLY the task description — no solution hints
- I pass specialist feedback verbatim to the next worker as constraints
- I do not edit code myself — I orchestrate workers and reviewers
- I am decisive: if 2 iterations produce no improvement, I submit what I have`

// ── Run one combo ───────────────────────────────────────────────────────

interface GridResult {
  combo: string
  orchModel: string
  workerModel: string
  pass: boolean
  orchTurns: number
  orchTokens: number
  workerRuns: number
  totalDurationSec: number
  specialistReviews: number
  error?: string
}

async function runCombo(combo: GridCombo): Promise<GridResult> {
  const startTime = Date.now()
  console.log(`\n${"=".repeat(70)}`)
  console.log(`  COMBO: ${combo.name}`)
  console.log(`  Orchestrator: ${combo.orch.model} @ ${combo.orch.endpoint}`)
  console.log(`  Worker: ${combo.worker.model} @ ${combo.worker.endpoint}`)
  console.log(`${"=".repeat(70)}`)

  const state: OrchestratorState = {
    workDir: null,
    workerRuns: [],
    specialistReviews: [],
    finalSummary: null,
  }

  const orchSend = createSendWithTools({
    baseUrl: combo.orch.endpoint,
    model: combo.orch.model,
    temperature: 0,
    timeoutMs: 300000,
  })

  const { executor, schemas } = createOrchTools(
    state,
    combo.worker.model,
    combo.worker.endpoint,
    orchSend,
    combo.orch.model,
  )

  const userMessage = `I need to fix the plugin registry in CodeRPG Desktop. I should set up a workspace, spawn a worker to fix the plugin system, verify the result, and iterate until the TypeScript compilation passes for plugin-related files.`

  try {
    const result = await runAgent({
      agentName: "Orchestrator",
      systemPrompt: ORCHESTRATOR_SYSTEM_PROMPT,
      userMessage,
      model: combo.orch.model,
      endpoint: combo.orch.endpoint,
      sendWithTools: orchSend,
      executor,
      tools: schemas,
      workDir: "/tmp",
      maxTurns: MAX_TURNS,
      allowedTools: ["setup_workspace", "spawn_worker", "run_specialist", "get_diff", "verify", "submit_result"],
      logLabel: `orch/${combo.name}`,
      notes: [],
      interceptors: false,
      compaction: true,
      verbose: VERBOSE,
    })

    // Final verification
    const finalPass = state.workDir ? verify(state.workDir).pass : false
    const durationSec = Math.round((Date.now() - startTime) / 1000)

    const gridResult: GridResult = {
      combo: combo.name,
      orchModel: combo.orch.model,
      workerModel: combo.worker.model,
      pass: finalPass,
      orchTurns: result.turns,
      orchTokens: result.totalTokens,
      workerRuns: state.workerRuns.length,
      totalDurationSec: durationSec,
      specialistReviews: state.specialistReviews.length,
    }

    console.log(`\n  RESULT: ${finalPass ? "PASS" : "FAIL"}`)
    console.log(`  Orch turns: ${result.turns}, Worker runs: ${state.workerRuns.length}`)
    console.log(`  Duration: ${durationSec}s`)

    // Cleanup on pass
    if (finalPass && state.workDir) {
      try { execSync(`rm -rf ${JSON.stringify(state.workDir)}`, { stdio: "pipe" }) } catch {}
    } else if (state.workDir) {
      console.log(`  [debug] Workspace preserved: ${state.workDir}`)
    }

    return gridResult
  } catch (err: any) {
    const durationSec = Math.round((Date.now() - startTime) / 1000)
    return {
      combo: combo.name,
      orchModel: combo.orch.model,
      workerModel: combo.worker.model,
      pass: false,
      orchTurns: 0,
      orchTokens: 0,
      workerRuns: state.workerRuns.length,
      totalDurationSec: durationSec,
      specialistReviews: state.specialistReviews.length,
      error: err.message?.slice(0, 200),
    }
  }
}

// ── Main ────────────────────────────────────────────────────────────────

async function main() {
  console.log("+====================================================================+")
  console.log("|  Orchestrator Grid Test — CodeRPG Desktop Plugin Creation          |")
  console.log("+====================================================================+")
  console.log(`  Max orchestrator turns: ${MAX_TURNS}`)
  console.log(`  Max worker turns: ${WORKER_TURNS}`)
  console.log(`  Grid size: ${GRID.length} combos`)

  const combos = COMBO_FILTER
    ? GRID.filter(g => g.name.includes(COMBO_FILTER))
    : GRID

  if (combos.length === 0) {
    console.error(`No combos match filter: ${COMBO_FILTER}`)
    console.error(`Available: ${GRID.map(g => g.name).join(", ")}`)
    process.exit(1)
  }

  console.log(`  Running: ${combos.map(c => c.name).join(", ")}`)

  const results: GridResult[] = []

  for (const combo of combos) {
    const result = await runCombo(combo)
    results.push(result)
  }

  // ── Summary Table ──
  console.log("\n" + "=".repeat(70))
  console.log("  GRID RESULTS — CodeRPG Desktop Plugin Creation")
  console.log("=".repeat(70))
  console.log()
  console.log("  Combo                    | Pass | Orch Turns | Workers | Duration")
  console.log("  " + "-".repeat(64))
  for (const r of results) {
    const pass = r.pass ? "YES " : "NO  "
    const pad = (s: string, n: number) => s.padEnd(n)
    console.log(`  ${pad(r.combo, 26)} | ${pass} | ${String(r.orchTurns).padStart(10)} | ${String(r.workerRuns).padStart(7)} | ${r.totalDurationSec}s`)
  }
  console.log()
  console.log(`  Pass rate: ${results.filter(r => r.pass).length}/${results.length}`)
  console.log("=".repeat(70))

  // Save results
  const resultsDir = "/Users/virtualmachine/plan-lab/results"
  await fs.mkdir(resultsDir, { recursive: true })
  const resultsFile = path.join(resultsDir, `orch-grid-coderpg-${Date.now()}.json`)
  await fs.writeFile(resultsFile, JSON.stringify({
    test: "coderpg-plugin-creation",
    timestamp: new Date().toISOString(),
    maxTurns: MAX_TURNS,
    workerTurns: WORKER_TURNS,
    results,
  }, null, 2))
  console.log(`  Results saved to: ${resultsFile}`)
}

await main()
