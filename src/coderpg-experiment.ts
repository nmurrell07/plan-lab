#!/usr/bin/env bun
/**
 * CodeRPG Desktop Plugin Experiment
 *
 * Tests whether local models can scaffold-plan and execute a real plugin task
 * against the CodeRPG-Desktop codebase.
 *
 * Usage:
 *   MODEL=qwen3.5-27b ENDPOINT=http://192.168.50.117:1234 bun run src/coderpg-experiment.ts
 *
 * Environment:
 *   MODEL       Model ID (default: qwen3.5-27b)
 *   ENDPOINT    LM Studio endpoint
 *   MAX_TURNS   Max agent turns (default: 25)
 *   VERBOSE     Set to 1 for full transcript
 *   PLAN_MODE   scaffold_plan | none (default: scaffold_plan)
 *   TOOL_MODE   base | meta (default: base)
 *   TASK        clock (default: clock)
 */

import { execSync } from "child_process"
import fs from "fs/promises"
import path from "path"
import { createSendWithTools } from "./client"
import { createToolExecutor, createEnhancedToolExecutor, getBaseToolSchemas } from "./tools"
import { createMetaToolExecutor, getMetaToolDefs } from "./meta-tools"
import { runAgent } from "./runner"
import { generateScaffoldPlan, injectPlanIntoPrompt } from "./planner"
import type { PlanArtifact } from "./planner"
import { createEmptyLedger } from "./ledger"

// ── Config ──────────────────────────────────────────────────────────────

const SOURCE_DIR = "/Users/virtualmachine/coderpg-desktop"
const WORK_DIR = "/Users/virtualmachine/coderpg-desktop-plantest"
const MODEL = process.env.MODEL || "qwen3.5-27b"
const ENDPOINT = process.env.ENDPOINT || "http://192.168.50.117:1234"
const MAX_TURNS = parseInt(process.env.MAX_TURNS || "25", 10)
const VERBOSE = process.env.VERBOSE === "1"
const PLAN_MODE = process.env.PLAN_MODE || "scaffold_plan"
const TOOL_MODE = process.env.TOOL_MODE || "base"

// ── Tasks ───────────────────────────────────────────────────────────────

const TSC_CHECK_CMD = `npx tsc --noEmit 2>&1 | grep -E 'src/plugins|src/windows/types' | grep 'error TS' || echo "PASS: no plugin errors"`

const TASKS: Record<string, { topic: string; verifyCmd: string }> = {
  clock: {
    topic: `Fix the plugin registry in CodeRPG Desktop so that both clock plugins (analog and digital) are properly registered and the plugin system compiles.

The codebase has these issues:
- src/plugins/index.ts imports types from ./types but src/plugins/types.ts does not exist
- The analog clock panel at src/panels/analog-clock.tsx is NOT registered in the plugin registry
- The digital clock at src/panels/clock.tsx IS registered but the registry is broken because of the missing types

What needs to happen:
1. Read src/windows/types.ts to understand what DesktopPlugin type should look like
2. Read src/plugins/index.ts to see the current registry pattern
3. Create src/plugins/types.ts with the DesktopPlugin type definition
4. Edit src/plugins/index.ts to also import and register the analog clock panel
5. Verify by running: npx tsc --noEmit 2>&1 | grep -E 'src/plugins|src/windows/types' | grep 'error TS' || echo "PASS: no plugin errors"
   If you see "PASS: no plugin errors", the fix is correct. Call complete.
   If you see error lines, read them and fix the issues.

IMPORTANT:
- Only modify files in src/plugins/. Do NOT touch other directories.
- All shell commands run in the working directory already -- do NOT use cd.
- The codebase has ~167 unrelated type errors in other files. Ignore those. Only care about errors in src/plugins/ and src/windows/types.ts.`,
    verifyCmd: "npx tsc --noEmit 2>&1 | grep -E 'src/plugins|src/windows/types' | grep 'error TS' || echo 'PASS: no plugin errors'",
  },
}

const TASK_NAME = process.env.TASK || "clock"
const TASK = TASKS[TASK_NAME]
if (!TASK) {
  console.error(`Unknown task: ${TASK_NAME}. Available: ${Object.keys(TASKS).join(", ")}`)
  process.exit(1)
}

// ── Reset workspace ─────────────────────────────────────────────────────

async function resetWorkspace(): Promise<string> {
  const runDir = `/tmp/coderpg-run-${Date.now()}`
  console.log(`  [setup] Creating isolated workspace: ${runDir}`)
  try {
    execSync(`git -C ${JSON.stringify(WORK_DIR)} checkout . 2>/dev/null`, { stdio: "pipe" })
    execSync(`git -C ${JSON.stringify(WORK_DIR)} clean -fd 2>/dev/null`, { stdio: "pipe" })
    execSync(`rsync -a ${JSON.stringify(WORK_DIR + "/")} ${JSON.stringify(runDir + "/")}`, { stdio: "pipe", timeout: 60000 })
    console.log("  [setup] Workspace ready")
  } catch (err: any) {
    console.error(`  [setup] Reset failed: ${err.message?.slice(0, 100)}`)
  }
  return runDir
}

// ── Verify ──────────────────────────────────────────────────────────────

function verify(workDir: string): { pass: boolean; output: string } {
  try {
    // Run tsc and capture ALL output
    let tscOutput = ""
    try {
      tscOutput = execSync("npx tsc --noEmit 2>&1", {
        cwd: workDir,
        encoding: "utf-8",
        timeout: 60000,
        shell: "/bin/bash",
      })
    } catch (err: any) {
      // tsc exits non-zero when there are errors, but we still get the output
      tscOutput = err.stdout || err.stderr || ""
    }

    // Filter for plugin-related errors only
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

// ── System Prompts ──────────────────────────────────────────────────────

const BASE_TOOL_SYSTEM_PROMPT = `You are a senior TypeScript developer. You fix code by reading files, making edits, and verifying with the compiler.

You have these tools:
- read(path) -- read a file, returns numbered lines
- write(path, content) -- create a new file with full content
- edit(path, old_string, new_string) -- find exact text in a file and replace it
- grep(pattern) -- search for a regex pattern across the codebase
- bash(command) -- run a shell command (already runs in the project directory)
- complete(summary) -- signal you are done (only after verification passes)

Rules:
- All paths are RELATIVE to the project root (e.g. "src/plugins/index.ts"), never absolute
- For edit: old_string must be an EXACT copy of text currently in the file (copy from what read shows you)
- For write: provide the COMPLETE file content
- Shell commands run in the project directory already. Do NOT prefix with cd.
- Always verify your changes compile before calling complete
- Work step by step: read first, understand the pattern, then make changes`

const META_TOOL_SYSTEM_PROMPT = `I am a Senior Developer working on the CodeRPG Desktop codebase.

I have these tools:
- investigate(target, how) -- read files, search code, list dirs. I use relative paths like "src/plugins/index.ts"
- modify(file, old_text, new_text) -- edit code (find and replace). I use relative paths
- execute(command) -- run shell commands
- note(text) -- save a finding to my scratchpad
- finish(summary) -- signal completion (must compile first)

My workflow:
1. I investigate the existing code using relative paths (e.g. investigate("src/plugins/index.ts", "read"))
2. I make changes following the patterns I find
3. I run the TypeScript compiler to verify
4. When it compiles, I call finish

IMPORTANT: I use relative file paths for investigate and modify (e.g. "src/plugins/index.ts"), NOT absolute paths.`

// ── Main ────────────────────────────────────────────────────────────────

async function main() {
  console.log("+====================================================================+")
  console.log("|  CodeRPG Desktop Plugin Experiment                                 |")
  console.log("+====================================================================+")
  console.log(`  Model: ${MODEL}`)
  console.log(`  Endpoint: ${ENDPOINT}`)
  console.log(`  Task: ${TASK_NAME}`)
  console.log(`  Plan mode: ${PLAN_MODE}`)
  console.log(`  Tool mode: ${TOOL_MODE}`)
  console.log(`  Max turns: ${MAX_TURNS}`)

  const runDir = await resetWorkspace()

  // Verify baseline fails (incomplete types)
  const baseline = verify(runDir)
  console.log(`  [baseline] Compilation: ${baseline.pass ? "PASS" : "FAIL"}`)
  if (baseline.pass) {
    console.log("  [baseline] Already compiles — nothing to fix")
    return
  }
  console.log(`  [baseline] Error: ${baseline.output.slice(0, 200)}`)

  const send = createSendWithTools({
    baseUrl: ENDPOINT,
    model: MODEL,
    temperature: 0,
    timeoutMs: 300000,
  })

  let systemPrompt = TOOL_MODE === "base" ? BASE_TOOL_SYSTEM_PROMPT : META_TOOL_SYSTEM_PROMPT
  let plan: PlanArtifact | null = null
  const startTime = Date.now()

  // ── Planning phase ──
  if (PLAN_MODE === "scaffold_plan") {
    console.log("\n  [planning] Generating scaffold plan...")
    plan = await generateScaffoldPlan({
      mode: "scaffold_plan",
      task: TASK.topic,
      workDir: runDir,
      model: MODEL,
      endpoint: ENDPOINT,
      workerModel: MODEL,
      workerEndpoint: ENDPOINT,
      temperature: 0,
    })
    if (plan) {
      console.log(`  [planning] Plan generated (${plan.inferenceCalls} calls, ${plan.tokenCost} tokens)`)
      console.log(`  [planning] Plan content:\n${plan.plan}`)
      systemPrompt = injectPlanIntoPrompt(systemPrompt, plan, runDir)
    }
  }

  // ── Execution phase ──
  console.log("\n  [execution] Starting agent...")
  const ledger = createEmptyLedger(`coderpg-${TASK_NAME}`)
  const baseExecutor = createToolExecutor()

  let executor, tools, notes: string[]

  if (TOOL_MODE === "base") {
    // Base tools with surgical conveyor for edits
    executor = createEnhancedToolExecutor({ sendWithTools: send, model: MODEL })
    tools = getBaseToolSchemas()
    notes = []
  } else {
    // Meta-tool mode (original)
    const meta = createMetaToolExecutor({
      baseExecutor,
      testCmd: TASK.verifyCmd,
      sendWithTools: send,
      model: MODEL,
      surgicalConveyor: true,
    })
    executor = meta.executor
    tools = meta.tools
    notes = meta.state.notes
  }

  const result = await runAgent({
    agentName: "Dev",
    systemPrompt,
    userMessage: `Working directory: ${runDir}\n\n${TASK.topic}`,
    model: MODEL,
    endpoint: ENDPOINT,
    sendWithTools: send,
    executor,
    tools,
    workDir: runDir,
    maxTurns: MAX_TURNS,
    testCmd: TASK.verifyCmd,
    allowedTools: ["read", "write", "edit", "grep", "list", "glob", "bash", "complete"],
    logLabel: `${MODEL.split("/").pop()}/${TASK_NAME}/${PLAN_MODE}/${TOOL_MODE}`,
    notes,
    interceptors: false,
    failureLedger: ledger,
    compaction: true,
    verbose: VERBOSE,
    recoverySend: PLAN_MODE !== "none" ? send : undefined,
    recoveryModel: PLAN_MODE !== "none" ? MODEL : undefined,
    evoObserve: PLAN_MODE !== "none" ? { send, model: MODEL, intervalTurns: 3 } : undefined,
  })

  // ── Final verification ──
  const final = verify(runDir)
  const durationSec = Math.round((Date.now() - startTime) / 1000)
  const totalTokens = (plan?.tokenCost || 0) + result.totalTokens

  console.log("\n" + "=".repeat(70))
  console.log(`  RESULT: ${final.pass ? "PASS" : "FAIL"}`)
  console.log(`  Turns: ${result.turns} (${result.turnsWithToolCalls} with tools)`)
  console.log(`  Tokens: ${totalTokens} (plan: ${plan?.tokenCost || 0}, execution: ${result.totalTokens})`)
  console.log(`  Duration: ${durationSec}s`)
  console.log(`  Loops: ${result.loopsDetected}, Recoveries: ${result.adaptiveRecoveries}`)
  if (!final.pass) {
    console.log(`  Error: ${final.output.slice(0, 300)}`)
  }
  console.log("=".repeat(70))

  // Save result
  const resultFile = path.join("/Users/virtualmachine/plan-lab/results", `coderpg-${TASK_NAME}-${Date.now()}.json`)
  await fs.writeFile(resultFile, JSON.stringify({
    task: TASK_NAME,
    model: MODEL,
    planMode: PLAN_MODE,
    toolMode: TOOL_MODE,
    pass: final.pass,
    turns: result.turns,
    turnsWithToolCalls: result.turnsWithToolCalls,
    totalTokens,
    planTokens: plan?.tokenCost || 0,
    durationSec,
    loopsDetected: result.loopsDetected,
    adaptiveRecoveries: result.adaptiveRecoveries,
    planContent: plan?.plan,
    error: final.pass ? undefined : final.output.slice(0, 500),
  }, null, 2))

  // Cleanup temp workspace only on pass
  if (final.pass) {
    try { execSync(`rm -rf ${JSON.stringify(runDir)}`, { stdio: "pipe" }) } catch {}
  } else {
    console.log(`  [debug] Workspace preserved for inspection: ${runDir}`)
  }

  process.exit(final.pass ? 0 : 1)
}

await main()
