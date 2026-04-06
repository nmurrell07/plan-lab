#!/usr/bin/env bun
/**
 * Plan-Lab Experiment Bench
 *
 * Runs scenarios with and without planning, across models, with meta-tools
 * or base tools, and outputs comparison tables.
 *
 * Usage:
 *   bun run src/bench.ts
 *
 * Environment:
 *   ENDPOINT        LM Studio endpoint (default: http://192.168.50.117:1234)
 *   MODELS          Comma-separated model list (default: all known models)
 *   SCENARIOS       Comma-separated scenario filter (e.g. q01-easy,q05-medium)
 *   PLAN_MODES      Comma-separated plan modes (default: none,evo_plan)
 *   TOOL_MODES      Comma-separated tool modes: base,meta (default: meta)
 *   MAX_TURNS       Max turns per run (default: 15)
 *   PLAN_ONLY       If 1, only run planning phase (no execution)
 *   RESULTS_DIR     Where to write results (default: ./results)
 */

import fs from "fs/promises"
import { existsSync } from "fs"
import os from "os"
import path from "path"
import { createSendWithTools, healthCheck } from "./client"
import type { SendWithToolsFn } from "./client"
import { createToolExecutor, getBaseToolSchemas } from "./tools"
import { createMetaToolExecutor, getMetaToolDefs } from "./meta-tools"
import { runAgent } from "./runner"
import type { RunResult } from "./runner"
import { generatePlan, injectPlanIntoPrompt } from "./planner"
import type { PlanningMode, PlanArtifact } from "./planner"
import { createEmptyLedger, printLedgerScorecard } from "./ledger"
import type { FailureLedger } from "./ledger"

// ── Config ────────────────────────────────────────────────────────────

const ENDPOINT_A = "http://192.168.50.117:1234"
const ENDPOINT_B = "http://192.168.50.206:1234"

interface ModelEntry {
  label: string
  model: string
  endpoint: string
}

const ALL_MODELS: ModelEntry[] = [
  { label: "9B-qwen",  model: "qwen/qwen3.5-9b",              endpoint: ENDPOINT_A },
  { label: "9B-flash", model: "zai-org/glm-4.7-flash",         endpoint: ENDPOINT_B },
  { label: "27B",      model: "qwen3.5-27b@q8_0",              endpoint: ENDPOINT_A },
  { label: "35B",      model: "qwen/qwen3.5-35b-a3b",          endpoint: ENDPOINT_B },
  { label: "120B",     model: "openai/gpt-oss-120b",            endpoint: ENDPOINT_A },
  { label: "397B",     model: "qwen3.5-397b-a17b",             endpoint: ENDPOINT_B },
]

const MODEL_FILTER = process.env.MODELS?.split(",").map(s => s.trim()) ?? null
const MAX_TURNS = parseInt(process.env.MAX_TURNS ?? "15", 10)
const TIMEOUT_MS = parseInt(process.env.TIMEOUT_MS ?? "300000", 10)
const RESULTS_DIR = process.env.RESULTS_DIR || "./results"
const PLAN_ONLY = process.env.PLAN_ONLY === "1"

type ToolMode = "base" | "meta"

const PLAN_MODES: PlanningMode[] = (process.env.PLAN_MODES?.split(",") as PlanningMode[])
  ?? ["none", "evo_plan"]
const TOOL_MODES: ToolMode[] = (process.env.TOOL_MODES?.split(",") as ToolMode[])
  ?? ["meta"]
const SCENARIO_FILTER = process.env.SCENARIOS?.split(",").map(s => s.trim()) ?? null

// ── Scenario loading ──────────────────────────────────────────────────

interface Scenario {
  id: string
  tier: string
  topic: string
  files: Record<string, string>
  buggyFiles: string[]
  testCmd: string
}

async function loadScenarios(): Promise<Scenario[]> {
  const scenarioDir = path.resolve("scenarios")
  const entries = await fs.readdir(scenarioDir)
  const jsonFiles = entries.filter(f => f.endsWith(".json")).sort()

  const scenarios: Scenario[] = []
  for (const file of jsonFiles) {
    const name = file.replace(".json", "")
    if (SCENARIO_FILTER && !SCENARIO_FILTER.includes(name)) continue
    const content = await fs.readFile(path.join(scenarioDir, file), "utf-8")
    scenarios.push(JSON.parse(content))
  }
  return scenarios
}

// ── Workspace setup ───────────────────────────────────────────────────

async function createWorkspace(scenario: Scenario): Promise<string> {
  const workDir = await fs.mkdtemp(path.join(os.tmpdir(), `plan-lab-${scenario.id}-`))
  for (const [relPath, content] of Object.entries(scenario.files)) {
    const fullPath = path.join(workDir, relPath)
    await fs.mkdir(path.dirname(fullPath), { recursive: true })
    await fs.writeFile(fullPath, content)
  }
  return workDir
}

// ── System prompts ────────────────────────────────────────────────────

const BASE_SYSTEM_PROMPT = `You are a Senior Developer debugging code.

You have standard development tools: read files, edit code, run commands, search code, list directories.

Workflow:
1. Read the buggy files to understand the code
2. Run the test command to see failures
3. Fix the bugs by editing the code
4. Run tests again to verify
5. When tests pass, call complete with a summary`

const META_SYSTEM_PROMPT = `You are a Senior Developer debugging code.

You have exactly 5 tools:
- investigate(target, how) -- read files, search code, list dirs
- modify(file, old_text, new_text) -- edit code (find and replace)
- execute(command) -- run tests and commands
- note(text) -- save a finding to your scratchpad
- finish(summary) -- signal completion (tests must pass first)

Workflow:
1. investigate the buggy files (read them)
2. execute the test command to see failures
3. modify the code to fix bugs
4. execute tests again
5. When tests pass, call finish`

// ── Single run ────────────────────────────────────────────────────────

interface ExperimentResult {
  scenarioId: string
  tier: string
  modelLabel: string
  model: string
  planMode: PlanningMode
  toolMode: ToolMode
  testsPass: boolean
  filesModified: boolean
  turns: number
  turnsWithToolCalls: number
  totalTokens: number
  durationSec: number
  planTokens: number
  planDurationMs: number
  planContent?: string
  error?: string
  interceptorFirings: number
  loopsDetected: number
  dandelionsDetected: number
  compactions: number
  recoveries: number
}

async function runExperiment(
  scenario: Scenario,
  modelEntry: ModelEntry,
  planMode: PlanningMode,
  toolMode: ToolMode,
): Promise<ExperimentResult> {
  const workDir = await createWorkspace(scenario)
  const startTime = Date.now()

  const result: ExperimentResult = {
    scenarioId: scenario.id,
    tier: scenario.tier,
    modelLabel: modelEntry.label,
    model: modelEntry.model,
    planMode,
    toolMode,
    testsPass: false,
    filesModified: false,
    turns: 0,
    turnsWithToolCalls: 0,
    totalTokens: 0,
    durationSec: 0,
    planTokens: 0,
    planDurationMs: 0,
    interceptorFirings: 0,
    loopsDetected: 0,
    dandelionsDetected: 0,
    compactions: 0,
    recoveries: 0,
  }

  // Create a fresh failure ledger per experiment
  const ledger = createEmptyLedger(`${scenario.id}-${modelEntry.label}`)

  try {
    // ── Planning phase ──
    let plan: PlanArtifact | null = null
    if (planMode !== "none") {
      plan = await generatePlan({
        mode: planMode,
        task: scenario.topic,
        workDir,
        model: modelEntry.model,
        endpoint: modelEntry.endpoint,
        workerModel: modelEntry.model,
        workerEndpoint: modelEntry.endpoint,
        temperature: 0,
      })
      if (plan) {
        result.planTokens = plan.tokenCost
        result.planDurationMs = plan.durationMs
        result.planContent = plan.plan
        result.totalTokens += plan.tokenCost
      }
    }

    if (PLAN_ONLY) {
      result.durationSec = Math.round((Date.now() - startTime) / 1000)
      return result
    }

    // ── Re-create workspace (planning workers might have modified it) ──
    await fs.rm(workDir, { recursive: true, force: true })
    const execWorkDir = await createWorkspace(scenario)

    // ── Execution phase ──
    const send = createSendWithTools({
      baseUrl: modelEntry.endpoint,
      model: modelEntry.model,
      temperature: 0,
      timeoutMs: TIMEOUT_MS,
    })

    let basePrompt = toolMode === "meta" ? META_SYSTEM_PROMPT : BASE_SYSTEM_PROMPT
    if (plan) {
      basePrompt = injectPlanIntoPrompt(basePrompt, plan, execWorkDir)
    }

    const baseExecutor = createToolExecutor()

    let executor, tools
    if (toolMode === "meta") {
      const meta = createMetaToolExecutor({
        baseExecutor,
        testCmd: scenario.testCmd,
        buggyFiles: scenario.buggyFiles,
      })
      executor = meta.executor
      tools = meta.tools
    } else {
      executor = baseExecutor
      tools = getBaseToolSchemas()
    }

    const runResult = await runAgent({
      agentName: "Dev",
      systemPrompt: basePrompt,
      userMessage: scenario.topic,
      model: modelEntry.model,
      endpoint: modelEntry.endpoint,
      sendWithTools: send,
      executor,
      tools,
      workDir: execWorkDir,
      maxTurns: MAX_TURNS,
      testCmd: scenario.testCmd,
      allowedTools: ["read", "write", "edit", "grep", "list", "glob", "bash", "complete"],
      logLabel: `${modelEntry.label}/${scenario.id}/${planMode}/${toolMode}`,
      interceptors: true,
      failureLedger: ledger,
      compaction: true,
      runId: `${scenario.id}-${modelEntry.label}-${planMode}-${toolMode}`,
    })

    result.testsPass = runResult.testsPass
    result.filesModified = runResult.filesModified
    result.turns = runResult.turns
    result.turnsWithToolCalls = runResult.turnsWithToolCalls
    result.totalTokens += runResult.totalTokens
    result.error = runResult.error
    result.interceptorFirings = runResult.interceptorFirings
    result.loopsDetected = runResult.loopsDetected
    result.dandelionsDetected = runResult.dandelionsDetected
    result.compactions = runResult.compactions
    result.recoveries = runResult.recoveries

    // Print ledger scorecard if there were failures
    if (ledger.meta.totalEntries > 0) {
      console.log(`  ${printLedgerScorecard(ledger)}`)
    }

    await fs.rm(execWorkDir, { recursive: true, force: true }).catch(() => {})
  } catch (err: any) {
    result.error = err.message?.slice(0, 200)
    console.error(`  ERROR: ${result.error}`)
  }

  result.durationSec = Math.round((Date.now() - startTime) / 1000)

  // Cleanup
  await fs.rm(workDir, { recursive: true, force: true }).catch(() => {})

  return result
}

// ── Report ────────────────────────────────────────────────────────────

function printReport(results: ExperimentResult[], scenarios: Scenario[]) {
  const scenarioIds = scenarios.map(s => s.id)

  console.log("\n" + "=".repeat(100))
  console.log("  PLAN-LAB EXPERIMENT RESULTS")
  console.log("=".repeat(100))

  // Group by plan+tool mode combination
  const combos = [...new Set(results.map(r => `${r.planMode}+${r.toolMode}`))]

  for (const combo of combos) {
    const [planMode, toolMode] = combo.split("+")
    const comboResults = results.filter(r => r.planMode === planMode && r.toolMode === toolMode)
    const models = [...new Set(comboResults.map(r => r.modelLabel))]

    console.log(`\n  --- ${planMode} + ${toolMode} tools ---`)
    console.log(
      "  " + "Model".padEnd(12) +
      scenarioIds.map(id => id.padEnd(8)).join("") +
      "Score".padEnd(8) + "Time".padEnd(8) + "Tokens"
    )
    console.log("  " + "-".repeat(80))

    for (const label of models) {
      let line = "  " + label.padEnd(12)
      let passed = 0

      for (const sid of scenarioIds) {
        const r = comboResults.find(cr => cr.modelLabel === label && cr.scenarioId === sid)
        if (!r) { line += "--".padEnd(8); continue }
        const icon = r.testsPass ? "PASS" : r.filesModified ? "FIX" : "FAIL"
        line += icon.padEnd(8)
        if (r.testsPass) passed++
      }

      const totalTime = comboResults.filter(r => r.modelLabel === label).reduce((s, r) => s + r.durationSec, 0)
      const totalTok = comboResults.filter(r => r.modelLabel === label).reduce((s, r) => s + r.totalTokens, 0)
      line += `${passed}/${scenarioIds.length}`.padEnd(8) + `${totalTime}s`.padEnd(8) + `${totalTok}`
      console.log(line)
    }
  }

  // ── Delta report (planning vs no planning) ──
  const noneResults = results.filter(r => r.planMode === "none")
  const planResults = results.filter(r => r.planMode !== "none")

  if (noneResults.length > 0 && planResults.length > 0) {
    console.log("\n  --- PLANNING DELTA ---")
    console.log("  " + "Model".padEnd(12) + "Scenario".padEnd(10) + "No Plan".padEnd(10) + "With Plan".padEnd(12) + "Plan Mode".padEnd(15) + "Delta Tokens")
    console.log("  " + "-".repeat(70))

    const models = [...new Set(results.map(r => r.modelLabel))]
    for (const label of models) {
      for (const sid of scenarioIds) {
        const baseline = noneResults.find(r => r.modelLabel === label && r.scenarioId === sid)
        const planned = planResults.find(r => r.modelLabel === label && r.scenarioId === sid)
        if (!baseline || !planned) continue

        const bIcon = baseline.testsPass ? "PASS" : "FAIL"
        const pIcon = planned.testsPass ? "PASS" : "FAIL"
        const delta = planned.totalTokens - baseline.totalTokens
        const deltaStr = delta > 0 ? `+${delta}` : `${delta}`
        const improved = !baseline.testsPass && planned.testsPass ? " << IMPROVED" : ""

        console.log(
          "  " + label.padEnd(12) + sid.padEnd(10) + bIcon.padEnd(10) +
          pIcon.padEnd(12) + planned.planMode.padEnd(15) + deltaStr + improved
        )
      }
    }
  }

  const totalPassed = results.filter(r => r.testsPass).length
  console.log("\n" + "=".repeat(100))
  console.log(`  ${totalPassed}/${results.length} experiments passed`)
  console.log("=".repeat(100))
}

// ── Save results ──────────────────────────────────────────────────────

async function saveResults(results: ExperimentResult[]) {
  await fs.mkdir(RESULTS_DIR, { recursive: true })
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-")
  const filePath = path.join(RESULTS_DIR, `bench-${timestamp}.json`)
  await fs.writeFile(filePath, JSON.stringify(results, null, 2))
  console.log(`\n  Results saved to ${filePath}`)

  // Also append to NDJSON log for time-series analysis
  const ndjsonPath = path.join(RESULTS_DIR, "bench-log.ndjson")
  const lines = results.map(r => JSON.stringify({ ...r, timestamp: new Date().toISOString() })).join("\n") + "\n"
  await fs.appendFile(ndjsonPath, lines)
}

// ── Main ──────────────────────────────────────────────────────────────

async function main() {
  console.log("+" + "=".repeat(68) + "+")
  console.log("|  Plan-Lab Experiment Bench                                         |")
  console.log("+" + "=".repeat(68) + "+")

  // Load scenarios
  const scenarios = await loadScenarios()
  if (scenarios.length === 0) {
    console.log("  No scenarios found. Check the scenarios/ directory.")
    process.exit(1)
  }

  // Determine models
  let models: ModelEntry[]
  if (process.env.MODEL && process.env.ENDPOINT) {
    models = [{
      label: process.env.MODEL.split("/").pop() || process.env.MODEL,
      model: process.env.MODEL,
      endpoint: process.env.ENDPOINT,
    }]
  } else if (MODEL_FILTER) {
    models = ALL_MODELS.filter(m => MODEL_FILTER.includes(m.label))
  } else {
    models = ALL_MODELS
  }

  console.log(`  Scenarios: ${scenarios.map(s => `${s.id}[${s.tier}]`).join(", ")}`)
  console.log(`  Models: ${models.map(m => m.label).join(", ")}`)
  console.log(`  Plan modes: ${PLAN_MODES.join(", ")}`)
  console.log(`  Tool modes: ${TOOL_MODES.join(", ")}`)
  console.log(`  Max turns: ${MAX_TURNS}`)
  if (PLAN_ONLY) console.log("  MODE: Plan-only (no execution)")

  // Health check endpoints
  const endpoints = [...new Set(models.map(m => m.endpoint))]
  for (const ep of endpoints) {
    const health = await healthCheck(ep)
    if (!health.ok) {
      console.log(`  [WARN] ${ep} unreachable: ${health.error}`)
      models = models.filter(m => m.endpoint !== ep)
    } else {
      console.log(`  [health] ${ep} OK (${health.models.length} models)`)
    }
  }
  if (models.length === 0) {
    console.log("  No reachable models. Exiting.")
    process.exit(1)
  }

  // Run experiments
  const allResults: ExperimentResult[] = []
  const totalExperiments = models.length * scenarios.length * PLAN_MODES.length * TOOL_MODES.length
  let completed = 0

  for (const model of models) {
    for (const scenario of scenarios) {
      for (const planMode of PLAN_MODES) {
        for (const toolMode of TOOL_MODES) {
          completed++
          console.log(`\n  == [${completed}/${totalExperiments}] ${model.label} / ${scenario.id} / ${planMode} / ${toolMode} ==`)

          const result = await runExperiment(scenario, model, planMode, toolMode)
          allResults.push(result)

          const status = result.testsPass ? "PASS" : result.filesModified ? "FIX" : "FAIL"
          console.log(`  == ${status} (${result.durationSec}s, ${result.totalTokens} tok) ==`)
        }
      }
    }
  }

  // Report
  printReport(allResults, scenarios)
  await saveResults(allResults)
}

await main()
