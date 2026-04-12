#!/usr/bin/env bun
/**
 * Orchestrator Grid Test — Skyrim Mod Registration (Novel Pattern Discovery)
 *
 * Tests whether the orchestrator can discover a novel registration pattern
 * (Skyrim mod system) by studying existing mods, then create a new mod
 * following the discovered pattern.
 *
 * Grid:
 *   Orchestrators: 27B, 35B
 *   Workers: 9B, 27B, 35B
 *   = 6 combinations
 *
 * Task: Create a custom power/shout mod by studying existing mod patterns.
 *
 * Usage:
 *   bun run src/orch-grid-skyrim.ts
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
  "9B":  { label: "9B",  model: "qwen/qwen3.5-9b",      endpoint: ENDPOINT_A },
  "27B": { label: "27B", model: "qwen3.5-27b",             endpoint: ENDPOINT_A },
  "35B": { label: "35B", model: "qwen/qwen3.5-35b-a3b",  endpoint: ENDPOINT_B },
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
]

// ── Skyrim Mod Structure Scaffold ───────────────────────────────────────

/**
 * Creates a mock Skyrim mod workspace with existing mods as reference patterns.
 * The model must discover the registration pattern from these examples.
 */
async function createSkyrimWorkspace(): Promise<string> {
  const workDir = `/tmp/skyrim-modtest-${Date.now()}`
  await fs.mkdir(workDir, { recursive: true })

  // ── Load order file (master registration) ──
  await fs.writeFile(path.join(workDir, "plugins.txt"), `# Skyrim Plugin Load Order
# Each line is a plugin (.esp/.esm) filename
# Plugins are loaded in order from top to bottom
# Lines starting with # are comments
# A * prefix means the plugin is active
*Skyrim.esm
*Update.esm
*Dawnguard.esm
*HearthFires.esm
*Dragonborn.esm
*FireballSpell.esp
*FrostCloakSpell.esp
`)

  // ── Mod manager manifest ──
  await fs.writeFile(path.join(workDir, "modlist.json"), JSON.stringify({
    version: 2,
    gameVersion: "1.6.1170",
    mods: [
      {
        name: "FireballSpell",
        plugin: "FireballSpell.esp",
        author: "ModAuthor1",
        version: "1.0.0",
        description: "Adds an enhanced fireball destruction spell",
        files: [
          "Data/FireballSpell.esp",
          "Data/Scripts/FireballSpellScript.psc",
          "Data/Scripts/FireballSpellScript.pex",
          "Data/Meshes/FireballSpell/fireballprojectile.nif",
          "Data/Textures/FireballSpell/fireballtex.dds"
        ],
        registration: {
          type: "spell",
          school: "destruction",
          formId: "0x000D62",
          editorId: "EnhancedFireball"
        }
      },
      {
        name: "FrostCloakSpell",
        plugin: "FrostCloakSpell.esp",
        author: "ModAuthor2",
        version: "2.1.0",
        description: "Adds an improved frost cloak spell with visual effects",
        files: [
          "Data/FrostCloakSpell.esp",
          "Data/Scripts/FrostCloakScript.psc",
          "Data/Scripts/FrostCloakScript.pex",
          "Data/Meshes/FrostCloak/frostcloakeffect.nif",
          "Data/Textures/FrostCloak/frostcloaktex.dds"
        ],
        registration: {
          type: "spell",
          school: "destruction",
          formId: "0x000E73",
          editorId: "ImprovedFrostCloak"
        }
      }
    ]
  }, null, 2))

  // ── Data directory structure ──
  const dirs = [
    "Data",
    "Data/Scripts",
    "Data/Scripts/Source",
    "Data/Meshes/FireballSpell",
    "Data/Meshes/FrostCloak",
    "Data/Textures/FireballSpell",
    "Data/Textures/FrostCloak",
  ]
  for (const dir of dirs) {
    await fs.mkdir(path.join(workDir, dir), { recursive: true })
  }

  // ── Existing mod: FireballSpell ──
  // Plugin file (mock .esp — just metadata since we can't create real Creation Kit data)
  await fs.writeFile(path.join(workDir, "Data/FireballSpell.esp"), `; TES5 Plugin: FireballSpell
; Author: ModAuthor1
; Version: 1.0.0
; Description: Adds an enhanced fireball destruction spell
;
; Record: SPEL (Spell)
;   EditorID: EnhancedFireball
;   FormID: 0x000D62
;   Name: "Enhanced Fireball"
;   Type: SpellAbility
;   School: Destruction
;   Cost: 85
;   CastType: FireAndForget
;   DeliveryType: AimedProjectile
;   BaseDamage: 75
;   EffectArea: 15
;
; Record: MGEF (Magic Effect)
;   EditorID: EnhancedFireballEffect
;   FormID: 0x000D63
;   Name: "Enhanced Fireball Effect"
;   School: Destruction
;   BaseCost: 12.5
;   Archetype: ValueModifier
;   ActorValue: Health
;   Script: FireballSpellScript
;
; Record: PROJ (Projectile)
;   EditorID: EnhancedFireballProjectile
;   FormID: 0x000D64
;   Model: Meshes/FireballSpell/fireballprojectile.nif
;   Speed: 1200
;   Gravity: 0.1
;   ExplosionType: FireballExplosion
`)

  // Papyrus script source
  await fs.writeFile(path.join(workDir, "Data/Scripts/Source/FireballSpellScript.psc"), `Scriptname FireballSpellScript extends ActiveMagicEffect
{Script for the Enhanced Fireball spell effect}

; Properties
float Property BaseDamage = 75.0 Auto
float Property EffectArea = 15.0 Auto
string Property SpellName = "Enhanced Fireball" Auto

; Events
Event OnEffectStart(Actor akTarget, Actor akCaster)
  Debug.Trace("[FireballSpell] Enhanced Fireball cast by " + akCaster.GetDisplayName())
  ; Apply fire damage in area
  akTarget.DamageActorValue("Health", BaseDamage)
  ; Visual effect
  Game.ShakeCamera(akTarget, 0.5, 0.3)
EndEvent

Event OnEffectFinish(Actor akTarget, Actor akCaster)
  Debug.Trace("[FireballSpell] Enhanced Fireball effect ended")
EndEvent
`)

  // Compiled script (mock .pex)
  await fs.writeFile(path.join(workDir, "Data/Scripts/FireballSpellScript.pex"), `; Compiled Papyrus script
; Source: Scripts/Source/FireballSpellScript.psc
; Compiler: PapyrusCompiler v2.7.1
; Timestamp: 2024-01-15T10:30:00Z
; [binary data would be here in a real .pex file]
`)

  // Mesh placeholder
  await fs.writeFile(path.join(workDir, "Data/Meshes/FireballSpell/fireballprojectile.nif"), `; NIF Model placeholder
; FireballSpell projectile mesh
`)

  // Texture placeholder
  await fs.writeFile(path.join(workDir, "Data/Textures/FireballSpell/fireballtex.dds"), `; DDS Texture placeholder
; FireballSpell texture
`)

  // ── Existing mod: FrostCloakSpell ──
  await fs.writeFile(path.join(workDir, "Data/FrostCloakSpell.esp"), `; TES5 Plugin: FrostCloakSpell
; Author: ModAuthor2
; Version: 2.1.0
; Description: Adds an improved frost cloak spell with visual effects
;
; Record: SPEL (Spell)
;   EditorID: ImprovedFrostCloak
;   FormID: 0x000E73
;   Name: "Improved Frost Cloak"
;   Type: SpellAbility
;   School: Destruction
;   Cost: 60
;   CastType: FireAndForget
;   DeliveryType: Self
;   Duration: 45
;   DamagePerSecond: 12
;
; Record: MGEF (Magic Effect)
;   EditorID: ImprovedFrostCloakEffect
;   FormID: 0x000E74
;   Name: "Improved Frost Cloak Effect"
;   School: Destruction
;   BaseCost: 8.0
;   Archetype: Cloak
;   ActorValue: Health
;   Script: FrostCloakScript
;
; Record: PROJ (Projectile)
;   EditorID: FrostCloakVFX
;   FormID: 0x000E75
;   Model: Meshes/FrostCloak/frostcloakeffect.nif
;   Type: BarrierEffect
`)

  await fs.writeFile(path.join(workDir, "Data/Scripts/Source/FrostCloakScript.psc"), `Scriptname FrostCloakScript extends ActiveMagicEffect
{Script for the Improved Frost Cloak spell effect}

; Properties
float Property DamagePerSecond = 12.0 Auto
float Property Duration = 45.0 Auto
float Property CloakRadius = 10.0 Auto
string Property SpellName = "Improved Frost Cloak" Auto

; State tracking
bool isActive = false

; Events
Event OnEffectStart(Actor akTarget, Actor akCaster)
  Debug.Trace("[FrostCloak] Improved Frost Cloak activated by " + akCaster.GetDisplayName())
  isActive = true
  RegisterForUpdate(1.0)
EndEvent

Event OnUpdate()
  if isActive
    ; Find nearby actors and apply frost damage
    Actor self = GetTargetActor()
    if self
      ; Apply frost damage to nearby enemies
      Debug.Trace("[FrostCloak] Frost cloak pulse - dealing " + DamagePerSecond + " damage")
    endif
  endif
EndEvent

Event OnEffectFinish(Actor akTarget, Actor akCaster)
  Debug.Trace("[FrostCloak] Improved Frost Cloak deactivated")
  isActive = false
  UnregisterForUpdate()
EndEvent
`)

  await fs.writeFile(path.join(workDir, "Data/Scripts/FrostCloakScript.pex"), `; Compiled Papyrus script
; Source: Scripts/Source/FrostCloakScript.psc
; Compiler: PapyrusCompiler v2.7.1
; Timestamp: 2024-03-22T14:15:00Z
; [binary data would be here in a real .pex file]
`)

  await fs.writeFile(path.join(workDir, "Data/Meshes/FrostCloak/frostcloakeffect.nif"), `; NIF Model placeholder
; FrostCloak visual effect mesh
`)

  await fs.writeFile(path.join(workDir, "Data/Textures/FrostCloak/frostcloaktex.dds"), `; DDS Texture placeholder
; FrostCloak texture
`)

  // ── README for the modding workspace ──
  await fs.writeFile(path.join(workDir, "README.txt"), `Skyrim Mod Development Workspace
=================================

This workspace contains Skyrim mods. Each mod consists of:

1. A plugin file (.esp) in Data/ - defines the mod's records (spells, effects, etc.)
2. Papyrus scripts in Data/Scripts/Source/ (.psc files) and compiled in Data/Scripts/ (.pex)
3. Assets (meshes, textures) in Data/Meshes/ and Data/Textures/
4. Registration in plugins.txt (load order) and modlist.json (mod manifest)

To create a new mod:
- Study the existing mods to understand the file structure and patterns
- Follow the same conventions for naming, directory structure, and registration
- All mods must be registered in both plugins.txt and modlist.json

Existing mods:
- FireballSpell: Enhanced fireball destruction spell
- FrostCloakSpell: Improved frost cloak with visual effects
`)

  // ── Verification script ──
  await fs.writeFile(path.join(workDir, "verify-mod.sh"), `#!/bin/bash
# Verify that a new mod is properly created and registered
# Usage: bash verify-mod.sh

ERRORS=0

# Check 1: New .esp file exists in Data/
NEW_ESPS=$(find Data/ -maxdepth 1 -name "*.esp" | grep -v "FireballSpell\\.esp" | grep -v "FrostCloakSpell\\.esp")
if [ -z "$NEW_ESPS" ]; then
  echo "FAIL: No new .esp plugin file found in Data/"
  ERRORS=$((ERRORS + 1))
else
  echo "PASS: New plugin found: $NEW_ESPS"
fi

# Check 2: New Papyrus script source exists
NEW_SCRIPTS=$(find Data/Scripts/Source/ -name "*.psc" | grep -v "FireballSpellScript\\.psc" | grep -v "FrostCloakScript\\.psc")
if [ -z "$NEW_SCRIPTS" ]; then
  echo "FAIL: No new Papyrus script source (.psc) found in Data/Scripts/Source/"
  ERRORS=$((ERRORS + 1))
else
  echo "PASS: New script found: $NEW_SCRIPTS"
fi

# Check 3: plugins.txt has a new entry
BASELINE_COUNT=7  # number of original entries (including Skyrim.esm etc.)
CURRENT_COUNT=$(grep -c "^\\*" plugins.txt 2>/dev/null || echo 0)
if [ "$CURRENT_COUNT" -le "$BASELINE_COUNT" ]; then
  echo "FAIL: No new entry in plugins.txt (expected > $BASELINE_COUNT active plugins, got $CURRENT_COUNT)"
  ERRORS=$((ERRORS + 1))
else
  echo "PASS: plugins.txt has new entry ($CURRENT_COUNT active plugins)"
fi

# Check 4: modlist.json has a new mod entry
BASELINE_MODS=2  # FireballSpell + FrostCloakSpell
CURRENT_MODS=$(grep -c '"name"' modlist.json 2>/dev/null || echo 0)
if [ "$CURRENT_MODS" -le "$BASELINE_MODS" ]; then
  echo "FAIL: No new mod in modlist.json (expected > $BASELINE_MODS mods, got $CURRENT_MODS)"
  ERRORS=$((ERRORS + 1))
else
  echo "PASS: modlist.json has new mod ($CURRENT_MODS mods)"
fi

# Check 5: New .esp has proper structure (contains Record sections)
if [ -n "$NEW_ESPS" ]; then
  for esp in $NEW_ESPS; do
    if grep -q "Record: SPEL" "$esp" 2>/dev/null; then
      echo "PASS: $esp contains spell record definition"
    else
      echo "FAIL: $esp missing spell record definition (Record: SPEL)"
      ERRORS=$((ERRORS + 1))
    fi
    if grep -q "EditorID:" "$esp" 2>/dev/null; then
      echo "PASS: $esp has EditorID"
    else
      echo "FAIL: $esp missing EditorID"
      ERRORS=$((ERRORS + 1))
    fi
    if grep -q "FormID:" "$esp" 2>/dev/null; then
      echo "PASS: $esp has FormID"
    else
      echo "FAIL: $esp missing FormID"
      ERRORS=$((ERRORS + 1))
    fi
  done
fi

# Check 6: New script has proper Papyrus structure
if [ -n "$NEW_SCRIPTS" ]; then
  for script in $NEW_SCRIPTS; do
    if grep -q "Scriptname" "$script" 2>/dev/null; then
      echo "PASS: $script has Scriptname declaration"
    else
      echo "FAIL: $script missing Scriptname declaration"
      ERRORS=$((ERRORS + 1))
    fi
    if grep -q "Event " "$script" 2>/dev/null; then
      echo "PASS: $script has event handlers"
    else
      echo "FAIL: $script missing event handlers"
      ERRORS=$((ERRORS + 1))
    fi
  done
fi

echo ""
if [ "$ERRORS" -eq 0 ]; then
  echo "ALL CHECKS PASSED"
  exit 0
else
  echo "FAILED: $ERRORS check(s) failed"
  exit 1
fi
`)
  execSync(`chmod +x ${JSON.stringify(path.join(workDir, "verify-mod.sh"))}`)

  // Initialize git for diff tracking
  execSync(`git init ${JSON.stringify(workDir)}`, { stdio: "pipe" })
  execSync(`git -C ${JSON.stringify(workDir)} add -A`, { stdio: "pipe" })
  execSync(`git -C ${JSON.stringify(workDir)} commit -m "baseline skyrim mod workspace"`, { stdio: "pipe" })

  return workDir
}

// ── Task Description ────────────────────────────────────────────────────

const TASK_DESCRIPTION = `Create a new Skyrim mod that adds a custom power or shout. Study the existing mod structure to understand the registration pattern, then create a new mod following the same conventions.

I need to:
1. Explore the workspace to understand the mod structure
2. Study the existing mods (FireballSpell and FrostCloakSpell) to learn the pattern
3. Look at how mods are registered in plugins.txt and modlist.json
4. Create a new mod with all required files following the discovered pattern
5. Register the new mod in plugins.txt and modlist.json
6. Verify by running: bash verify-mod.sh

The verification script checks that:
- A new .esp plugin file exists in Data/
- A new Papyrus script source (.psc) exists in Data/Scripts/Source/
- plugins.txt has a new active entry
- modlist.json has a new mod entry
- The .esp has proper record structure (SPEL record, EditorID, FormID)
- The script has proper Papyrus structure (Scriptname, Event handlers)

IMPORTANT: I must discover the pattern by reading existing files. I should NOT make assumptions about the format.`

const VERIFY_CMD = "bash verify-mod.sh"

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

// ── Verification ────────────────────────────────────────────────────────

function verify(workDir: string): { pass: boolean; output: string } {
  try {
    let output = ""
    try {
      output = execSync("bash verify-mod.sh 2>&1", {
        cwd: workDir,
        encoding: "utf-8",
        timeout: 30000,
        shell: "/bin/bash",
      })
      return { pass: output.includes("ALL CHECKS PASSED"), output }
    } catch (err: any) {
      output = err.stdout || err.stderr || ""
      return { pass: false, output }
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
      const workDir = await createSkyrimWorkspace()
      state.workDir = workDir

      const listing = shellExec(`find ${JSON.stringify(workDir)} -not -path '*/.git/*' -type f | sort`)
      const relativeListing = listing.replace(new RegExp(workDir + "/?", "g"), "")

      return {
        tool: "setup_workspace",
        success: true,
        output: `Workspace ready at: ${workDir}\n\nFile listing:\n${relativeListing}`,
      }
    } catch (err: any) {
      return { tool: "setup_workspace", success: false, output: "", error: err.message?.slice(0, 300) }
    }
  }

  async function toolSpawnWorker(args: Record<string, string>): Promise<ToolCallResult> {
    const task = args.task || ""
    const workDir = state.workDir || ""
    if (!task) {
      return { tool: "spawn_worker", success: false, output: "", error: "Missing task. Usage: spawn_worker(task='description')" }
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

    const workerSystemPrompt = `I am a Skyrim modder creating a new mod. I study existing mods to learn the file structure and registration pattern, then create a new mod following those conventions.

I have these tools:
- read(path) -- read a file, returns numbered lines
- write(path, content) -- create a new file with full content
- edit(path, old_string, new_string) -- find exact text in a file and replace it
- grep(pattern) -- search for a regex pattern across the workspace
- bash(command) -- run a shell command (already in the workspace directory)
- list(path) -- list directory contents
- glob(pattern) -- find files by name pattern
- complete(summary) -- signal I am done

My rules:
- All paths are RELATIVE to the workspace root (e.g. "Data/MyMod.esp"), never absolute
- For edit: old_string must be an EXACT copy of text currently in the file
- Shell commands run in the workspace directory. I do NOT prefix with cd.
- I explore FIRST to discover the pattern, then create files following it
- I make sure all registration files are updated
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

    const ledger = createEmptyLedger("skyrim-worker")
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
        logLabel: `skyrim-worker/${workerModel.split("/").pop()}`,
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
        verifyResult.output,
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
      architect: `I am reviewing a Skyrim mod creation diff as a senior modder. I check:
- File structure: does the new mod follow the same directory patterns as existing mods?
- Registration: is the mod properly registered in both plugins.txt and modlist.json?
- Plugin format: does the .esp follow the same record structure?
- Script format: does the Papyrus script follow conventions?

I produce CONCRETE findings. If it looks correct, I say so briefly.`,

      qa: `I am a QA reviewer checking a Skyrim mod for completeness:
- Are all required files present? (.esp, .psc, .pex, mesh, texture)
- Is the load order correct in plugins.txt?
- Is the modlist.json entry complete with all required fields?
- Do FormIDs avoid conflicts with existing mods?

I produce CONCRETE findings. If everything checks out, I say so briefly.`,
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
      const severity = noIssues ? "info" : "warning"

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
    return { tool: "verify", success: result.pass, output: result.output }
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
            results.push({ tool: call.tool, success: false, output: "", error: `Unknown tool: ${call.tool}` })
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
        description: "Create an isolated Skyrim mod development workspace with existing reference mods. Must be called first.",
        parameters: { type: "object", properties: {} },
      },
    },
    {
      type: "function",
      function: {
        name: "spawn_worker",
        description: "Launch a coding worker to create the new Skyrim mod. Returns execution summary, verification result, and diff.",
        parameters: {
          type: "object",
          properties: {
            task: { type: "string", description: "Full task description for the worker" },
          },
          required: ["task"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "run_specialist",
        description: "Run a specialist review on the latest worker's diff.",
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
        description: "Run the verification script to check if the new mod is properly created and registered.",
        parameters: { type: "object", properties: {} },
      },
    },
    {
      type: "function",
      function: {
        name: "submit_result",
        description: "Finalize the workflow.",
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

const ORCHESTRATOR_SYSTEM_PROMPT = `I am an orchestrator managing a Skyrim mod creation workflow.

My tools:
- setup_workspace() -- create the Skyrim mod development workspace with existing reference mods
- spawn_worker(task) -- launch a worker to create the new mod
- run_specialist(role) -- review the worker's output (architect or qa)
- get_diff() -- see the current changes
- verify() -- run the verification script
- submit_result(summary) -- finalize the workflow

My process:
1. I set up the workspace
2. I spawn a worker with a clear task: study existing mods and create a new one
3. If verification passes, I submit the result
4. If it fails, I run specialist reviews and spawn another worker with feedback
5. I iterate until verification passes or I have made 3 attempts

Key principles:
- I give the worker ONLY the task description — no solution hints about file formats
- The worker must discover the mod structure by reading existing files
- I pass specialist feedback to help the next worker iteration
- I am decisive: if 2 iterations show no progress, I submit what I have`

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

  const userMessage = `I need to create a new Skyrim mod that adds a custom power or shout. I should set up the workspace, spawn a worker to study the existing mods and create a new one, verify the result, and iterate until the mod passes all verification checks.`

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
      logLabel: `skyrim-orch/${combo.name}`,
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
  console.log("|  Orchestrator Grid Test — Skyrim Mod Registration                  |")
  console.log("+====================================================================+")
  console.log(`  Max orchestrator turns: ${MAX_TURNS}`)
  console.log(`  Max worker turns: ${WORKER_TURNS}`)
  console.log(`  Grid size: ${GRID.length} combos`)

  const combos = COMBO_FILTER
    ? GRID.filter(g => g.name.includes(COMBO_FILTER))
    : GRID

  if (combos.length === 0) {
    console.error(`No combos match filter: ${COMBO_FILTER}`)
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
  console.log("  GRID RESULTS — Skyrim Mod Registration")
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
  const resultsFile = path.join(resultsDir, `orch-grid-skyrim-${Date.now()}.json`)
  await fs.writeFile(resultsFile, JSON.stringify({
    test: "skyrim-mod-registration",
    timestamp: new Date().toISOString(),
    maxTurns: MAX_TURNS,
    workerTurns: WORKER_TURNS,
    results,
  }, null, 2))
  console.log(`  Results saved to: ${resultsFile}`)
}

await main()
