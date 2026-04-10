/**
 * Meta-Tool System — 5 intent-based tools instead of 8.
 *
 * Reduces cognitive load for smaller models by presenting 5 high-level intents:
 *   investigate(target, how) → read, grep, list, glob
 *   modify(file, old_text, new_text) → edit, write
 *   execute(command) → bash
 *   note(text) → scratchpad (survives context pressure)
 *   checkoff(step, status) → checklist progress tracker
 *   finish(summary) → complete (with optional gate)
 *
 * The model thinks in intents; the router translates to real tools.
 */

import type { OpenAIToolDef, SendWithToolsFn, ChatMessage } from "./client"
import type { ToolCall, ToolCallResult, ToolExecutor, ExecutionContext } from "./tools"
import { runSurgicalEditConveyor } from "./conveyor"

// ── Meta-tool FC definitions ──────────────────────────────────────────

export function getMetaToolDefs(options?: { includeCheckoff?: boolean; includeRegister?: boolean }): OpenAIToolDef[] {
  const defs: OpenAIToolDef[] = [
    {
      type: "function",
      function: {
        name: "investigate",
        description: "Understand code, find files, or search patterns. Use to READ files, SEARCH for patterns, LIST directories, or FIND files by name.",
        parameters: {
          type: "object",
          properties: {
            target: {
              type: "string",
              description: "What to investigate: a file path, search pattern, directory, or glob pattern.",
            },
            how: {
              type: "string",
              description: "How to investigate: 'read' (read a file), 'search' (grep for pattern), 'list' (show directory), 'find' (glob for filename).",
              enum: ["read", "search", "list", "find"],
            },
          },
          required: ["target"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "modify",
        description: "Change code in a file. EDIT existing code (find and replace), CREATE a new file, or REWRITE entire content.",
        parameters: {
          type: "object",
          properties: {
            file: {
              type: "string",
              description: "The file path to modify.",
            },
            old_text: {
              type: "string",
              description: "For editing: the EXACT text to find and replace. Copy from the file.",
            },
            new_text: {
              type: "string",
              description: "The replacement text (for edit) or full content (for new file).",
            },
            mode: {
              type: "string",
              description: "How to modify: 'edit' (find+replace), 'create' (new file), 'rewrite' (replace entire file).",
              enum: ["edit", "create", "rewrite"],
            },
          },
          required: ["file"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "execute",
        description: "Run a command in the terminal. Use for tests, installs, or scripts.",
        parameters: {
          type: "object",
          properties: {
            command: {
              type: "string",
              description: "The bash command to run.",
            },
          },
          required: ["command"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "note",
        description: "Save a finding to your scratchpad. Notes survive context compression. Write what you LEARNED, not what you DID.",
        parameters: {
          type: "object",
          properties: {
            text: {
              type: "string",
              description: "What you learned. Be specific: names, values, constraints.",
            },
          },
          required: ["text"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "finish",
        description: "Signal that you are done. Tests must pass before finishing.",
        parameters: {
          type: "object",
          properties: {
            summary: {
              type: "string",
              description: "Brief summary of what you did.",
            },
          },
          required: ["summary"],
        },
      },
    },
  ]

  if (options?.includeRegister) {
    defs.splice(defs.length - 1, 0, {
      type: "function",
      function: {
        name: "register",
        description: "Declare what I intend to do before I do it. I must register each file I plan to modify.",
        parameters: {
          type: "object",
          properties: {
            intent: {
              type: "string",
              description: "What I intend to do.",
              enum: ["fix", "add", "refactor", "remove"],
            },
            file: {
              type: "string",
              description: "The file I will modify.",
            },
            reason: {
              type: "string",
              description: "Why I need to make this change (what I learned from exploration).",
            },
          },
          required: ["intent", "file", "reason"],
        },
      },
    })
  }

  if (options?.includeCheckoff) {
    defs.splice(defs.length - 1, 0, {
      type: "function",
      function: {
        name: "checkoff",
        description: "Update checklist progress. Use when starting, completing, or getting blocked on a checklist step.",
        parameters: {
          type: "object",
          properties: {
            step: {
              type: "string",
              description: "The checklist step text you are updating. Use the exact step or its leading phrase.",
            },
            status: {
              type: "string",
              description: "Current status for that checklist step.",
              enum: ["in_progress", "done", "blocked"],
            },
            note: {
              type: "string",
              description: "Optional short note explaining progress or the blocker.",
            },
          },
          required: ["step", "status"],
        },
      },
    })
  }

  return defs
}

// ── Meta-tool → real tool translation ─────────────────────────────────

function translateMetaCall(call: ToolCall): ToolCall[] {
  switch (call.tool) {
    case "investigate": {
      const target = call.args.target || ""
      const how = call.args.how || "read"

      switch (how) {
        case "read":
          return [{ tool: "read", args: { path: target }, raw: call.raw }]
        case "search":
          return [{ tool: "grep", args: { pattern: target, glob: call.args.glob || "" }, raw: call.raw }]
        case "list":
          return [{ tool: "list", args: { path: target || "." }, raw: call.raw }]
        case "find":
          return [{ tool: "glob", args: { pattern: target }, raw: call.raw }]
        default:
          // Default: if target looks like a file path, read it; otherwise grep
          if (target.includes("/") || target.includes(".")) {
            return [{ tool: "read", args: { path: target }, raw: call.raw }]
          }
          return [{ tool: "grep", args: { pattern: target }, raw: call.raw }]
      }
    }

    case "modify": {
      const file = call.args.file || call.args.path || ""
      const mode = call.args.mode || "edit"
      const oldText = call.args.old_text || call.args.old_string || ""
      const newText = call.args.new_text || call.args.new_string || call.args.content || ""

      switch (mode) {
        case "create":
          return [{ tool: "write", args: { path: file, content: newText }, raw: call.raw }]
        case "rewrite":
          return [{ tool: "write", args: { path: file, content: newText }, raw: call.raw }]
        case "edit":
        default:
          return [{ tool: "edit", args: { path: file, old_string: oldText, new_string: newText }, raw: call.raw }]
      }
    }

    case "execute":
      return [{ tool: "bash", args: { command: call.args.command || "" }, raw: call.raw }]

    case "finish":
      return [{ tool: "complete", args: { summary: call.args.summary || "" }, raw: call.raw }]

    case "note":
      // Handled directly in the meta executor, not routed to base tools
      return [{ tool: "note", args: { text: call.args.text || "" }, raw: call.raw }]

    default:
      return [{ tool: call.tool, args: call.args, raw: call.raw }]
  }
}

// ── Meta-tool executor ────────────────────────────────────────────────

// ── Conveyor phase types ─────────────────────────────────────────────

export type ConveyorPhase = "explore" | "execute" | "verify"

export interface ConveyorRegistration {
  intent: "fix" | "add" | "refactor" | "remove"
  file: string
  reason: string
}

export interface ConveyorPhaseState {
  phase: ConveyorPhase
  registrations: ConveyorRegistration[]
  phaseTransitions: Array<{ from: string; to: string; turn: number }>
  gateBlocks: number
  verifyAttempts: number
  loopBacks: number
  testsPassed: boolean
}

// ── Meta-tool config ─────────────────────────────────────────────────

export interface MetaToolConfig {
  baseExecutor: ToolExecutor
  testCmd?: string
  buggyFiles?: string[]
  checklistSteps?: string[]
  enableCheckoff?: boolean
  requireCompletedStepForFinish?: boolean
  requireInProgressStepForModify?: boolean
  /** SendWithTools for surgical edit conveyor stations (optional — falls back to direct edit) */
  sendWithTools?: SendWithToolsFn
  /** Model ID for conveyor station calls */
  model?: string
  /** Enable surgical edit conveyor for modify calls (default: true if sendWithTools provided) */
  surgicalConveyor?: boolean
  /** Enable conveyor phase machine (explore → register → execute → verify) */
  conveyorPhase?: boolean
}

export interface MetaExecutorState {
  notes: string[]
  checklistSteps: Array<{ text: string; status: "pending" | "in_progress" | "done" | "blocked" }>
  checkoffCalls: number
  completedSteps: number
  blockedSteps: number
  progressPercent: number
  modifyGateBlocks: number
  finishGateBlocks: number
  consecutiveTestFails: number
  lastEditFile: string | null
  commandHistory: Array<{ sig: string; success: boolean }>
  surgicalEdits: number
  surgicalSuccesses: number
  directEditFallbacks: number
  conveyor?: ConveyorPhaseState
}

export function createMetaToolExecutor(config: MetaToolConfig): {
  executor: ToolExecutor
  tools: OpenAIToolDef[]
  state: MetaExecutorState
} {
  const useSurgical = (config.surgicalConveyor ?? true) && !!config.sendWithTools && !!config.model

  const state: MetaExecutorState = {
    notes: [],
    checklistSteps: (config.checklistSteps || []).map(text => ({ text, status: "pending" })),
    checkoffCalls: 0,
    completedSteps: 0,
    blockedSteps: 0,
    progressPercent: 0,
    modifyGateBlocks: 0,
    finishGateBlocks: 0,
    consecutiveTestFails: 0,
    lastEditFile: null,
    commandHistory: [],
    surgicalEdits: 0,
    surgicalSuccesses: 0,
    directEditFallbacks: 0,
    conveyor: config.conveyorPhase ? {
      phase: "explore",
      registrations: [],
      phaseTransitions: [],
      gateBlocks: 0,
      verifyAttempts: 0,
      loopBacks: 0,
      testsPassed: false,
    } : undefined,
  }

  const executor: ToolExecutor = {
    async executeAll(calls, context) {
      const results: ToolCallResult[] = []

      for (const call of calls) {
        const callSig = `${call.tool}:${JSON.stringify(call.args)}`

        // ── Conveyor phase gating ──
        if (state.conveyor) {
          const cv = state.conveyor
          const phase = cv.phase

          // Register tool handler
          if (call.tool === "register") {
            const intent = (call.args.intent || "fix") as ConveyorRegistration["intent"]
            const file = (call.args.file || "").replace(/`/g, "").trim()
            const reason = call.args.reason || ""

            if (!file) {
              results.push({ tool: "register", success: false, output: "I must specify a file to register.", error: "Missing file" })
              continue
            }

            cv.registrations.push({ intent, file, reason })

            // Transition explore → execute on first registration
            if (phase === "explore") {
              cv.phaseTransitions.push({ from: "explore", to: "execute", turn: -1 })
              cv.phase = "execute"
              console.log(`    [conveyor] explore → execute (registered: ${file})`)
            } else {
              console.log(`    [conveyor] registered additional file: ${file} (phase: ${phase})`)
            }

            const fileList = cv.registrations.map(r => `${r.intent}: ${r.file}`).join(", ")
            results.push({
              tool: "register",
              success: true,
              output: `Registered: I will ${intent} ${file} — ${reason}. (${cv.registrations.length} files registered: ${fileList})`,
            })
            continue
          }

          // Phase gate checks
          const normalizeFile = (f: string) => f.replace(/`/g, "").trim().replace(/^\.\//, "")

          if (phase === "explore") {
            if (call.tool === "modify") {
              cv.gateBlocks++
              results.push({ tool: "modify", success: false, output: "I haven't explored yet. I should investigate the code first, then call register to declare what I will change.", error: "Phase: explore — modify blocked" })
              continue
            }
            if (call.tool === "execute") {
              cv.gateBlocks++
              results.push({ tool: "execute", success: false, output: "I should understand the problem before running commands. I need to investigate first.", error: "Phase: explore — execute blocked" })
              continue
            }
            if (call.tool === "finish") {
              cv.gateBlocks++
              results.push({ tool: "finish", success: false, output: "I haven't done any work yet. I should explore, register, execute, then verify first.", error: "Phase: explore — finish blocked" })
              continue
            }
          }

          if (phase === "execute") {
            if (call.tool === "modify") {
              const modFile = normalizeFile(call.args.file || call.args.path || "")
              const registered = cv.registrations.some(r => {
                const regFile = normalizeFile(r.file)
                const regBase = regFile.split("/").pop() || regFile
                const modBase = modFile.split("/").pop() || modFile
                return regFile === modFile || regBase === modBase
              })
              if (!registered) {
                cv.gateBlocks++
                results.push({
                  tool: "modify",
                  success: false,
                  output: `I haven't registered ${modFile}. I should call register(intent, file, reason) first to declare my intent.`,
                  error: "Phase: execute — unregistered file",
                })
                continue
              }
            }
            if (call.tool === "finish") {
              cv.gateBlocks++
              results.push({ tool: "finish", success: false, output: "I need to verify my changes by running the tests before finishing.", error: "Phase: execute — finish blocked" })
              continue
            }
          }

          if (phase === "verify") {
            if (call.tool === "modify") {
              cv.gateBlocks++
              results.push({ tool: "modify", success: false, output: "Tests failed — I should go back to exploring to understand why, not just retry the edit.", error: "Phase: verify — modify blocked" })
              // Reset to explore on blocked modify during verify
              cv.phase = "explore"
              cv.registrations = []
              cv.loopBacks++
              cv.phaseTransitions.push({ from: "verify", to: "explore", turn: -1 })
              console.log(`    [conveyor] verify → explore (loop-back #${cv.loopBacks}: modify attempted during verify)`)
              continue
            }
            if (call.tool === "finish" && !cv.testsPassed) {
              cv.gateBlocks++
              results.push({ tool: "finish", success: false, output: "Tests haven't passed yet. I should run the test command first.", error: "Phase: verify — tests not passed" })
              continue
            }
          }
        }

        // ── Stuck detection: same failing command 3+ times ──
        const recentFails = state.commandHistory.filter(h => h.sig === callSig && !h.success)
        if (recentFails.length >= 3) {
          results.push({
            tool: call.tool,
            success: false,
            output: `This command has failed ${recentFails.length} times. Try a DIFFERENT approach.`,
            error: "Repeated failure — change your approach.",
          })
          state.commandHistory.push({ sig: callSig, success: false })
          continue
        }

        // ── Note tool — direct scratchpad write ──
        if (call.tool === "note") {
          const text = call.args.text || ""
          if (text) state.notes.push(text)
          results.push({
            tool: "note",
            success: true,
            output: `Noted (${state.notes.length} total). Notes survive context compression.`,
          })
          continue
        }

        // ── Checklist progress tracker ──
        if (call.tool === "checkoff") {
          const stepText = (call.args.step || "").trim()
          const status = (call.args.status || "in_progress") as "in_progress" | "done" | "blocked"
          const note = (call.args.note || "").trim()

          if (state.checklistSteps.length === 0) {
            results.push({
              tool: "checkoff",
              success: false,
              output: "No checklist is active for this run.",
              error: "No checklist configured",
            })
            continue
          }

          const index = findChecklistStepIndex(state.checklistSteps.map(s => s.text), stepText)
          if (index < 0) {
            results.push({
              tool: "checkoff",
              success: false,
              output: `Checklist step not recognized: ${stepText}`,
              error: "Unknown checklist step",
            })
            continue
          }

          state.checkoffCalls++
          state.checklistSteps[index].status = status
          state.completedSteps = state.checklistSteps.filter(s => s.status === "done").length
          state.blockedSteps = state.checklistSteps.filter(s => s.status === "blocked").length
          state.progressPercent = Math.round((state.completedSteps / state.checklistSteps.length) * 100)

          const parts = [
            `Checklist progress: ${state.completedSteps}/${state.checklistSteps.length} done (${state.progressPercent}%)`,
            `Updated step ${index + 1}: ${state.checklistSteps[index].text} -> ${status}`,
          ]
          if (note) parts.push(`Note: ${note}`)
          results.push({
            tool: "checkoff",
            success: true,
            output: parts.join("\n"),
          })
          continue
        }

        if (call.tool === "modify" && config.requireInProgressStepForModify && state.checklistSteps.length > 0) {
          const hasInProgressStep = state.checklistSteps.some(step => step.status === "in_progress")
          if (!hasInProgressStep) {
            state.modifyGateBlocks++
            results.push({
              tool: "modify",
              success: false,
              output: "Modify blocked — first mark a checklist step in progress with checkoff(step=\"...\", status=\"in_progress\").",
              error: "Checklist protocol requires an in-progress step before modifying code.",
            })
            state.commandHistory.push({ sig: callSig, success: false })
            continue
          }
        }

        // ── Surgical edit conveyor for modify calls ──
        if (call.tool === "modify" && useSurgical) {
          const file = call.args.file || call.args.path || ""
          const oldText = call.args.old_text || call.args.old_string || ""
          const searchHint = oldText.split("\n")[0]?.trim().slice(0, 60) || ""

          if (file && searchHint) {
            state.surgicalEdits++
            const surgicalMessages: ChatMessage[] = [
              { role: "system", content: "You are editing code. Answer each station question precisely." },
              { role: "user", content: `Edit ${file}: find "${searchHint}" and replace the matching section.` },
            ]

            try {
              const surgicalResult = await runSurgicalEditConveyor(
                surgicalMessages,
                config.sendWithTools!,
                config.model!,
                context.workDir,
                2,
                {
                  file,
                  searchKeyword: searchHint,
                  // Pass the model's original new_text — it was written with full context
                  newText: call.args.new_text || call.args.new_string || call.args.content || undefined,
                },
              )

              if (surgicalResult.success) {
                state.surgicalSuccesses++
                console.log(`    [surgical] SUCCESS: ${surgicalResult.output}`)
                results.push({ ...surgicalResult, tool: call.tool })
                state.commandHistory.push({ sig: callSig, success: true })
                state.lastEditFile = file
                state.consecutiveTestFails = 0
                // Skip to auto-test (handled below)
                goto_auto_test: {
                  // Auto-test is handled after this block
                }
              } else {
                console.log(`    [surgical] FAIL: ${surgicalResult.error} — falling back to direct edit`)
                state.directEditFallbacks++
                // Fall through to direct edit below
              }
            } catch (err: any) {
              console.log(`    [surgical] ERROR: ${err.message?.slice(0, 60)} — falling back to direct edit`)
              state.directEditFallbacks++
              // Fall through to direct edit below
            }

            // If surgical succeeded, skip direct edit
            if (results.length > 0 && results[results.length - 1].tool === call.tool && results[results.length - 1].success) {
              // Already handled — skip to auto-test
            } else {
              // Fall through to direct edit
              const translated = translateMetaCall(call)
              for (const realCall of translated) {
                const realResults = await config.baseExecutor.executeAll([realCall], context)
                for (const r of realResults) {
                  results.push({ ...r, tool: call.tool })
                  state.commandHistory.push({ sig: callSig, success: r.success })
                  if (realCall.tool === "edit" && r.success) {
                    state.lastEditFile = file
                    state.consecutiveTestFails = 0
                  }
                }
              }
            }
          } else {
            // No search hint — go direct
            const translated = translateMetaCall(call)
            for (const realCall of translated) {
              const realResults = await config.baseExecutor.executeAll([realCall], context)
              for (const r of realResults) {
                results.push({ ...r, tool: call.tool })
                state.commandHistory.push({ sig: callSig, success: r.success })
                if (realCall.tool === "edit" && r.success) {
                  state.lastEditFile = call.args.file || call.args.path || null
                  state.consecutiveTestFails = 0
                }
              }
            }
          }
        } else {
          // ── Standard path: translate meta-tool to real tool(s) ──
          const translated = translateMetaCall(call)

          for (const realCall of translated) {
            const realResults = await config.baseExecutor.executeAll([realCall], context)
            for (const r of realResults) {
              results.push({ ...r, tool: call.tool })
              state.commandHistory.push({ sig: callSig, success: r.success })

              // Track edit/test cycles for stuck detection
              if (realCall.tool === "edit" && r.success) {
                state.lastEditFile = call.args.file || call.args.path || null
                state.consecutiveTestFails = 0
              }
              if (realCall.tool === "bash") {
                const isTestPass = r.success && r.output?.includes("tests passed") && !r.output?.includes("FAIL")
                if (isTestPass) {
                  state.consecutiveTestFails = 0
                } else if (state.lastEditFile) {
                  state.consecutiveTestFails++
                }
              }
            }
          }
        }

        // ── Auto-test after successful modify ──
        if (call.tool === "modify" && config.testCmd) {
          const hadSuccessEdit = results.some(r => r.tool === "modify" && r.success)
          const hadExplicitExec = calls.some(c => c.tool === "execute")
          if (hadSuccessEdit && !hadExplicitExec) {
            const [testResult] = await config.baseExecutor.executeAll(
              [{ tool: "bash", args: { command: config.testCmd }, raw: "" }],
              context,
            )
            const autoTestPassed = !!testResult.output?.includes("tests passed") && !testResult.output?.includes("FAIL")
            if (autoTestPassed) {
              results.push({ ...testResult, tool: "auto-test" })
            } else {
              results.push({
                tool: "auto-test",
                success: false,
                output: `[Auto-test] ${config.testCmd}: ${testResult.output?.slice(0, 2000)}`,
              })
            }

            // Conveyor: auto-test transitions execute → verify
            if (state.conveyor && state.conveyor.phase === "execute") {
              state.conveyor.phase = "verify"
              state.conveyor.verifyAttempts++
              state.conveyor.phaseTransitions.push({ from: "execute", to: "verify", turn: -1 })
              console.log(`    [conveyor] execute → verify via auto-test (${autoTestPassed ? "PASSED" : "FAILED"})`)
              if (autoTestPassed) {
                state.conveyor.testsPassed = true
              } else {
                state.conveyor.testsPassed = false
                state.conveyor.phase = "explore"
                state.conveyor.registrations = []
                state.conveyor.loopBacks++
                state.conveyor.phaseTransitions.push({ from: "verify", to: "explore", turn: -1 })
                console.log(`    [conveyor] verify → explore (loop-back #${state.conveyor.loopBacks}: auto-test failed)`)
              }
            }
          }
        }

        // ── Conveyor: detect test execution and transition phases ──
        if (state.conveyor && call.tool === "execute") {
          const cv = state.conveyor
          const cmd = (call.args.command || "").trim()
          const isTestCmd = config.testCmd && cmd.includes(config.testCmd)
          const lastResult = results[results.length - 1]

          if (isTestCmd && lastResult) {
            const testPassed = lastResult.success && lastResult.output?.includes("tests passed") && !lastResult.output?.includes("FAIL")

            if (cv.phase === "execute") {
              cv.phase = "verify"
              cv.verifyAttempts++
              cv.phaseTransitions.push({ from: "execute", to: "verify", turn: -1 })
              console.log(`    [conveyor] execute → verify (test ${testPassed ? "PASSED" : "FAILED"})`)
            }

            if (testPassed) {
              cv.testsPassed = true
            } else if (cv.phase === "verify") {
              // Test failed during verify — loop back to explore
              cv.testsPassed = false
              cv.phase = "explore"
              cv.registrations = []
              cv.loopBacks++
              cv.phaseTransitions.push({ from: "verify", to: "explore", turn: -1 })
              console.log(`    [conveyor] verify → explore (loop-back #${cv.loopBacks}: test failed)`)
            }
          }
        }

        // ── Finish gate: verify tests pass before accepting ──
        if (call.tool === "finish" && config.requireCompletedStepForFinish && state.checklistSteps.length > 0 && state.completedSteps === 0) {
          state.finishGateBlocks++
          results[results.length - 1] = {
            tool: "finish",
            success: false,
            output: "Finish blocked — mark at least one checklist step done with checkoff(step=\"...\", status=\"done\") before finishing.",
            error: "Checklist protocol requires at least one completed step before finishing.",
          }
          continue
        }

        if (call.tool === "finish" && config.testCmd) {
          const [gateResult] = await config.baseExecutor.executeAll(
            [{ tool: "bash", args: { command: config.testCmd }, raw: "" }],
            context,
          )
          const pass = gateResult.success && gateResult.output?.includes("tests passed") && !gateResult.output?.includes("FAIL")
          if (!pass) {
            // Override the complete result — tests haven't passed
            results[results.length - 1] = {
              tool: "finish",
              success: false,
              output: `Finish rejected — tests still failing:\n${gateResult.output?.slice(0, 2000)}`,
              error: "Tests must pass before finishing.",
            }
          }
        }
      }

      return results
    },
  }

  return {
    executor,
    tools: getMetaToolDefs({
      includeCheckoff: !!config.enableCheckoff,
      includeRegister: !!config.conveyorPhase,
    }),
    state,
  }
}

function findChecklistStepIndex(steps: string[], query: string): number {
  const normalizedQuery = query.trim().toLowerCase()
  if (!normalizedQuery) return -1

  const exact = steps.findIndex(step => step.trim().toLowerCase() === normalizedQuery)
  if (exact >= 0) return exact

  const contains = steps.findIndex(step => step.trim().toLowerCase().includes(normalizedQuery))
  if (contains >= 0) return contains

  return steps.findIndex(step => normalizedQuery.includes(step.trim().toLowerCase().slice(0, Math.min(step.length, 24))))
}
