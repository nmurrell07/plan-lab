/**
 * Meta-Tool System — 5 intent-based tools instead of 8.
 *
 * Reduces cognitive load for smaller models by presenting 5 high-level intents:
 *   investigate(target, how) → read, grep, list, glob
 *   modify(file, old_text, new_text) → edit, write
 *   execute(command) → bash
 *   note(text) → scratchpad (survives context pressure)
 *   finish(summary) → complete (with optional gate)
 *
 * The model thinks in intents; the router translates to real tools.
 */

import type { OpenAIToolDef, SendWithToolsFn } from "./client"
import type { ToolCall, ToolCallResult, ToolExecutor, ExecutionContext } from "./tools"

// ── Meta-tool FC definitions ──────────────────────────────────────────

export function getMetaToolDefs(): OpenAIToolDef[] {
  return [
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

export interface MetaToolConfig {
  baseExecutor: ToolExecutor
  testCmd?: string
  buggyFiles?: string[]
}

export interface MetaExecutorState {
  notes: string[]
  consecutiveTestFails: number
  lastEditFile: string | null
  commandHistory: Array<{ sig: string; success: boolean }>
}

export function createMetaToolExecutor(config: MetaToolConfig): {
  executor: ToolExecutor
  tools: OpenAIToolDef[]
  state: MetaExecutorState
} {
  const state: MetaExecutorState = {
    notes: [],
    consecutiveTestFails: 0,
    lastEditFile: null,
    commandHistory: [],
  }

  const executor: ToolExecutor = {
    async executeAll(calls, context) {
      const results: ToolCallResult[] = []

      for (const call of calls) {
        const callSig = `${call.tool}:${JSON.stringify(call.args)}`

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

        // ── Translate meta-tool to real tool(s) ──
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

        // ── Auto-test after successful modify ──
        if (call.tool === "modify" && config.testCmd) {
          const hadSuccessEdit = results.some(r => r.tool === "modify" && r.success)
          const hadExplicitExec = calls.some(c => c.tool === "execute")
          if (hadSuccessEdit && !hadExplicitExec) {
            const [testResult] = await config.baseExecutor.executeAll(
              [{ tool: "bash", args: { command: config.testCmd }, raw: "" }],
              context,
            )
            if (testResult.output?.includes("tests passed") && !testResult.output?.includes("FAIL")) {
              results.push({ ...testResult, tool: "auto-test" })
            } else {
              results.push({
                tool: "auto-test",
                success: false,
                output: `[Auto-test] ${config.testCmd}: ${testResult.output?.slice(0, 2000)}`,
              })
            }
          }
        }

        // ── Finish gate: verify tests pass before accepting ──
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

  return { executor, tools: getMetaToolDefs(), state }
}
