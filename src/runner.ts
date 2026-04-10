/**
 * Agent Runner — ReAct loop with full system integration.
 *
 * Supports:
 * - Multi-turn tool calling (FC mode)
 * - Text-to-FC recovery (when model emits text instead of tool calls)
 * - Stall detection (empty responses, text-only nudges)
 * - Configurable tool set (base tools or meta-tools)
 * - Interceptors (transform/inject/block before execution)
 * - Failure ledger (error bucketing, loop/dandelion detection)
 * - Context compression (microcompact + full summarization)
 * - Surgical edit conveyor (4-station decomposed edits)
 */

import type { ChatMessage, OpenAIToolDef, SendWithToolsFn } from "./client"
import type { ToolCall, ToolCallResult, ToolExecutor, ExecutionContext } from "./tools"
import { compactIfNeeded } from "./compactor"
import { recoverToolCallsFromText } from "./conveyor"
import {
  checkInterceptors, recordInterceptorOutcome, recordInterceptorMiss,
} from "./interceptors"
import type { FailureLedger, LoopSignal, DandelionSignal } from "./ledger"
import {
  addEntry, detectLoop, detectDandelion, formatDandelionRedirect,
  normalizeErrorKey, matchRoutingRule, recordRuleActivation, getDisabledRules,
} from "./ledger"

// ── Types ─────────────────────────────────────────────────────────────

export interface RunConfig {
  agentName: string
  systemPrompt: string
  userMessage: string
  model: string
  endpoint: string
  sendWithTools: SendWithToolsFn
  executor: ToolExecutor
  tools: OpenAIToolDef[]
  workDir: string
  maxTurns?: number
  temperature?: number
  testCmd?: string
  /** Allowed tool IDs for the execution context */
  allowedTools?: string[]
  /** Log prefix for console output */
  logLabel?: string
  /** Inject notes from scratchpad into nudges */
  notes?: string[]
  /** Enable interceptors (default: true) */
  interceptors?: boolean
  /** Failure ledger for error tracking (default: none) */
  failureLedger?: FailureLedger
  /** Enable context compression (default: true) */
  compaction?: boolean
  /** Run ID for ledger entries */
  runId?: string
  /** Print full transcript (system prompt, assistant text, tool calls, results) */
  verbose?: boolean
  /** Callback to get current conveyor phase for context-aware nudges */
  phaseHint?: () => string | undefined
}

export interface RunResult {
  agentName: string
  turns: number
  turnsWithToolCalls: number
  toolCalls: Array<{ tool: string; success: boolean; realTool?: string }>
  touchedFiles: string[]
  totalTokens: number
  durationSec: number
  testsPass: boolean
  filesModified: boolean
  finalText: string
  error?: string
  /** Number of interceptor firings */
  interceptorFirings: number
  /** Number of loops detected */
  loopsDetected: number
  /** Number of dandelions detected */
  dandelionsDetected: number
  /** Number of context compactions */
  compactions: number
  /** Number of text-to-FC recoveries */
  recoveries: number
}

// ── Trajectory entry for loop/dandelion detection ─────────────────────

interface TrajectoryEntry {
  turn: number
  translatedTool: string
  argsDigest: string
  success: boolean
  metaTool?: string
  phase?: string
}

const PERSPECTIVE = process.env.PERSPECTIVE || "1p"

// ── Runner ─────────────────────────────────────────────────────────────

export async function runAgent(config: RunConfig): Promise<RunResult> {
  const {
    agentName,
    systemPrompt,
    userMessage,
    sendWithTools,
    executor,
    tools,
    workDir,
    maxTurns = 15,
    testCmd,
    allowedTools = ["read", "write", "edit", "grep", "list", "glob", "bash", "complete"],
    logLabel = agentName,
    notes = [],
    interceptors: interceptorsEnabled = true,
    failureLedger,
    compaction: compactionEnabled = true,
    runId = `run-${Date.now()}`,
    verbose = false,
  } = config

  const label = `[${logLabel}]`
  const startTime = Date.now()
  const availableToolNames = tools.map(t => t.function.name)
  const trajectory: TrajectoryEntry[] = []

  const context: ExecutionContext = { workDir, allowedTools }

  if (verbose) {
    console.log(`\n${"─".repeat(80)}`)
    console.log(`TRANSCRIPT: ${agentName}`)
    console.log(`${"─".repeat(80)}`)
    console.log(`\n[SYSTEM PROMPT]\n${systemPrompt}\n`)
    console.log(`[USER MESSAGE]\n${userMessage}\n`)
    console.log(`${"─".repeat(80)}`)
  }

  const messages: ChatMessage[] = [
    { role: "system", content: systemPrompt },
    { role: "user", content: userMessage },
  ]

  const result: RunResult = {
    agentName,
    turns: 0,
    turnsWithToolCalls: 0,
    toolCalls: [],
    touchedFiles: [],
    totalTokens: 0,
    durationSec: 0,
    testsPass: false,
    filesModified: false,
    finalText: "",
    interceptorFirings: 0,
    loopsDetected: 0,
    dandelionsDetected: 0,
    compactions: 0,
    recoveries: 0,
  }

  let consecutiveEmpty = 0
  let consecutiveTextOnly = 0

  for (let turn = 0; turn < maxTurns; turn++) {
    result.turns = turn + 1

    // ── Context compression ──
    if (compactionEnabled) {
      const compactResult = await compactIfNeeded(messages, userMessage, {
        sendWithTools,
        summaryModel: config.model,
        notes,
      })
      if (compactResult.result.tier !== "none") {
        result.compactions++
        console.log(`  ${label} T${turn + 1}: COMPACTED (${compactResult.result.tier}: ${compactResult.result.messagesBefore} -> ${compactResult.result.messagesAfter} msgs)`)
        // Replace messages array contents
        messages.length = 0
        messages.push(...compactResult.messages)
      }
    }

    let turnResult
    try {
      turnResult = await sendWithTools(agentName, messages, tools)
    } catch (err: any) {
      console.log(`  ${label} T${turn + 1}: ERROR ${err.message?.slice(0, 80)}`)
      result.error = err.message
      break
    }

    result.totalTokens += turnResult.tokens?.total ?? 0

    // ── Verbose: print assistant response ──
    if (verbose) {
      console.log(`\n[TURN ${turn + 1}] Assistant:`)
      if (turnResult.content) console.log(`  Text: ${turnResult.content.slice(0, 500)}`)
      if (turnResult.toolCalls?.length) {
        for (const tc of turnResult.toolCalls) {
          const args = tc.function.arguments
          console.log(`  Tool: ${tc.function.name}(${args.length > 200 ? args.slice(0, 200) + '...' : args})`)
        }
      }
      if (!turnResult.content?.trim() && !turnResult.toolCalls?.length) {
        console.log(`  (empty response)`)
      }
    }

    // ── No tool calls — handle text-only or empty ──
    if (!turnResult.toolCalls || turnResult.toolCalls.length === 0) {
      const text = turnResult.content ?? ""

      if (!text.trim()) {
        consecutiveEmpty++
        console.log(`  ${label} T${turn + 1}: EMPTY (${consecutiveEmpty} consecutive)`)

        if (consecutiveEmpty >= 4) {
          result.error = "4 consecutive empty responses — agent stalled"
          break
        }

        // Context-aware nudge: tell the model exactly what to do next
        const lastToolResult = result.toolCalls[result.toolCalls.length - 1]
        let nudge: string

        // Phase-aware nudges for conveyor planner
        const currentPhase = config.phaseHint?.()
        if (currentPhase) {
          if (currentPhase === "explore") {
            nudge = "I should investigate the code to understand the problem. I'll call investigate now."
          } else if (currentPhase === "execute") {
            if (!result.filesModified) {
              nudge = "I've registered my intent. Now I should call modify on the registered files."
            } else {
              nudge = `I should run the tests to verify: execute(command="${testCmd || 'node test/*.test.js'}").`
            }
          } else if (currentPhase === "verify") {
            nudge = `I should run the tests: execute(command="${testCmd || 'node test/*.test.js'}").`
          } else {
            nudge = "I need to make progress. I should call a tool now."
          }
        } else if (PERSPECTIVE === "2p") {
          if (!lastToolResult) {
            nudge = "Call investigate to read the buggy files. Start now."
          } else if (!result.filesModified && result.toolCalls.filter(t => t.tool === "investigate" || t.realTool === "read").length > 0) {
            nudge = "You've read the files. Now call modify to fix the bug. Use the exact text from the file."
          } else if (result.filesModified && !result.testsPass) {
            nudge = `Run the tests: execute(command="${testCmd || 'node test/*.test.js'}"). If they fail, read the error and fix it.`
          } else {
            nudge = "Use your tools to make progress. Act now — don't think, just call a tool."
          }
        } else {
          if (!lastToolResult) {
            nudge = "I should call investigate to read the buggy files. Time to start."
          } else if (!result.filesModified && result.toolCalls.filter(t => t.tool === "investigate" || t.realTool === "read").length > 0) {
            nudge = "I've read the files. Now I should call modify to fix the bug. I'll use the exact text from the file."
          } else if (result.filesModified && !result.testsPass) {
            nudge = `I should run the tests now: execute(command="${testCmd || 'node test/*.test.js'}"). If they fail, I'll read the error and fix it.`
          } else {
            nudge = "I need to make progress. I should call a tool now instead of thinking."
          }
        }

        // Don't count empty turns toward maxTurns for small models — they're not real turns
        // Instead, just inject the nudge and retry without incrementing the turn counter
        if (verbose) console.log(`  [NUDGE] ${nudge}`)
        messages.push({ role: "user", content: nudge })
        // Give back the turn — empty responses shouldn't count
        turn--
        continue
      }

      consecutiveEmpty = 0

      // Try text-to-FC recovery (from conveyor module)
      const recovered = recoverToolCallsFromText(text, availableToolNames)
      if (recovered && recovered.length > 0) {
        result.recoveries++
        console.log(`  ${label} T${turn + 1}: RECOVERED ${recovered.map(r => r.tool).join(", ")} from text`)
        const toolResults = await executor.executeAll(
          recovered.map(r => ({ tool: r.tool, args: r.args, raw: r.raw })),
          context,
        )

        messages.push({ role: "assistant", content: text })
        for (let i = 0; i < toolResults.length; i++) {
          const r = toolResults[i]
          result.toolCalls.push({ tool: recovered[i]?.tool || "?", realTool: r.tool, success: r.success })
          const recLimit = (r.tool === "read" || r.tool === "investigate") ? 6000 : 4000
          messages.push({
            role: "user",
            content: r.success ? `[Result] ${r.output?.slice(0, recLimit)}` : `[Error] ${r.output} ${r.error || ""}`,
          })
          trajectory.push({
            turn, translatedTool: r.tool, argsDigest: JSON.stringify(recovered[i]?.args || {}).slice(0, 120),
            success: r.success, metaTool: recovered[i]?.tool,
          })
          const touched = extractTouchedPath(recovered[i], r)
          if (touched && !result.touchedFiles.includes(touched)) result.touchedFiles.push(touched)
          if (checkTestPass(r)) result.testsPass = true
          if (r.tool === "edit" || r.tool === "write") result.filesModified = true
        }
        result.turnsWithToolCalls++
        if (result.testsPass) break
        continue
      }

      // Pure text — nudge toward tool use
      consecutiveTextOnly++
      console.log(`  ${label} T${turn + 1}: text-only (${text.length} chars, ${consecutiveTextOnly} consecutive)`)
      result.finalText = text

      if (consecutiveTextOnly >= 3) {
        result.error = "3 consecutive text-only responses — agent not using tools"
        break
      }

      messages.push({ role: "assistant", content: text })
      const nudge = PERSPECTIVE === "2p"
        ? (notes.length > 0
          ? `Use your tools. Your notes so far:\n${notes.map((n, i) => `${i + 1}. ${n}`).join("\n")}\n\nAct, don't explain.`
          : "Use your tools (investigate, modify, execute, finish). Act, don't explain.")
        : (notes.length > 0
          ? `I should use my tools. My notes so far:\n${notes.map((n, i) => `${i + 1}. ${n}`).join("\n")}\n\nI need to act, not explain.`
          : "I should use my tools (investigate, modify, execute, finish). I need to act, not explain.")
      messages.push({ role: "user", content: nudge })
      continue
    }

    // ── Has tool calls ──
    consecutiveEmpty = 0
    consecutiveTextOnly = 0
    result.turnsWithToolCalls++

    const parsedCalls: ToolCall[] = turnResult.toolCalls.map(tc => {
      let args: Record<string, string> = {}
      try { args = JSON.parse(tc.function.arguments) } catch {}
      return { tool: tc.function.name, args, raw: tc.function.arguments }
    })

    // ── Loop detection (before execution) ──
    if (failureLedger) {
      const loop = detectLoop(trajectory)
      if (loop) {
        result.loopsDetected++
        console.log(`  ${label} T${turn + 1}: LOOP detected (${loop.count}x): ${loop.pattern.slice(0, 60)}`)
        // Check if current call matches the looping signature
        const thisCallSig = `${parsedCalls[0]?.tool}:${JSON.stringify(parsedCalls[0]?.args).slice(0, 40)}`
        if (thisCallSig === loop.pattern) {
          messages.push({
            role: "assistant",
            content: turnResult.content || null,
            tool_calls: turnResult.toolCalls,
          })
          messages.push({
            role: "tool",
            content: PERSPECTIVE === "2p"
              ? `You are repeating the same command (${loop.count} times). Try a DIFFERENT approach.`
              : `I've repeated this same command ${loop.count} times now. I should try a different approach.`,
            tool_call_id: turnResult.toolCalls[0].id,
          })
          continue
        }
      }

      // Dandelion detection
      const dandelion = detectDandelion(trajectory)
      if (dandelion) {
        result.dandelionsDetected++
        console.log(`  ${label} T${turn + 1}: DANDELION on "${dandelion.intent}" (${dandelion.count} seeds)`)
        messages.push({
          role: "assistant",
          content: turnResult.content || null,
          tool_calls: turnResult.toolCalls,
        })
        messages.push({
          role: "tool",
          content: formatDandelionRedirect(dandelion),
          tool_call_id: turnResult.toolCalls[0].id,
        })
        continue
      }
    }

    // ── Interceptors (before execution) ──
    let intercepted = false
    if (interceptorsEnabled) {
      for (const call of parsedCalls) {
        const hit = checkInterceptors(call.tool, call.args, { workDir })
        if (hit) {
          result.interceptorFirings++
          console.log(`  ${label} T${turn + 1}: INTERCEPTOR ${hit.interceptor.id} -> ${hit.action.type}`)

          if (hit.action.type === "block") {
            messages.push({
              role: "assistant",
              content: turnResult.content || null,
              tool_calls: turnResult.toolCalls,
            })
            messages.push({
              role: "tool",
              content: `[BLOCKED by ${hit.interceptor.id}]: ${hit.action.explanation}`,
              tool_call_id: turnResult.toolCalls[0].id,
            })
            intercepted = true
            break
          }

          if (hit.action.type === "transform" && hit.action.command) {
            // Replace the command with the transformed one
            call.args.command = hit.action.command
          }
        }
      }
    }
    if (intercepted) continue

    // ── Execute tool calls ──
    const toolResults = await executor.executeAll(parsedCalls, context)
    const summary = toolResults.map(r => `${r.tool}:${r.success ? "ok" : "ERR"}`).join(", ")
    console.log(`  ${label} T${turn + 1}: ${parsedCalls.map(c => c.tool).join(",")} -> ${summary}`)

    // ── Verbose: print tool results ──
    if (verbose) {
      for (let i = 0; i < toolResults.length; i++) {
        const r = toolResults[i]
        if (r) {
          const output = r.output?.slice(0, 300) || "(no output)"
          console.log(`  [RESULT ${parsedCalls[i]?.tool} → ${r.tool}] ${r.success ? "OK" : "ERR"}: ${output}${r.error ? ` | error: ${r.error}` : ""}`)
        }
      }
    }

    // Record results and update trajectory
    for (let i = 0; i < parsedCalls.length; i++) {
      const r = toolResults[i]
      if (r) {
        result.toolCalls.push({ tool: parsedCalls[i].tool, realTool: r.tool, success: r.success })
        if (r.tool === "edit" || r.tool === "write") result.filesModified = true

        // Track in trajectory for loop/dandelion detection
        trajectory.push({
          turn,
          translatedTool: r.tool,
          argsDigest: JSON.stringify(parsedCalls[i].args).slice(0, 120),
          success: r.success,
          metaTool: parsedCalls[i].tool,
        })
        const touched = extractTouchedPath(parsedCalls[i], r)
        if (touched && !result.touchedFiles.includes(touched)) result.touchedFiles.push(touched)

        // Record failures in ledger
        if (!r.success && failureLedger) {
          addEntry(failureLedger, {
            tool: parsedCalls[i].tool,
            args: parsedCalls[i].args,
            error: r.error || r.output?.slice(0, 200) || "unknown",
            intent: `${parsedCalls[i].tool}(${JSON.stringify(parsedCalls[i].args).slice(0, 100)})`,
            model: config.model,
            runId,
            timestamp: new Date().toISOString(),
          })
        }

        // Record interceptor outcomes
        if (interceptorsEnabled) {
          const hit = checkInterceptors(parsedCalls[i].tool, parsedCalls[i].args, { workDir })
          if (hit) {
            recordInterceptorOutcome(hit.interceptor.id, r.success)
            if (!r.success) {
              recordInterceptorMiss(hit.interceptor.id, parsedCalls[i].args.command || "", r.error || "")
            }
          }
        }
      }
    }

    // Build messages
    messages.push({
      role: "assistant",
      content: turnResult.content || null,
      tool_calls: turnResult.toolCalls,
    })
    for (let i = 0; i < turnResult.toolCalls.length; i++) {
      const tc = turnResult.toolCalls[i]
      const r = toolResults[i]
      // Use larger limit for read operations (files need to be fully visible)
      const outputLimit = (r?.tool === "read" || r?.tool === "investigate") ? 6000 : 4000
      const content = r
        ? (r.success ? `OK: ${r.output?.slice(0, outputLimit)}` : `ERROR: ${r.output}${r.error ? ` (${r.error})` : ""}`)
        : "ERROR: No result"
      messages.push({ role: "tool", content, tool_call_id: tc.id })
    }

    // Check for test pass or completion
    for (const r of toolResults) {
      if (checkTestPass(r)) result.testsPass = true
      if (r.output?.includes("Complete:")) result.finalText = r.output
    }

    if (result.testsPass) break
  }

  result.durationSec = Math.round((Date.now() - startTime) / 1000)
  return result
}

// ── Helpers ────────────────────────────────────────────────────────────

function checkTestPass(r: ToolCallResult): boolean {
  if (!r.output) return false
  if (r.output.includes("tests passed") && !r.output.includes("FAIL")) return true
  if (r.output.includes("Tests pass")) return true
  if (r.output.includes("Complete approved")) return true
  return false
}

function extractTouchedPath(call: ToolCall | undefined, result: ToolCallResult): string | null {
  if (!call) return null

  const normalize = (value: string | undefined): string | null => {
    if (!value) return null
    const trimmed = value.replace(/`/g, "").trim().replace(/^\.\//, "")
    return trimmed || null
  }

  if (result.tool === "read" || result.tool === "write" || result.tool === "edit") {
    return normalize(call.args.path || call.args.file || call.args.target)
  }

  if (call.tool === "investigate" && (call.args.how || "read") === "read") {
    return normalize(call.args.target)
  }

  if (call.tool === "modify") {
    return normalize(call.args.file || call.args.path)
  }

  return null
}
