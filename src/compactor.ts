/**
 * Context Compactor — two-tier context compression.
 *
 * Tier 1 (Microcompact): Clears old droppable tool results, truncates large messages.
 *   Fast, no LLM call. Triggered at 60 messages.
 *
 * Tier 2 (Full Summarization): LLM summarizes all messages before the recent window.
 *   Requires LLM endpoint. Triggered at 80 messages.
 *
 * Stripped from roundtable compactor.ts for plan-lab.
 */

import type { ChatMessage, SendWithToolsFn } from "./client"

// ── Types ─────────────────────────────────────────────────────────────

export interface CompactionConfig {
  /** Message count threshold for Tier 1 microcompact (default: 60) */
  microCompactThreshold: number
  /** Message count threshold for Tier 2 full summarization (default: 80) */
  fullCompactThreshold: number
  /** Always keep this many recent messages (default: 10) */
  preserveRecentCount: number
  /** Tool result roles safe to clear during microcompact */
  droppableToolResults: string[]
  /** SendWithTools function for Tier 2 summarization */
  sendWithTools?: SendWithToolsFn
  /** Model for summarization */
  summaryModel?: string
  /** Scratchpad notes to inject into summary */
  notes?: string[]
}

export interface CompactionResult {
  tier: "micro" | "full" | "none"
  messagesBefore: number
  messagesAfter: number
  droppedCount: number
  summary?: string
}

// ── Defaults ──────────────────────────────────────────────────────────

const DEFAULT_CONFIG: CompactionConfig = {
  microCompactThreshold: 60,
  fullCompactThreshold: 80,
  preserveRecentCount: 10,
  droppableToolResults: [
    "read", "grep", "list", "glob", "bash",
    "investigate", "execute", "search",
  ],
}

const DROPPABLE_TOOL_SET = new Set(DEFAULT_CONFIG.droppableToolResults)

// ── Main entry point ──────────────────────────────────────────────────

export async function compactIfNeeded(
  messages: ChatMessage[],
  task: string,
  config: Partial<CompactionConfig> = {},
): Promise<{ messages: ChatMessage[]; result: CompactionResult }> {
  const cfg = { ...DEFAULT_CONFIG, ...config }
  const msgCount = messages.length

  // No compaction needed
  if (msgCount < cfg.microCompactThreshold) {
    return {
      messages,
      result: { tier: "none", messagesBefore: msgCount, messagesAfter: msgCount, droppedCount: 0 },
    }
  }

  // Tier 2: Full summarization
  if (msgCount >= cfg.fullCompactThreshold && cfg.sendWithTools && cfg.summaryModel) {
    return fullCompact(messages, task, cfg)
  }

  // Tier 1: Microcompact
  return microCompact(messages, cfg)
}

// ── Tier 1: Microcompact ──────────────────────────────────────────────

function microCompact(
  messages: ChatMessage[],
  config: CompactionConfig,
): { messages: ChatMessage[]; result: CompactionResult } {
  const before = messages.length
  const preserveStart = Math.max(0, messages.length - config.preserveRecentCount)
  let droppedCount = 0

  const compacted = messages.map((msg, idx) => {
    // Always keep system messages and recent messages
    if (msg.role === "system" || idx >= preserveStart) return msg

    // Always keep assistant messages (model reasoning)
    if (msg.role === "assistant") return msg

    // Clear droppable tool results
    if (msg.role === "tool") {
      const content = msg.content ?? ""
      // Check if this is a large tool result worth compacting
      if (content.length > 500) {
        droppedCount++
        return { ...msg, content: "[compacted -- tool result cleared to save context]" }
      }
      return msg
    }

    // Truncate large user messages (typically file reads injected as context)
    if (msg.role === "user" && (msg.content?.length ?? 0) > 2000) {
      const content = msg.content ?? ""
      droppedCount++
      return {
        ...msg,
        content: content.slice(0, 200) + "\n[... content compacted ...]",
      }
    }

    return msg
  })

  return {
    messages: compacted,
    result: {
      tier: "micro",
      messagesBefore: before,
      messagesAfter: compacted.length,
      droppedCount,
    },
  }
}

// ── Tier 2: Full summarization ────────────────────────────────────────

const SUMMARY_SYSTEM_PROMPT = `You are a context summarizer for a coding agent. Summarize the conversation so far.

Include:
- Files that were read and their key contents
- Edits that were made and to which files
- Commands that were run and their results
- Errors encountered and how they were addressed
- Current state of progress toward the task

Format as a bulleted list. Be specific -- include file paths, function names, error messages.
Keep it under 30 lines.`

async function fullCompact(
  messages: ChatMessage[],
  task: string,
  config: CompactionConfig,
): Promise<{ messages: ChatMessage[]; result: CompactionResult }> {
  const before = messages.length
  const preserveStart = Math.max(1, messages.length - config.preserveRecentCount) // keep system[0]

  // Split: to-summarize vs to-keep
  const systemMsg = messages[0] // always keep system prompt
  const toSummarize = messages.slice(1, preserveStart)
  const toKeep = messages.slice(preserveStart)

  // Build summary request
  const conversationText = toSummarize
    .map(m => `[${m.role}]: ${(m.content ?? "").slice(0, 500)}`)
    .join("\n")

  let summary: string

  try {
    const result = await config.sendWithTools!(
      "Compactor",
      [
        { role: "system", content: SUMMARY_SYSTEM_PROMPT },
        {
          role: "user",
          content: `## Task\n${task}\n\n## Conversation to summarize (${toSummarize.length} messages)\n${conversationText.slice(0, 8000)}`,
        },
      ],
      [],
      config.summaryModel,
    )
    summary = result.text || generateBasicSummary(toSummarize)
  } catch {
    summary = generateBasicSummary(toSummarize)
  }

  // Inject notes if available
  if (config.notes && config.notes.length > 0) {
    summary += "\n\n## Scratchpad Notes\n" + config.notes.map((n, i) => `${i + 1}. ${n}`).join("\n")
  }

  // Rebuild messages
  const summaryMessage: ChatMessage = {
    role: "user",
    content: `[Context was compacted. Here is a summary of what happened before this point:]\n\n${summary}`,
  }

  const compacted = [systemMsg, summaryMessage, ...toKeep]

  return {
    messages: compacted,
    result: {
      tier: "full",
      messagesBefore: before,
      messagesAfter: compacted.length,
      droppedCount: toSummarize.length,
      summary,
    },
  }
}

// ── Basic summary fallback (no LLM) ──────────────────────────────────

function generateBasicSummary(messages: ChatMessage[]): string {
  const toolCounts = new Map<string, number>()
  const edits: string[] = []
  const errors: string[] = []

  for (const msg of messages) {
    const content = msg.content ?? ""

    // Count tool mentions
    if (msg.role === "tool" || msg.role === "user") {
      const toolMatch = content.match(/\[(?:Result|Error)\].*?(\w+)/i)
      if (toolMatch) {
        const tool = toolMatch[1]
        toolCounts.set(tool, (toolCounts.get(tool) || 0) + 1)
      }
    }

    // Track edits
    if (content.includes("Replaced") || content.includes("Written")) {
      const fileMatch = content.match(/(?:in|to)\s+(\S+\.\w+)/i)
      if (fileMatch) edits.push(fileMatch[1])
    }

    // Track errors
    if (content.includes("ERROR") || content.includes("FAIL")) {
      errors.push(content.slice(0, 80))
    }
  }

  const lines: string[] = [`Summary of ${messages.length} compacted messages:`]

  if (toolCounts.size > 0) {
    lines.push("- Tool usage: " + [...toolCounts].map(([t, c]) => `${t}(${c})`).join(", "))
  }
  if (edits.length > 0) {
    lines.push("- Files edited: " + [...new Set(edits)].join(", "))
  }
  if (errors.length > 0) {
    lines.push("- Errors seen: " + errors.slice(0, 3).join("; "))
  }

  return lines.join("\n")
}
