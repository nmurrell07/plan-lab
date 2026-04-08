/**
 * Surgical Edit Conveyor — 4-station search-then-edit pipeline.
 *
 * Instead of asking the model to emit (file, old_string, new_string) in one shot,
 * decompose into single-inference stations:
 *   Station 1: Intent — model specifies file + search keyword
 *   Station 2: Locate — system reads file, greps keyword, shows context (no LLM)
 *   Station 3: Confirm — model picks start/end line range
 *   Station 4: Write — model provides replacement code
 *
 * Also includes text-to-FC recovery for when models emit text instead of tool calls.
 *
 * Stripped from roundtable tool-conveyor.ts for plan-lab.
 */

import fs from "fs/promises"
import path from "path"
import type { ChatMessage, OpenAIToolDef, SendWithToolsFn } from "./client"
import type { ToolCallResult } from "./tools"

// ── Station answer tool ───────────────────────────────────────────────

function makeAnswerTool(fieldName: string, description: string): OpenAIToolDef {
  return {
    type: "function",
    function: {
      name: "answer",
      description: `Provide the ${fieldName}`,
      parameters: {
        type: "object",
        properties: {
          value: { type: "string", description },
        },
        required: ["value"],
      },
    },
  }
}

// ── Ask a station question ────────────────────────────────────────────

async function askStationValue(
  messages: ChatMessage[],
  prompt: string,
  fieldName: string,
  sendWithTools: SendWithToolsFn,
  model: string,
  maxRetries: number,
): Promise<string | null> {
  const tool = makeAnswerTool(fieldName, `The ${fieldName} value`)

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const stationMessages: ChatMessage[] = [
      ...messages,
      { role: "user", content: prompt },
    ]

    const result = await sendWithTools("Station", stationMessages, [tool], model)

    // Extract from tool call
    if (result.toolCalls?.length) {
      try {
        const args = JSON.parse(result.toolCalls[0].function.arguments)
        const value = args.value ?? args[fieldName] ?? ""
        if (typeof value === "string" && value.trim()) return value.trim()
        if (typeof value === "object") return JSON.stringify(value)
      } catch { /* retry */ }
    }

    // Extract from text
    if (result.text?.trim()) {
      const firstLine = result.text.trim().split("\n")[0].trim()
      if (firstLine.length > 0 && firstLine.length < 500) return firstLine
    }
  }

  return null
}

// ── Surgical Edit Pipeline ────────────────────────────────────────────

export interface SurgicalEditSeed {
  file?: string
  searchKeyword?: string
  /** Model's original replacement text — if provided, skip Station 4 (re-derivation) and use this directly */
  newText?: string
}

export async function runSurgicalEditConveyor(
  messages: ChatMessage[],
  sendWithTools: SendWithToolsFn,
  model: string,
  workDir: string,
  maxRetries: number = 2,
  seed?: SurgicalEditSeed,
): Promise<ToolCallResult> {
  const CONTEXT_LINES = 10

  // ── Station 1: Intent (file + keyword) ──
  let file = seed?.file || null
  let keyword = seed?.searchKeyword || null

  if (!file) {
    file = await askStationValue(
      messages,
      "What file do you want to edit? Give the relative file path only.",
      "file",
      sendWithTools,
      model,
      maxRetries,
    )
  }

  if (!file) {
    return { tool: "edit", success: false, output: "", error: "Conveyor: could not determine file" }
  }

  // Resolve file path
  const resolved = path.resolve(workDir, file)
  if (!resolved.startsWith(path.resolve(workDir))) {
    return { tool: "edit", success: false, output: "", error: `Path outside workspace: ${file}` }
  }

  let fileContent: string
  try {
    fileContent = await fs.readFile(resolved, "utf-8")
  } catch {
    return { tool: "edit", success: false, output: "", error: `File not found: ${file}` }
  }

  if (!keyword) {
    keyword = await askStationValue(
      messages,
      `File "${file}" exists (${fileContent.split("\n").length} lines). What text or keyword should I search for to find the section to edit?`,
      "keyword",
      sendWithTools,
      model,
      maxRetries,
    )
  }

  if (!keyword) {
    return { tool: "edit", success: false, output: "", error: "Conveyor: could not determine search keyword" }
  }

  // ── Station 2: Locate (system-only, no LLM) ──
  const lines = fileContent.split("\n")
  let matchLineIdx = -1

  // Find best matching line
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes(keyword)) {
      matchLineIdx = i
      break
    }
  }

  // Fallback: case-insensitive, then token-based
  if (matchLineIdx === -1) {
    const kwLower = keyword.toLowerCase()
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].toLowerCase().includes(kwLower)) {
        matchLineIdx = i
        break
      }
    }
  }

  if (matchLineIdx === -1) {
    // Try splitting keyword into tokens and finding best partial match
    const tokens = keyword.split(/[\s.(){}[\]:;,]+/).filter(t => t.length > 2)
    let bestScore = 0
    for (let i = 0; i < lines.length; i++) {
      const lineLower = lines[i].toLowerCase()
      const score = tokens.filter(t => lineLower.includes(t.toLowerCase())).length
      if (score > bestScore) {
        bestScore = score
        matchLineIdx = i
      }
    }
  }

  if (matchLineIdx === -1) {
    return {
      tool: "edit",
      success: false,
      output: `Could not find "${keyword}" in ${file}`,
      error: "Keyword not found in file",
    }
  }

  // Build context window (±CONTEXT_LINES around match)
  const ctxStart = Math.max(0, matchLineIdx - CONTEXT_LINES)
  const ctxEnd = Math.min(lines.length, matchLineIdx + CONTEXT_LINES + 1)
  const contextWindow = lines
    .slice(ctxStart, ctxEnd)
    .map((line, i) => {
      const lineNum = ctxStart + i + 1
      const marker = ctxStart + i === matchLineIdx ? " >>>" : "    "
      return `${String(lineNum).padStart(4)}${marker} ${line}`
    })
    .join("\n")

  // ── Station 3: Confirm line range ──
  const rangeStr = await askStationValue(
    messages,
    [
      `Found "${keyword}" in ${file} at line ${matchLineIdx + 1}. Here is the context:`,
      "",
      contextWindow,
      "",
      "What line range do you want to replace? Format: START-END (e.g., 5-8) or a single line number.",
    ].join("\n"),
    "line_range",
    sendWithTools,
    model,
    maxRetries,
  )

  if (!rangeStr) {
    return { tool: "edit", success: false, output: "", error: "Conveyor: could not determine line range" }
  }

  // Parse range
  const rangeMatch = rangeStr.match(/(\d+)\s*[-–]\s*(\d+)/) || rangeStr.match(/(\d+)/)
  if (!rangeMatch) {
    return { tool: "edit", success: false, output: "", error: `Invalid line range: ${rangeStr}` }
  }

  const startLine = parseInt(rangeMatch[1], 10)
  const endLine = rangeMatch[2] ? parseInt(rangeMatch[2], 10) : startLine
  const startIdx = startLine - 1
  const removeCount = endLine - startLine + 1

  if (startIdx < 0 || startIdx >= lines.length || endLine > lines.length) {
    return { tool: "edit", success: false, output: "", error: `Line range ${startLine}-${endLine} out of bounds (file has ${lines.length} lines)` }
  }

  // Show the exact lines being replaced
  const oldLines = lines.slice(startIdx, startIdx + removeCount)
  const oldText = oldLines.map((l, i) => `${String(startLine + i).padStart(4)} | ${l}`).join("\n")

  // ── Station 4: Write replacement ──
  // If the model already provided new_text in the original call, USE IT.
  // The model wrote new_text in the main conversation where it has full context
  // (test output, plan, error messages). Re-deriving in a stripped station loses that.
  let replacement: string | null = null

  if (seed?.newText?.trim()) {
    replacement = seed.newText
    console.log(`    [surgical] Station 4: using model's original new_text (${replacement.length} chars)`)
  } else {
    replacement = await askStationValue(
      messages,
      [
        `Replace lines ${startLine}-${endLine} in ${file}:`,
        "",
        oldText,
        "",
        "Write the replacement code. Only the code — no markdown fences, no explanation.",
      ].join("\n"),
      "replacement_code",
      sendWithTools,
      model,
      maxRetries,
    )
  }

  if (!replacement) {
    return { tool: "edit", success: false, output: "", error: "Conveyor: no replacement provided" }
  }

  // Strip markdown fences if present
  let cleanReplacement = replacement
    .replace(/^```\w*\n?/, "")
    .replace(/\n?```$/, "")

  // Re-read file (may have changed since Station 2)
  try {
    fileContent = await fs.readFile(resolved, "utf-8")
  } catch {
    return { tool: "edit", success: false, output: "", error: `File disappeared: ${file}` }
  }

  // Apply the edit
  const freshLines = fileContent.split("\n")
  const newLines = cleanReplacement.split("\n")
  freshLines.splice(startIdx, removeCount, ...newLines)

  await fs.writeFile(resolved, freshLines.join("\n"), "utf-8")

  return {
    tool: "edit",
    success: true,
    output: `Replaced lines ${startLine}-${endLine} in ${file} (${removeCount} lines -> ${newLines.length} lines)`,
  }
}

// ── Text-to-FC Recovery ───────────────────────────────────────────────

/**
 * Recover tool calls from model text when function calling fails.
 *
 * 3-tier strategy:
 *   1. Channel markers (<|tool_call_begin|>...)
 *   2. Quick regex (file paths, bash commands)
 *   3. Intent detection via LLM (ask model to express as FC)
 */
export function recoverToolCallsFromText(
  text: string,
  availableTools: string[],
): Array<{ tool: string; args: Record<string, string>; raw: string }> | null {
  const calls: Array<{ tool: string; args: Record<string, string>; raw: string }> = []

  // ── Tier 1: Channel markers ──
  const channelPattern = /<\|tool_call_begin\|>functions\.(\w+)(?::\d+)?<\|tool_call_argument_begin\|>([\s\S]*?)<\|tool_call_argument_end\|>/g
  let match
  while ((match = channelPattern.exec(text)) !== null) {
    const toolName = match[1]
    try {
      const args = JSON.parse(match[2])
      if (availableTools.includes(toolName)) {
        calls.push({ tool: toolName, args, raw: match[0] })
      }
    } catch { /* skip malformed */ }
  }
  if (calls.length > 0) return calls

  // Lenient channel markers (truncated end tag)
  const lenientPattern = /<\|tool_call_begin\|>functions\.(\w+)(?::\d+)?<\|tool_call_argument_begin\|>(\{[^<]*\})/g
  while ((match = lenientPattern.exec(text)) !== null) {
    const toolName = match[1]
    try {
      const args = JSON.parse(match[2])
      if (availableTools.includes(toolName)) {
        calls.push({ tool: toolName, args, raw: match[0] })
      }
    } catch { /* skip */ }
  }
  if (calls.length > 0) return calls

  // ── Tier 2: Quick regex patterns ──

  // "read foo.js" or "cat foo.js" patterns
  const readMatch = text.match(/\b(?:read|cat|view)\s+([^\s]+\.\w{1,5})\b/i)
  if (readMatch && (availableTools.includes("read") || availableTools.includes("investigate"))) {
    const tool = availableTools.includes("investigate") ? "investigate" : "read"
    const args: Record<string, string> = tool === "investigate"
      ? { target: readMatch[1], how: "read" }
      : { path: readMatch[1] }
    return [{ tool, args, raw: readMatch[0] }]
  }

  // "run npm test" or "execute node ..." patterns
  const execMatch = text.match(/\b(?:run|execute|npm|node|bun|python|bash)\s+(.+?)(?:\.|$)/im)
  if (execMatch && (availableTools.includes("bash") || availableTools.includes("execute"))) {
    const cmd = execMatch[0].replace(/^(?:run|execute)\s+/i, "").trim().replace(/\.$/, "")
    const tool = availableTools.includes("execute") ? "execute" : "bash"
    const args = tool === "execute" ? { command: cmd } : { command: cmd }
    return [{ tool, args, raw: execMatch[0] }]
  }

  // ── Tier 3: JSON-like tool calls in text ──
  const jsonPattern = /\{[\s]*"(?:name|function)"[\s]*:[\s]*"(\w+)"[\s\S]*?"arguments"[\s]*:[\s]*(\{[^}]+\})/g
  while ((match = jsonPattern.exec(text)) !== null) {
    const toolName = match[1]
    if (availableTools.includes(toolName)) {
      try {
        const args = JSON.parse(match[2])
        calls.push({ tool: toolName, args, raw: match[0] })
      } catch { /* skip */ }
    }
  }
  if (calls.length > 0) return calls

  // ── Tier 4: Marker-based (---TOOL_CALL---) ──
  const markerPattern = /---TOOL_CALL---\s*\n([\s\S]*?)---TOOL_CALL_END---/g
  while ((match = markerPattern.exec(text)) !== null) {
    const block = match[1]
    const toolMatch = /^Tool:\s*(\w+)/mi.exec(block)
    if (toolMatch) {
      const tool = toolMatch[1].toLowerCase()
      const args: Record<string, string> = {}
      const blockLines = block.split("\n")
      let collectingContent = false
      const contentLines: string[] = []

      for (const line of blockLines) {
        if (collectingContent) { contentLines.push(line); continue }
        const kv = /^(\w+):\s*(.*)$/i.exec(line.trim())
        if (kv) {
          const key = kv[1].toLowerCase()
          if (key === "tool") continue
          if (key === "content" && !kv[2].trim()) { collectingContent = true }
          else { args[key] = kv[2].trim() }
        }
      }
      if (collectingContent) args.content = contentLines.join("\n")
      calls.push({ tool, args, raw: match[0] })
    }
  }

  return calls.length > 0 ? calls : null
}
