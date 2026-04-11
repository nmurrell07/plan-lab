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
import { execSync } from "child_process"
import type { ChatMessage, OpenAIToolDef, SendWithToolsFn } from "./client"
import type { ToolCallResult } from "./tools"

// ── Station 5: Syntax validation ─────────────────────────────────────

export interface ValidationResult {
  valid: boolean
  error?: string
  /** Normalized bucket key for the failure ledger */
  bucket?: string
}

const VALIDATION_EXTENSIONS = new Set([".js", ".mjs", ".cjs", ".jsx"])

/**
 * Validate file content before committing to disk.
 * Returns { valid: true } or { valid: false, error, bucket }.
 */
export function validateSyntax(content: string, filePath: string): ValidationResult {
  const ext = path.extname(filePath).toLowerCase()
  if (!VALIDATION_EXTENSIONS.has(ext)) return { valid: true }

  try {
    execSync("node --check -", { input: content, encoding: "utf-8", timeout: 5000, stdio: ["pipe", "pipe", "pipe"] })
    return { valid: true }
  } catch (err: any) {
    const stderr = err.stderr || ""
    const firstLines = stderr.split("\n").slice(0, 4).join("\n").trim()

    // Classify the error for ledger bucketing
    let bucket = "syntax_unknown"
    if (/unexpected (token|identifier)/i.test(stderr)) bucket = "syntax_unexpected_token"
    else if (/unterminated|unexpected end/i.test(stderr)) bucket = "syntax_unterminated"
    else if (/duplicate/i.test(stderr)) bucket = "syntax_duplicate"
    else if (/unexpected string/i.test(stderr)) bucket = "syntax_unexpected_string"

    return { valid: false, error: firstLines, bucket }
  }
}

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

  // ── Station 5: Validate + retry loop ──
  const MAX_STATION5_RETRIES = 1

  for (let s5attempt = 0; s5attempt <= MAX_STATION5_RETRIES; s5attempt++) {
    // Re-read file (may have changed since Station 2 or previous attempt)
    try {
      fileContent = await fs.readFile(resolved, "utf-8")
    } catch {
      return { tool: "edit", success: false, output: "", error: `File disappeared: ${file}` }
    }

    // Apply the edit
    const freshLines = fileContent.split("\n")
    const newLines = cleanReplacement.split("\n")
    freshLines.splice(startIdx, removeCount, ...newLines)

    const newContent = freshLines.join("\n")

    // Validate
    const validation = validateSyntax(newContent, file)
    if (validation.valid) {
      await fs.writeFile(resolved, newContent, "utf-8")
      return {
        tool: "edit",
        success: true,
        output: `Replaced lines ${startLine}-${endLine} in ${file} (${removeCount} lines -> ${newLines.length} lines)`,
      }
    }

    // Validation failed
    const bucketTag = validation.bucket || "syntax_unknown"
    console.log(`    [surgical] Station 5 FAILED (${bucketTag}, attempt ${s5attempt + 1}/${MAX_STATION5_RETRIES + 1}): ${validation.error?.slice(0, 80)}`)

    // If we have retries left, ask the model for a corrected replacement
    if (s5attempt < MAX_STATION5_RETRIES && sendWithTools) {
      const correctionPrompt = [
        `My replacement code for lines ${startLine}-${endLine} in ${file} has a syntax error:`,
        "",
        validation.error,
        "",
        `Original lines I am replacing:`,
        oldText,
        "",
        `My broken replacement:`,
        cleanReplacement,
        "",
        `I need to fix the syntax error. Write the corrected replacement code only — no explanation, no markdown fences.`,
      ].join("\n")

      const corrected = await askStationValue(
        messages,
        correctionPrompt,
        "corrected_code",
        sendWithTools,
        model,
        0, // no retries within the retry
      )

      if (corrected?.trim()) {
        cleanReplacement = corrected.replace(/^```\w*\n?/, "").replace(/\n?```$/, "")
        console.log(`    [surgical] Station 5 retry: got corrected replacement (${cleanReplacement.length} chars)`)
        continue
      }
    }

    // Out of retries — fail with bucket info for the ledger
    return {
      tool: "edit",
      success: false,
      output: `Edit would create a syntax error in ${file} — rolling back. Error: ${validation.error}`,
      error: `Surgical edit produced invalid syntax [${bucketTag}]`,
    }
  }

  // Should not reach here, but just in case
  return { tool: "edit", success: false, output: "", error: "Station 5: unexpected exit" }
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

  // ── Tier 0: Kimi Linear markers ──
  // Kimi uses <|tool_calls_section_begin|> with a different structure than standard channel markers
  const kimiSectionPattern = /<\|tool_calls_section_begin\|>([\s\S]*?)<\|tool_calls_section_end\|>/g
  let match
  while ((match = kimiSectionPattern.exec(text)) !== null) {
    const section = match[1]
    // Kimi wraps individual calls within the section
    const kimiCallPattern = /<\|tool_call_begin\|>([\s\S]*?)<\|tool_call_end\|>/g
    let callMatch
    while ((callMatch = kimiCallPattern.exec(section)) !== null) {
      const callBody = callMatch[1]
      // Extract function name and arguments
      const nameMatch = callBody.match(/(\w+)\s*\n/)
      const argsMatch = callBody.match(/```json\s*\n?([\s\S]*?)```/) || callBody.match(/(\{[\s\S]*\})/)
      if (nameMatch && argsMatch) {
        const toolName = nameMatch[1].trim()
        try {
          const args = JSON.parse(argsMatch[1].trim())
          if (availableTools.includes(toolName)) {
            calls.push({ tool: toolName, args, raw: callMatch[0] })
          }
        } catch { /* skip malformed */ }
      }
    }
  }
  if (calls.length > 0) return calls

  // Lenient Kimi: section begin without section end (truncated)
  const kimiLenientPattern = /<\|tool_calls_section_begin\|>([\s\S]*?)(?:<\|tool_calls_section_end\|>|$)/g
  while ((match = kimiLenientPattern.exec(text)) !== null) {
    const section = match[1]
    // Try to find function name + JSON args
    const fnPattern = /(\w+)\s*\n\s*```(?:json)?\s*\n?([\s\S]*?)```/g
    let fnMatch
    while ((fnMatch = fnPattern.exec(section)) !== null) {
      const toolName = fnMatch[1].trim()
      try {
        const args = JSON.parse(fnMatch[2].trim())
        if (availableTools.includes(toolName)) {
          calls.push({ tool: toolName, args, raw: fnMatch[0] })
        }
      } catch { /* skip */ }
    }
    // Also try without code fences — just name + bare JSON
    if (calls.length === 0) {
      const barePattern = /(\w+)\s*\n\s*(\{[\s\S]*?\})\s*(?:\n|$)/g
      let bareMatch
      while ((bareMatch = barePattern.exec(section)) !== null) {
        const toolName = bareMatch[1].trim()
        try {
          const args = JSON.parse(bareMatch[2].trim())
          if (availableTools.includes(toolName)) {
            calls.push({ tool: toolName, args, raw: bareMatch[0] })
          }
        } catch { /* skip */ }
      }
    }
  }
  if (calls.length > 0) return calls

  // ── Tier 1: Channel markers (Qwen-style) ──
  const channelPattern = /<\|tool_call_begin\|>functions\.(\w+)(?::\d+)?<\|tool_call_argument_begin\|>([\s\S]*?)<\|tool_call_argument_end\|>/g
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
