/**
 * Minimal Tool Executor — standalone tool implementations for plan-lab.
 *
 * Supports: read, write, edit, grep, list, glob, bash, complete
 * No desktop tools, no conveyor, no interceptors.
 */

import fs from "fs/promises"
import { existsSync, readFileSync } from "fs"
import path from "path"
import { spawn } from "child_process"
import type { OpenAIToolDef } from "./client"

// ── Types ─────────────────────────────────────────────────────────────

export interface ToolCall {
  tool: string
  args: Record<string, string>
  raw: string
}

export interface ToolCallResult {
  tool: string
  success: boolean
  output: string
  error?: string
}

export interface ExecutionContext {
  workDir: string
  allowedTools: string[]
}

export interface ToolExecutor {
  executeAll(calls: ToolCall[], context: ExecutionContext): Promise<ToolCallResult[]>
}

// ── Path safety ───────────────────────────────────────────────────────

function safePath(workDir: string, filePath: string): string | null {
  const resolved = path.resolve(workDir, filePath)
  if (resolved.startsWith(path.resolve(workDir))) return resolved
  // Allow tmp dirs (cloned repos)
  const tmp = process.platform === "darwin" ? "/private/tmp" : "/tmp"
  if (resolved.startsWith(tmp)) return resolved
  return null
}

// ── Shell execution ───────────────────────────────────────────────────

const SHELL_TIMEOUT_MS = 30_000

function runShell(command: string, cwd: string, timeoutMs = SHELL_TIMEOUT_MS): Promise<{ stdout: string; stderr: string; code: number }> {
  return new Promise((resolve) => {
    const proc = spawn("sh", ["-c", command], {
      cwd,
      env: { ...process.env, HOME: process.env.HOME },
      stdio: ["ignore", "pipe", "pipe"],
    })

    let stdout = ""
    let stderr = ""
    let killed = false

    const timer = setTimeout(() => {
      killed = true
      proc.kill("SIGKILL")
    }, timeoutMs)

    proc.stdout.on("data", (d: Buffer) => { stdout += d.toString() })
    proc.stderr.on("data", (d: Buffer) => { stderr += d.toString() })

    proc.on("close", (code) => {
      clearTimeout(timer)
      if (killed) {
        resolve({ stdout, stderr: stderr + `\n[Timeout after ${timeoutMs}ms]`, code: 124 })
      } else {
        resolve({ stdout, stderr, code: code ?? 1 })
      }
    })

    proc.on("error", (err) => {
      clearTimeout(timer)
      resolve({ stdout, stderr: err.message, code: 1 })
    })
  })
}

// ── Tool implementations ──────────────────────────────────────────────

async function toolRead(args: Record<string, string>, ctx: ExecutionContext): Promise<ToolCallResult> {
  const filePath = args.path || args.target || ""
  if (!filePath) return { tool: "read", success: false, output: "", error: "Missing path" }

  const resolved = safePath(ctx.workDir, filePath)
  if (!resolved) return { tool: "read", success: false, output: "", error: `Path outside workspace: ${filePath}` }

  try {
    const content = await fs.readFile(resolved, "utf-8")
    const numbered = content.split("\n").map((line, i) => `${String(i + 1).padStart(4)} | ${line}`).join("\n")
    return { tool: "read", success: true, output: numbered }
  } catch (e: any) {
    return { tool: "read", success: false, output: "", error: e.message }
  }
}

async function toolWrite(args: Record<string, string>, ctx: ExecutionContext): Promise<ToolCallResult> {
  const filePath = args.path || args.file || ""
  const content = args.content || ""
  if (!filePath) return { tool: "write", success: false, output: "", error: "Missing path" }

  const resolved = safePath(ctx.workDir, filePath)
  if (!resolved) return { tool: "write", success: false, output: "", error: `Path outside workspace: ${filePath}` }

  try {
    await fs.mkdir(path.dirname(resolved), { recursive: true })
    await fs.writeFile(resolved, content, "utf-8")
    return { tool: "write", success: true, output: `Written ${content.length} chars to ${filePath}` }
  } catch (e: any) {
    return { tool: "write", success: false, output: "", error: e.message }
  }
}

async function toolEdit(args: Record<string, string>, ctx: ExecutionContext): Promise<ToolCallResult> {
  const filePath = args.path || args.file || ""
  if (!filePath) return { tool: "edit", success: false, output: "", error: "Missing path" }

  const resolved = safePath(ctx.workDir, filePath)
  if (!resolved) return { tool: "edit", success: false, output: "", error: `Path outside workspace: ${filePath}` }

  try {
    let content = await fs.readFile(resolved, "utf-8")

    // Line-range mode
    if (args.lines) {
      const lines = content.split("\n")
      const match = args.lines.match(/^(\d+)(?:-(\d+))?$/)
      if (!match) return { tool: "edit", success: false, output: "", error: `Invalid lines: ${args.lines}` }

      const start = parseInt(match[1], 10) - 1
      const end = match[2] ? parseInt(match[2], 10) : start + 1
      const replacement = args.content || args.new_text || args.new_string || ""
      const newLines = replacement.split("\n")

      lines.splice(start, end - start, ...newLines)
      content = lines.join("\n")
      await fs.writeFile(resolved, content, "utf-8")
      return { tool: "edit", success: true, output: `Replaced lines ${start + 1}-${end} in ${filePath}` }
    }

    // Search-and-replace mode (old_string / old_text)
    const oldStr = args.old_string || args.old_text || ""
    const newStr = args.new_string || args.new_text || args.content || ""

    if (oldStr) {
      if (!content.includes(oldStr)) {
        // Try trimmed match
        const trimmed = oldStr.trim()
        if (trimmed && content.includes(trimmed)) {
          content = content.replace(trimmed, newStr.trim())
          await fs.writeFile(resolved, content, "utf-8")
          return { tool: "edit", success: true, output: `Replaced (trimmed match) in ${filePath}` }
        }
        return { tool: "edit", success: false, output: `old_string not found in ${filePath}`, error: "No match" }
      }
      content = content.replace(oldStr, newStr)
      await fs.writeFile(resolved, content, "utf-8")
      return { tool: "edit", success: true, output: `Replaced in ${filePath}` }
    }

    // <<<SEARCH>>>...<<<REPLACE>>> mode
    const srMatch = (args.content || "").match(/<<<SEARCH>>>\n?([\s\S]*?)<<<REPLACE>>>\n?([\s\S]*)/)
    if (srMatch) {
      const search = srMatch[1].trimEnd()
      const replace = srMatch[2].trimEnd()
      if (!content.includes(search)) {
        return { tool: "edit", success: false, output: `Search text not found in ${filePath}`, error: "No match" }
      }
      content = content.replace(search, replace)
      await fs.writeFile(resolved, content, "utf-8")
      return { tool: "edit", success: true, output: `Search/replace in ${filePath}` }
    }

    return { tool: "edit", success: false, output: "", error: "No edit operation specified (need lines, old_string, or <<<SEARCH>>>/<<<REPLACE>>>)" }
  } catch (e: any) {
    return { tool: "edit", success: false, output: "", error: e.message }
  }
}

async function toolGrep(args: Record<string, string>, ctx: ExecutionContext): Promise<ToolCallResult> {
  const pattern = args.pattern || args.target || ""
  if (!pattern) return { tool: "grep", success: false, output: "", error: "Missing pattern" }

  const globFilter = args.glob ? `--glob '${args.glob}'` : ""
  const contextFlag = args.context ? `-C ${args.context}` : ""
  const cmd = `rg --no-heading -n ${contextFlag} ${globFilter} ${JSON.stringify(pattern)} . 2>/dev/null | head -80`

  const result = await runShell(cmd, ctx.workDir)
  const output = (result.stdout || result.stderr).trim()
  return { tool: "grep", success: output.length > 0, output: output || "No matches found." }
}

async function toolList(args: Record<string, string>, ctx: ExecutionContext): Promise<ToolCallResult> {
  const dir = args.path || args.target || "."
  const resolved = safePath(ctx.workDir, dir)
  if (!resolved) return { tool: "list", success: false, output: "", error: `Path outside workspace: ${dir}` }

  const result = await runShell(`find ${JSON.stringify(resolved)} -maxdepth 3 -not -path '*/node_modules/*' -not -path '*/.git/*' | head -200 | sort`, ctx.workDir)
  // Make paths relative
  const output = result.stdout.replace(new RegExp(ctx.workDir + "/?", "g"), "").trim()
  return { tool: "list", success: true, output: output || "(empty directory)" }
}

async function toolGlob(args: Record<string, string>, ctx: ExecutionContext): Promise<ToolCallResult> {
  const pattern = args.pattern || args.target || ""
  if (!pattern) return { tool: "glob", success: false, output: "", error: "Missing pattern" }

  const startDir = args.path || "."
  const cmd = `find ${JSON.stringify(startDir)} -name ${JSON.stringify(pattern)} -not -path '*/node_modules/*' -not -path '*/.git/*' 2>/dev/null | head -50`
  const result = await runShell(cmd, ctx.workDir)
  const output = result.stdout.trim()
  return { tool: "glob", success: output.length > 0, output: output || "No files found." }
}

async function toolBash(args: Record<string, string>, ctx: ExecutionContext): Promise<ToolCallResult> {
  const command = args.command || ""
  if (!command) return { tool: "bash", success: false, output: "", error: "Missing command" }

  const result = await runShell(command, ctx.workDir)
  const output = (result.stdout + (result.stderr ? `\nSTDERR: ${result.stderr}` : "")).trim()

  // False success detection: exit 0 but stderr has error indicators
  const hasErrorSignals = /error|exception|FAIL|fatal|panic|traceback/i.test(result.stderr)
  const success = result.code === 0 && !hasErrorSignals

  return { tool: "bash", success, output: output.slice(0, 4000), error: success ? undefined : `Exit ${result.code}` }
}

function toolComplete(args: Record<string, string>): ToolCallResult {
  return { tool: "complete", success: true, output: `Complete: ${args.summary || args.message || "Done."}` }
}

// ── Executor factory ──────────────────────────────────────────────────

export function createToolExecutor(): ToolExecutor {
  return {
    async executeAll(calls, context) {
      const results: ToolCallResult[] = []

      for (const call of calls) {
        if (!context.allowedTools.includes(call.tool)) {
          results.push({ tool: call.tool, success: false, output: "", error: `Tool "${call.tool}" not allowed` })
          continue
        }

        switch (call.tool) {
          case "read":     results.push(await toolRead(call.args, context)); break
          case "write":    results.push(await toolWrite(call.args, context)); break
          case "edit":     results.push(await toolEdit(call.args, context)); break
          case "grep":     results.push(await toolGrep(call.args, context)); break
          case "list":     results.push(await toolList(call.args, context)); break
          case "glob":     results.push(await toolGlob(call.args, context)); break
          case "bash":     results.push(await toolBash(call.args, context)); break
          case "complete": results.push(toolComplete(call.args)); break
          default:
            results.push({ tool: call.tool, success: false, output: "", error: `Unknown tool: ${call.tool}` })
        }
      }

      return results
    },
  }
}

// ── OpenAI-compatible tool schemas for FC mode ────────────────────────

export function getBaseToolSchemas(): OpenAIToolDef[] {
  return [
    {
      type: "function",
      function: {
        name: "read",
        description: "Read a file's contents. Returns numbered lines.",
        parameters: {
          type: "object",
          properties: {
            path: { type: "string", description: "Relative file path" },
          },
          required: ["path"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "write",
        description: "Create a new file with the given content.",
        parameters: {
          type: "object",
          properties: {
            path: { type: "string", description: "Relative file path" },
            content: { type: "string", description: "Full file content" },
          },
          required: ["path", "content"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "edit",
        description: "Edit an existing file. Find old_string and replace with new_string.",
        parameters: {
          type: "object",
          properties: {
            path: { type: "string", description: "Relative file path" },
            old_string: { type: "string", description: "Exact text to find" },
            new_string: { type: "string", description: "Replacement text" },
          },
          required: ["path", "old_string", "new_string"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "grep",
        description: "Search file contents by regex pattern.",
        parameters: {
          type: "object",
          properties: {
            pattern: { type: "string", description: "Regex search pattern" },
            glob: { type: "string", description: "Optional file glob filter (e.g. *.js)" },
          },
          required: ["pattern"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "list",
        description: "List directory contents.",
        parameters: {
          type: "object",
          properties: {
            path: { type: "string", description: "Directory path (default: .)" },
          },
        },
      },
    },
    {
      type: "function",
      function: {
        name: "glob",
        description: "Find files by name or glob pattern.",
        parameters: {
          type: "object",
          properties: {
            pattern: { type: "string", description: "Glob pattern (e.g. *.test.js)" },
          },
          required: ["pattern"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "bash",
        description: "Run a shell command.",
        parameters: {
          type: "object",
          properties: {
            command: { type: "string", description: "The command to run" },
          },
          required: ["command"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "complete",
        description: "Signal that you are done. Tests must pass first.",
        parameters: {
          type: "object",
          properties: {
            summary: { type: "string", description: "Brief summary of what you did" },
          },
          required: ["summary"],
        },
      },
    },
  ]
}
