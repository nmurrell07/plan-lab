/**
 * Failure Ledger — error bucketing, loop/dandelion detection, routing rules.
 *
 * Records all tool failures with normalized error keys, detects repeated patterns,
 * and learns routing rules to correct common mistakes.
 *
 * Stripped from roundtable failure-ledger.ts for plan-lab.
 */

import fs from "fs/promises"

// ── Types ─────────────────────────────────────────────────────────────

export interface IntentEntry {
  tool: string
  args: Record<string, string>
  error: string
  intent: string
  model: string
  runId: string
  timestamp: string
  phase?: string
}

export interface Lookout {
  question: string
  matchPattern: string
  _stats?: { blocks: number; falsePositives: number }
}

export interface RoutingRule {
  id: string
  match: { tool: string; errorPattern: string }
  action: { tool: string; argsTransform: string }
  source: "learned"
  createdAt: string
  activations: number
  successes: number
  phase?: string
}

export interface BucketAnalysis {
  pattern: string
  rootCause: string
  recommendation: "route" | "tree" | "prompt" | "ignore"
  confidence: number
}

export interface FailureBucket {
  key: string
  entries: IntentEntry[]
  analysisResult?: BucketAnalysis
  routingRule?: RoutingRule
  lookouts?: Lookout[]
  analyzedAt?: string
  analyzedCount: number
}

export interface FailureLedger {
  domain: string
  version: number
  buckets: Record<string, FailureBucket>
  routingRules: RoutingRule[]
  meta: {
    createdAt: string
    lastUpdated: string
    totalEntries: number
    totalAnalyses: number
    totalRulesLearned: number
  }
}

export interface LoopSignal {
  type: "command-repeat" | "lookout-block"
  pattern: string
  count: number
  lookoutQuestion?: string
}

export interface DandelionSignal {
  intent: string
  attempts: string[]
  count: number
  suggestion: "lateral" | "backtrack"
}

// ── Factory ───────────────────────────────────────────────────────────

export function createEmptyLedger(domain: string): FailureLedger {
  return {
    domain,
    version: 0,
    buckets: {},
    routingRules: [],
    meta: {
      createdAt: new Date().toISOString(),
      lastUpdated: new Date().toISOString(),
      totalEntries: 0,
      totalAnalyses: 0,
      totalRulesLearned: 0,
    },
  }
}

// ── Error key normalization ───────────────────────────────────────────

const ERROR_PATTERNS: Array<[RegExp, string]> = [
  [/old_string.*not found|search text not found|no match/i, "old_string_not_found"],
  [/path.*travers|outside.*workspace/i, "path_traversal"],
  [/command not found/i, "command_not_found"],
  [/tool.*not.*recogni|unknown tool/i, "tool_not_recognized"],
  [/no such file|ENOENT|file not found/i, "file_not_found"],
  [/permission denied|EACCES/i, "permission_denied"],
  [/syntax.?error|unexpected token/i, "syntax_error"],
  [/timeout|timed out/i, "timeout"],
  [/empty.*content|missing.*content/i, "empty_content"],
  [/already exists/i, "already_exists"],
  [/cannot find module|module not found/i, "module_not_found"],
  [/not a git repo/i, "not_git_repo"],
  [/nothing to commit/i, "nothing_to_commit"],
  [/FAIL|tests? fail/i, "test_failure"],
]

export function normalizeErrorKey(tool: string, error: string): string {
  for (const [pattern, key] of ERROR_PATTERNS) {
    if (pattern.test(error)) return `${tool}:${key}`
  }
  // Fallback: first few words
  const words = error.replace(/[^a-zA-Z0-9\s]/g, " ").trim().split(/\s+/).slice(0, 4).join("_").toLowerCase()
  return `${tool}:${words || "unknown"}`
}

// ── Entry management ──────────────────────────────────────────────────

export function addEntry(ledger: FailureLedger, entry: IntentEntry): { key: string; count: number } {
  const key = normalizeErrorKey(entry.tool, entry.error)

  if (!ledger.buckets[key]) {
    ledger.buckets[key] = { key, entries: [], analyzedCount: 0 }
  }

  ledger.buckets[key].entries.push(entry)
  ledger.meta.totalEntries++
  ledger.meta.lastUpdated = new Date().toISOString()
  ledger.version++

  return { key, count: ledger.buckets[key].entries.length }
}

export function getBucketsReadyForAnalysis(ledger: FailureLedger, threshold = 3): FailureBucket[] {
  return Object.values(ledger.buckets).filter(
    b => b.entries.length >= threshold && b.entries.length > b.analyzedCount,
  )
}

// ── Loop detection ────────────────────────────────────────────────────

interface TrajectoryEntry {
  translatedTool: string
  argsDigest: string
  success: boolean
  metaTool?: string
  phase?: string
  turn?: number
}

export function detectLoop(
  trajectory: TrajectoryEntry[],
  window = 10,
): LoopSignal | null {
  if (trajectory.length < 3) return null

  const recent = trajectory.slice(-window)
  const sigs = recent.map(t => `${t.translatedTool}:${t.argsDigest.slice(0, 40)}`)

  // Count signature frequencies
  const counts = new Map<string, number>()
  for (const sig of sigs) {
    counts.set(sig, (counts.get(sig) || 0) + 1)
  }

  // Find strongest loop (3+ repetitions)
  let bestSig = ""
  let bestCount = 0
  for (const [sig, count] of counts) {
    if (count >= 3 && count > bestCount) {
      bestSig = sig
      bestCount = count
    }
  }

  if (bestCount >= 3) {
    return { type: "command-repeat", pattern: bestSig, count: bestCount }
  }

  return null
}

// ── Dandelion detection (exhausted intent groups) ─────────────────────

export function extractTarget(args: Record<string, string>): string {
  // Extract meaningful target from args
  const combined = JSON.stringify(args)

  // Repo patterns
  const repoMatch = combined.match(/--repo[= ](\S+)/i) || combined.match(/([a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+)/)
  if (repoMatch) return repoMatch[1]

  // File paths
  const fileMatch = combined.match(/([a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+\.\w+)/)
  if (fileMatch) return fileMatch[1]

  // Command targets
  const cmdMatch = combined.match(/(?:grep|find|search|read)\s+(.{5,30})/i)
  if (cmdMatch) return cmdMatch[1].trim()

  return combined.slice(0, 50)
}

export function detectDandelion(
  trajectory: TrajectoryEntry[],
  maxSeeds = 4,
  window = 15,
): DandelionSignal | null {
  if (trajectory.length < maxSeeds) return null

  const recent = trajectory.slice(-window).filter(t => !t.success)

  // Group by phase + target
  const groups = new Map<string, string[]>()
  for (const entry of recent) {
    const phase = entry.phase || "unknown"
    const target = entry.argsDigest.slice(0, 30)
    const groupKey = `${phase}:${target}`

    const attempts = groups.get(groupKey) || []
    attempts.push(`${entry.translatedTool}(${entry.argsDigest.slice(0, 50)})`)
    groups.set(groupKey, attempts)
  }

  // Find groups with enough seeds
  for (const [intent, attempts] of groups) {
    if (attempts.length >= maxSeeds) {
      return {
        intent,
        attempts: attempts.slice(-6),
        count: attempts.length,
        suggestion: attempts.length > maxSeeds * 2 ? "backtrack" : "lateral",
      }
    }
  }

  return null
}

export function formatDandelionRedirect(signal: DandelionSignal): string {
  const lines = [
    `[STUCK — ${signal.count} failed attempts on "${signal.intent}"]`,
    "",
    "Attempts so far:",
    ...signal.attempts.map(a => `  - ${a}`),
    "",
    signal.suggestion === "lateral"
      ? "Try a DIFFERENT approach to achieve the same goal."
      : "BACKTRACK — this path is exhausted. Pick a completely different strategy.",
  ]
  return lines.join("\n")
}

// ── Routing rules ─────────────────────────────────────────────────────

export function matchRoutingRule(
  tool: string,
  error: string,
  args: Record<string, string>,
  rules: RoutingRule[],
  currentPhase?: string,
): { rule: RoutingRule; translatedTool: string; translatedArgs: Record<string, string> } | null {
  const normalizedKey = normalizeErrorKey(tool, error)

  for (const rule of rules) {
    // Phase check
    if (rule.phase && currentPhase && rule.phase !== currentPhase) continue

    // Tool match
    if (rule.match.tool !== "*" && rule.match.tool !== tool) continue

    // Error pattern match
    try {
      const pattern = new RegExp(rule.match.errorPattern, "i")
      const matchTarget = `${error} ${normalizedKey} ${JSON.stringify(args)}`
      if (!pattern.test(matchTarget)) continue
    } catch {
      continue // bad regex
    }

    // Reject overly broad patterns
    if (/^\.\*$|^exit\s*\d*$/.test(rule.match.errorPattern)) continue

    // Apply transform
    const translatedArgs = applyArgsTransform(rule.action.argsTransform, args, tool)

    return {
      rule,
      translatedTool: rule.action.tool,
      translatedArgs,
    }
  }

  return null
}

function applyArgsTransform(
  transform: string,
  args: Record<string, string>,
  originalTool: string,
): Record<string, string> {
  const result: Record<string, string> = {}

  // Parse "key = value" assignments
  const assignments = transform.split(/\s*[;\n]\s*/).filter(Boolean)

  for (const assignment of assignments) {
    const eqIdx = assignment.indexOf("=")
    if (eqIdx === -1) continue

    const key = assignment.slice(0, eqIdx).trim()
    let value = assignment.slice(eqIdx + 1).trim()

    // Substitute args.X tokens
    value = value.replace(/args\.(\w+)/g, (_, argName) => args[argName] || "")
    value = value.replace(/originalTool/g, originalTool)

    // Handle .replace() patterns
    const replaceMatch = value.match(/(.+)\.replace\(["'](.+?)["']\s*,\s*["'](.*?)["']\)/)
    if (replaceMatch) {
      const base = replaceMatch[1].replace(/args\.(\w+)/g, (_, n) => args[n] || "").trim()
      value = base.replace(new RegExp(replaceMatch[2]), replaceMatch[3])
    }

    // Handle concatenation with +
    if (value.includes("+")) {
      value = value.split("+").map(part => {
        const trimmed = part.trim()
        if (trimmed.startsWith("'") || trimmed.startsWith('"')) {
          return trimmed.slice(1, -1)
        }
        return trimmed
      }).join("")
    }

    result[key] = value.trim()
  }

  return result
}

export function recordRuleActivation(rule: RoutingRule, success: boolean): void {
  rule.activations++
  if (success) rule.successes++
}

export function getDisabledRules(rules: RoutingRule[], minActivations = 5, minSuccessRate = 0.5): string[] {
  return rules
    .filter(r => r.activations >= minActivations && (r.successes / r.activations) < minSuccessRate)
    .map(r => r.id)
}

// ── Lookout system ────────────────────────────────────────────────────

export function getMatchingLookouts(command: string, ledger: FailureLedger): Lookout[] {
  const matches: Lookout[] = []
  for (const bucket of Object.values(ledger.buckets)) {
    if (!bucket.lookouts) continue
    for (const lookout of bucket.lookouts) {
      try {
        if (new RegExp(lookout.matchPattern, "i").test(command)) {
          matches.push(lookout)
        }
      } catch { /* bad regex, skip */ }
    }
  }
  return matches
}

export function trackLookoutEffectiveness(lookout: Lookout, commandSucceeded: boolean): void {
  if (!lookout._stats) lookout._stats = { blocks: 0, falsePositives: 0 }
  lookout._stats.blocks++
  if (commandSucceeded) lookout._stats.falsePositives++
}

export function getCounterproductiveLookouts(
  ledger: FailureLedger,
  minBlocks = 3,
  maxFalsePositiveRate = 0.7,
): Array<{ bucketKey: string; question: string; falsePositiveRate: number }> {
  const results: Array<{ bucketKey: string; question: string; falsePositiveRate: number }> = []

  for (const bucket of Object.values(ledger.buckets)) {
    if (!bucket.lookouts) continue
    for (const lookout of bucket.lookouts) {
      if (!lookout._stats || lookout._stats.blocks < minBlocks) continue
      const rate = lookout._stats.falsePositives / lookout._stats.blocks
      if (rate >= maxFalsePositiveRate) {
        results.push({ bucketKey: bucket.key, question: lookout.question, falsePositiveRate: rate })
      }
    }
  }

  return results
}

export function disableLookout(ledger: FailureLedger, bucketKey: string, question: string): boolean {
  const bucket = ledger.buckets[bucketKey]
  if (!bucket?.lookouts) return false
  const idx = bucket.lookouts.findIndex(l => l.question === question)
  if (idx === -1) return false
  bucket.lookouts.splice(idx, 1)
  ledger.version++
  return true
}

// ── Persistence ───────────────────────────────────────────────────────

export async function loadLedger(filePath: string): Promise<FailureLedger | null> {
  try {
    const content = await fs.readFile(filePath, "utf-8")
    return JSON.parse(content)
  } catch {
    return null
  }
}

export async function saveLedger(ledger: FailureLedger, filePath: string): Promise<void> {
  const dir = (await import("path")).dirname(filePath)
  await fs.mkdir(dir, { recursive: true })
  await fs.writeFile(filePath, JSON.stringify(ledger, null, 2))
}

// ── Reporting ─────────────────────────────────────────────────────────

export function printLedgerScorecard(ledger: FailureLedger): string {
  const lines: string[] = [
    `Failure Ledger: ${ledger.domain} (v${ledger.version})`,
    `  Entries: ${ledger.meta.totalEntries}  Buckets: ${Object.keys(ledger.buckets).length}  Rules: ${ledger.routingRules.length}`,
    "",
  ]

  const sorted = Object.values(ledger.buckets).sort((a, b) => b.entries.length - a.entries.length)
  for (const bucket of sorted.slice(0, 15)) {
    const analysis = bucket.analysisResult ? ` [${bucket.analysisResult.recommendation}]` : ""
    lines.push(`  ${bucket.key}: ${bucket.entries.length} entries${analysis}`)
  }

  if (ledger.routingRules.length > 0) {
    lines.push("")
    lines.push("  Routing Rules:")
    for (const rule of ledger.routingRules) {
      const rate = rule.activations > 0 ? `${Math.round(rule.successes / rule.activations * 100)}%` : "unused"
      lines.push(`    ${rule.id}: ${rule.activations} fires, ${rate} success`)
    }
  }

  return lines.join("\n")
}
