/**
 * Interceptor Registry — transform/inject/block commands before execution.
 *
 * Interceptors match error bucket patterns and apply deterministic corrections.
 * They can be hand-written (core) or ML-generated (evolved).
 *
 * Stripped from roundtable interceptor-registry.ts for plan-lab.
 */

import fs from "fs/promises"
import type { OpenAIToolDef } from "./client"
import type { RoutingRule } from "./ledger"

// ── Types ─────────────────────────────────────────────────────────────

export interface InterceptorAction {
  type: "transform" | "inject" | "block"
  command?: string
  explanation: string
}

export type InterceptorFn = (
  tool: string,
  args: Record<string, string>,
  context: { workDir: string; activeDir?: string; repo?: string },
) => InterceptorAction | null

export interface Interceptor {
  id: string
  description: string
  triggerPatterns: string[]
  activationThreshold: number
  active: boolean
  firings: number
  successes: number
  origin: "core" | "evolved"
  sourceRuleId?: string
  evolvedAt?: string
  failureStreak: number
}

export interface InterceptorSpec {
  id: string
  description: string
  matchTool: string
  matchPattern: string
  actionType: "transform" | "inject" | "block"
  commandTemplate?: string
  rewrites?: Array<{ find: string; replace: string }>
  rationale: string
  sourceBucket: string
  sourceRuleId?: string
}

export interface InterceptorMiss {
  id: string
  transformedCommand: string
  error: string
  timestamp: string
}

interface RegisteredInterceptor {
  meta: Interceptor
  spec?: InterceptorSpec
  fn: InterceptorFn
}

// ── Registry (in-memory) ──────────────────────────────────────────────

const registry: RegisteredInterceptor[] = []
const missLog: InterceptorMiss[] = []

// ── Core interceptor registration ─────────────────────────────────────

export function registerCoreInterceptor(
  id: string,
  description: string,
  triggerPatterns: string[],
  fn: InterceptorFn,
): void {
  registry.push({
    meta: {
      id,
      description,
      triggerPatterns,
      activationThreshold: 0,
      active: true,
      firings: 0,
      successes: 0,
      origin: "core",
      failureStreak: 0,
    },
    fn,
  })
}

// ── Runtime execution ─────────────────────────────────────────────────

export function checkInterceptors(
  tool: string,
  args: Record<string, string>,
  context: { workDir: string; activeDir?: string; repo?: string },
): { interceptor: Interceptor; action: InterceptorAction } | null {
  for (const reg of registry) {
    if (!reg.meta.active) continue
    const action = reg.fn(tool, args, context)
    if (action) {
      reg.meta.firings++
      return { interceptor: reg.meta, action }
    }
  }
  return null
}

export function recordInterceptorOutcome(id: string, prevented: boolean): void {
  const reg = registry.find(r => r.meta.id === id)
  if (!reg) return
  if (prevented) {
    reg.meta.successes++
    reg.meta.failureStreak = 0
  } else {
    reg.meta.failureStreak++
  }
}

export function recordInterceptorMiss(id: string, transformedCommand: string, error: string): void {
  missLog.push({ id, transformedCommand, error, timestamp: new Date().toISOString() })
}

export function getMissLog(id?: string): InterceptorMiss[] {
  if (id) return missLog.filter(m => m.id === id)
  return [...missLog]
}

export function clearMissLog(): void {
  missLog.length = 0
}

// ── Listing and scoring ───────────────────────────────────────────────

export function listInterceptors(): Interceptor[] {
  return registry.map(r => ({ ...r.meta }))
}

export function scoreInterceptor(interceptor: Interceptor): {
  verdict: "effective" | "neutral" | "failing" | "untested"
  rate: number
  signal: string
} {
  if (interceptor.firings === 0) {
    return { verdict: "untested", rate: 0, signal: "No firings yet" }
  }

  const rate = interceptor.successes / interceptor.firings

  if (rate >= 0.5) {
    return { verdict: "effective", rate, signal: `${Math.round(rate * 100)}% success (${interceptor.successes}/${interceptor.firings})` }
  }

  if (interceptor.failureStreak >= 3) {
    return { verdict: "failing", rate, signal: `${interceptor.failureStreak} consecutive failures` }
  }

  return { verdict: "neutral", rate, signal: `${Math.round(rate * 100)}% success — needs more data` }
}

export function interceptorSummary(): string {
  const lines: string[] = ["Interceptor Registry:"]
  for (const reg of registry) {
    const { verdict, signal } = scoreInterceptor(reg.meta)
    const status = reg.meta.active ? "ON" : "OFF"
    lines.push(`  [${status}] ${reg.meta.id}: ${reg.meta.firings} fires, ${verdict} — ${signal}`)
  }
  return lines.join("\n")
}

// ── Evolved interceptor installation ──────────────────────────────────

export function installEvolvedInterceptor(spec: InterceptorSpec): boolean {
  // Don't install duplicates
  if (registry.some(r => r.meta.id === spec.id)) return false

  // Validate pattern
  try { new RegExp(spec.matchPattern) } catch { return false }

  const fn: InterceptorFn = (tool, args, context) => {
    // Tool match
    if (spec.matchTool !== "*" && tool !== spec.matchTool) return null

    // Command match
    const command = args.command || args.target || JSON.stringify(args)
    try {
      if (!new RegExp(spec.matchPattern, "i").test(command)) return null
    } catch { return null }

    // Apply rewrites
    let cmd = command
    if (spec.rewrites) {
      for (const rewrite of spec.rewrites) {
        try {
          cmd = cmd.replace(new RegExp(rewrite.find, "g"), rewrite.replace)
        } catch { /* skip bad regex */ }
      }
    }

    // Apply template
    if (spec.commandTemplate) {
      cmd = spec.commandTemplate
        .replace(/\$CMD/g, cmd)
        .replace(/\$DIR/g, context.activeDir || context.workDir)
        .replace(/\$REPO/g, context.repo || "")
    }

    return {
      type: spec.actionType,
      command: cmd,
      explanation: spec.rationale,
    }
  }

  registry.push({
    meta: {
      id: spec.id,
      description: spec.description,
      triggerPatterns: [spec.sourceBucket],
      activationThreshold: 3,
      active: true,
      firings: 0,
      successes: 0,
      origin: "evolved",
      sourceRuleId: spec.sourceRuleId,
      evolvedAt: new Date().toISOString(),
      failureStreak: 0,
    },
    spec,
    fn,
  })

  return true
}

export function refineInterceptor(
  id: string,
  newCommandTemplate: string,
  reason: string,
  newRewrites?: Array<{ find: string; replace: string }>,
): boolean {
  const reg = registry.find(r => r.meta.id === id)
  if (!reg || !reg.spec) return false

  reg.spec.commandTemplate = newCommandTemplate
  if (newRewrites) reg.spec.rewrites = newRewrites
  reg.meta.failureStreak = 0 // reset after refinement

  return true
}

// ── Evolution (promotion / demotion) ──────────────────────────────────

const PROMOTION_MIN_ACTIVATIONS = 3
const PROMOTION_MIN_SUCCESS_RATE = 0.5
const DEMOTION_MIN_FIRINGS = 3
const FAILURE_STREAK_LIMIT = 3

export function promoteRule(rule: RoutingRule): InterceptorSpec | null {
  if (rule.activations < PROMOTION_MIN_ACTIVATIONS) return null
  if (rule.successes / rule.activations < PROMOTION_MIN_SUCCESS_RATE) return null

  return {
    id: `evolved-${rule.id}`,
    description: `Promoted from routing rule: ${rule.match.errorPattern}`,
    matchTool: rule.match.tool === "*" ? "bash" : rule.match.tool,
    matchPattern: rule.match.errorPattern,
    actionType: "transform",
    commandTemplate: rule.action.argsTransform,
    rationale: `Routing rule ${rule.id} hit ${rule.activations} activations with ${Math.round(rule.successes / rule.activations * 100)}% success`,
    sourceBucket: rule.match.errorPattern,
    sourceRuleId: rule.id,
  }
}

export function evolveInterceptors(routingRules?: RoutingRule[]): {
  promoted: InterceptorSpec[]
  demoted: string[]
  events: string[]
} {
  const events: string[] = []
  const promoted: InterceptorSpec[] = []
  const demoted: string[] = []

  // Demote failing evolved interceptors
  for (const reg of registry) {
    if (reg.meta.origin !== "evolved") continue
    if (reg.meta.firings < DEMOTION_MIN_FIRINGS) continue

    if (reg.meta.failureStreak >= FAILURE_STREAK_LIMIT) {
      reg.meta.active = false
      demoted.push(reg.meta.id)
      events.push(`Demoted ${reg.meta.id}: ${FAILURE_STREAK_LIMIT} consecutive failures`)
    }
  }

  // Promote eligible routing rules
  if (routingRules) {
    for (const rule of routingRules) {
      if (registry.some(r => r.meta.sourceRuleId === rule.id)) continue // already promoted
      const spec = promoteRule(rule)
      if (spec) {
        if (installEvolvedInterceptor(spec)) {
          promoted.push(spec)
          events.push(`Promoted ${rule.id} -> ${spec.id}`)
        }
      }
    }
  }

  return { promoted, demoted, events }
}

// ── Persistence ───────────────────────────────────────────────────────

interface EvolvedState {
  version: number
  interceptors: Array<InterceptorSpec & {
    active: boolean
    firings: number
    successes: number
    evolvedAt: string
    failureStreak: number
  }>
  meta: {
    totalPromotions: number
    totalDemotions: number
    totalRefinements: number
    lastEvolved: string
  }
}

export async function saveEvolvedInterceptors(filePath: string): Promise<void> {
  const evolved = registry.filter(r => r.meta.origin === "evolved" && r.spec)
  const state: EvolvedState = {
    version: 1,
    interceptors: evolved.map(r => ({
      ...r.spec!,
      active: r.meta.active,
      firings: r.meta.firings,
      successes: r.meta.successes,
      evolvedAt: r.meta.evolvedAt || new Date().toISOString(),
      failureStreak: r.meta.failureStreak,
    })),
    meta: {
      totalPromotions: evolved.length,
      totalDemotions: registry.filter(r => r.meta.origin === "evolved" && !r.meta.active).length,
      totalRefinements: 0,
      lastEvolved: new Date().toISOString(),
    },
  }

  const dir = (await import("path")).dirname(filePath)
  await (await import("fs/promises")).mkdir(dir, { recursive: true })
  await (await import("fs/promises")).writeFile(filePath, JSON.stringify(state, null, 2))
}

export async function loadEvolvedInterceptors(filePath: string): Promise<number> {
  try {
    const content = await (await import("fs/promises")).readFile(filePath, "utf-8")
    const state: EvolvedState = JSON.parse(content)
    let loaded = 0

    for (const entry of state.interceptors) {
      if (registry.some(r => r.meta.id === entry.id)) continue
      const spec: InterceptorSpec = {
        id: entry.id,
        description: entry.description,
        matchTool: entry.matchTool,
        matchPattern: entry.matchPattern,
        actionType: entry.actionType,
        commandTemplate: entry.commandTemplate,
        rewrites: entry.rewrites,
        rationale: entry.rationale,
        sourceBucket: entry.sourceBucket,
        sourceRuleId: entry.sourceRuleId,
      }
      if (installEvolvedInterceptor(spec)) {
        // Restore stats
        const reg = registry.find(r => r.meta.id === entry.id)
        if (reg) {
          reg.meta.active = entry.active
          reg.meta.firings = entry.firings
          reg.meta.successes = entry.successes
          reg.meta.failureStreak = entry.failureStreak
        }
        loaded++
      }
    }

    return loaded
  } catch {
    return 0
  }
}

// ── ML tool definitions ───────────────────────────────────────────────

export const PROPOSE_INTERCEPTOR_TOOL: OpenAIToolDef = {
  type: "function",
  function: {
    name: "propose_interceptor",
    description: "Propose a new command interceptor based on observed failure patterns",
    parameters: {
      type: "object",
      properties: {
        id: { type: "string", description: "Unique interceptor ID" },
        description: { type: "string", description: "What this interceptor does" },
        matchTool: { type: "string", description: "Tool to match (e.g. bash)" },
        matchPattern: { type: "string", description: "Regex to match command" },
        actionType: { type: "string", description: "Action type", enum: ["transform", "inject", "block"] },
        commandTemplate: { type: "string", description: "Template with $CMD, $DIR, $REPO variables" },
        rewrites: { type: "string", description: "JSON array of {find, replace} rewrites" },
        rationale: { type: "string", description: "Why this interceptor is needed" },
      },
      required: ["id", "description", "matchTool", "matchPattern", "actionType", "commandTemplate", "rationale"],
    },
  },
}

export const REFINE_INTERCEPTOR_TOOL: OpenAIToolDef = {
  type: "function",
  function: {
    name: "refine_interceptor",
    description: "Refine a failing interceptor with a better command template",
    parameters: {
      type: "object",
      properties: {
        id: { type: "string", description: "Interceptor ID to refine" },
        newCommandTemplate: { type: "string", description: "New command template" },
        newRewrites: { type: "string", description: "JSON array of {find, replace} (optional)" },
        reason: { type: "string", description: "Why the old template failed" },
      },
      required: ["id", "newCommandTemplate", "reason"],
    },
  },
}

// ── Prompt builders ───────────────────────────────────────────────────

export function buildInterceptorProposalPrompt(
  bucketKey: string,
  entries: Array<{ tool: string; args: Record<string, string>; error: string; intent: string }>,
  existingInterceptors: Interceptor[],
): string {
  const examples = entries.slice(0, 5).map((e, i) =>
    `${i + 1}. Tool: ${e.tool}, Args: ${JSON.stringify(e.args).slice(0, 150)}\n   Error: ${e.error.slice(0, 100)}\n   Intent: ${e.intent.slice(0, 100)}`
  ).join("\n\n")

  const existing = existingInterceptors
    .filter(i => i.active)
    .map(i => `  - ${i.id}: ${i.description}`)
    .join("\n")

  return [
    `Error bucket "${bucketKey}" has ${entries.length} failures. Propose an interceptor.`,
    "",
    "Failures:",
    examples,
    "",
    existing ? `Existing interceptors:\n${existing}` : "No existing interceptors.",
    "",
    "Rules:",
    "- matchPattern must be GENERIC (command shape, not specific paths)",
    "- commandTemplate must use $CMD, $DIR, $REPO variables",
    "- Think about the ROOT CAUSE, not just the symptom",
    "",
    "Call propose_interceptor with your design.",
  ].join("\n")
}

export function buildRefinementPrompt(
  interceptor: Interceptor,
  recentFailures: Array<{ command: string; error: string }>,
): string {
  const failures = recentFailures.slice(0, 5).map((f, i) =>
    `${i + 1}. Command: ${f.command.slice(0, 150)}\n   Error: ${f.error.slice(0, 100)}`
  ).join("\n\n")

  return [
    `Interceptor "${interceptor.id}" is failing. ${interceptor.firings} firings, ${interceptor.successes} successes.`,
    "",
    `Description: ${interceptor.description}`,
    "",
    "Recent failures after interception:",
    failures,
    "",
    "Call refine_interceptor with a better commandTemplate that addresses these failures.",
  ].join("\n")
}
