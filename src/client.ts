/**
 * Minimal LM Studio Client — stripped from roundtable for plan-lab.
 *
 * Only what's needed: chat completions with tool support, health check,
 * per-model serialization, and stall watchdog.
 */

// ── Types ─────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool"
  content?: string | null
  tool_calls?: ToolCallResponse[]
  tool_call_id?: string
}

export interface ToolCallResponse {
  id: string
  type: "function"
  function: { name: string; arguments: string }
}

export interface OpenAIToolDef {
  type: "function"
  function: {
    name: string
    description: string
    parameters: {
      type: "object"
      properties: Record<string, { type: string; description?: string; enum?: string[] }>
      required?: string[]
    }
  }
}

export interface SendResult {
  content: string
  text: string // alias for content
  tokens: { prompt: number; completion: number; total: number }
  usage?: { total_tokens: number }
  latencyMs: number
  model: string
  toolCalls?: ToolCallResponse[]
}

export interface SendWithToolsFn {
  (agentName: string, messages: ChatMessage[], tools: OpenAIToolDef[], modelId?: string): Promise<SendResult>
}

export interface ClientConfig {
  baseUrl: string
  model: string
  temperature?: number
  maxTokens?: number
  timeoutMs?: number
}

export interface HealthResult {
  ok: boolean
  models: string[]
  error?: string
}

// ── Per-model serialization ───────────────────────────────────────────

const modelQueues = new Map<string, Promise<void>>()

async function acquireSlot(key: string): Promise<() => void> {
  const prev = modelQueues.get(key) ?? Promise.resolve()
  let release!: () => void
  const next = new Promise<void>(resolve => { release = resolve })
  modelQueues.set(key, next)
  await prev
  return release
}

// ── Stall watchdog ────────────────────────────────────────────────────

const STALL_MS = 20_000

async function fetchWithStallDetection(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<{ data: unknown; latencyMs: number }> {
  const controller = new AbortController()
  const start = Date.now()

  const hardTimeout = setTimeout(() => controller.abort("timeout"), timeoutMs)

  try {
    const response = await fetch(url, { ...init, signal: controller.signal })
    if (!response.ok) {
      const text = await response.text().catch(() => "")
      throw new Error(`HTTP ${response.status}: ${text.slice(0, 200)}`)
    }

    // Stream with stall watchdog
    if (response.body) {
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let text = ""
      let sawChunk = false
      let stallTimer: ReturnType<typeof setTimeout> | undefined

      const armStall = () => {
        if (!sawChunk) return
        if (stallTimer) clearTimeout(stallTimer)
        stallTimer = setTimeout(() => {
          controller.abort("stalled")
          void reader.cancel("stalled")
        }, STALL_MS)
      }

      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          text += decoder.decode(value, { stream: true })
          sawChunk = true
          armStall()
        }
      } finally {
        if (stallTimer) clearTimeout(stallTimer)
      }

      text += decoder.decode()
      const data = JSON.parse(text)
      return { data, latencyMs: Date.now() - start }
    }

    const data = await response.json()
    return { data, latencyMs: Date.now() - start }
  } finally {
    clearTimeout(hardTimeout)
  }
}

// ── Send with tools ───────────────────────────────────────────────────

export function createSendWithTools(config: ClientConfig): SendWithToolsFn {
  const { baseUrl, model, temperature = 0, maxTokens, timeoutMs = 120_000 } = config

  return async (_agentName, messages, tools, modelId) => {
    const resolvedModel = modelId || model
    const slotKey = `${baseUrl}|${resolvedModel}`

    const reqBody: Record<string, unknown> = {
      model: resolvedModel,
      messages,
      temperature,
    }
    if (maxTokens !== undefined) reqBody.max_tokens = maxTokens

    if (tools.length > 0) {
      reqBody.tools = tools
      reqBody.tool_choice = "auto"
      reqBody.parallel_tool_calls = false
    }

    const release = await acquireSlot(slotKey)
    try {
      const { data: rawData, latencyMs } = await fetchWithStallDetection(
        `${baseUrl}/v1/chat/completions`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(reqBody),
        },
        timeoutMs,
      )

      const data = rawData as {
        choices?: Array<{
          message?: {
            content?: string | null
            tool_calls?: ToolCallResponse[]
            reasoning?: string
          }
        }>
        usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number }
        model?: string
      }

      const choice = data.choices?.[0]?.message
      let content = choice?.content ?? ""

      // Strip reasoning tags if present (reasoning models like gpt-oss)
      content = content
        .replace(/<thinking>[\s\S]*?<\/thinking>/gi, "")
        .replace(/<\|begin_of_thought\|>[\s\S]*?<\|end_of_thought\|>/gi, "")
        .trim()

      const tokens = {
        prompt: data.usage?.prompt_tokens ?? 0,
        completion: data.usage?.completion_tokens ?? 0,
        total: data.usage?.total_tokens ?? 0,
      }

      return {
        content,
        text: content,
        tokens,
        usage: { total_tokens: tokens.total },
        latencyMs,
        model: data.model || resolvedModel,
        toolCalls: choice?.tool_calls,
      }
    } finally {
      release()
    }
  }
}

// ── Health check ──────────────────────────────────────────────────────

export async function healthCheck(baseUrl: string): Promise<HealthResult> {
  try {
    const response = await fetch(`${baseUrl}/v1/models`, {
      signal: AbortSignal.timeout(10_000),
    })
    if (!response.ok) {
      return { ok: false, models: [], error: `HTTP ${response.status}` }
    }
    const data = (await response.json()) as { data?: Array<{ id: string }> }
    const models = data.data?.map(m => m.id) ?? []
    return { ok: true, models }
  } catch (e) {
    return { ok: false, models: [], error: e instanceof Error ? e.message : String(e) }
  }
}
