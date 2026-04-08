# Plan-Lab

Isolated experimentation harness for testing LLM planning modes with LM Studio models.

## Architecture Diagram (Canonical)

**FigJam**: https://www.figma.com/board/JNMdIkRw3RhDRrOrGAfVsZ

This is the canonical architecture diagram. When making structural changes to the harness (adding/removing systems, changing data flow), update the diagram to match. See AGENTS.md for the update protocol.

## What This Is

A stripped-down fork of the roundtable harness from CodeRPG-Desktop, focused entirely on:
- Testing planning arms (A: evo_plan, B: worker_plan, C: multi_explore)
- Comparing meta-tools (5 intents) vs base tools (8 tools)
- A/B testing across model sizes (9B through 397B)
- Measuring whether planning improves task completion rates

## Quick Start

```bash
# Single model, single scenario
MODEL=qwen/qwen3.5-9b ENDPOINT=http://192.168.50.117:1234 \
  SCENARIOS=q01-easy bun run src/bench.ts

# All models, all scenarios, planning vs no planning
bun run bench

# Just observe planning output (no execution)
PLAN_ONLY=1 bun run bench

# New harder scenarios (feature builds requiring planning)
SCENARIOS=f01-feature-add,f02-multi-file-refactor,f03-plugin-system,f04-event-system bun run bench
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | (all) | Single model ID to test |
| `ENDPOINT` | (all) | Single endpoint URL |
| `MODELS` | (all) | Comma-separated model labels to include |
| `SCENARIOS` | (all) | Comma-separated scenario names (e.g. q01-easy) |
| `PLAN_MODES` | none,evo_plan | Planning modes to test |
| `TOOL_MODES` | meta | Tool presentation: base or meta |
| `MAX_TURNS` | 15 | Max agent turns per run |
| `PLAN_ONLY` | 0 | Set to 1 to only run planning (skip execution) |
| `TIMEOUT_MS` | 300000 | Per-inference timeout |

## Architecture

```
src/
  client.ts        LM Studio client (serialized, stall watchdog)
  tools.ts         Tool catalog + executor (read, write, edit, grep, list, glob, bash, complete)
  meta-tools.ts    Meta-tool router (investigate, modify, execute, note, finish)
  planner.ts       Planning arms A/B/C + plan injection + path validation
  runner.ts        ReAct agent loop with all systems integrated
  conveyor.ts      Surgical edit conveyor (4-station) + text-to-FC recovery
  ledger.ts        Failure ledger (error bucketing, loop/dandelion detection, routing rules)
  interceptors.ts  Interceptor registry (transform/inject/block, evolution, persistence)
  compactor.ts     Context compactor (two-tier: microcompact + full summarization)
  bench.ts         Experiment bench (entry point)

scenarios/         JSON challenge files — bugfix (Q01-Q11) and feature build (F01-F04)
results/           Experiment output (NDJSON log + per-run JSON)
```

## System Integration

All systems are wired into the runner and fire on every turn:

| System | File | What it does | Key finding |
|--------|------|-------------|-------------|
| **Planning** | planner.ts | Pre-execution plan generation (3 arms) | Saves 9B 16K tokens on hard tasks; adaptive injection critical |
| **Meta-tools** | meta-tools.ts | 5 intents instead of 8 tools | Reduces tool choice complexity for small models |
| **Surgical conveyor** | conveyor.ts | 4-station edit: locate lines + apply model's original new_text | Must use model's new_text, not re-derive in stripped context |
| **Failure ledger** | ledger.ts | Error bucketing + loop/dandelion detection | Catches fixation loops, suggests alternatives |
| **Interceptors** | interceptors.ts | Transform/block commands before execution | Evolved from failure patterns |
| **Context compactor** | compactor.ts | Two-tier compression at 60/80 messages | Preserves recent 10 + scratchpad notes |
| **Stall nudge** | runner.ts | Context-aware nudge on empty responses | Empty turns don't count; 9B recovers ~50% of wasted turns |
| **Text-to-FC recovery** | conveyor.ts | Recovers tool calls from model text | Channel markers, regex, JSON patterns |

## Key Lessons Learned

1. **Plan paths must not have backticks** — small models copy markdown formatting literally into tool calls
2. **Plan injection must be adaptive** — "follow closely" prevents recovery; "use as guidance" lets models adapt
3. **Surgical conveyor Station 4 must use model's original new_text** — re-deriving in a stripped context loses diagnostic reasoning
4. **Empty responses should not count as turns** — 9B stalls ~50% of turns; penalizing this cuts effective turn budget in half
5. **Planning helps most on tasks requiring multi-file coordination** — single-file bugfixes don't need plans

## LM Studio Endpoints

| Endpoint | Hardware |
|----------|----------|
| `http://192.168.50.117:1234` | 512GB M3 Ultra Mac Studio |
| `http://192.168.50.206:1234` | 512GB M3 Ultra Mac Studio |
