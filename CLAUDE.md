# Plan-Lab

Isolated experimentation harness for testing LLM planning modes with LM Studio models.

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
  client.ts      Minimal LM Studio client (chat completions + health check)
  tools.ts       Tool catalog + executor (read, write, edit, grep, list, glob, bash, complete)
  meta-tools.ts  Meta-tool layer (investigate, modify, execute, note, finish)
  planner.ts     Planning arms A/B/C
  runner.ts      Minimal ReAct agent loop
  bench.ts       Experiment bench (entry point)

scenarios/       JSON challenge files with buggy code + tests
results/         Experiment output (NDJSON log + per-run JSON)
```

## Key Differences from Roundtable

What's **included**:
- LM Studio client (serialized, stall watchdog)
- Base tool executor (8 tools)
- Meta-tool router (5 intents)
- Planner (3 arms)
- Agent runner (ReAct loop with text-to-FC recovery)
- Bench runner with comparison tables

What's **stripped out** (add back one-at-a-time to measure impact):
- Interceptors
- Surgical edit conveyor
- Failure ledger
- Context compression
- Exit gates / knowledge graphs
- Orchestrator (Evo supervisor)
- Desktop tools
- Workflows
- Real-time refinement

## LM Studio Endpoints

| Endpoint | Hardware |
|----------|----------|
| `http://192.168.50.117:1234` | 512GB M3 Ultra Mac Studio |
| `http://192.168.50.206:1234` | 512GB M3 Ultra Mac Studio |
