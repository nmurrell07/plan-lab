#!/bin/bash
# Run remaining grid test combos sequentially
# Results append to a summary log
set -e

cd /Users/virtualmachine/plan-lab
LOG="/Users/virtualmachine/plan-lab/results/grid-batch-$(date +%s).log"

echo "=== Grid Batch Run Started: $(date) ===" | tee "$LOG"

# Combo 2: 35B-orch/35B-worker CodeRPG
echo "" | tee -a "$LOG"
echo ">>> [$(date +%H:%M:%S)] Starting: 35B-orch/35B-worker CodeRPG" | tee -a "$LOG"
COMBO="35B-orch/35B-worker" WORKER_TURNS=50 MAX_TURNS=30 bun run src/orch-grid-coderpg.ts 2>&1 | tee -a "$LOG" || echo "FAILED: 35B-orch/35B-worker CodeRPG" | tee -a "$LOG"

# Combo 3: 35B-orch/9B-worker Skyrim
echo "" | tee -a "$LOG"
echo ">>> [$(date +%H:%M:%S)] Starting: 35B-orch/9B-worker Skyrim" | tee -a "$LOG"
COMBO="35B-orch/9B-worker" WORKER_TURNS=50 MAX_TURNS=30 bun run src/orch-grid-skyrim.ts 2>&1 | tee -a "$LOG" || echo "FAILED: 35B-orch/9B-worker Skyrim" | tee -a "$LOG"

# Combo 4: 35B-orch/27B-worker Skyrim
echo "" | tee -a "$LOG"
echo ">>> [$(date +%H:%M:%S)] Starting: 35B-orch/27B-worker Skyrim" | tee -a "$LOG"
COMBO="35B-orch/27B-worker" WORKER_TURNS=50 MAX_TURNS=30 bun run src/orch-grid-skyrim.ts 2>&1 | tee -a "$LOG" || echo "FAILED: 35B-orch/27B-worker Skyrim" | tee -a "$LOG"

# Combo 5: 35B-orch/35B-worker Skyrim
echo "" | tee -a "$LOG"
echo ">>> [$(date +%H:%M:%S)] Starting: 35B-orch/35B-worker Skyrim" | tee -a "$LOG"
COMBO="35B-orch/35B-worker" WORKER_TURNS=50 MAX_TURNS=30 bun run src/orch-grid-skyrim.ts 2>&1 | tee -a "$LOG" || echo "FAILED: 35B-orch/35B-worker Skyrim" | tee -a "$LOG"

# Combo 6: 27B-orch/35B-worker CodeRPG
echo "" | tee -a "$LOG"
echo ">>> [$(date +%H:%M:%S)] Starting: 27B-orch/35B-worker CodeRPG" | tee -a "$LOG"
COMBO="27B-orch/35B-worker" WORKER_TURNS=50 MAX_TURNS=30 bun run src/orch-grid-coderpg.ts 2>&1 | tee -a "$LOG" || echo "FAILED: 27B-orch/35B-worker CodeRPG" | tee -a "$LOG"

# Combo 7: 27B-orch/35B-worker Skyrim
echo "" | tee -a "$LOG"
echo ">>> [$(date +%H:%M:%S)] Starting: 27B-orch/35B-worker Skyrim" | tee -a "$LOG"
COMBO="27B-orch/35B-worker" WORKER_TURNS=50 MAX_TURNS=30 bun run src/orch-grid-skyrim.ts 2>&1 | tee -a "$LOG" || echo "FAILED: 27B-orch/35B-worker Skyrim" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Grid Batch Run Complete: $(date) ===" | tee -a "$LOG"
echo "Log saved to: $LOG"
