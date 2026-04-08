# Agents Guide — Plan-Lab

## Canonical Architecture Diagram

**FigJam URL**: https://www.figma.com/board/JNMdIkRw3RhDRrOrGAfVsZ
**File key**: `JNMdIkRw3RhDRrOrGAfVsZ`

This diagram is the single source of truth for how the harness components connect. It lives in Figma so both the developer and Claude can read/write to it.

### Two-Way Bridge Protocol

**Developer -> Claude**: Add comments or sticky notes to the FigJam board. Claude reads them via the Figma MCP (`get_figjam` tool) and responds or makes changes.

**Claude -> Developer**: When making structural code changes, Claude updates the diagram via the Figma MCP (`use_figma` tool) to keep it in sync.

### When to Update the Diagram

Update the FigJam diagram when ANY of the following happen:

1. **New system added** — A new `.ts` file in `src/` that represents a harness component (e.g., adding an orchestrator, knowledge graph)
2. **System removed** — A component is deleted or disabled
3. **Data flow changed** — The way components communicate changes (e.g., conveyor now receives model's new_text instead of re-deriving)
4. **New integration point** — A system starts feeding data to another system that it didn't before (e.g., failure ledger feeding interceptor evolution)
5. **Configuration surface changed** — New toggles, thresholds, or modes that affect harness behavior

### How to Update

```
1. Read the current diagram state:
   Use get_figjam with fileKey "JNMdIkRw3RhDRrOrGAfVsZ" and nodeId "0:1"

2. Identify what changed in the code

3. Update the diagram via use_figma:
   - Add new sticky notes for new components
   - Update text on existing stickies if behavior changed
   - Add connector stickies to show new data flow

4. Note the update in the commit message
```

### Diagram Structure

The diagram uses FigJam sections and sticky notes:

| Section | Color | Contains |
|---------|-------|----------|
| **Planning Phase** | Green | Arms A/B/C, Plan Injection |
| **Agent Runner** | Blue | LM Studio Client, Turn Handler, Stall Nudge, Text Recovery |
| **Meta-Tool Router** | Blue | 5 intent tools (investigate, modify, execute, note, finish) |
| **Pre-Execution Guards** | Red | Loop Detection, Dandelion Detection, Interceptors |
| **Execution Layer** | Gold | Surgical Conveyor, Direct Edit, Tool Executor |
| **Feedback Systems** | (standalone) | Failure Ledger, Context Compactor |

## Agent Behavior Rules

### Experiment Protocol
- Always run grid tests after structural changes
- Compare before/after with the same model set (minimum: 9B, 27B, 397B)
- Record results in `results/` — never delete previous results
- Note regressions immediately

### Code Change Rules
- All inference runs must go through the agent runner (`runAgent` in runner.ts)
- Never bypass the meta-tool router for direct tool execution in experiments
- New systems should be toggleable via RunConfig (default: enabled)
- Test with 9B first — if it breaks the smallest model, it's wrong
