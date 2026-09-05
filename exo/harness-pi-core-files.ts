// pi-core harness, plus the write/edit/read file tools.
//
// Identical to exo/harness-pi-core.ts's own default as of the file-tools
// benchmarking documented there - this module is kept as an explicit,
// separately-addressable name for existing "pi-core-files-*" agents rather
// than being removed as redundant.
//
// The comparison matrix this repo's benchmarking actually used:
//
//   exo/harness.ts                    Exo loop      + shell only
//   exo/harness-pi-core-shell-only.ts pi-core loop  + shell only
//   exo/harness-pi-core.ts / this      pi-core loop  + shell + file tools
//   pi-harness.ts (examples)          real Pi       + Pi's own tool set
//
// Comparing rows 2 and 3 isolates the tool surface; comparing rows 1 and 2
// isolates the loop. Benchmarking on openai/gpt-5-nano put exo/harness.ts at
// 2/6 on tier-5 tasks against 6/6 for both pi-core and real Pi, with the
// failures tracing back to shell-quoting collapses the loop then failed to
// recover from - two separate causes that a single combined harness would
// have left entangled.

import { defineHarness } from "@exo/harness";

import { runPiCoreTurn } from "./harness-pi-core";

export default defineHarness({
  async runTurn(context) {
    await runPiCoreTurn(context);
  },
});
