// pi-core harness, plus the write/edit/read file tools.
//
// Exists purely to keep two variables separable in benchmarks:
//
//   exo/harness.ts            Exo loop      + shell only
//   exo/harness-pi-core.ts    pi-core loop  + shell only
//   this module               pi-core loop  + shell + file tools
//   pi-harness.ts (examples)  real Pi       + Pi's own tool set
//
// Comparing the middle two isolates the tool surface; comparing the first two
// isolates the loop. Benchmarking on openai/gpt-5-nano put exo/harness.ts at
// 2/6 on tier-5 tasks against 6/6 for both pi-core and real Pi, with the
// failures tracing back to shell-quoting collapses the loop then failed to
// recover from - two separate causes that a single combined harness would
// have left entangled.

import { defineHarness } from "@exo/harness";

import { runPiCoreTurn } from "./harness-pi-core";

export default defineHarness({
  async runTurn(context) {
    await runPiCoreTurn(context, { fileTools: true });
  },
});
