// The pi-core loop with Exo's shell-only tool surface - kept as an explicit,
// separately-addressable variant purely so benchmark comparisons don't lose
// their shell-only data point now that exo/harness-pi-core.ts's own default
// switched to file tools (see that file's header for why: file tools won 27/27
// vs 26/27 on gpt-5-nano and 23/27 vs 14/27 on Gemini 2.5 Flash Lite, with no
// case where shell-only won).
//
// Point existing "pi-core-bench-*" comparison agents at this module so they
// keep meaning what their name says - shell-only - regardless of what the
// unsuffixed module's default becomes next.

import { defineHarness } from "@exo/harness";

import { runPiCoreTurn } from "./harness-pi-core";

export default defineHarness({
  async runTurn(context) {
    await runPiCoreTurn(context, { fileTools: false });
  },
});
