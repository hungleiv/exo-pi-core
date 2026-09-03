// Pi-style extension exposing exo's sandbox task scheduler as a
// runtime-loadable toolset.

import type { PiExtensionApi } from "../pi-compat";
import { registerSchedulerTools } from "../scheduler-tools";
import { toolInstancesFromRegistrar } from "./helpers";

export default function schedulerToolsExtension(pi: PiExtensionApi): void {
  for (const tool of toolInstancesFromRegistrar(registerSchedulerTools)) {
    pi.registerToolInstance(tool);
  }
}
