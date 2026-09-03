// Pi-style extension exposing exo's host-side web tools (web_search,
// web_fetch) as a runtime-loadable toolset.

import type { PiExtensionApi } from "../pi-compat";
import { registerWebTools } from "../web-tools";
import { toolInstancesFromRegistrar } from "./helpers";

export default function webToolsExtension(pi: PiExtensionApi): void {
  for (const tool of toolInstancesFromRegistrar(registerWebTools)) {
    pi.registerToolInstance(tool);
  }
}
