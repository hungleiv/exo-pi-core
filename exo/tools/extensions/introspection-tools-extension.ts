// Pi-style extension exposing exo's conversation/adapter event introspection
// tools as a runtime-loadable toolset.

import type { PiExtensionApi } from "../pi-compat";
import { registerIntrospectionTools } from "../introspection-tools";
import { toolInstancesFromRegistrar } from "./helpers";

export default function introspectionToolsExtension(pi: PiExtensionApi): void {
  for (const tool of toolInstancesFromRegistrar(registerIntrospectionTools)) {
    pi.registerToolInstance(tool);
  }
}
