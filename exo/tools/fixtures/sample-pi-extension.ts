import type { PiExtensionApi } from "../pi-compat";

export default function sampleExtension(api: PiExtensionApi): void {
  api.registerTool({
    name: "shout",
    description: "Uppercase some text.",
    parameters: {
      type: "object",
      properties: {
        text: { type: "string", description: "Text to shout." },
        loud: { type: "boolean", description: "Add exclamation marks." },
      },
      required: ["text"],
    },
    execute: (args, ctx) => ({
      ok: true,
      text: `${String(args.text).toUpperCase()}${args.loud === true ? "!!!" : ""}`,
      route: ctx.route,
    }),
  });

  api.registerCommand({
    name: "greet",
    description: "Greet someone.",
    execute: (args) => ({ ok: true, text: `hello ${args}` }),
  });

  api.on("tool_call", () => {});
  api.on("session_start", () => {});
}
