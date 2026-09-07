// pi-core, but with pi-agent-core's own read/write/edit tools instead of
// Exo's reimplementations in tools/file-tools.ts.
//
// A separate module rather than a change to harness-pi-core-files.ts on
// purpose: the whole point is to measure the two against each other, and
// mutating the existing variant would destroy the baseline the rest of this
// session's numbers were taken against - the same reason
// harness-pi-core-shell-only.ts exists.
//
// What changes, and why it is worth measuring (see tools/pi-execution-env.ts
// for the environment that makes it possible):
//   - read gains offset/limit. Exo's read has neither, which is why
//     recovering a truncated command's tail needed a spill file here while
//     real Pi just paged through its own log with read(path, offset, limit).
//   - read gains image support.
//   - edit returns a diff, a unified patch, and the first changed line
//     instead of only edits_applied.
//   - truncation and path resolution come from pi's tested code rather than
//     a second local implementation.
//
// The shell tool stays Exo's. pi ships createBashTool too, but Exo's shell is
// what every benchmark in this session ran against, and the one advantage
// pi's has - a recoverable output log - is already covered by the spill file
// added to Exo's shell earlier. Swapping it would add risk without a
// measured gain; swap it when there is a number saying it helps.

import type { AgentTool } from "@earendil-works/pi-agent-core";
import {
  createEditTool,
  createReadTool,
  createWriteTool,
} from "@earendil-works/pi-agent-core";

import {
  defineHarness,
  registerBuiltInTools,
  type HarnessToolRegistry,
  type TurnContext,
} from "@exo/harness";
import { registerConfiguredAgentTools } from "@exo/model-runtime/turn-loop";

import { registerGuardianTools } from "./tools/guardian-tools";
import { registerSandboxTools } from "./tools/sandbox-tools";
import { runPiCoreTurn } from "./harness-pi-core";
import { createSandboxExecutionEnv } from "./tools/pi-execution-env";

export default defineHarness({
  async runTurn(context) {
    await runPiCoreTurn(context, {
      registerTools: registerExoToolsWithPiFileTools,
      // pi's tools are AgentHarnessTool, which is AgentTool with an extra
      // resolved-context argument on execute. They cannot go through the
      // ToolInstance registry Exo's own tools use, so they are handed to the
      // Agent directly and the registry carries only Exo's.
      extraAgentTools: (context) => piFileTools(context),
    });
  },
});

// Everything registerExoTools would register except file-tools: the built-in
// shell, sandbox and guardian tools, plus whatever the agent has installed.
// Deliberately not calling the practical profile's registerTools, because
// that is what pulls in Exo's write/edit/read - the tools being replaced.
async function registerExoToolsWithPiFileTools(
  tools: HarnessToolRegistry,
  context: TurnContext,
): Promise<void> {
  registerBuiltInTools(tools, context, ["shell"]);
  registerSandboxTools(tools);
  registerGuardianTools(tools);
  await registerConfiguredAgentTools(tools, context);
}

function piFileTools(context: TurnContext): AgentTool[] {
  const env = createSandboxExecutionEnv(context);
  const toolContext = { env };
  return [createReadTool(), createWriteTool(), createEditTool()].map(
    (tool) =>
      ({
        ...tool,
        parameters: toStrictSchema(tool.parameters),
        // Bind the resolved context, which is the only difference between
        // AgentHarnessTool and AgentTool.
        execute: (
          toolCallId: string,
          params: never,
          signal?: AbortSignal,
          onUpdate?: never,
        ) =>
          tool.execute(
            toolCallId,
            dropNulls(params) as never,
            signal,
            onUpdate,
            toolContext,
          ),
      }) as unknown as AgentTool,
  );
}

// pi's tool schemas are written for pi-ai's own provider dispatch, which does
// not use OpenAI strict mode. Exo's model path does, and it rejects them:
// read's offset/limit are optional and therefore missing from `required`, and
// no object declares additionalProperties. Sending them unchanged returns
// "400 Provider returned error" on the very first call - measured, before
// this conversion existed.
//
// Strict mode's rule is that every property is required, so an optional one
// has to be expressed as nullable instead. dropNulls then turns the nulls the
// model sends back into absent keys, which is what pi's tools expect.
export function toStrictSchema(schema: unknown): unknown {
  if (Array.isArray(schema)) {
    return schema.map(toStrictSchema);
  }
  if (schema === null || typeof schema !== "object") {
    return schema;
  }
  const node = { ...(schema as Record<string, unknown>) };
  for (const [key, value] of Object.entries(node)) {
    node[key] = toStrictSchema(value);
  }
  if (node.type === "object" && node.properties) {
    const properties = node.properties as Record<string, unknown>;
    const names = Object.keys(properties);
    const required = new Set((node.required as string[] | undefined) ?? []);
    for (const name of names) {
      if (!required.has(name)) {
        properties[name] = nullable(properties[name]);
      }
    }
    node.required = names;
    node.additionalProperties = false;
  }
  return node;
}

function nullable(property: unknown): unknown {
  if (property === null || typeof property !== "object") {
    return property;
  }
  const node = property as Record<string, unknown>;
  if (typeof node.type === "string" && node.type !== "null") {
    return { ...node, type: [node.type, "null"] };
  }
  return node;
}

// A null offset means "not provided", not "line null": pi reads these with
// `if (params.offset)`-style checks, so leaving the key present with a null
// value changes behaviour.
export function dropNulls(params: unknown): unknown {
  if (params === null || typeof params !== "object" || Array.isArray(params)) {
    return params;
  }
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(
    params as Record<string, unknown>,
  )) {
    if (value !== null) {
      out[key] = value;
    }
  }
  return out;
}
