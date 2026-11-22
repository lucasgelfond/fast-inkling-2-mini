# Lean pi prompt for local models

Prompt tokens are slow to read on a Mac (about 200 tokens/s here). Pi's default system prompt and tool schemas are about 10K tokens, so every new session waits close to a minute before the first token. This pi extension gives local models a short system prompt and four tools, about 1K tokens in total. Cloud models keep the full prompt.

Commands:

- `/lean`: show the current mode
- `/lean on`: always lean
- `/lean off`: never lean
- `/lean auto`: lean only for models served from `localhost` or `127.0.0.1` (default)

To install, save the code below as `~/.pi/agent/extensions/lean.ts`.

```ts
// Lean mode for local models: prompt tokens are slow to read on this Mac
// (about 200 tokens/s), so local models get four tools and a short system
// prompt (about 1K tokens instead of about 10K). Cloud models keep everything.
//
//   /lean        show the current mode
//   /lean on     always lean
//   /lean off    never lean
//   /lean auto   lean only for models served from this machine (default)
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

type Mode = "auto" | "on" | "off";

const LEAN_TOOLS = ["read", "bash", "edit", "write"];

const leanPrompt = (cwd: string) => `You are a coding agent running in pi. Tools: read, bash, edit, write.
- Explore with bash: rg -n, ls, git diff --stat. Cap long output with head or tail.
- Read only the range you need (read with offset and limit). Do not re-read a file that has not changed.
- edit: each oldText must match the file exactly and only once. Keep it short. Put all edits to one file in one call.
- Use write only for new files or full rewrites.
- After a change, run the smallest check that proves it. Report only output you saw.
- Reading input is slow on this machine and earlier context is cached, so read less.
- Think briefly, then act. Be concise. Show file paths.

Working directory: ${cwd}`;

// Short tool descriptions. Parameter descriptions are dropped except where the
// name alone does not say enough.
const SHORT_TOOLS: Record<string, { description: string; params?: Record<string, string> }> = {
  read: {
    description: "Read a text file or image. Text stops at 2000 lines; continue with offset.",
    params: { offset: "First line, 1-indexed", limit: "Number of lines" },
  },
  bash: {
    description: "Run a shell command in the working directory. Returns stdout and stderr (last 2000 lines).",
    params: { timeout: "Seconds" },
  },
  edit: {
    description: "Replace exact text in one file. Each oldText must occur once in the original file. Edits must not overlap.",
  },
  write: { description: "Create or overwrite a file. Creates parent folders." },
};

type JsonObject = Record<string, unknown>;
const isObject = (value: unknown): value is JsonObject =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const withoutDescriptions = (schema: unknown, keep: Record<string, string> = {}): unknown => {
  if (Array.isArray(schema)) return schema.map((item) => withoutDescriptions(item, keep));
  if (!isObject(schema)) return schema;
  const result: JsonObject = {};
  for (const [key, value] of Object.entries(schema)) {
    if (key === "description") continue;
    if (key === "properties" && isObject(value)) {
      result.properties = Object.fromEntries(
        Object.entries(value).map(([name, property]) => {
          const short = withoutDescriptions(property, keep);
          return [name, keep[name] && isObject(short) ? { ...short, description: keep[name] } : short];
        }),
      );
    } else {
      result[key] = withoutDescriptions(value, keep);
    }
  }
  return result;
};

const shortenTool = (tool: unknown): unknown => {
  if (!isObject(tool) || !isObject(tool.function)) return tool;
  const short = SHORT_TOOLS[String(tool.function.name)];
  if (!short) return tool;
  return {
    ...tool,
    function: {
      ...tool.function,
      description: short.description,
      parameters: withoutDescriptions(tool.function.parameters, short.params),
    },
  };
};

const isLocal = (ctx: ExtensionContext) =>
  /^https?:\/\/(127\.0\.0\.1|localhost)[:/]/.test(ctx.model?.baseUrl ?? "");

export default function (pi: ExtensionAPI) {
  let mode: Mode = "auto";
  let fullTools: string[] | undefined;

  const isLean = (ctx: ExtensionContext) => mode === "on" || (mode === "auto" && isLocal(ctx));

  pi.registerCommand("lean", {
    description: "Lean prompt and tools: /lean on | off | auto",
    handler: async (args, ctx) => {
      const requested = args.trim();
      if (requested === "on" || requested === "off" || requested === "auto") mode = requested;
      ctx.ui.notify(`lean: ${mode} (${isLean(ctx) ? "lean" : "full"} for ${ctx.model?.id ?? "no model"})`, "info");
    },
  });

  pi.on("before_agent_start", (event, ctx) => {
    if (!isLean(ctx)) {
      if (fullTools) {
        pi.setActiveTools(fullTools);
        fullTools = undefined;
      }
      return;
    }
    fullTools ??= pi.getActiveTools();
    pi.setActiveTools(LEAN_TOOLS);
    return { systemPrompt: leanPrompt(event.systemPromptOptions.cwd) };
  });

  pi.on("before_provider_request", (event, ctx) => {
    if (!isLean(ctx) || !isObject(event.payload) || !Array.isArray(event.payload.tools)) return;
    return { ...event.payload, tools: event.payload.tools.map(shortenTool) };
  });
}
```
