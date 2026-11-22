// Inkling 2 Mini 9.6GB (local), served by `inkling-16gb serve` on 127.0.0.1:8090.
//
// - Selecting the model (or starting pi on it) starts the server if it isn't running, and every request waits until
//   it is ready; switching to another model or quitting pi stops it again (if this pi started it), giving its ~9.5 GB
//   back to the Mac.
// - Selecting it switches thinking to "low"; switching away restores the level you had before.
//   (settings.json `modelThinkingLevels` covers startup; this covers /model and Ctrl+P mid-session.)
// - The lean prompt and tools come from lean.ts, which applies to every model served from localhost.
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { spawn, type ChildProcess } from "node:child_process";
import { mkdirSync, openSync } from "node:fs";
import { connect } from "node:net";
import { homedir } from "node:os";
import { join } from "node:path";

const PROVIDER = "inkling-16gb";
const DEFAULT_THINKING = "low";
const PORT = 8090;
const BIN = process.env.INKLING16_BIN ?? join(homedir(), ".local", "bin", "inkling-16gb");
const LOG = join(homedir(), ".cache", "inkling-16gb", "server.log");
const READY_TIMEOUT_MS = 240_000; // first-ever launch compiles Metal kernels (~50 s); later ~15 s

const isOurs = (m?: { provider?: string }) => m?.provider === PROVIDER;

const listening = () =>
  new Promise<boolean>((resolve) => {
    const s = connect(PORT, "127.0.0.1");
    s.once("connect", () => (s.destroy(), resolve(true)));
    s.once("error", () => resolve(false));
  });

const ready = async () => {
  try {
    const r = await fetch(`http://127.0.0.1:${PORT}/health`, { signal: AbortSignal.timeout(2000) });
    return r.ok;
  } catch {
    return false;
  }
};

export default function (pi: ExtensionAPI) {
  let before: string | undefined;
  let child: ChildProcess | undefined;
  let starting: Promise<boolean> | undefined;

  const ensure = (ctx: ExtensionContext): Promise<boolean> => {
    starting ??= (async () => {
      if (await ready()) return true;
      if (!(await listening())) {
        mkdirSync(join(homedir(), ".cache", "inkling-16gb"), { recursive: true });
        const log = openSync(LOG, "a");
        child = spawn(BIN, ["serve", "--port", String(PORT)], { detached: true, stdio: ["ignore", log, log] });
        child.unref();
        child.once("exit", () => (child = undefined));
      }
      ctx.ui.setStatus("inkling", "Inkling: loading model…");
      const t0 = Date.now();
      while (Date.now() - t0 < READY_TIMEOUT_MS) {
        if (await ready()) {
          ctx.ui.setStatus("inkling", "Inkling: ready");
          return true;
        }
        if (child === undefined && !(await listening())) break; // server exited
        await new Promise((r) => setTimeout(r, 1000));
      }
      ctx.ui.setStatus("inkling", undefined);
      ctx.ui.notify(`Inkling server did not start; see ${LOG}`, "error");
      return false;
    })().finally(() => (starting = undefined));
    return starting;
  };

  const stop = (ctx?: ExtensionContext) => {
    if (child?.pid) {
      try {
        process.kill(child.pid, "SIGTERM");
      } catch {}
      child = undefined;
      ctx?.ui.setStatus("inkling", undefined);
    }
  };

  pi.on("session_start", async (_event, ctx) => {
    if (isOurs(ctx.model)) void ensure(ctx);
  });

  pi.on("model_select", async (event, ctx) => {
    const now = isOurs(event.model);
    const was = isOurs(event.previousModel);
    if (now && !was) {
      before = pi.getThinkingLevel();
      pi.setThinkingLevel(DEFAULT_THINKING);
      void ensure(ctx);
    } else if (!now && was) {
      if (before) pi.setThinkingLevel(before as Parameters<typeof pi.setThinkingLevel>[0]);
      before = undefined;
      stop(ctx);
    }
  });

  pi.on("before_agent_start", async (_event, ctx) => {
    if (isOurs(ctx.model)) await ensure(ctx);
  });

  pi.on("session_shutdown", async (event) => {
    if (event.reason === "quit") stop();
  });
}
