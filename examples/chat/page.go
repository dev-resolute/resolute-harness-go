package main

// page is the whole client: one HTML document, no build step, no external
// assets. It speaks the harness's plain HTTP surface — POST dispatches,
// POST steers, and an EventSource on the conversation's SSE feed whose
// event names are the canonical record kinds.
const page = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>resolute-harness chat</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.45 system-ui, sans-serif; background: #101418; color: #e6e9ec;
         display: flex; flex-direction: column; height: 100vh; }
  header { display: flex; gap: .6rem; align-items: center; padding: .6rem 1rem;
           background: #171c22; border-bottom: 1px solid #242b33; flex-wrap: wrap; }
  header h1 { font-size: 1rem; margin: 0 auto 0 0; font-weight: 600; }
  #dot { width: .6rem; height: .6rem; border-radius: 50%; background: #e0a030; }
  #dot.live { background: #3fb950; }
  input, button { font: inherit; border-radius: 6px; border: 1px solid #2c353f;
                  background: #1c232b; color: inherit; padding: .35rem .6rem; }
  button { cursor: pointer; background: #263241; }
  button:hover { background: #2f3e51; }
  #log { flex: 1; overflow-y: auto; padding: 1rem; display: flex; flex-direction: column; gap: .5rem; }
  .msg { max-width: 72%; padding: .5rem .75rem; border-radius: 10px; white-space: pre-wrap; word-break: break-word; }
  .user { align-self: flex-end; background: #2b4a6f; }
  .assistant { align-self: flex-start; background: #1e2730; }
  .thinking { color: #8b98a5; font-style: italic; font-size: .85rem; }
  .chip { align-self: flex-start; font-size: .8rem; color: #9fb0c0; background: #161d24;
          border: 1px dashed #2c353f; padding: .25rem .6rem; border-radius: 999px; }
  .divider { align-self: center; font-size: .75rem; color: #7a8794; }
  .divider.failed { color: #e0707a; }
  footer { display: grid; grid-template-columns: 1fr auto; gap: .5rem; padding: .75rem 1rem;
           background: #171c22; border-top: 1px solid #242b33; }
  footer .steer-row input { border-color: #5a4a26; }
  #toast { position: fixed; bottom: 5.5rem; left: 50%; transform: translateX(-50%);
           background: #3a2b2b; border: 1px solid #6b4040; padding: .5rem .9rem; border-radius: 8px;
           opacity: 0; transition: opacity .3s; pointer-events: none; }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
<header>
  <h1>resolute-harness chat <small style="color:#7a8794">agents/chat/demo</small></h1>
  <span id="dot" title="stream status"></span>
  <label>session <input id="session" value="default" size="10"></label>
  <button id="switch">switch</button>
</header>
<div id="log"></div>
<footer>
  <div style="display:grid; gap:.5rem;">
    <input id="prompt" placeholder='say something — or "research quantum kettles"' autofocus>
    <div class="steer-row" style="display:grid; grid-template-columns:1fr auto; gap:.5rem;">
      <input id="steerText" placeholder="steer the run mid-flight (works while a tool is running)">
      <button id="steer">Steer</button>
    </div>
  </div>
  <button id="send" style="align-self:start; padding:.35rem 1.2rem;">Send</button>
</footer>
<div id="toast"></div>
<script>
"use strict";
const $ = id => document.getElementById(id);
const log = $("log");
let session = "default";
let es = null;
const seen = new Set();
let bubble = null, thinkingLine = null;

function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2600);
}

function el(cls, text) {
  const d = document.createElement("div");
  d.className = cls;
  if (text !== undefined) d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

function handle(fn) {
  return ev => {
    if (seen.has(ev.lastEventId)) return;
    seen.add(ev.lastEventId);
    fn(JSON.parse(ev.data));
  };
}

function connect() {
  if (es) es.close();
  log.replaceChildren();
  seen.clear();
  bubble = thinkingLine = null;
  es = new EventSource("/agents/chat/demo?session=" + encodeURIComponent(session));
  es.onopen = () => $("dot").classList.add("live");
  es.onerror = () => $("dot").classList.remove("live"); // retries automatically
  es.addEventListener("user_message", handle(p => el("msg user", p.payload.body)));
  es.addEventListener("assistant_message_started", handle(() => { bubble = null; thinkingLine = null; }));
  es.addEventListener("assistant_thinking_delta", handle(p => {
    if (!thinkingLine) thinkingLine = el("thinking");
    thinkingLine.textContent += p.payload.text;
  }));
  es.addEventListener("assistant_text_delta", handle(p => {
    if (!bubble) bubble = el("msg assistant");
    bubble.textContent += p.payload.text;
    log.scrollTop = log.scrollHeight;
  }));
  es.addEventListener("assistant_tool_call", handle(p =>
    el("chip", "🔧 " + p.payload.toolName + " " + (p.payload.args ? JSON.stringify(p.payload.args) : ""))));
  es.addEventListener("tool_outcome", handle(p =>
    el("chip", (p.payload.isError ? "⚠️ " : "✅ ") + p.payload.toolName + ": " + (p.payload.content || ""))));
  es.addEventListener("submission_settled", handle(p => {
    const s = p.payload.status;
    el("divider" + (s === "failed" ? " failed" : ""),
       "— settled: " + s + (p.payload.error ? " (" + p.payload.error + ")" : "") + " —");
    bubble = thinkingLine = null;
  }));
}

async function post(path, body) {
  const resp = await fetch(path, { method: "POST", body: JSON.stringify(body) });
  if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).error || resp.status);
  return resp;
}

$("send").onclick = async () => {
  const body = $("prompt").value.trim();
  if (!body) return;
  $("prompt").value = "";
  try { await post("/agents/chat/demo", { kind: "user", body, session }); }
  catch (err) { toast("dispatch failed: " + err.message); }
};
$("prompt").addEventListener("keydown", ev => { if (ev.key === "Enter") $("send").click(); });

$("steer").onclick = async () => {
  const body = $("steerText").value.trim();
  if (!body) return;
  try {
    await post("/agents/chat/demo/steer", { body, session });
    $("steerText").value = "";
  } catch (err) {
    toast("steer: " + err.message + " — steer while the research tool is running");
  }
};
$("steerText").addEventListener("keydown", ev => { if (ev.key === "Enter") $("steer").click(); });

$("switch").onclick = () => { session = $("session").value.trim() || "default"; connect(); };

connect();
</script>
</body>
</html>
`
