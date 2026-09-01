import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the learning workspace", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /Ling&#x27;s Lab/);
  assert.match(html, /Hot 100/);
  assert.match(html, /项目知识库/);
  assert.match(html, /两数之和/);
  assert.match(html, /14 天讲解计划/);
  assert.match(html, /8(?:<!-- -->)? 张/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("ships the curated HTML lesson library", async () => {
  const [harness, hot100, defenseScript, defenseManual, lrBestStructures, paperMap, misoPrimitives, agentResearchMap, autoResearchFramework, paperGuides, agentResearchGuides] = await Promise.all([
    readFile(new URL("../public/lessons/harness-opd-evi.html", import.meta.url), "utf8"),
    readFile(new URL("../public/lessons/hot100-14days.html", import.meta.url), "utf8"),
    readFile(new URL("../public/lessons/defense-script-qa.html", import.meta.url), "utf8"),
    readFile(new URL("../public/lessons/defense-manual.html", import.meta.url), "utf8"),
    readFile(new URL("../public/lessons/two-lr-best-structures.html", import.meta.url), "utf8"),
    readFile(new URL("../public/lessons/paper-methods-map.html", import.meta.url), "utf8"),
    readFile(new URL("../public/lessons/miso-primitives.html", import.meta.url), "utf8"),
    readFile(new URL("../public/lessons/agent-mle-auto-research.html", import.meta.url), "utf8"),
    readFile(new URL("../public/lessons/auto-research-framework.html", import.meta.url), "utf8"),
    readdir(new URL("../public/lessons/paper-guides/", import.meta.url)),
    readdir(new URL("../public/lessons/agent-research-guides/", import.meta.url)),
  ]);
  assert.match(harness, /RecHarness/);
  assert.match(hot100, /14 天刷完 Hot 100/);
  assert.match(defenseScript, /转正答辩逐页讲稿/);
  assert.match(defenseScript, /29 页 · 40 问/);
  assert.match(defenseScript, /RecHarness 动机/);
  assert.match(defenseScript, /为什么使用 Thompson Sampling/);
  assert.match(defenseScript, /货架“曝光下降、消耗上升”/);
  assert.match(defenseManual, /defense-script-qa\.html/);
  assert.match(lrBestStructures, /最好的结构/);
  assert.match(lrBestStructures, /ESCM²-DR-SNIPS/);
  assert.match(paperMap, /论文地图/);
  assert.match(misoPrimitives, /Ranking：哪些神经元重要/);
  assert.match(agentResearchMap, /Agent-as-MLE × Auto Research/);
  assert.match(agentResearchMap, /三层 Memory 设计/);
  assert.match(autoResearchFramework, /Typed Research Graph/);
  assert.ok(paperGuides.filter((name) => name.endsWith(".html")).length >= 12);
  assert.equal(agentResearchGuides.filter((name) => name.endsWith(".html")).length, 7);
});
