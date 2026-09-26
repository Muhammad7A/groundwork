/** Endpoint tests for the sidecar server — node:test, zero dependencies. */
import test, { after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import crypto from "node:crypto";

import { createApp, main, PORT_DEFAULT, SECRET_ENV } from "./server.mjs";

const SECRET = "test-secret-0123456789";
const AUTH = { "Authorization": `Bearer ${SECRET}` };

const _servers = [];
after(() => {
  for (const s of _servers) s.close();
});

/** Drive the in-process app through node:http on an ephemeral port. */
function listen(app) {
  const server = http.createServer((req, res) => {
    app.handle(req, res).catch(() => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "unhandled", code: "internal" }));
    });
  });
  _servers.push(server);
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve({ server, port: server.address().port })));
}

async function post(port, path, body, headers = AUTH) {
  const started = performance.now();
  const res = await fetch(`http://127.0.0.1:${port}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
  const json = await res.json();
  return { status: res.status, json, ms: performance.now() - started };
}

const PROFILE = { namaUsaha: "Warung Bu Rina", jenisUsaha: "kuliner", cicilanBulanan: 450000 };
const METRIC = { cicilanTercatat: 0, prive: 300000, labaBersih: 900000, bulanTercatat: 4 };

test("happy path: intake → turns → examiner, contract fields exact", async () => {
  const app = createApp({ secret: SECRET });
  const { port } = await listen(app);

  const intake = await post(port, "/v1/intakes", { profil: PROFILE, metrik: METRIC });
  assert.equal(intake.status, 200);
  assert.match(intake.json.intakeId, /^int_[0-9a-f]{12}$/);
  assert.equal(intake.json.jumlahDiscrepancy, 1); // cicilan declared, never recorded
  assert.equal(typeof intake.json.pertanyaan, "string");
  assert.ok(intake.json.pertanyaan.includes("walk me through"));

  // walk every turn until the interview closes
  let turn = await post(port, `/v1/intakes/${intake.json.intakeId}/turns`,
    { jawaban: "I record sales on paper and buy stock from the market every two days myself." });
  assert.equal(turn.status, 200);
  let guard = 0;
  while (!turn.json.selesai && guard < 20) {
    assert.equal(typeof turn.json.pertanyaan, "string");
    assert.ok(turn.json.pertanyaan.length > 10);
    assert.ok(turn.json.coverage >= 0 && turn.json.coverage <= 1);
    turn = await post(port, `/v1/intakes/${intake.json.intakeId}/turns`,
      { jawaban: "The supplier delivery arrives late every Friday and I wait without help, that is the honest story of the stall." });
    guard += 1;
    assert.ok(guard < 20, "interview must close within the question budget");
  }
  assert.equal(turn.json.selesai, true);
  assert.equal(turn.json.pertanyaan, null);
  assert.ok(turn.json.coverage >= 0 && turn.json.coverage <= 1);
  assert.equal(turn.json.jumlahDiscrepancy, 1); // final discrepancy echo, contract-optional

  const examiner = await post(port, `/v1/examiner/${intake.json.intakeId}`, {});
  assert.equal(examiner.status, 200);
  assert.ok(["ok", "insufficient"].includes(examiner.json.verdict));
  assert.ok(examiner.json.coverage >= 0 && examiner.json.coverage <= 1);
  assert.ok(Array.isArray(examiner.json.perArea));
  for (const row of examiner.json.perArea) {
    assert.ok(row.area.length <= 40);
    assert.ok(row.coverage >= 0 && row.coverage <= 1);
  }
});

test("wrong secret → 401 typed error (auth-gagal on the Novaro side)", async () => {
  const app = createApp({ secret: SECRET });
  const { port } = await listen(app);
  const res = await post(port, "/v1/intakes", { profil: PROFILE }, { Authorization: "Bearer wrong" });
  assert.equal(res.status, 401);
  assert.equal(res.json.code, "unauthorized");
  assert.equal(typeof res.json.error, "string");
});

test("missing secret → 401 typed error", async () => {
  const app = createApp({ secret: SECRET });
  const { port } = await listen(app);
  const res = await post(port, "/v1/intakes", { profil: PROFILE }, {});
  assert.equal(res.status, 401);
  assert.equal(res.json.code, "unauthorized");
});

test("unknown intakeId → 404 typed error (never another meaning)", async () => {
  const app = createApp({ secret: SECRET });
  const { port } = await listen(app);
  for (const path of ["/v1/intakes/int_000000000000/turns", "/v1/examiner/int_000000000000"]) {
    const res = await post(port, path, { jawaban: "anything" });
    assert.equal(res.status, 404);
    assert.equal(res.json.code, "unknown-intake");
    assert.equal(typeof res.json.error, "string");
  }
});

test("timeout semantics: every endpoint answers far below the client's 10 s budget", async () => {
  const app = createApp({ secret: SECRET });
  const { port } = await listen(app);
  const intake = await post(port, "/v1/intakes", { profil: PROFILE, metrik: METRIC });
  const id = intake.json.intakeId;
  const turn = await post(port, `/v1/intakes/${id}/turns`, { jawaban: "The morning stock run happens before the stall opens, every single day." });
  const examiner = await post(port, `/v1/examiner/${id}`, {});
  for (const r of [intake, turn, examiner]) {
    assert.ok(r.ms < 10_000, `endpoint must answer inside the client's 10 s timeout, took ${r.ms}ms`);
  }
});

test("invalid JSON body → 400 typed error", async () => {
  const app = createApp({ secret: SECRET });
  const { port, server } = await listen(app);
  const res = await fetch(`http://127.0.0.1:${port}/v1/intakes`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...AUTH },
    body: "{not json",
  });
  const json = await res.json();
  assert.equal(res.status, 400);
  assert.equal(json.code, "invalid-body");
  server.close();
});

test("GET on a sidecar path → 405 typed error", async () => {
  const app = createApp({ secret: SECRET });
  const { port, server } = await listen(app);
  const res = await fetch(`http://127.0.0.1:${port}/v1/intakes`, { headers: AUTH });
  const json = await res.json();
  assert.equal(res.status, 405);
  assert.equal(json.code, "method-not-allowed");
  server.close();
});

test("without NOVA_INTERVIEW_SECRET the server refuses to start (fail-closed)", async () => {
  const { spawn } = await import("node:child_process");
  const { fileURLToPath } = await import("node:url");
  const self = fileURLToPath(import.meta.url);
  const serverPath = fileURLToPath(new URL("./server.mjs", import.meta.url));
  const clean = { ...process.env };
  delete clean[SECRET_ENV];
  delete clean.PORT;
  const result = await new Promise((resolve) => {
    const child = spawn(process.execPath, [serverPath], { env: clean });
    let stderr = "";
    child.stderr.on("data", (d) => { stderr += d; });
    child.on("exit", (code) => resolve({ code, stderr }));
  });
  assert.notEqual(result.code, 0, "startup must fail without the secret");
  assert.ok(result.stderr.includes(SECRET_ENV), "the refusal must name the missing variable");
});

test("sessions are ephemeral: a fresh process sees no intakes (BY DESIGN)", async () => {
  // Two apps, same secret: sessions never cross process boundaries.
  const a = createApp({ secret: SECRET });
  const b = createApp({ secret: SECRET });
  const intake = a._startIntake({ profil: PROFILE, metrik: METRIC });
  assert.ok(b.sessions.size === 0, "another process (or a restarted one) holds no sessions");
  assert.equal(b._examiner(intake.intakeId).status, 404);
});

test("port default constant is 8090 per contract", () => {
  assert.equal(PORT_DEFAULT, 8090);
});
