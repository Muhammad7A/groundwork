/**
 * SIDECAR INTERVIEW API v1 — server implementation.
 *
 * Contract source of truth: Novaro `server/lib/interview.js` (the client that
 * consumes this API validates every field; this server must satisfy it
 * byte-per-field). Summary of the contract implemented here:
 *
 *   POST /v1/intakes
 *     Authorization : Bearer <NOVA_INTERVIEW_SECRET>
 *     body          : { profil: <object|null>, metrik: <object|null> }
 *     200           : { intakeId: "int_[0-9a-f]{12}", jumlahDiscrepancy: <int>,
 *                       pertanyaan: <string> }
 *
 *   POST /v1/intakes/{intakeId}/turns
 *     body          : { jawaban: <string> }  (free text, relayed — never stored
 *                     on the Novaro side)
 *     200           : { pertanyaan: <string|null>, selesai: <bool>,
 *                       coverage: <0..1>, jumlahDiscrepancy?: <int> }
 *       selesai=true → pertanyaan null, coverage final.
 *
 *   POST /v1/examiner/{intakeId}
 *     200           : { verdict: "ok"|"insufficient", coverage: <0..1>,
 *                       perArea: [{ area: <string ≤40>, coverage: <0..1> }] }
 *
 *   Errors are typed: { error: <string>, code: <string> } with codes
 *   "unauthorized" (401), "unknown-intake" (404), "invalid-body" (400),
 *   "method-not-allowed" (405), "internal" (500). The Novaro client maps
 *   401/403 → auth-gagal and any other non-ok → sidecar-mati; 404 is reserved
 *   for unknown intakes and never carries another meaning.
 *
 * SESSIONS ARE EPHEMERAL AND IN-MEMORY, BY DESIGN: one process restart and
 * every intakeId is gone (the Novaro client then reports sidecar-mati and the
 * user simply re-intakes). No session state is ever written to disk. This is
 * a deliberate privacy property, not a limitation to fix.
 *
 * The question bank below is ported VERBATIM from the Groundwork engine
 * (apps/ai-engine/ai_engine/interview/engine.py — _OPENING, _BANK,
 * _SPECIFICITY_BY_AREA). A cross-language parity test
 * (apps/ai-engine/tests/test_sidecar_contract.py) fails if either side
 * drifts — one bank, one wording, two runtimes.
 *
 * Zero third-party dependencies: node:http + node:crypto only.
 */
import http from "node:http";
import crypto from "node:crypto";
import { pathToFileURL } from "node:url";

export const PORT_DEFAULT = 8090;
export const SECRET_ENV = "NOVA_INTERVIEW_SECRET";

/** Session budget: the opening + one question per bank area. */
export const MAX_QUESTIONS = 7;

/** The contract's intake-id shape (validated before any URL composition). */
export const INTAKE_ID_PATTERN = /^int_[0-9a-f]{12}$/;

/**
 * The discovery question bank, ported verbatim from
 * apps/ai-engine/ai_engine/interview/engine.py. Do not reword here without
 * the same change there — the parity test guards it.
 */
export const OPENING =
  "Thanks for making the time — nothing you say gets back to your employer with your name on it. To start, can you walk me through how your work actually gets done day to day?";

export const BANK = [
  {
    area: "process_reality",
    tier: 1,
    text: "Where does the real version of that differ from how it's officially supposed to work?",
    probe: "When did it last differ from how it's officially supposed to work — what actually happened that time?",
  },
  {
    area: "workarounds",
    tier: 2,
    text: "When the official tools or process get in your way, what do you do instead — any workarounds you've built for yourself?",
    probe: "When did you last do something instead of the official way — what was the workaround, exactly?",
  },
  {
    area: "bottlenecks",
    tier: 2,
    text: "Where do things most often stall or pile up waiting on someone or something?",
    probe: "When did things last stall or pile up — what were you waiting on, and for how long?",
  },
  {
    area: "wasted_effort",
    tier: 2,
    text: "When was the last time you spent real effort on something that felt redundant or got redone? Walk me through it.",
    probe: "What was the last thing that felt redundant or got redone — what was it, and how long did it take?",
  },
  {
    area: "ai_opportunity",
    tier: 2,
    text: "Which parts of your week are the most repetitive or rules-based — the stuff you could almost do in your sleep?",
    probe: "Which repetitive, rules-based task did you do most recently — walk me through it step by step?",
  },
  {
    area: "friction",
    tier: 3,
    text: "Was that more about the tools, the process, or a decision someone made? There's no wrong answer — I'm just mapping where the friction sits.",
    probe: "What was the last decision someone made that got in your way — what happened after that?",
  },
  {
    area: "friction",
    tier: 4,
    text: "A lot of people quietly work around all this. What's your own honest version — anything you do differently than you're 'supposed' to?",
    probe: "What was the last decision someone made that got in your way — what happened after that?",
  },
];

/** Vague/deflection markers — ported verbatim from the engine's assessors. */
const VAGUE_MARKERS = [
  "mostly fine", "nothing jumps out", "pretty standard", "standard stuff",
  "nothing really", "can't think of", "not sure", "hard to say",
  "can't really say",
];
const DEFLECTION_MARKERS = [
  "rather not", "prefer not", "no comment", "don't want to get into",
  "not comfortable", "won't answer", "skip that", "pass on that",
];

const isVague = (answer) => {
  const low = answer.toLowerCase();
  return (
    DEFLECTION_MARKERS.some((m) => low.includes(m)) ||
    VAGUE_MARKERS.some((m) => low.includes(m)) ||
    answer.trim().split(/\s+/).length < 6
  );
};

/**
 * Discrepancy scan: stated profile vs the books' numbers digest. Deterministic
 * rules only — the same conservatism as metrics.js (never inflate the business).
 */
export function scanDiscrepancies(profil, metrik) {
  const out = [];
  const p = profil ?? {};
  const m = metrik ?? {};
  const declaredCicilan = Number(p.cicilanBulanan ?? 0);
  const recordedCicilan = Number(m.cicilanTercatat ?? 0);
  if (declaredCicilan > 0 && recordedCicilan <= 0) {
    out.push("cicilan-dideklarasikan-tapi-tidak-tercatat");
  }
  const prive = Number(m.prive ?? 0);
  const laba = Number(m.labaBersih ?? 0);
  if (prive > laba) {
    out.push("prive-melebihi-laba");
  }
  const bulan = Number(m.bulanTercatat ?? 0);
  if (bulan > 0 && bulan < 3) {
    out.push("bulan-tercatat-kurang-dari-tiga");
  }
  return out;
}

/** Constant-time secret comparison (hash both sides to equal length first). */
function secretMatches(supplied, secret) {
  const a = crypto.createHash("sha256").update(String(supplied)).digest();
  const b = crypto.createHash("sha256").update(String(secret)).digest();
  return crypto.timingSafeEqual(a, b);
}

function authorized(req, secret) {
  const header = req.headers.authorization ?? "";
  if (!header.startsWith("Bearer ")) return false;
  return secretMatches(header.slice("Bearer ".length).trim(), secret);
}

function send(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
    "Connection": "close",
  });
  res.end(body);
}

const fail = (res, status, code, message) => send(res, status, { error: message, code });

/** One intake session — ephemeral, in-memory, never persisted. */
class Session {
  constructor(intakeId, discrepancies) {
    this.intakeId = intakeId;
    this.discrepancies = discrepancies;
    this.pending = { area: "process_reality", tier: 1 };
    this.asked = new Set();          // bank questions already asked
    this.probed = new Set();         // areas that already had their probe
    this.coverage = {};              // area -> 0..1
    this.turns = 0;
    this.done = false;
  }

  get coverageOverall() {
    const values = Object.values(this.coverage);
    if (!values.length) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  perArea() {
    return Object.entries(this.coverage).map(([area, coverage]) => ({
      area,
      coverage: Math.round(coverage * 100) / 100,
    }));
  }

  recordAnswer(answer) {
    const substantive = !isVague(answer);
    const area = this.pending.area;
    this.coverage[area] = substantive
      ? 1
      : this.probed.has(area)
        ? 0
        : this.coverage[area] ?? 0;
    return substantive;
  }

  nextQuestion() {
    // one retry per area via the probe, then move on
    if (!this.probed.has(this.pending.area) && !this.coverage[this.pending.area]) {
      this.probed.add(this.pending.area);
      const entry = BANK.find((b) => b.area === this.pending.area);
      return entry ? entry.probe : null;
    }
    const next = BANK.find(
      (b) => !this.asked.has(b.text) && this.coverage[b.area] === undefined,
    );
    if (!next) return null;
    this.asked.add(next.text);
    this.pending = { area: next.area, tier: next.tier };
    return next.text;
  }
}

/** The application: pure route handling over the session map (test-friendly). */
export function createApp({ secret }) {
  if (!secret || !String(secret).trim()) {
    throw new Error(
      "NOVA_INTERVIEW_SECRET is not set — the sidecar refuses to start without its auth secret.",
    );
  }
  /** Sessions are in-memory BY DESIGN: restart = sessions gone (re-intake). */
  const sessions = new Map();

  function startIntake(body) {
    const discrepancies = scanDiscrepancies(body?.profil, body?.metrik);
    let intakeId = "";
    do {
      intakeId = `int_${crypto.randomBytes(6).toString("hex")}`;
    } while (sessions.has(intakeId));
    sessions.set(intakeId, new Session(intakeId, discrepancies));
    return {
      intakeId,
      jumlahDiscrepancy: discrepancies.length,
      pertanyaan: OPENING,
    };
  }

  function turn(intakeId, body) {
    const session = sessions.get(intakeId);
    if (!session) return { status: 404, payload: { error: "unknown intake id", code: "unknown-intake" } };
    const jawaban = String(body?.jawaban ?? "");
    if (session.done || session.turns >= MAX_QUESTIONS) {
      return {
        status: 200,
        payload: {
          pertanyaan: null,
          selesai: true,
          coverage: Math.round(session.coverageOverall * 100) / 100,
          jumlahDiscrepancy: session.discrepancies.length,
        },
      };
    }
    session.turns += 1;
    session.recordAnswer(jawaban);
    const pertanyaan = session.nextQuestion();
    const selesai = pertanyaan === null || session.turns >= MAX_QUESTIONS;
    const payload = {
      pertanyaan: selesai ? null : pertanyaan,
      selesai,
      coverage: Math.round(session.coverageOverall * 100) / 100,
    };
    if (selesai) payload.jumlahDiscrepancy = session.discrepancies.length;
    return { status: 200, payload };
  }

  function examiner(intakeId) {
    const session = sessions.get(intakeId);
    if (!session) return { status: 404, payload: { error: "unknown intake id", code: "unknown-intake" } };
    const coverage = Math.round(session.coverageOverall * 100) / 100;
    return {
      status: 200,
      payload: {
        verdict: coverage >= 0.6 ? "ok" : "insufficient",
        coverage,
        perArea: session.perArea(),
      },
    };
  }

  async function handle(req, res) {
    const url = new URL(req.url ?? "/", "http://localhost");
    const path = url.pathname;

    if (!authorized(req, secret)) {
      fail(res, 401, "unauthorized", "Bearer secret missing or wrong.");
      return;
    }

    const intakeTurn = path.match(/^\/v1\/intakes\/(int_[0-9a-f]{12})\/turns$/);
    const examinerRoute = path.match(/^\/v1\/examiner\/(int_[0-9a-f]{12})$/);

    if (req.method === "POST" && path === "/v1/intakes") {
      let body;
      try {
        body = await readJson(req);
      } catch {
        fail(res, 400, "invalid-body", "Request body is not valid JSON.");
        return;
      }
      const payload = startIntake(body);
      send(res, 200, payload);
      return;
    }

    if (req.method === "POST" && intakeTurn) {
      let body;
      try {
        body = await readJson(req);
      } catch {
        fail(res, 400, "invalid-body", "Request body is not valid JSON.");
        return;
      }
      const { status, payload } = turn(intakeTurn[1], body);
      send(res, status, payload);
      return;
    }

    if (req.method === "POST" && examinerRoute) {
      const { status, payload } = examiner(examinerRoute[1]);
      send(res, status, payload);
      return;
    }

    if (path.startsWith("/v1/")) {
      if (req.method !== "POST") {
        fail(res, 405, "method-not-allowed", "Use POST for sidecar endpoints.");
        return;
      }
      fail(res, 404, "unknown-intake", "Unknown sidecar path or intake id.");
      return;
    }

    fail(res, 404, "not-found", "Unknown path.");
  }

  return { handle, sessions, _startIntake: startIntake, _turn: turn, _examiner: examiner };
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
      if (data.length > 1_000_000) reject(new Error("body too large"));
    });
    req.on("end", () => {
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch (err) {
        reject(err);
      }
    });
    req.on("error", reject);
  });
}

/** Entry point: refuses to start without the auth secret (fail-closed). */
export function main(argv = process.argv) {
  const secret = (process.env[SECRET_ENV] ?? "").trim();
  if (!secret) {
    console.error(
      `[sidecar] ${SECRET_ENV} is not set — refusing to start. ` +
        "The sidecar's only authentication is this shared secret; it must be " +
        "provided by the Novaro server and kept out of logs.",
    );
    return 1;
  }
  const port = Number(process.env.PORT) || PORT_DEFAULT;
  const app = createApp({ secret });
  const server = http.createServer((req, res) => {
    app.handle(req, res).catch(() => {
      fail(res, 500, "internal", "Unhandled sidecar error.");
    });
  });
  server.listen(port, "127.0.0.1", () => {
    console.log(`[sidecar] interview API v1 listening on 127.0.0.1:${port}`);
  });
  return server;
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  process.exitCode = main();
}
