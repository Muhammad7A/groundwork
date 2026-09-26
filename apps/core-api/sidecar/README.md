# Sidecar Interview API v1 (server)

The Groundwork interviewer engine exposed as a three-endpoint sidecar for the
Novaro server. Contract source of truth: Novaro's `server/lib/interview.js`
(the client validates every response field; this server satisfies it
byte-per-field).

## Run

```bash
NOVA_INTERVIEW_SECRET=<secret> node server.mjs        # listens on 127.0.0.1:8090
PORT=9000 NOVA_INTERVIEW_SECRET=<secret> node server.mjs
```

**Fail-closed:** `NOVA_INTERVIEW_SECRET` unset/empty → the server refuses to
start with a message naming the variable. The secret is the sidecar's only
authentication (Bearer header) and is compared in constant time.

## Endpoints

| Endpoint | Body | 200 response |
|---|---|---|
| `POST /v1/intakes` | `{profil, metrik}` — stated business profile + the books' numbers digest | `{intakeId, jumlahDiscrepancy, pertanyaan}` |
| `POST /v1/intakes/{intakeId}/turns` | `{jawaban}` — free text | `{pertanyaan, selesai, coverage, jumlahDiscrepancy?}` |
| `POST /v1/examiner/{intakeId}` | `{}` | `{verdict: "ok"\|"insufficient", coverage, perArea: [{area, coverage}]}` |

`intakeId` matches `int_[0-9a-f]{12}`. `selesai=true` closes the intake:
`pertanyaan` is `null` and `coverage` is final. Errors are typed
`{error, code}` with codes `unauthorized` (401), `unknown-intake` (404),
`invalid-body` (400), `method-not-allowed` (405), `internal` (500).

## Engine mapping

- **intake** → deterministic discrepancy scan of stated profile vs books digest
  (declared installment never recorded; drawings exceeding profit; fewer than
  three recorded months) + the opening question.
- **turn** → the next question from the discovery interview question bank
  (ported verbatim from `apps/ai-engine/ai_engine/interview/engine.py`; a
  cross-language parity test fails if either side drifts). A vague answer
  earns one specificity-probe retry, then the area scores 0 and the interview
  moves on — no dead ends, no repeated questions.
- **examiner** → per-area coverage from the collected answers; overall verdict
  `ok` at coverage ≥ 0.6, else the honest `insufficient`.

## Sessions are ephemeral — BY DESIGN

Sessions live in process memory only. **A process restart loses every
`intakeId`**; the Novaro client then reports `sidecar-mati` and the user
simply re-intakes. Nothing about an interview is ever written to disk by this
server. This is a deliberate privacy property of the sidecar, not a
limitation — do not "fix" it by persisting sessions.

## Tests

```bash
node --test --test-force-exit server.test.mjs
```

Covers: happy path intake → turns → examiner; wrong/missing secret → 401
typed; unknown intakeId → 404 typed; timeout semantics (every endpoint answers
far inside the client's 10 s budget); invalid body → 400; wrong method → 405;
fail-closed startup without the secret; session ephemerality across
"processes".
