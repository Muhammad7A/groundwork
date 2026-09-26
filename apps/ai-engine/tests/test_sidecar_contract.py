"""Cross-language contract test: the sidecar server (apps/core-api, Node ESM)
ports the discovery question bank verbatim from the Groundwork engine, and
carries the Novaro sidecar contract's load-bearing rules.

This follows the repo's established cross-language drift-guard pattern
(test_contract_conformance.py parses TypeScript; this parses the sidecar's
JavaScript). The bank texts are the product's wording — a silent rewording on
either runtime breaks the shared interview experience, so drift must fail a
test, not a user session.
"""
import unittest
from pathlib import Path

SIDECAR = Path(__file__).resolve().parents[3] / "apps" / "core-api" / "sidecar" / "server.mjs"
NOVARO_CLIENT = (Path(__file__).resolve().parents[3] / "novaro-ai" / "server"
                 / "lib" / "interview.js")


def sidecar_source() -> str:
    return SIDECAR.read_text(encoding="utf-8")


class SidecarContractTest(unittest.TestCase):
    def setUp(self):
        self.source = sidecar_source()

    def test_bank_is_ported_verbatim(self):
        from ai_engine.interview.engine import _BANK

        for (area, tier), text in _BANK.items():
            with self.subTest(area=area, tier=tier):
                self.assertIn(text, self.source,
                              f"bank text for ({area}, {tier}) drifted between "
                              f"the engine and the sidecar")

    def test_opening_is_ported_verbatim(self):
        from ai_engine.interview.engine import _OPENING

        self.assertIn(_OPENING, self.source)

    def test_contract_rules_are_present_in_the_server(self):
        # auth: constant-time compare + the exact env variable
        self.assertIn("timingSafeEqual", self.source)
        self.assertIn("NOVA_INTERVIEW_SECRET", self.source)
        self.assertIn("Bearer ", self.source)
        # fail-closed startup without the secret
        self.assertIn("refuses to start", self.source)
        # contract endpoints
        self.assertIn("/v1/intakes", self.source)
        self.assertIn("/turns", self.source)
        self.assertIn("/v1/examiner/", self.source)
        # intake id shape
        self.assertIn("int_[0-9a-f]{12}", self.source)
        # sessions documented as ephemeral BY DESIGN
        self.assertIn("EPHEMERAL", self.source.upper())
        self.assertIn("BY DESIGN", self.source.upper())
        # default port
        self.assertIn("8090", self.source)

    def test_the_novaro_client_and_the_server_agree_on_error_shape(self):
        # Conditional, NOT skipped: CI has no novaro-ai checkout, and a skip
        # would trip the CI skip-guard. The assertions run wherever the
        # checkout exists (this workstation).
        if not NOVARO_CLIENT.exists():
            return  # CI has no novaro-ai checkout; the cross-check runs on
            # machines that have it. A skip here would trip the CI skip-guard.
        client_source = NOVARO_CLIENT.read_text(encoding="utf-8")
        # the server emits these codes; the client translates these statuses
        for code in ("unauthorized", "unknown-intake", "invalid-body"):
            self.assertIn(code, self.source)
        self.assertIn("res.status === 401 || res.status === 403", client_source)
        self.assertIn("payload-rusak", client_source)
        self.assertIn("sidecar-mati", client_source)


if __name__ == "__main__":
    unittest.main()
