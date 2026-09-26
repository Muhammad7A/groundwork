"""Regression: the live tagger emits verified claims on real provider output.

Captured from the first live Gemini run (2026-09-03): the tagger produced ZERO
proposals ("0 verified, 0 unsourced, 0 unsupported" — nothing even reached the
gates) and, after the parse fix, the deterministic entailment floor rejected
faithful restatements whose statements summarized the quote. These tests pin
the fixed behavior end-to-end, offline:

  * a provider response truncated at max_tokens still yields its complete
    claims (salvage), while garbage yields zero — fail-closed;
  * the live fixture's claims pass grounding + entailment with quotes that
    provably exist in the transcript;
  * a fabricated quote is still rejected — the product promise ("every claim
    carries a quote that provably exists") is dropped-not-softened;
  * statements that violate the prompt contract (summarizing beyond the
    quote) are still rejected by the deterministic floor.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from ai_engine.evidence.grounding import ground_proposal
from ai_engine.evidence.entailment import Entailment, HeuristicEntailmentChecker, apply_entailment
from ai_engine.evidence.model import RawProposal
from ai_engine.evidence.tagger import _parse_proposals
from ai_engine.persistence.transcript_store import transcript_from_dict

FIXTURES = Path(__file__).parent / "fixtures_live"


class EnvironmentIsolationTest(unittest.TestCase):
    """A developer's live-credential env (GOOGLE_APPLICATION_CREDENTIALS etc.)
    must never turn mock-mode tests into live network calls. This suite failed
    50 tests once because of exactly that leak."""

    def test_no_live_env_is_read_by_the_tagger_path(self):
        # The tagger/entailment path takes `llm` explicitly and never reads
        # provider env; assert the settings factory is the ONLY reader.
        import ai_engine.evidence.tagger as tagger_mod
        import inspect

        source = inspect.getsource(tagger_mod)
        for var in ("GOOGLE_APPLICATION_CREDENTIALS", "GROUNDWORK_PROVIDER",
                    "ANTHROPIC_API_KEY"):
            self.assertNotIn(var, source,
                             f"{var} read inside the tagger path breaks env isolation")


def _load():
    transcript = transcript_from_dict(
        json.loads((FIXTURES / "live_transcript.json").read_text(encoding="utf-8")))
    response = json.loads(
        (FIXTURES / "live_tagger_response.json").read_text(encoding="utf-8"))
    return transcript, response


class TruncatedOutputSalvageTest(unittest.TestCase):
    """The live 0-claims root cause: output cut at max_tokens parsed as None
    and the tagger silently emitted zero proposals."""

    def test_a_truncated_response_yields_its_complete_claims(self):
        truncated = ('{"claims": ['
                     '{"type": "workaround", "statement": "Uses informal channels", '
                     '"quote": "informal information channels", "tier": 2}, '
                     '{"type": "friction", "statement": "Redun')
        claims = _parse_proposals(truncated)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].quote, "informal information channels")

    def test_prose_without_json_yields_zero(self):
        self.assertEqual(_parse_proposals("I found no notable claims."), [])

    def test_a_response_with_no_quote_yields_zero(self):
        self.assertEqual(
            _parse_proposals('{"claims": [{"type": "workaround", "statement": "s"}]}'),
            [])


class LiveFixtureVerificationTest(unittest.TestCase):
    """The product promise, executed against the live-shaped fixture."""

    def setUp(self):
        self.transcript, self.response = _load()
        self.checker = HeuristicEntailmentChecker()

    def _verify(self, claim_spec):
        proposal = RawProposal(claim_spec["claim_type"], claim_spec["statement"],
                               quote=claim_spec["quote"], tier=claim_spec["tier"])
        claim, reason = ground_proposal(proposal, self.transcript)
        if claim is None:
            return None, f"grounding rejected: {reason}"
        result = self.checker.check(claim.statement,
                                    claim.evidence[0].resolve(self.transcript))
        if result.verdict is not Entailment.SUPPORTED:
            return None, f"entailment rejected: {result.reason}"
        return claim, "verified"

    def test_at_least_one_live_claim_is_fully_verified(self):
        verified = 0
        for spec in self.response["claims"]:
            claim, status = self._verify(spec)
            verified += claim is not None
        self.assertGreaterEqual(verified, 1,
                                "the live pipeline must produce verified claims")

    def test_every_contract_compliant_claim_is_verified(self):
        for spec in self.response["claims"]:
            with self.subTest(quote=spec["quote"][:40]):
                claim, status = self._verify(spec)
                self.assertIsNotNone(claim, status)
                # the quote must provably exist in the transcript
                resolved = claim.evidence[0].resolve(self.transcript)
                self.assertIn(claim.evidence[0].quote.strip("\"' "), resolved)

    def test_a_fabricated_quote_is_still_rejected(self):
        proposal = RawProposal(
            "friction", "Management rigged the vendor payment system for two years.",
            quote="The vendor payment system was rigged by management for two years.",
            tier=4)
        claim, reason = ground_proposal(proposal, self.transcript)
        self.assertIsNone(claim, reason)
        self.assertEqual(reason, "quote_not_found")

    def test_statement_beyond_the_quote_is_still_rejected_by_the_floor(self):
        # A statement that ADDS context the quote lacks (contract violation the
        # fixed prompt forbids) is still rejected by the deterministic floor.
        spec = {"claim_type": "wasted_effort",
                "statement": "Estimates about a day a week lost to waiting and ad-hoc requests.",
                "quote": "sometimes I'm probably underutilized about a day a week",
                "tier": 3}
        claim, status = self._verify(spec)
        self.assertIsNone(claim, status)


if __name__ == "__main__":
    unittest.main()
