"""The consultant's application layer, exercised headless — no web server.

These tests run the use-cases the way any surface (routes, CLI, a future API)
would: invite → start → answer → finish → tag → validate → report → release.
If orchestration is correct here, the routes cannot drift from it, because
they hold none of it.
"""
import tempfile
import unittest
from pathlib import Path

from ai_engine.application.service import (
    ANSWER_INACTIVE,
    ANSWER_RECORDED,
    ANSWER_TOO_LONG,
    ConsultantService,
)
from ai_engine.config import Runtime, Settings
from ai_engine.llm.retry import LLMUnavailable
from ai_engine.validation.model import Verdict


def _settings(data_dir: Path, **kw) -> Settings:
    return Settings(api_key=None, store_key=None, runtime=Runtime.DEV,
                    provider="", data_dir=data_dir, **kw)


class InterviewUseCaseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        self.service = ConsultantService(_settings(self.data_dir))

    def _run_simulated(self) -> str:
        transcript_id = self.service.start("Dana Example", simulated=True)
        return self.service.finish(transcript_id)

    def test_a_simulated_interview_runs_to_a_stored_transcript(self):
        transcript_id = self._run_simulated()
        stored = self.service.load_transcript(transcript_id)
        self.assertIsNotNone(stored)
        self.assertTrue(stored.finalized)
        self.assertGreater(len(stored.segments), 2)
        self.assertNotIn("Dana Example", stored.render(),
                         "the real name must never reach the transcript")

    def test_a_manual_interview_advances_turn_by_turn(self):
        transcript_id = self.service.start("Eli")
        first = self.service.view(transcript_id)
        self.assertIsNotNone(first["question"])

        self.assertEqual(self.service.submit_answer(
            transcript_id, "I keep a private spreadsheet, honestly."), ANSWER_RECORDED)
        second = self.service.view(transcript_id)
        self.assertIn("private spreadsheet",
                      " ".join(s["text"] for s in second["segments"]))

        self.assertEqual(
            self.service.submit_answer(transcript_id, "x" * 50_000),
            ANSWER_TOO_LONG, "an oversized paste must not be recorded")
        self.assertEqual(self.service.submit_answer("no-such-id", "hi"),
                         ANSWER_INACTIVE)

    def test_finishing_twice_cannot_duplicate_the_transcript(self):
        transcript_id = self.service.start("Eli")
        self.assertIsNotNone(self.service.finish(transcript_id))
        self.assertIsNone(self.service.finish(transcript_id))

    def test_validation_flows_through_the_gate(self):
        transcript_id = self._run_simulated()
        tagging = self.service.tag(transcript_id)
        self.assertGreater(len(tagging.claims), 0)
        claim = tagging.claims[0]

        # The gate requires a reason to amend; without one the verdict is refused.
        self.assertFalse(self.service.record_verdict(
            transcript_id, claim.id, Verdict.AMENDED.value, reason=""))
        self.assertTrue(self.service.record_verdict(
            transcript_id, claim.id, Verdict.ACCEPTED.value))
        verdicts = self.service.verdicts(transcript_id)
        self.assertEqual(verdicts[claim.id].verdict, Verdict.ACCEPTED.value)

        findings = self.service.validated_findings(transcript_id)
        self.assertEqual([f.claim.id for f in findings], [claim.id])

    def test_reports_obey_the_firewall(self):
        transcript_id = self._run_simulated()
        tagging = self.service.tag(transcript_id)
        for claim in tagging.claims:
            self.service.record_verdict(transcript_id, claim.id,
                                        Verdict.ACCEPTED.value)

        consultant_md = self.service.consultant_report_markdown(transcript_id)
        self.assertIn("## Summary", consultant_md)
        # Saved under the at-rest policy like any testimony-derived artifact.
        report_files = list(self.data_dir.glob("*.report.md*"))
        self.assertEqual(len(report_files), 1)

        employer_md = self.service.employer_release_markdown([transcript_id])
        # One participant cannot satisfy k-anonymity: everything is suppressed,
        # and nothing verbatim reaches the employer's document.
        for claim in tagging.claims:
            self.assertNotIn(claim.statement, employer_md)

    def test_the_report_links_every_finding_to_its_verbatim_moment(self):
        transcript_id = self._run_simulated()
        tagging = self.service.tag(transcript_id)
        for claim in tagging.claims:
            self.service.record_verdict(transcript_id, claim.id,
                                        Verdict.ACCEPTED.value)
        claim = tagging.claims[0]
        segment_id = claim.evidence[0].ref.segment_id

        markdown = self.service.consultant_report_markdown(transcript_id)
        self.assertIn(f"/transcripts/{transcript_id}#seg-{segment_id}", markdown)
        self.assertIn("Grounded by construction", markdown)
        self.assertIn(str(tagging.report.total), markdown)

        page = self.service.consultant_report_page(transcript_id)
        self.assertEqual(page["grounding"]["proposed"], tagging.report.total)
        self.assertEqual(len(page["findings"]), len(tagging.claims))

    def test_dashboard_rows_count_claims_and_verdicts(self):
        self._run_simulated()
        rows = self.service.dashboard_rows()
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["claims"], 0)
        self.assertEqual(rows[0]["validated"], 0)

    def test_live_interviews_survive_only_in_memory_until_finished(self):
        transcript_id = self.service.start("Eli")
        self.assertIn(transcript_id, [i["id"] for i in self.service.live_summary()])
        # A fresh service (a restarted process) has no live state — that is the
        # consent-preserving failure mode, not a bug.
        fresh = ConsultantService(_settings(self.data_dir))
        self.assertEqual(fresh.live_summary(), [])
        self.assertIsNone(fresh.view(transcript_id))


class PseudonymPersistenceTest(unittest.TestCase):
    def test_the_service_survives_a_restart_with_its_pseudonyms(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = ConsultantService(_settings(Path(tmp)))
            token = first.invite("Dana Example")
            invitation = first.list_invitations()[0]
            pseudonym = invitation.pseudonym

            restarted = ConsultantService(_settings(Path(tmp)))
            same_person = restarted.pseudonymizer.pseudonym("Dana Example")
            self.assertEqual(same_person, pseudonym,
                             "a restart must not re-pseudonymize the same person")
            self.assertEqual(restarted.list_invitations()[0].token, token)


if __name__ == "__main__":
    unittest.main()


class EngagementUseCaseTest(unittest.TestCase):
    """Engagements are the consultant's unit of work — named, grouped, reported."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        self.service = ConsultantService(_settings(self.data_dir))

    def test_interviews_group_under_a_named_engagement(self):
        engagement_id = self.service.create_engagement("Acme discovery")["id"]
        tid = self.service.start("Dana", simulated=True, engagement_id=engagement_id)
        self.service.finish(tid)
        rows = {r["id"]: r for r in self.service.dashboard_rows()}
        self.assertEqual(rows[tid]["engagement"], "Acme discovery")
        listing = self.service.list_engagements()
        acme = next(e for e in listing if e["id"] == engagement_id)
        self.assertEqual(acme["interviews"], 1)

    def test_resolving_a_name_reuses_the_engagement(self):
        first = self.service.resolve_engagement("Acme")
        second = self.service.resolve_engagement("acme")   # case-insensitive
        self.assertEqual(first, second)
        blank = self.service.resolve_engagement("  ")
        self.assertEqual(self.service.engagement_name(blank), "Ad-hoc interviews")

    def test_batch_intake_creates_one_invitation_per_line(self):
        tokens = self.service.invite_batch("Ana\nBo\n\nCitra\n")
        self.assertEqual(len(tokens), 3, "blank lines must be skipped")
        self.assertEqual(len(self.service.list_invitations()), 3)


class BatchIntakeWebTest(unittest.TestCase):
    def test_the_batch_route_creates_invitations(self):
        try:
            from fastapi.testclient import TestClient
        except Exception:
            self.skipTest("requires fastapi")
            return
        from ai_engine.webapp.app import create_app

        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(_settings(Path(tmp))))
            r = client.post("/invitations/batch",
                            data={"names": "Ana\nBo\nCitra"}, follow_redirects=False)
            self.assertEqual(r.status_code, 303)
            page = client.get("/invitations").text
            self.assertEqual(page.count("/i/"), 3)


class DemoEngagementTest(unittest.TestCase):
    """The one-click demo: parallel interviews in, both deliverables out."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        self.service = ConsultantService(_settings(self.data_dir))

    def test_the_whole_engagement_runs_offline_in_one_call(self):
        from ai_engine.application.demo import _personas, run_demo_engagement

        result = run_demo_engagement(self.service)
        self.assertEqual(len(result.interviews), len(_personas()))
        self.assertGreater(result.claims, 0)
        # Stored, pseudonymized, grouped under the engagement.
        for item in result.interviews:
            transcript = self.service.load_transcript(item["transcript_id"])
            self.assertIsNotNone(transcript)
            self.assertTrue(transcript.finalized)
            self.assertNotIn(item["participant"], transcript.render())
            self.assertEqual(transcript.engagement_id, result.engagement_id)
            # Verdicts were recorded through the real ledger, labelled auto-sim.
            verdicts = self.service.verdicts(item["transcript_id"])
            self.assertTrue(verdicts)
            self.assertTrue(all(v.validator_kind == "auto-sim"
                                for v in verdicts.values()))

    def test_the_synthesis_reports_the_engagement(self):
        from ai_engine.application.demo import run_demo_engagement

        result = run_demo_engagement(self.service)
        md = self.service.engagement_synthesis_markdown(result.engagement_id)
        self.assertIn("Speedrun demo", md)
        self.assertIn(f"Interviews:** {len(result.interviews)}", md)
        # Consultant is inside the firewall: verbatim detail with attribution.
        self.assertIn("evidence:", md)

    def test_the_demo_is_deterministic_in_mock_mode(self):
        from ai_engine.application.demo import run_demo_engagement

        first = run_demo_engagement(self.service, engagement_name="Demo A")
        second = run_demo_engagement(self.service, engagement_name="Demo B")
        self.assertEqual([i["claims"] for i in first.interviews],
                         [i["claims"] for i in second.interviews],
                         "the mock demo must not vary between pitches")


class EmployerReleaseContractTest(unittest.TestCase):
    """The contract that used to fail silently: an engagement id passed where a
    list of transcript ids goes iterated character-by-character and rendered a
    plausible, completely empty report. Now the contract is loud."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        self.service = ConsultantService(_settings(self.data_dir))

    def _demo(self):
        from ai_engine.application.demo import run_demo_engagement
        return run_demo_engagement(self.service)

    def test_an_engagement_id_is_not_a_list_of_ids(self):
        result = self._demo()
        with self.assertRaises(ValueError):
            self.service.employer_release_markdown(result.engagement_id)

    def test_both_scopes_at_once_is_refused(self):
        result = self._demo()
        with self.assertRaises(ValueError):
            self.service.employer_release_markdown(
                [result.interviews[0]["transcript_id"]],
                engagement_id=result.engagement_id)

    def test_unknown_ids_are_refused_not_rendered_empty(self):
        self._demo()
        with self.assertRaises(ValueError):
            self.service.employer_release_markdown(["txn-doesnotexist1"])

    def test_engagement_scope_renders_the_named_release(self):
        result = self._demo()
        md = self.service.employer_release_markdown(engagement_id=result.engagement_id)
        self.assertIn(f"Interviews:** {len(result.interviews)}", md)
        self.assertIn(result.engagement_name, md)

    def test_an_empty_engagement_is_refused_at_the_service_layer(self):
        eid = self.service.create_engagement("Nothing yet")["id"]
        with self.assertRaises(ValueError):
            self.service.employer_release_markdown(engagement_id=eid)


class DemoFailureIsolationTest(unittest.TestCase):
    """One failed interview used to escape the thread pool and 500 the whole
    route, leaving a random half of the engagement stored. Failures are now
    isolated per interview and reported."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        self.service = ConsultantService(_settings(self.data_dir))

    def _patch(self, attr, value):
        import ai_engine.application.demo as demo_mod

        original = getattr(demo_mod, attr)
        setattr(demo_mod, attr, value)
        self.addCleanup(setattr, demo_mod, attr, original)

    def test_a_dead_model_fails_every_interview_without_raising(self):
        from ai_engine.application import demo as demo_mod

        class _Dead:
            def complete(self, **k):
                raise LLMUnavailable(4, RuntimeError("down"))

        self._patch("get_llm_client", lambda s: _Dead())
        result = demo_mod.run_demo_engagement(self.service)
        self.assertEqual(len(result.failures), 9)
        self.assertEqual(len(result.interviews), 0)
        self.assertTrue(result.all_failed)
        # Failed interviews take their unconsented partial testimony with them.
        leftovers = [p.name for p in (self.data_dir).glob("*.testimony.jsonl")]
        self.assertEqual(leftovers, [], "partial testimony of failed interviews must not persist")

    def test_one_failed_interview_does_not_sink_the_engagement(self):
        import ai_engine.application.demo as demo_mod
        from ai_engine.interview.engine import InterviewEngine as _Real
        from ai_engine.llm.retry import LLMUnavailable

        class _OneFails(_Real):
            _calls = {"n": 0}

            def next_turn(self, **kwargs):
                if _OneFails._calls["n"] == 0:
                    _OneFails._calls["n"] = 1
                    raise LLMUnavailable(1, RuntimeError("flaky"))
                return super().next_turn(**kwargs)

        self._patch("InterviewEngine", _OneFails)
        result = demo_mod.run_demo_engagement(self.service)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(len(result.interviews) + len(result.failures), 9,
                         "every persona is accounted for: completed or failed")
        self.assertIn("error", result.failures[0])
        stored = [i for i in result.interviews
                  if self.service.load_transcript(i["transcript_id"])]
        self.assertEqual(len(stored), 8)


class EngagementRegistryCorruptionTest(unittest.TestCase):
    """A corrupt index used to read as empty AND be overwritten by the next
    create — every engagement name gone without a trace."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        self.service = ConsultantService(_settings(self.data_dir))

    def test_corrupt_index_is_quarantined_not_overwritten(self):
        self.service.create_engagement("Keep me")
        index = self.data_dir / "engagements" / "index.json"
        index.write_text("{broken", encoding="utf-8")

        entry = self.service.create_engagement("After corruption")

        backups = list((self.data_dir / "engagements").glob("index.json.corrupt-*"))
        self.assertEqual(len(backups), 1, "the corrupt file must be preserved")
        self.assertIn("{broken", backups[0].read_text(encoding="utf-8"))
        self.assertEqual(entry["name"], "After corruption")
        # And the registry is usable afterwards.
        self.assertEqual(
            [e["name"] for e in self.service.list_engagements()],
            ["After corruption"])
