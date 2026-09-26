"""The consultant workspace.

Exercises the whole MVP loop through HTTP — start an interview, answer turns,
finalise, review evidence, validate, generate both reports — and asserts the
properties the app must not break, above all that the employer-facing page does not
leak what the consultant-facing one legitimately shows.
"""
import re
import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    _HAS_CLIENT = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_CLIENT = False

from ai_engine.config import Runtime, Settings
from ai_engine.evidence.grounding import claim_id_for
from ai_engine.persistence.ledger import read_verdicts


class ClaimIdentityTest(unittest.TestCase):
    """Content-addressed ids are what make recorded verdicts survive a restart."""

    def test_same_content_gives_the_same_id(self):
        args = dict(transcript_id="txn-1", segment_id="seg-1", start=0, end=10,
                    statement="a claim")
        self.assertEqual(claim_id_for(**args), claim_id_for(**args))

    def test_different_span_gives_a_different_id(self):
        base = dict(transcript_id="txn-1", segment_id="seg-1", start=0, end=10,
                    statement="a claim")
        other = dict(base, end=11)
        self.assertNotEqual(claim_id_for(**base), claim_id_for(**other))

    def test_different_transcript_gives_a_different_id(self):
        base = dict(transcript_id="txn-1", segment_id="seg-1", start=0, end=10,
                    statement="a claim")
        self.assertNotEqual(claim_id_for(**base),
                            claim_id_for(**dict(base, transcript_id="txn-2")))


@unittest.skipUnless(_HAS_CLIENT, "requires fastapi and httpx")
class WorkspaceFlowTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        from ai_engine.webapp.app import create_app

        settings = Settings(api_key=None, store_key=None, runtime=Runtime.DEV,
                            provider="", data_dir=self.data_dir)
        self.settings = settings
        self.client = TestClient(create_app(settings))

    def _run_simulated_interview(self) -> str:
        """Start a simulated interview, finalise it, return the transcript id."""
        r = self.client.post("/interviews/new",
                             data={"participant": "Dana Example", "mode": "simulated"},
                             follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        interview_id = r.headers["location"].rsplit("/", 1)[-1]
        r = self.client.post(f"/interviews/{interview_id}/finish",
                             follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        return interview_id

    # -- basics ------------------------------------------------------------
    def test_dashboard_renders(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Consultant workspace", r.text)

    def test_posture_is_shown_and_warns_in_mock_mode(self):
        r = self.client.get("/")
        self.assertIn("cognition=MOCK", r.text)
        self.assertIn("not a real interview", r.text)

    def test_manual_interview_asks_a_question_then_accepts_an_answer(self):
        r = self.client.post("/interviews/new",
                             data={"participant": "Eli", "mode": "manual"},
                             follow_redirects=False)
        interview_id = r.headers["location"].rsplit("/", 1)[-1]
        page = self.client.get(f"/interviews/{interview_id}").text
        self.assertIn("Send answer", page)
        self.client.post(f"/interviews/{interview_id}/answer",
                         data={"answer": "I keep a private spreadsheet, honestly."},
                         follow_redirects=False)
        page = self.client.get(f"/interviews/{interview_id}").text
        self.assertIn("private spreadsheet", page)

    def test_real_name_is_pseudonymised_immediately(self):
        r = self.client.post("/interviews/new",
                             data={"participant": "Dana Example", "mode": "manual"},
                             follow_redirects=False)
        interview_id = r.headers["location"].rsplit("/", 1)[-1]
        page = self.client.get(f"/interviews/{interview_id}").text
        self.assertNotIn("Dana Example", page)
        self.assertRegex(page, r"P-[0-9a-f]{6}")

    def test_finalising_stores_a_retrievable_transcript(self):
        tid = self._run_simulated_interview()
        self.assertIn(tid, [p.name.split(".")[0]
                            for p in (self.data_dir / "transcripts").iterdir()])
        self.assertEqual(self.client.get(f"/transcripts/{tid}").status_code, 200)

    # -- review + validation ----------------------------------------------
    def test_review_shows_evidence_before_the_controls(self):
        tid = self._run_simulated_interview()
        page = self.client.get(f"/transcripts/{tid}/review").text
        self.assertEqual(page.count("awaiting review") > 0, True)
        # The quote must appear before the Accept button for each claim.
        first_quote = page.index('class="quote')
        first_accept = page.index('value="accepted"')
        self.assertLess(first_quote, first_accept)

    def test_accepting_a_claim_is_recorded_and_shown(self):
        tid = self._run_simulated_interview()
        page = self.client.get(f"/transcripts/{tid}/review").text
        claim_id = re.search(r'name="claim_id" value="(clm-[0-9a-f]+)"', page).group(1)
        self.client.post(f"/transcripts/{tid}/validate",
                         data={"claim_id": claim_id, "verdict": "accepted",
                               "reason": "", "statement": ""},
                         follow_redirects=False)
        verdicts = read_verdicts(self.data_dir, tid)
        self.assertIn(claim_id, verdicts)
        self.assertEqual(verdicts[claim_id].verdict, "accepted")
        self.assertTrue(verdicts[claim_id].is_human)
        self.assertIn("ACCEPTED", self.client.get(f"/transcripts/{tid}/review").text)

    def test_rejecting_without_a_reason_is_refused_by_the_gate(self):
        tid = self._run_simulated_interview()
        page = self.client.get(f"/transcripts/{tid}/review").text
        claim_id = re.search(r'name="claim_id" value="(clm-[0-9a-f]+)"', page).group(1)
        self.client.post(f"/transcripts/{tid}/validate",
                         data={"claim_id": claim_id, "verdict": "rejected",
                               "reason": "   ", "statement": ""},
                         follow_redirects=False)
        # The gate requires a reason, so nothing is recorded.
        self.assertNotIn(claim_id, read_verdicts(self.data_dir, tid))

    def test_verdicts_survive_a_new_app_instance(self):
        # The point of content-addressed claim ids.
        tid = self._run_simulated_interview()
        page = self.client.get(f"/transcripts/{tid}/review").text
        claim_id = re.search(r'name="claim_id" value="(clm-[0-9a-f]+)"', page).group(1)
        self.client.post(f"/transcripts/{tid}/validate",
                         data={"claim_id": claim_id, "verdict": "accepted",
                               "reason": "", "statement": ""},
                         follow_redirects=False)
        from ai_engine.webapp.app import create_app

        fresh = TestClient(create_app(self.settings))
        self.assertIn("ACCEPTED", fresh.get(f"/transcripts/{tid}/review").text)

    # -- documents ---------------------------------------------------------
    def test_consultant_report_contains_only_validated_findings(self):
        tid = self._run_simulated_interview()
        before = self.client.get(f"/transcripts/{tid}/report").text
        self.assertIn("No validated findings", before)

        page = self.client.get(f"/transcripts/{tid}/review").text
        claim_id = re.search(r'name="claim_id" value="(clm-[0-9a-f]+)"', page).group(1)
        self.client.post(f"/transcripts/{tid}/validate",
                         data={"claim_id": claim_id, "verdict": "accepted",
                               "reason": "", "statement": ""},
                         follow_redirects=False)
        after = self.client.get(f"/transcripts/{tid}/report").text
        self.assertIn("verdict accepted", after)
        self.assertIn("open in transcript", after)
        self.assertNotIn("No validated findings", after)

    def test_employer_release_leaks_no_verbatim_testimony(self):
        """The invariant that matters most in the whole app."""
        tid = self._run_simulated_interview()
        page = self.client.get(f"/transcripts/{tid}/review").text
        for claim_id in set(re.findall(r'name="claim_id" value="(clm-[0-9a-f]+)"', page)):
            self.client.post(f"/transcripts/{tid}/validate",
                             data={"claim_id": claim_id, "verdict": "accepted",
                                   "reason": "", "statement": ""},
                             follow_redirects=False)

        transcript = self.settings.__class__(
            api_key=None, store_key=None, runtime=Runtime.DEV, data_dir=self.data_dir)
        from ai_engine.persistence.transcript_store import TranscriptStore
        from ai_engine.transcript.model import Speaker

        stored = TranscriptStore(self.data_dir).load(tid)
        utterances = [s.text for s in stored.segments
                      if s.speaker is Speaker.SUBJECT and len(s.text.split()) >= 6]
        self.assertGreater(len(utterances), 0)

        employer = " ".join(self.client.get(f"/transcripts/{tid}/employer").text.lower().split())
        for utterance in utterances:
            words = utterance.lower().split()
            for i in range(len(words) - 4):
                phrase = " ".join(words[i:i + 5])
                with self.subTest(phrase=phrase):
                    self.assertNotIn(phrase, employer)

    def test_engagement_report_is_the_employer_view(self):
        self._run_simulated_interview()
        page = self.client.get("/engagement").text
        self.assertEqual(self.client.get("/engagement").status_code, 200)
        self.assertIn("Employer view", page)

    def test_no_employer_login_surface_exists(self):
        # Consultant-only by construction: the employer gets a document, not access.
        for path in ("/login", "/employer", "/admin"):
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_unknown_transcript_is_a_clean_404(self):
        self.assertEqual(self.client.get("/transcripts/txn-nope/review").status_code, 404)

    def test_testimony_is_escaped_not_injected(self):
        r = self.client.post("/interviews/new",
                             data={"participant": "X", "mode": "manual"},
                             follow_redirects=False)
        interview_id = r.headers["location"].rsplit("/", 1)[-1]
        self.client.post(f"/interviews/{interview_id}/answer",
                         data={"answer": "<script>alert('x')</script> is my workaround"},
                         follow_redirects=False)
        page = self.client.get(f"/interviews/{interview_id}").text
        self.assertNotIn("<script>alert", page)
        self.assertIn("&lt;script&gt;", page)


@unittest.skipUnless(_HAS_CLIENT, "requires fastapi and httpx")
class LazyExportTest(unittest.TestCase):
    """The counterpart to the zero-dependency import contract.

    ``ai_engine.webapp`` resolves ``create_app`` lazily so the package imports without
    FastAPI. That must not break the public name.
    """

    def test_create_app_is_reachable_from_the_package(self):
        from ai_engine.webapp import create_app

        self.assertTrue(callable(create_app))

    def test_unknown_attribute_still_raises_attribute_error(self):
        import ai_engine.webapp as pkg

        with self.assertRaises(AttributeError):
            pkg.no_such_thing


@unittest.skipUnless(_HAS_CLIENT, "requires fastapi and httpx")
class ProductionGuardTest(unittest.TestCase):
    def test_app_refuses_to_start_in_unsafe_production(self):
        from ai_engine.config import ConfigurationError
        from ai_engine.webapp.app import create_app

        with self.assertRaises(ConfigurationError):
            create_app(Settings(api_key=None, store_key=None,
                                runtime=Runtime.PRODUCTION))


if __name__ == "__main__":
    unittest.main()


class ProvenanceUXTest(unittest.TestCase):
    """The demo spine: a claim clicks through to the exact verbatim moment."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        from ai_engine.webapp.app import create_app

        settings = Settings(api_key=None, store_key=None, runtime=Runtime.DEV,
                            provider="", data_dir=self.data_dir)
        self.client = TestClient(create_app(settings))

    def _simulated_transcript(self, client) -> str:
        r = client.post("/interviews/new",
                        data={"participant": "Dana Example", "mode": "simulated"},
                        follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        interview_id = r.headers["location"].rsplit("/", 1)[-1]
        client.post(f"/interviews/{interview_id}/finish", follow_redirects=False)
        return interview_id

    def test_review_quotes_link_to_the_transcript_anchor(self):
        tid = self._simulated_transcript(self.client)
        page = self.client.get(f"/transcripts/{tid}/review").text
        self.assertIn("#seg-", page, "review quotes must deep-link to the transcript")
        self.assertIn("gate 1 grounded", page)

    def test_transcript_page_anchors_every_segment(self):
        tid = self._simulated_transcript(self.client)
        page = self.client.get(f"/transcripts/{tid}").text
        self.assertIn('id="seg-', page)
        self.assertIn(".seg:target", page, "the highlight must exist for anchors")

    def test_report_page_shows_what_the_gates_admitted_and_refused(self):
        tid = self._simulated_transcript(self.client)
        page = self.client.get(f"/transcripts/{tid}/report").text
        self.assertIn("Grounded by construction", page)
        self.assertIn("rejected as unsourced", page)


class DemoRunTest(unittest.TestCase):
    """POST /demo/run — the speedrun demo: one click, a full engagement."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        from ai_engine.webapp.app import create_app

        settings = Settings(api_key=None, store_key=None, runtime=Runtime.DEV,
                            provider="", data_dir=self.data_dir)
        self.client = TestClient(create_app(settings))

    def test_the_demo_runs_end_to_end_and_lands_on_both_deliverables(self):
        r = self.client.post("/demo/run", follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        url = r.headers["location"]
        self.assertIn("/engagements/", url)
        self.assertIn("/deliverables", url)

        page = self.client.get(url)
        self.assertEqual(page.status_code, 200)
        # The contrast IS the page: consultant and employer, side by side.
        self.assertIn("Consultant", page.text)
        self.assertIn("Employer", page.text)
        self.assertIn("k-anonymity", page.text)
        self.assertIn("auto-sim", page.text,
                      "the page must say which verdicts were machine-made")

        # The engagement shows up on the dashboard with its interviews.
        dash = self.client.get("/").text
        self.assertIn("Speedrun demo", dash)

    def test_engagement_level_employer_route_renders_the_firewall(self):
        r = self.client.post("/demo/run", follow_redirects=False)
        eid = r.headers["location"].split("/")[2]
        page = self.client.get(f"/engagements/{eid}/employer")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Employer release", page.text)
        self.assertIn("withheld", page.text)

        # The firewall invariant, over the WHOLE payload: no verbatim testimony.
        from ai_engine.persistence.transcript_store import TranscriptStore

        store = TranscriptStore(self.data_dir)
        for tid in store.list_ids():
            t = store.load(tid)
            for seg in t.segments:
                if seg.speaker.value == "subject" and len(seg.text) > 25:
                    self.assertNotIn(seg.text, page.text)

    def test_unknown_engagement_is_a_404_not_a_blank_document(self):
        self.assertEqual(
            self.client.get("/engagements/eng-nothing/employer").status_code, 404)
        self.assertEqual(
            self.client.get("/engagements/eng-nothing/deliverables").status_code, 404)
