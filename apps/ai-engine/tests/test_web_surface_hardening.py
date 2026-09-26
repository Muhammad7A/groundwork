"""Hardening of the two web surfaces.

Each test pins a failure mode an unattended interview could actually hit: one
transient model failure bricking the consultant's loop, an oversized paste
wedging a participant's interview permanently, a withdrawal that lied about a
completed interview, a route parameter escaping the transcript directory, and a
bad model id surfacing at the participant's first question instead of startup.
"""
import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    _HAS_CLIENT = True
except Exception:  # pragma: no cover
    _HAS_CLIENT = False

from ai_engine.config import ModelUnavailable, Runtime, Settings, preflight_model
from ai_engine.llm.retry import LLMUnavailable
from ai_engine.persistence.invitations import InvitationStatus, InvitationStore
from ai_engine.persistence.transcript_store import TranscriptStore


def _settings(data_dir: Path, **kw) -> Settings:
    kw.setdefault("api_key", None)
    kw.setdefault("store_key", None)
    kw.setdefault("runtime", Runtime.DEV)
    kw.setdefault("provider", "")  # hermetic: a developer's live-credential env must not flip tests to live
    return Settings(data_dir=data_dir, **kw)


class _FailingLLM:
    def complete(self, **kwargs):
        raise LLMUnavailable("the model is down", None)


class _FlakyEngine:
    """An engine whose model dies exactly once — on the second turn.

    Installed as a subclass so the route behaviour — pause, recover, continue —
    is what is under test, against the real interview engine otherwise.
    """

    def __init__(self, *args, **kwargs):
        import ai_engine.interview.engine as engine_mod

        self._inner = engine_mod.InterviewEngine(*args, **kwargs)
        self._calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def next_turn(self, **kwargs):
        self._calls += 1
        if self._calls in (2, 3):
            # Down for the answer's follow-up AND the first retry; back after.
            raise LLMUnavailable("the model is down", None)
        return self._inner.next_turn(**kwargs)


@unittest.skipUnless(_HAS_CLIENT, "requires fastapi and httpx")
class ConsultantModelFailureTest(unittest.TestCase):
    """One transient model failure must pause the interview, not brick it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        # The use-cases live in the application layer now — that is the seam to
        # patch; the webapp only parses forms and renders HTML.
        import ai_engine.application.service as service_mod

        self._module = service_mod
        self._original_client = service_mod.get_llm_client
        self._original_engine = service_mod.InterviewEngine
        self.addCleanup(self._restore)

    def _restore(self):
        self._module.get_llm_client = self._original_client
        self._module.InterviewEngine = self._original_engine

    def test_start_survives_a_dead_model(self):
        self._module.get_llm_client = lambda s: _FailingLLM()
        from ai_engine.webapp.app import create_app

        client = TestClient(create_app(_settings(self.data_dir)))
        r = client.post("/interviews/new", data={"participant": "Dana", "mode": "manual"},
                        follow_redirects=False)
        self.assertEqual(r.status_code, 503)
        self.assertIn("temporarily unavailable", r.text)

    def test_a_failed_next_question_pauses_and_recovers(self):
        from ai_engine.webapp.app import create_app

        self._module.InterviewEngine = _FlakyEngine
        client = TestClient(create_app(_settings(self.data_dir)))
        r = client.post("/interviews/new", data={"participant": "Dana", "mode": "manual"},
                        follow_redirects=False)
        interview_id = r.headers["location"].rsplit("/", 1)[-1]

        # The next model call fails, but the answer was already recorded.
        r = client.post(f"/interviews/{interview_id}/answer",
                        data={"answer": "I keep a private spreadsheet, honestly."},
                        follow_redirects=False)
        self.assertEqual(r.status_code, 303)

        # The interview page reports a recoverable pause, not a 500.
        page = client.get(f"/interviews/{interview_id}")
        self.assertEqual(page.status_code, 503)
        self.assertIn("intact", page.text)

        # The model is back (transient failure): the same page continues —
        # nothing re-entered, nothing asked twice.
        page = client.get(f"/interviews/{interview_id}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("private spreadsheet", page.text)


@unittest.skipUnless(_HAS_CLIENT, "requires fastapi and httpx")
class AnswerLengthCapTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)

    def test_an_oversized_consultant_answer_is_refused(self):
        from ai_engine.webapp.app import create_app

        client = TestClient(create_app(_settings(self.data_dir, max_answer_chars=100)))
        r = client.post("/interviews/new", data={"participant": "D", "mode": "manual"},
                        follow_redirects=False)
        interview_id = r.headers["location"].rsplit("/", 1)[-1]
        r = client.post(f"/interviews/{interview_id}/answer",
                        data={"answer": "word " * 40}, follow_redirects=False)
        self.assertEqual(r.status_code, 413)
        page = client.get(f"/interviews/{interview_id}")
        self.assertNotIn("word " * 10, page.text, "the oversized answer was recorded")

    def test_an_oversized_participant_answer_is_refused(self):
        from ai_engine.employee.app import create_employee_app

        invitations = InvitationStore(self.data_dir)
        token = invitations.create(pseudonym="P-abc123def456").token
        client = TestClient(create_employee_app(
            _settings(self.data_dir, max_answer_chars=50)))
        client.post(f"/i/{token}/begin", follow_redirects=False)
        r = client.post(f"/i/{token}/answer", data={"answer": "word " * 20})
        self.assertEqual(r.status_code, 413)
        self.assertIn("not recorded", r.text)


@unittest.skipUnless(_HAS_CLIENT, "requires fastapi and httpx")
class WithdrawAfterCompleteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        self.invitations = InvitationStore(self.data_dir)
        self.token = self.invitations.create(pseudonym="P-abc123def456").token
        from ai_engine.employee.app import create_employee_app

        self.client = TestClient(create_employee_app(_settings(self.data_dir)))

    def test_a_completed_interview_cannot_be_withdrawn_into_a_lie(self):
        self.client.post(f"/i/{self.token}/begin", follow_redirects=False)
        self.client.post(f"/i/{self.token}/answer",
                         data={"answer": "I keep a private spreadsheet, honestly."})
        self.client.post(f"/i/{self.token}/finish")

        r = self.client.post(f"/i/{self.token}/withdraw")
        self.assertEqual(r.status_code, 409)
        self.assertIn("already complete", r.text)
        self.assertEqual(self.invitations.get(self.token).status,
                         InvitationStatus.COMPLETED.value)
        transcript_id = self.invitations.get(self.token).transcript_id
        self.assertTrue((self.data_dir / "transcripts").exists())


class TranscriptIdTraversalTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TranscriptStore(Path(self._tmp.name))

    def test_separator_ids_are_refused(self):
        for bad in ("../escape", "a/b", "a\\b", ".hidden", "..", ""):
            with self.subTest(transcript_id=bad):
                with self.assertRaises(ValueError):
                    self.store.load(bad)
                with self.assertRaises(ValueError):
                    self.store.exists(bad)

    @unittest.skipUnless(_HAS_CLIENT, "requires fastapi and httpx")
    def test_a_traversal_route_parameter_is_a_404_not_an_escape(self):
        from ai_engine.webapp.app import create_app

        client = TestClient(create_app(_settings(Path(self._tmp.name))))
        r = client.get("/transcripts/%5C..%5C..%5Csecrets")
        self.assertEqual(r.status_code, 404)


class PreflightAtStartupTest(unittest.TestCase):
    def test_mock_mode_preflights_as_mock(self):
        settings = Settings(api_key=None, store_key=None, runtime=Runtime.DEV)
        self.assertEqual(preflight_model(settings), "mock")

    def test_an_unreachable_model_fails_preflight_with_the_model_named(self):
        import ai_engine.llm.client as llm_client_mod

        class _Dead:
            def __init__(self, **kwargs):
                pass

            def complete(self, **kwargs):
                raise LLMUnavailable("404 not found")

        original = llm_client_mod.AnthropicClient
        llm_client_mod.AnthropicClient = _Dead
        try:
            settings = Settings(api_key="sk-test", store_key="k", runtime=Runtime.DEV)
            with self.assertRaises(ModelUnavailable) as ctx:
                preflight_model(settings)
            self.assertIn("did not answer", str(ctx.exception))
        finally:
            llm_client_mod.AnthropicClient = original


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(_HAS_CLIENT, "requires fastapi and httpx")
class OperatorAuthTest(unittest.TestCase):
    """GROUNDWORK_OPERATOR_PASSWORD turns on HTTP Basic for every route."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        from ai_engine.webapp.app import create_app

        self._create_app = create_app

    def test_open_by_default_in_dev(self):
        client = TestClient(self._create_app(_settings(self.data_dir)))
        self.assertEqual(client.get("/").status_code, 200)

    def test_password_set_demands_basic_auth(self):
        client = TestClient(self._create_app(
            _settings(self.data_dir, operator_password="s3cret")))
        r = client.get("/")
        self.assertEqual(r.status_code, 401)
        self.assertIn("WWW-Authenticate", r.headers)
        self.assertIn("testimony", r.text)

        # Browsers prompt natively via WWW-Authenticate; the right password passes.
        ok = TestClient(self._create_app(
            _settings(self.data_dir, operator_password="s3cret"))).get(
            "/", auth=("op", "s3cret"))
        self.assertEqual(ok.status_code, 200)
        # and a wrong password still fails
        r = TestClient(self._create_app(
            _settings(self.data_dir, operator_password="s3cret"))).get(
            "/", auth=("op", "wrong"))
        self.assertEqual(r.status_code, 401)

    def test_production_refuses_to_start_without_a_password(self):
        from ai_engine.config import ConfigurationError

        s = _settings(self.data_dir, operator_password=None,
                      api_key="sk-test", store_key="k",
                      runtime=Runtime.PRODUCTION)
        with self.assertRaises(ConfigurationError) as ctx:
            self._create_app(s)
        self.assertIn("OPERATOR_PASSWORD", str(ctx.exception))
