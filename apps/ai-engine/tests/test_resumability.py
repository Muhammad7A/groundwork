"""Resumable interviews, without weakening the right to withdraw.

Two guarantees that pull against each other, and both must hold:

  * A dropped connection, a slept phone, or a restarted process must not cost the
    participant fifteen minutes.
  * Withdrawing must still leave nothing behind.

The resolution is to persist drafts under the same guarantees as everything else —
encrypted, never treated as a consented record, deleted on withdrawal, swept when
abandoned — rather than to refuse to persist at all.
"""
import base64
import re
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    _HAS_CLIENT = True
except Exception:  # pragma: no cover
    _HAS_CLIENT = False

from ai_engine.config import Runtime, Settings
from ai_engine.interview.driver import InterviewDriver
from ai_engine.interview.engine import InterviewEngine
from ai_engine.persistence.event_log import EventLog, NullEventLog
from ai_engine.persistence.invitations import InvitationStatus, InvitationStore
from ai_engine.persistence.session_store import (
    DEFAULT_TTL,
    SavedSession,
    SessionStore,
)
from ai_engine.transcript.model import Speaker, Transcript

SECRET = "The director's approval adds three days and everyone routes around it."


class _StubCipher:
    """A real reversible transformation, so the encrypted path runs everywhere."""

    name = "stub"

    @property
    def protects_at_rest(self) -> bool:
        return True

    def encrypt(self, plaintext: bytes) -> bytes:
        return base64.b64encode(plaintext[::-1])

    def decrypt(self, blob: bytes) -> bytes:
        return base64.b64decode(blob)[::-1]


def _driver(max_turns: int = 14) -> InterviewDriver:
    return InterviewDriver(engine=InterviewEngine(llm=None, max_turns=max_turns),
                           transcript=Transcript(), objective="test",
                           event_log=NullEventLog(), max_turns=max_turns)


class SessionStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _saved(self, driver, token="tok") -> SavedSession:
        return SavedSession(token=token, transcript=driver.transcript,
                            pending_question=driver.pending_question,
                            turn_count=driver.state.turn_count,
                            closed=driver.closed)

    def test_round_trips_a_draft(self):
        store = SessionStore(self.dir)
        driver = _driver()
        driver.next_question()
        driver.submit_answer(SECRET)
        store.save(self._saved(driver))

        loaded = store.load("tok")
        self.assertIsNotNone(loaded)
        self.assertIn(SECRET, loaded.transcript.render())
        self.assertFalse(loaded.transcript.finalized,
                         "a draft must resume as unfinalized")

    def test_draft_is_encrypted_when_a_cipher_is_configured(self):
        store = SessionStore(self.dir, _StubCipher())
        driver = _driver()
        driver.next_question()
        driver.submit_answer(SECRET)
        path = store.save(self._saved(driver))
        self.assertNotIn(SECRET, path.read_text(encoding="utf-8", errors="ignore"))
        self.assertIn(SECRET, store.load("tok").transcript.render())

    def test_delete_removes_everything(self):
        store = SessionStore(self.dir)
        driver = _driver()
        driver.next_question()
        driver.submit_answer(SECRET)
        store.save(self._saved(driver))
        self.assertTrue(store.delete("tok"))
        self.assertIsNone(store.load("tok"))
        for path in self.dir.rglob("*"):
            if path.is_file():
                self.assertNotIn(SECRET, path.read_text(encoding="utf-8", errors="ignore"))

    def test_expired_drafts_are_swept(self):
        store = SessionStore(self.dir)
        driver = _driver()
        driver.next_question()
        store.save(self._saved(driver))
        later = datetime.now(timezone.utc) + DEFAULT_TTL + timedelta(minutes=1)
        self.assertEqual(store.sweep(now=later), 1)
        self.assertIsNone(store.load("tok"))

    def test_fresh_drafts_survive_a_sweep(self):
        store = SessionStore(self.dir)
        driver = _driver()
        driver.next_question()
        store.save(self._saved(driver))
        self.assertEqual(store.sweep(), 0)
        self.assertIsNotNone(store.load("tok"))

    def test_traversal_tokens_are_refused(self):
        store = SessionStore(self.dir)
        for bad in ("../escape", "a/b", ".hidden", ""):
            with self.subTest(token=bad):
                self.assertIsNone(store.load(bad))


class DriverResumeTest(unittest.TestCase):
    def test_resumed_driver_continues_where_it_stopped(self):
        original = _driver()
        q1 = original.next_question()
        original.submit_answer("I keep a private spreadsheet, it takes hours weekly.")
        q2 = original.next_question()

        resumed = InterviewDriver.resume(
            engine=InterviewEngine(llm=None, max_turns=14),
            transcript=original.transcript, objective="test",
            pending_question=q2, turn_count=original.state.turn_count,
            event_log=NullEventLog(), max_turns=14)

        self.assertEqual(resumed.pending_question, q2)
        self.assertEqual(resumed.state.turn_count, original.state.turn_count)
        self.assertEqual(len(resumed.transcript.segments),
                         len(original.transcript.segments))

    def test_coverage_is_reconstructed_not_lost(self):
        # Replayed through the same assessment the engine uses, so a resumed
        # interview cannot diverge from one that never stopped. Asserting only
        # history length once let a resume through that attributed every replayed
        # answer to no area at tier 0 — all six areas "untouched".
        original = _driver()
        original.next_question()
        original.submit_answer("I keep a private spreadsheet, it takes hours weekly.")
        original.next_question()

        resumed = InterviewDriver.resume(
            engine=InterviewEngine(llm=None, max_turns=14),
            transcript=original.transcript, objective="test",
            pending_question=original.pending_question,
            turn_count=original.state.turn_count,
            event_log=NullEventLog(), max_turns=14)
        self.assertEqual(len(resumed.state.history), len(original.state.history))
        self.assertEqual(
            {a: (c.level, c.max_tier) for a, c in resumed.state.coverage.items()},
            {a: (c.level, c.max_tier) for a, c in original.state.coverage.items()},
            "coverage diverged from the interview that never stopped")
        self.assertEqual(resumed.state.pending_area, original.state.pending_area)
        self.assertEqual(resumed.state.pending_tier, original.state.pending_tier)
        self.assertEqual(resumed.state.history[-1].area,
                         original.state.history[-1].area)

    def test_resume_restores_question_targets_from_the_event_log(self):
        # The testimony log records what each asked question targeted; resume must
        # prefer it so attribution survives even for questions the bank cannot
        # classify (live-model wording).
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp), "int-resume-1")
            original = InterviewDriver(
                engine=InterviewEngine(llm=None, max_turns=14),
                transcript=Transcript(), objective="test",
                event_log=log, max_turns=14)
            original.next_question()
            original.submit_answer("I keep a private spreadsheet, it takes hours weekly.")
            original.next_question()

            resumed = InterviewDriver.resume(
                engine=InterviewEngine(llm=None, max_turns=14),
                transcript=original.transcript, objective="test",
                pending_question=original.pending_question,
                turn_count=original.state.turn_count,
                event_log=EventLog(Path(tmp), "int-resume-1"), max_turns=14)
            self.assertEqual(resumed.state.pending_area,
                             original.state.pending_area)
            self.assertEqual(
                {a: (c.level, c.max_tier) for a, c in resumed.state.coverage.items()},
                {a: (c.level, c.max_tier) for a, c in original.state.coverage.items()})

    def test_a_resume_cannot_tell_its_answers_were_replayed(self):
        # The resumed driver must answer its pending question and move on exactly
        # as the never-stopped one would.
        original = _driver()
        q1 = original.next_question()
        original.submit_answer("I keep a private spreadsheet, it takes hours weekly.")
        q2 = original.next_question()

        resumed = InterviewDriver.resume(
            engine=InterviewEngine(llm=None, max_turns=14),
            transcript=original.transcript, objective="test",
            pending_question=q2, turn_count=original.state.turn_count,
            event_log=NullEventLog(), max_turns=14)
        resumed.submit_answer("And the weekly report is rebuilt by hand each Monday.")
        q3_resumed = resumed.next_question()

        never_stopped = original
        never_stopped.submit_answer("And the weekly report is rebuilt by hand each Monday.")
        q3_live = never_stopped.next_question()
        self.assertEqual(q3_resumed, q3_live,
                         "the resumed interview diverged from the uninterrupted one")

    def test_resuming_does_not_re_answer_or_duplicate_turns(self):
        original = _driver()
        original.next_question()
        original.submit_answer("Something substantive about the weekly report cycle.")
        pending = original.next_question()
        before = len(original.transcript.segments)

        resumed = InterviewDriver.resume(
            engine=InterviewEngine(llm=None, max_turns=14),
            transcript=original.transcript, objective="test",
            pending_question=pending, turn_count=original.state.turn_count,
            event_log=NullEventLog(), max_turns=14)
        # Asking again returns the pending question rather than burning a turn.
        self.assertEqual(resumed.next_question(), pending)
        self.assertEqual(len(resumed.transcript.segments), before)


@unittest.skipUnless(_HAS_CLIENT, "requires fastapi and httpx")
class SurfaceResumeTest(unittest.TestCase):
    """The behaviour a participant actually experiences."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data_dir = Path(self._tmp.name)
        self.settings = Settings(api_key=None, store_key=None, runtime=Runtime.DEV,
                            provider="",
                                 data_dir=self.data_dir)
        self.invitations = InvitationStore(self.data_dir)
        self.token = self.invitations.create(pseudonym="P-abc123").token
        self.client = self._new_client()

    def _new_client(self):
        """A brand-new app instance — as if the process had restarted."""
        from ai_engine.employee.app import create_employee_app

        return TestClient(create_employee_app(self.settings))

    def _answer(self, client, text):
        client.post(f"/i/{self.token}/answer", data={"answer": text},
                    follow_redirects=False)

    def test_interview_survives_a_process_restart(self):
        self.client.post(f"/i/{self.token}/begin", follow_redirects=False)
        self._answer(self.client, SECRET)

        restarted = self._new_client()          # no in-memory state at all
        page = restarted.get(f"/i/{self.token}").text
        self.assertIn("Question", page, "the interview did not resume")
        self.assertIn("director", page, "the participant's answer was lost")

    def test_the_participant_can_finish_after_a_restart(self):
        from ai_engine.persistence.transcript_store import TranscriptStore

        self.client.post(f"/i/{self.token}/begin", follow_redirects=False)
        self._answer(self.client, SECRET)

        restarted = self._new_client()
        self._answer(restarted, "And the weekly report is rebuilt by hand each Monday.")
        response = restarted.post(f"/i/{self.token}/finish")
        self.assertIn("Submitted", response.text)

        invitation = self.invitations.get(self.token)
        self.assertEqual(invitation.status, InvitationStatus.COMPLETED.value)
        stored = TranscriptStore(self.data_dir).load(invitation.transcript_id)
        self.assertIn("director", stored.render())

    def test_the_draft_is_removed_once_submitted(self):
        self.client.post(f"/i/{self.token}/begin", follow_redirects=False)
        self._answer(self.client, SECRET)
        self.client.post(f"/i/{self.token}/finish")
        self.assertIsNone(SessionStore(self.data_dir).load(self.token))

    def test_withdrawal_still_leaves_absolutely_nothing(self):
        # The guarantee resumability must not weaken.
        self.client.post(f"/i/{self.token}/begin", follow_redirects=False)
        self._answer(self.client, SECRET)
        self.client.post(f"/i/{self.token}/withdraw")

        self.assertIsNone(SessionStore(self.data_dir).load(self.token))
        transcripts = self.data_dir / "transcripts"
        self.assertEqual(list(transcripts.iterdir()) if transcripts.exists() else [], [])
        for path in self.data_dir.rglob("*"):
            if path.is_file():
                with self.subTest(path=path.name):
                    self.assertNotIn("routes around",
                                     path.read_text(encoding="utf-8", errors="ignore"))

    def test_a_withdrawn_interview_cannot_be_resumed(self):
        self.client.post(f"/i/{self.token}/begin", follow_redirects=False)
        self._answer(self.client, SECRET)
        self.client.post(f"/i/{self.token}/withdraw")
        restarted = self._new_client()
        self.assertEqual(restarted.get(f"/i/{self.token}").status_code, 404)


if __name__ == "__main__":
    unittest.main()
