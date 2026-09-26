"""The evaluation laboratory: schemas, dataset, rubrics, runner, regression.

The lab's own honesty rules are what these tests pin:
  * deterministic graders are stable and fail when the behavior is wrong;
  * heuristic graders label themselves as such in their evidence;
  * model graders report UNSCORED without a live provider — never a fake score;
  * a regression is detected exactly when behavior drifts, not when latency does.
"""
import json
import tempfile
import unittest
from pathlib import Path

from ai_engine.config import Runtime, Settings
from ai_engine.lab.dataset import load_dataset, load_golden
from ai_engine.lab.regression import compare
from ai_engine.lab.runner import run_suite, run_turn_case
from ai_engine.lab.schemas import (LAB_SCHEMA, Scenario, TurnEvaluation,
                                   load_results, write_results)


def _settings(data_dir: Path) -> Settings:
    return Settings(api_key=None, store_key=None, runtime=Runtime.DEV,
                            provider="",
                    data_dir=data_dir)


class DatasetTest(unittest.TestCase):
    def test_golden_dataset_loads_and_validates(self):
        version, scenarios = load_golden()
        self.assertEqual(version, "golden-v1")
        self.assertGreaterEqual(len(scenarios), 13)
        # load_golden gates on kind == "golden" only; edge/adversarial records
        # in the file are measurable but never fail the build.
        self.assertEqual({s.kind for s in scenarios}, {"golden"})
        units = {s.unit for s in scenarios}
        self.assertEqual(units, {"turn", "gate", "report", "full"})

    def test_the_raw_dataset_still_carries_edge_and_adversarial_cases(self):
        from ai_engine.lab.dataset import DATASET_SCHEMA

        raw = json.loads((Path("ai_engine/lab/datasets/golden.json")
                          .read_text(encoding="utf-8")))
        self.assertEqual(raw["schema"], DATASET_SCHEMA)
        kinds = {s["kind"] for s in raw["scenarios"]}
        self.assertIn("adversarial", kinds)
        self.assertIn("edge", kinds)

    def test_malformed_records_are_refused_not_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.json"
            path.write_text(json.dumps({
                "schema": "groundwork.lab.dataset/v1", "version": "x",
                "scenarios": [{"id": "s1", "title": "t", "kind": "golden",
                               "dimension": "reasoning", "candidate_answer": "",
                               "criteria": [], "bogus_field": 1}]}), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_dataset(path)
            self.assertIn("bogus_field", str(ctx.exception))

    def test_duplicate_scenario_ids_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.json"
            rec = {"id": "s1", "title": "t", "kind": "golden",
                   "dimension": "reasoning", "candidate_answer": "",
                   "criteria": []}
            path.write_text(json.dumps({
                "schema": "groundwork.lab.dataset/v1", "version": "x",
                "scenarios": [rec, rec]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(path)


class RubricTest(unittest.TestCase):
    def test_every_dimension_is_covered_by_at_least_one_rubric(self):
        from ai_engine.lab.rubrics import RUBRICS

        covered = {spec.dimension for spec in RUBRICS.values()}
        for dimension in ("relevance", "correctness", "reasoning", "communication",
                          "consistency", "interview_quality", "feedback_quality"):
            self.assertIn(dimension, covered, f"{dimension} has no rubric")

    def test_model_graders_report_unscored_never_a_score(self):
        from ai_engine.lab.rubrics import RUBRICS, TurnContext, Unscored

        for spec in RUBRICS.values():
            if spec.grader_class != "model":
                continue
            ctx = TurnContext(scenario=None, utterance="x", state=None,
                              move=None, transcript=None)
            with self.assertRaises(Unscored):
                spec.fn(ctx, {})


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.settings = _settings(Path(self._tmp.name))
        _version, self.scenarios = load_golden()
        self.by_id = {s.id: s for s in self.scenarios}

    def test_the_whole_golden_suite_passes_in_mock_mode(self):
        result = run_suite(self.scenarios, self.settings, suite_id="golden",
                           repeats=2)
        failed = [(e.scenario_id, [r.rubric_id for r in e.results if r.status == "fail"])
                  for e in result.evaluations if e.total_score is not None
                  and e.total_score < 1.0]
        self.assertEqual(failed, [], "golden scenarios must pass in mock mode")
        # Every scenario ran twice with identical output (deterministic engine).
        for v in result.variance:
            self.assertEqual(v.distinct_outputs, 1,
                             f"{v.scenario_id} is not deterministic in mock mode")

    def test_results_round_trip_through_the_stored_schema(self):
        _version, scenarios = load_golden()
        result = run_suite(scenarios[:3], self.settings, suite_id="golden")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            write_results(result.evaluations, path)
            loaded = load_results(path)
        self.assertEqual([e.to_dict() for e in result.evaluations],
                         [e.to_dict() for e in loaded])

    def test_a_wrong_behavior_is_caught_by_the_relevance_grader(self):
        # Run the guarded-deferral scenario but grade it against the WRONG
        # expected area — the grader must fail, proving it is not tautological.
        scenario = self.by_id["golden-guarded-deferral"]
        correctness = next(c for c in scenario.criteria
                           if c.rubric_id == "correctness/state")
        correctness.params["pending_area"] = "ai_opportunity"
        runs = run_turn_case(scenario, self.settings)
        correctness = next(r for r in runs[0].results
                           if r.rubric_id == "correctness/state")
        self.assertEqual(correctness.status, "fail")
        self.assertLess(correctness.score, 1.0)

    def test_repeatability_captures_latency_and_output_hashes(self):
        scenario = self.by_id["golden-open-intent"]
        runs = run_turn_case(scenario, self.settings, repeats=3)
        self.assertEqual(len(runs), 3)
        self.assertEqual(len({r.output_sha for r in runs}), 1)
        for r in runs:
            self.assertGreaterEqual(r.latency_ms, 0.0)


class RegressionTest(unittest.TestCase):
    def _eval(self, scenario_id, total, statuses, error=None, sha="aaaa"):
        evaluation = TurnEvaluation(
            scenario_id=scenario_id, suite_id="golden",
            rubric_version="lab.rubric/v1", mode="mock",
            produced_utterance="x", results=[], total_score=total,
            latency_ms=1.0, output_sha=sha, error=error)
        from ai_engine.lab.schemas import CriterionResult

        for rubric_id, status in statuses.items():
            evaluation.results.append(CriterionResult(
                rubric_id=rubric_id, dimension="reasoning",
                grader_class="deterministic",
                score=1.0 if status == "pass" else (0.0 if status == "fail" else None),
                status=status, evidence=""))
        return evaluation

    def test_score_drop_is_a_regression(self):
        baseline = [self._eval("s1", 1.0, {"r": "pass"})]
        current = [self._eval("s1", 0.5, {"r": "fail"})]
        report = compare(baseline, current)
        self.assertTrue(report.has_regression)
        self.assertIn("score-drop", report.summary())

    def test_latency_change_is_never_a_regression(self):
        baseline = [self._eval("s1", 1.0, {"r": "pass"})]
        slow = self._eval("s1", 1.0, {"r": "pass"})
        slow.latency_ms = 99999.0
        report = compare(baseline, [slow])
        self.assertFalse(report.has_regression)

    def test_new_failure_is_a_regression(self):
        baseline = [self._eval("s1", 1.0, {"r": "pass"})]
        current = [self._eval("s1", None, {}, error="LLMUnavailable: down")]
        report = compare(baseline, current)
        self.assertTrue(report.has_regression)
        self.assertIn("failure-new", report.summary())

    def test_variance_appearing_is_a_regression(self):
        baseline = [self._eval("s1", 1.0, {"r": "pass"})]
        current = [self._eval("s1", 1.0, {"r": "pass"}, sha="a"),
                   self._eval("s1", 1.0, {"r": "pass"}, sha="b")]
        report = compare(baseline, current)
        self.assertTrue(report.has_regression)
        self.assertIn("variance-appeared", report.summary())

    def test_identical_runs_are_not_a_regression(self):
        baseline = [self._eval("s1", 1.0, {"r": "pass"})]
        current = [self._eval("s1", 1.0, {"r": "pass"})]
        self.assertFalse(compare(baseline, current).has_regression)


if __name__ == "__main__":
    unittest.main()
