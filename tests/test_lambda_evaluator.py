"""T16 — custom code evaluator as a Lambda handler (SPEC §6.2)."""

from saes.evaluators import CodeVerdict, code_evaluator
from saes.online.lambda_evaluator import handle


def test_lambda_dispatches_to_registered_evaluator():
    @code_evaluator(id="lambda_paystub", level="trace")
    def check(case) -> CodeVerdict:
        ok = "$8,333.33" in str(case.actual_output)
        return CodeVerdict(
            score=1.0 if ok else 0.0,
            label="PASS" if ok else "FAIL",
            reason="amount present" if ok else "amount missing",
        )

    resp = handle(
        {"evaluator_id": "lambda_paystub", "actual_output": "Your net pay is $8,333.33."}
    )
    assert resp == {"score": 1.0, "label": "PASS", "reason": "amount present", "test_pass": True}


def test_lambda_negative_case():
    @code_evaluator(id="lambda_contains")
    def check(case) -> CodeVerdict:
        ok = "ok" in str(case.actual_output).lower()
        return CodeVerdict(1.0 if ok else 0.0, "PASS" if ok else "FAIL")

    resp = handle({"evaluator_id": "lambda_contains", "actual_output": "nope"})
    assert resp["score"] == 0.0
    assert resp["test_pass"] is False


def test_lambda_accepts_output_alias():
    @code_evaluator(id="lambda_alias")
    def check(case) -> CodeVerdict:
        return CodeVerdict(1.0 if case.actual_output == "hi" else 0.0)

    # event uses "output" instead of "actual_output"
    resp = handle({"evaluator_id": "lambda_alias", "output": "hi"})
    assert resp["score"] == 1.0


def test_lambda_missing_evaluator_id():
    resp = handle({"actual_output": "x"})
    assert resp["label"] == "ERROR"
    assert "evaluator_id" in resp["reason"]


def test_lambda_unregistered_id():
    resp = handle({"evaluator_id": "never_registered_lambda"})
    assert resp["label"] == "ERROR"
    assert "not registered" in resp["reason"]


def test_lambda_evaluator_exception_is_reported_not_raised():
    @code_evaluator(id="lambda_boom")
    def check(case) -> CodeVerdict:
        raise ValueError("kaboom")

    resp = handle({"evaluator_id": "lambda_boom", "actual_output": "x"})
    assert resp["label"] == "ERROR"
    assert "kaboom" in resp["reason"]
    assert resp["test_pass"] is False


def test_same_function_body_works_locally_and_as_lambda():
    """The point of T16: one function, two runtimes."""
    from strands_evals.types.evaluation import EvaluationData

    from saes.config.schema import EvaluatorRef
    from saes.evaluators import resolve_evaluator

    @code_evaluator(id="dual_runtime")
    def check(case) -> CodeVerdict:
        return CodeVerdict(1.0 if "yes" in str(case.actual_output) else 0.0, reason="r")

    # local path (native Evaluator)
    ev = resolve_evaluator(EvaluatorRef(id="dual_runtime", type="code"), object())
    local = ev.evaluate(EvaluationData(input="q", actual_output="yes please"))
    assert local[0].score == 1.0

    # lambda path (same fn)
    lam = handle({"evaluator_id": "dual_runtime", "actual_output": "yes please"})
    assert lam["score"] == 1.0
