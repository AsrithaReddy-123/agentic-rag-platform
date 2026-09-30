from app.guardrails import UNSUPPORTED_CLAIM, groundedness, is_supported


def test_extractive_answer_is_supported():
    evidence = "People employed in Germany may carry over a maximum of 40 hours of unused vacation."
    assert is_supported(evidence, evidence)


def test_added_claim_is_not_supported():
    evidence = "People employed in Germany may carry over a maximum of 40 hours of unused vacation."
    answer = evidence + UNSUPPORTED_CLAIM
    assert groundedness(answer, evidence) < 0.85
    assert not is_supported(answer, evidence)
