from evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank, token_f1


def test_recall_and_precision():
    assert recall_at_k(["a", "b", "c"], ["b", "d"], 2) == 0.5
    assert precision_at_k(["a", "b", "c"], ["b"], 2) == 0.5


def test_mrr():
    assert reciprocal_rank(["x", "gold"], ["gold"]) == 0.5
    assert reciprocal_rank(["x"], ["gold"]) == 0.0


def test_token_f1_identical():
    assert token_f1("vacation hours carry over", "vacation hours carry over") == 1.0
