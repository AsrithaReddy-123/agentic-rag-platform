from evaluation.dataset import build_knowledge_base


def test_corpus_size_and_unique_ids():
    chunks, questions = build_knowledge_base(10000, seed=7)
    assert len(chunks) == 10000
    assert len({chunk.id for chunk in chunks}) == 10000
    lexical = [q for q in questions if q.qtype == "lexical"]
    paraphrase = [q for q in questions if q.qtype == "paraphrase"]
    assert len(lexical) > 200
    assert len(paraphrase) > 200
    by_id = {chunk.id: chunk for chunk in chunks}
    sample = next(q for q in paraphrase)
    gold = by_id[sample.gold_ids[0]]
    pid = gold.metadata["policy_id"]
    assert pid in gold.text
    assert pid not in sample.text
    lexical_q = next(q for q in lexical if q.gold_ids == sample.gold_ids)
    hook = gold.metadata["alert"] or pid
    assert hook in lexical_q.text


def test_bm25_finds_policy_id():
    from app.retrieval.bm25 import BM25Index

    chunks, questions = build_knowledge_base(2000, seed=3)
    index = BM25Index(chunks)
    lexical = next(q for q in questions if q.qtype == "lexical")
    hits = index.search(lexical.text, k=5)
    assert lexical.gold_ids[0] in [hit["id"] for hit in hits]
