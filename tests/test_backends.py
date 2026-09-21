import os
import pytest
from target_loop.backends import parse_gene_list, cosine


def test_parse_gene_list():
    assert parse_gene_list("CD80, CD86") == ["CD80", "CD86"]
    assert parse_gene_list("The targets are TNF and IL6R.") == ["TNF", "IL6R"]
    assert parse_gene_list("NONE") == []
    assert parse_gene_list("S4") == []
    assert parse_gene_list("HLA-DRB1, TNF", exclude=["TNF"]) == ["HLA-DRB1"]
    assert parse_gene_list("Genes at this stage: IL6, IL17A and CSF2 are cytokines.") == ["IL6", "IL17A", "CSF2"]
    assert parse_gene_list("TNF, FOO", known_symbols={"TNF"}) == ["TNF"]


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0


TINY = os.environ.get("TINY_GGUF", "outputs/tiny_random.gguf")


@pytest.mark.skipif(not os.path.exists(TINY), reason="tiny GGUF not built (run scripts/make_tiny_gguf.py)")
def test_llama_cpp_backend_smoke():
    """Random-weight model: values are meaningless, but the code path (tokenize,
    eval, logits row, Yes/No aggregation, order swap, generate) must run."""
    from target_loop.backends import LlamaCppBackend
    b = LlamaCppBackend(TINY, n_ctx=128, n_gpu_layers=0, chat_wrap=False)
    p1 = b.yes_probability("Is TNF a drug target for rheumatoid arthritis?", "yes_first")
    p2 = b.yes_probability("Is TNF a drug target for rheumatoid arthritis?", "no_first")
    assert 0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0
    # deterministic: same prompt -> same probability
    assert b.yes_probability("Is TNF a drug target for rheumatoid arthritis?", "yes_first") == pytest.approx(p1)
    txt = b.generate("List genes:", max_tokens=4)
    assert isinstance(txt, str)
    b.close()
