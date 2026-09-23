import math
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


@pytest.mark.skipif(not os.path.exists(TINY), reason="tiny GGUF not built")
def test_guidance_backend_smoke():
    """guidance の select + top_k トレースで確率が読めること（値はランダム）。"""
    pytest.importorskip("guidance")
    from target_loop.backends import GuidanceBackend
    b = GuidanceBackend(TINY, top_k=20, n_ctx=128, n_gpu_layers=0)
    lp = b.option_logprobs("Answer Yes or No.\nIs TNF a target?\nAnswer:", [" Yes", " No"])
    assert set(lp) == {"Yes", "No"} and all(v <= 0 for v in lp.values())
    p = b.yes_probability("Is TNF a target?", "yes_first")
    assert 0.0 <= p <= 1.0
    assert isinstance(b.generate("List genes:", max_tokens=4), str)


def test_ollama_backend_parsing(monkeypatch):
    """Ollama の応答の形（chat 形式 / legacy 形式 / 非対応）を正しく読めること（HTTP は差し替え）。"""
    from target_loop.backends import OllamaBackend
    b = OllamaBackend("fake")
    calls = {}
    def fake_post(path, body):
        calls[path] = calls.get(path, 0) + 1
        if path == "/v1/completions":
            return {"choices": [{"logprobs": {"content": [{"token": " Yes", "logprob": -0.2,
                     "top_logprobs": [{"token": " Yes", "logprob": -0.2}, {"token": " No", "logprob": -1.8}]}]}}]}
        return {"response": " Yes"}
    monkeypatch.setattr(b, "_post", fake_post)
    p = b.yes_probability("Is TNF a target?")
    assert abs(p - math.exp(-0.2) / (math.exp(-0.2) + math.exp(-1.8))) < 1e-9 and not b.missing
    monkeypatch.setattr(b, "_post", lambda path, body: {"choices": [{"logprobs": {"top_logprobs": [{" Yes": -1.0, " No": -1.0}]}}]} if path == "/v1/completions" else {"response": "x"})
    assert abs(b.yes_probability("q") - 0.5) < 1e-9
    monkeypatch.setattr(b, "_post", lambda path, body: {"choices": [{"text": " Yes"}]} if path == "/v1/completions" else {"response": " Yes"})
    p = b.yes_probability("q")                       # logprobs 非対応 → サンプリング代用
    assert abs(p - (8 + 0.5) / 9) < 1e-9 and b.missing[-1]["option"].startswith("logprobs unsupported")
    assert b.generate("hi") == "Yes"


def test_ollama_backend_generate_endpoint_logprobs(monkeypatch):
    """/v1/completions が logprobs を返さず /api/generate が返す Ollama でも確率が読めること。"""
    from target_loop.backends import OllamaBackend
    b = OllamaBackend("fake")
    def fake_post(path, body):
        if path == "/v1/completions":
            return {"choices": [{"text": " Yes"}]}
        if body.get("logprobs"):
            return {"response": " Yes", "logprobs": [{"token": " Yes", "logprob": -0.2,
                    "top_logprobs": [{"token": " Yes", "logprob": -0.2}, {"token": " No", "logprob": -1.8}]}]}
        return {"response": " Yes"}
    monkeypatch.setattr(b, "_post", fake_post)
    p = b.yes_probability("q")
    assert abs(p - math.exp(-0.2) / (math.exp(-0.2) + math.exp(-1.8))) < 1e-9 and not b.missing
    assert b.lp_endpoint == "/api/generate"


def test_ollama_backend_caps_top_logprobs_at_20(monkeypatch):
    """top_k=50 でも Ollama には 20 で送る（21 以上は Ollama が拒否する）。/v1/completions では整数で送る。"""
    from target_loop.backends import OllamaBackend
    b = OllamaBackend("fake", top_k=50)
    seen = []
    def fake_post(path, body):
        seen.append((path, body))
        if path == "/api/generate" and body.get("logprobs"):
            if body["top_logprobs"] > 20: raise RuntimeError("400 top_logprobs must be 0-20")
            return {"response": " Yes", "logprobs": [{"token": " Yes", "logprob": -0.1, "top_logprobs": [{"token": " Yes", "logprob": -0.1}, {"token": " No", "logprob": -2.4}]}]}
        return {"response": " Yes"}
    monkeypatch.setattr(b, "_post", fake_post)
    assert b.yes_probability("q") > 0.9
    assert seen[0][0] == "/api/generate" and seen[0][1]["top_logprobs"] == 20


def test_ollama_backend_normalises_value_conventions(monkeypatch):
    """値が確率（0〜1）や −log p（正）で返っても、対数確率に揃えて同じ p_yes になること。"""
    from target_loop.backends import OllamaBackend
    def make(vals):
        b = OllamaBackend("fake")
        monkeypatch.setattr(b, "_post", lambda path, body: {"response": " Yes", "logprobs": [{"token": " Yes", "logprob": vals[0],
                            "top_logprobs": [{"token": " Yes", "logprob": vals[0]}, {"token": " No", "logprob": vals[1]}]}]})
        return b
    ref = make([math.log(0.9), math.log(0.1)]).yes_probability("q")            # 対数確率
    assert abs(ref - 0.9) < 1e-9
    assert abs(make([0.9, 0.1]).yes_probability("q") - 0.9) < 1e-9             # 確率
    assert abs(make([-math.log(0.9), -math.log(0.1)]).yes_probability("q") - 0.9) < 1e-9   # −log p
