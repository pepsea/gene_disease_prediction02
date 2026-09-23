"""AI 役（バックエンド）。手順（loop.py）と AI の実体を切り離す層。

中学生向けの説明:
    loop.py は「AI に質問して答えをもらう」ことしかしません。誰が答えるかはここで決めます。
      RecordedBackend  会話記録の答えを再生する（テスト・再現用。通信なし）
      LlamaCppBackend  手元の GGUF モデル（TxGemma など）に本当に聞く（通信なし）
    どちらも同じメソッド名なので、差し替えても手順のコードは1行も変わりません。

LLM backends.

Every backend answers two kinds of requests:
    yes_probability(prompt, option_order) -> float in [0, 1]
    generate(prompt) -> str

* RecordedBackend  replays answers stored in a JSON file (used for the demo that
  was judged by Claude in the original conversation, and for tests).
* GuidanceBackend  same model through the guidance library (select + top_k trace),
  i.e. the option_logprobs mechanism already used in the project's earlier code.
* OllamaBackend    a running Ollama server over HTTP (logprobs via /v1/completions when supported).
* LlamaCppBackend  runs a local GGUF model (e.g. TxGemma-9B-Chat Q6_K) with
  llama-cpp-python and reads the next-token probability of "Yes" vs "No"
  directly from the logits.  No text is generated for scoring questions, so a
  question costs one forward pass of the prompt.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

GENE_SYMBOL_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]{1,5})?\b")
# all-caps words that are not gene symbols but appear in short LLM answers
_STOPWORDS = {"YES", "NO", "NONE", "AND", "OR", "THE", "A", "AN", "OF", "IN", "IS", "ARE", "NOT", "NEAR",
              "MID", "FAR", "WORSE", "BETTER", "GENE", "GENES", "TARGET", "TARGETS", "S", "RA", "DNA", "RNA",
              "GWAS", "LLM", "N", "E", "G", "I", "II", "III", "IV"} | {f"S{i}" for i in range(20)}


def parse_gene_list(text: str, known_symbols: Optional[set] = None, exclude: Sequence[str] = ()) -> List[str]:
    """AI の答えの文章から遺伝子記号（大文字の TNF, IL6R など）だけを取り出す。

    小文字の普通の単語は無視し、YES/NO/NONE などの大文字語は除外リストで外します。
    known_symbols（HGNC の記号一覧など）を渡すと、それ以外は捨てるので誤認を防げます。
    Extract gene-symbol-like tokens (upper-case, as LLMs write official symbols).

    Prose in lower case is ignored on purpose: 'The targets are TNF and IL6R.' -> [TNF, IL6R].
    If `known_symbols` (e.g. an HGNC list) is given, anything outside it is dropped."""
    out: List[str] = []
    for tok in GENE_SYMBOL_RE.findall(text):
        if tok in _STOPWORDS:
            continue
        if known_symbols is not None and tok not in known_symbols:
            continue
        if tok in exclude or tok in out:
            continue
        out.append(tok)
    return out


class LLMBackend:
    """AI 役の共通の顔。loop.py はこの3つのメソッドだけを呼ぶので、AI を差し替えても手順は変わらない。
        yes_probability  「はい」の確率を返す（採点用。文章は生成しない）
        generate         文章で答える（種の抽出、拡張のリスト、反論役、対戦比較）
        embed            文章をベクトルにする（地図の近さ V 用。無ければ None）
    """
    name = "base"

    def yes_probability(self, prompt: str, option_order: str = "yes_first") -> float:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        raise NotImplementedError

    def embed(self, text: str) -> Optional[List[float]]:
        return None

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------
class RecordedBackend(LLMBackend):
    """記録を再生する AI 役。会話記録の答えやテストに使う。

    質問文（プロンプト）をキーにして答えを引くだけです。記録にない質問が来たら KeyError で止まります。
    「黙って既定値を返す」ことをしないので、再現漏れが隠れません。
    Replay answers from a JSON record.

    Record format (see demo/ra_recorded.json):
        {"yes": {"<key>": p, ...}, "text": {"<key>": "...", ...}}
    Keys are looked up exactly; a prompt that is not recorded raises KeyError so
    that silent fall-backs cannot fake a result.
    """
    name = "recorded"

    def __init__(self, record: Dict):
        self.yes: Dict[str, float] = dict(record.get("yes", {}))
        self.text: Dict[str, str] = dict(record.get("text", {}))
        self.vectors: Dict[str, List[float]] = dict(record.get("vectors", {}))
        self.calls: List[dict] = []

    @classmethod
    def from_file(cls, path: str | Path) -> "RecordedBackend":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def yes_probability(self, prompt: str, option_order: str = "yes_first") -> float:
        self.calls.append({"kind": "yes", "prompt": prompt, "order": option_order})
        if prompt in self.yes:
            return float(self.yes[prompt])
        raise KeyError(f"no recorded yes/no answer for: {prompt}")

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        self.calls.append({"kind": "text", "prompt": prompt})
        if prompt in self.text:
            return self.text[prompt]
        raise KeyError(f"no recorded text answer for: {prompt}")

    def embed(self, text: str) -> Optional[List[float]]:
        return self.vectors.get(text)


# ----------------------------------------------------------------------------
class LlamaCppBackend(LLMBackend):
    """手元の GGUF モデル（TxGemma など）を llama-cpp-python で動かす AI 役。

    採点質問は文章を生成せず、次のトークンの logits から「Yes」「No」の確率だけを読みます。
    そのため 1 問 ＝ プロンプト1回分の forward pass で済み、外部通信はありません。
    Local GGUF model via llama-cpp-python.

    Notes learned the hard way (these were the errors at the end of the original
    conversation):
    * Do NOT pass logits_all=True.  It allocates n_ctx * n_vocab float32 for the
      score buffer (Gemma vocab 262k * 2048 ctx = 2.1 GB) and is unnecessary: the
      last-row logits are always available after eval().
    * Keep n_ctx small for scoring (512 is enough for one question).
    * "Yes"/" Yes"/"yes" are different tokens; sum all single-token spellings.
    """
    name = "llama_cpp"

    def __init__(self, model_path: str, n_ctx: int = 512, n_gpu_layers: int = -1,
                 n_threads: Optional[int] = None, embedding: bool = False, verbose: bool = False,
                 chat_wrap: bool = True):
        from llama_cpp import Llama  # imported lazily so the package is optional
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                         n_threads=n_threads, logits_all=False, embedding=embedding, verbose=verbose)
        self.embedding_enabled = embedding
        self.chat_wrap = chat_wrap
        self.yes_ids = self._token_ids(["Yes", " Yes", "yes", " yes", "YES", " YES"])
        self.no_ids = self._token_ids(["No", " No", "no", " no", "NO", " NO"])
        if not self.yes_ids or not self.no_ids:
            raise RuntimeError("could not find single-token spellings of Yes/No in this tokenizer")

    def _token_ids(self, spellings: Sequence[str]) -> List[int]:
        """"Yes" " Yes" "yes" など、1トークンで表せる綴りのトークン番号を集める。"""
        ids: List[int] = []
        for s in spellings:
            toks = self.llm.tokenize(s.encode("utf-8"), add_bos=False, special=False)
            if len(toks) == 1 and toks[0] not in ids:
                ids.append(toks[0])
        return ids

    def _wrap(self, prompt: str) -> str:
        """チャット用モデル向けに Gemma 形式の会話テンプレートで包む（--no-chat-wrap で無効化）。"""
        # Gemma-style chat template; harmless for base models used with plain prompts.
        if not self.chat_wrap:
            return prompt
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    def _last_logits(self, text: str):
        """プロンプトを1回評価し、最後の位置の logits（全語彙のスコア）を取り出す。logits_all は不要。"""
        import numpy as np
        toks = self.llm.tokenize(text.encode("utf-8"), add_bos=True, special=True)
        if len(toks) >= self.llm.n_ctx():
            raise ValueError(f"prompt too long for n_ctx={self.llm.n_ctx()}: {len(toks)} tokens")
        self.llm.reset()
        self.llm.eval(toks)
        return np.asarray(self.llm.scores[self.llm.n_tokens - 1], dtype=np.float64)

    def yes_probability(self, prompt: str, option_order: str = "yes_first") -> float:
        """p_yes ＝ ΣYes系トークン / (ΣYes系 + ΣNo系)。option_order で「Yes or No」「No or Yes」を切り替える。"""
        options = "Answer with Yes or No." if option_order == "yes_first" else "Answer with No or Yes."
        logits = self._last_logits(self._wrap(f"{prompt}\n{options}"))
        ly = logits[self.yes_ids]
        ln = logits[self.no_ids]
        m = max(ly.max(), ln.max())
        sy = float(sum(math.exp(x - m) for x in ly))
        sn = float(sum(math.exp(x - m) for x in ln))
        return sy / (sy + sn)

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        out = self.llm.create_completion(self._wrap(prompt), max_tokens=max_tokens, temperature=0.0,
                                         stop=["<end_of_turn>"])
        return out["choices"][0]["text"].strip()

    def embed(self, text: str) -> Optional[List[float]]:
        if not self.embedding_enabled:
            return None
        v = self.llm.embed(text)
        # llama-cpp may return a list of per-token vectors; mean-pool in that case
        if v and isinstance(v[0], list):
            n = len(v)
            dim = len(v[0])
            return [sum(row[i] for row in v) / n for i in range(dim)]
        return list(v)

    def close(self) -> None:
        try:
            self.llm.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
class GuidanceBackend(LLMBackend):
    """guidance ライブラリ経由で GGUF モデルを動かす AI 役（お使いの option_logprobs 方式）。

    仕組み:
      state = llm.copy() + prompt            問いを足した状態（元の llm は変わらない）
      state += select(options, name=...)     答えを選択肢の1つに限る（制約付き生成）
      state._trace_nodes の TokenOutput      最初に生成したトークンの top_k に「制約をかける前の」
                                             上位 TOP_K 個の (token, prob, masked) が入っている
    そこから " Yes" と " No" の確率を取り出し、p_yes = pY / (pY + pN) にします。

    guidance 0.3 系では LlamaCpp(model, echo=True, top_k=TOP_K) としないと top_k が記録されません
    （enable_top_k=echo）。選択肢が top_k に入らなかった場合は、top_k の最小確率を上限値として
    代入し、missing に記録します（黙って落とさない）。
    """
    name = "guidance"

    def __init__(self, model_path: str, top_k: int = 50, n_ctx: int = 1024, n_gpu_layers: int = -1,
                 verbose: bool = False, yes_options: Sequence[str] = (" Yes", " No"), **llama_cpp_kwargs):
        from guidance.models import LlamaCpp  # imported lazily so the package is optional
        self.llm = LlamaCpp(model_path, echo=True, top_k=top_k, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                            verbose=verbose, **llama_cpp_kwargs)
        self.top_k = top_k
        self.options = list(yes_options)
        self.missing: List[dict] = []          # 選択肢が top_k に無かった記録

    def option_logprobs(self, prompt: str, options: Sequence[str]) -> Dict[str, float]:
        """お使いの関数と同じ。各選択肢の log 確率（制約前）を返す。"""
        from guidance import select
        state = self.llm.copy() + prompt
        state += select(list(options), name="answer")
        generated = [o for node in state._trace_nodes for o in node.output
                     if type(o).__name__ == "TokenOutput" and not o.is_input]
        candidates = generated[0].top_k or []
        floor = min((c.prob for c in candidates if c.prob and c.prob > 0), default=1e-6)
        result: Dict[str, float] = {}
        for option in options:
            probs = ([c.prob for c in candidates if c.token == option]
                     or [c.prob for c in candidates if c.token.strip() == option.strip()])
            if probs and probs[0] > 0:
                result[option.strip()] = math.log(probs[0])
            else:
                self.missing.append({"prompt": prompt[:80], "option": option, "floor": floor})
                result[option.strip()] = math.log(floor)     # 上限値（top_k の最小確率）で代用
        return result

    def yes_probability(self, prompt: str, option_order: str = "yes_first") -> float:
        options = "Answer with Yes or No." if option_order == "yes_first" else "Answer with No or Yes."
        lp = self.option_logprobs(f"{options}\n{prompt}\nAnswer:", self.options)
        py, pn = math.exp(lp["Yes"]), math.exp(lp["No"])
        return py / (py + pn)

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        from guidance import gen
        state = self.llm.copy() + prompt + "\n" + gen(name="out", max_tokens=max_tokens, temperature=0.0)
        return state["out"].strip()

    def close(self) -> None:
        self.llm = None


# ----------------------------------------------------------------------------
class OllamaBackend(LLMBackend):
    """起動中の Ollama（http://localhost:11434）を HTTP で使う AI 役。

    yes_probability は OpenAI 互換の /v1/completions に logprobs を要求して先頭トークンの上位 k 個から
    Yes/No を読む。Ollama の版が logprobs 非対応なら、温度 1.0 で n_fallback 回サンプリングした
    Yes 割合で代用し、missing に記録する。generate は /api/generate（温度 0）。
    """
    name = "ollama"

    def __init__(self, model: str, url: str = "http://localhost:11434", top_k: int = 20, n_fallback: int = 8,
                 timeout: int = 300):
        self.model, self.url, self.top_k, self.n_fallback, self.timeout = model, url.rstrip("/"), top_k, n_fallback, timeout
        self.missing: List[dict] = []

    @staticmethod
    def list_models(url: str = "http://localhost:11434", timeout: int = 3) -> List[dict]:
        """/api/tags からモデル一覧（name, size）を返す。サーバーが無ければ空リスト。"""
        import urllib.request
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=timeout) as r:
                return [{"name": m["name"], "size": m.get("size", 0)} for m in json.load(r).get("models", [])]
        except Exception:
            return []

    def _post(self, path: str, body: dict) -> dict:
        import urllib.request
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.0) -> str:
        out = self._post("/api/generate", {"model": self.model, "prompt": prompt, "stream": False,
                                           "options": {"temperature": temperature, "num_predict": max_tokens}})
        return out.get("response", "").strip()

    def top_logprobs(self, prompt: str) -> Optional[Dict[str, float]]:
        """先頭1トークンの上位 k 個の {token: logprob}。

        Ollama の仕様（api/types.go）: /api/generate は logprobs: true, top_logprobs: 0〜20（上限 20）。
        /v1/completions では logprobs は個数（整数）。/api/generate → /v1/completions の順に試す。"""
        k = max(1, min(int(self.top_k), 20))
        tries = [self.lp_endpoint] if getattr(self, "lp_endpoint", None) else ["/api/generate", "/v1/completions"]
        for ep in tries:
            try:
                if ep == "/api/generate":
                    out = self._post(ep, {"model": self.model, "prompt": prompt, "stream": False, "logprobs": True,
                                          "top_logprobs": k, "options": {"temperature": 0, "num_predict": 1}})
                    lps = out.get("logprobs") or []
                    top = {t["token"]: t["logprob"] for t in (lps[0].get("top_logprobs", []) if lps else [])}
                    if lps and not top:
                        top = {lps[0]["token"]: lps[0]["logprob"]}
                else:
                    ch = self._post(ep, {"model": self.model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
                                         "logprobs": k})["choices"][0]
                    lp = ch.get("logprobs") or {}
                    if lp.get("content"):
                        top = {t["token"]: t["logprob"] for t in lp["content"][0].get("top_logprobs", [])}
                    elif lp.get("top_logprobs"):
                        top = dict(lp["top_logprobs"][0])
                    else:
                        top = {}
                if top:
                    self.lp_endpoint = ep
                    return top
            except Exception:
                continue
        return None

    def yes_probability(self, prompt: str, option_order: str = "yes_first") -> float:
        options = "Answer with Yes or No." if option_order == "yes_first" else "Answer with No or Yes."
        full = f"{options}\n{prompt}\nAnswer:"
        top = self.top_logprobs(full)
        if top is None:
            yes = 0
            for _ in range(self.n_fallback):
                m = re.match(r"\s*([A-Za-z]+)", self.generate(full, max_tokens=3, temperature=1.0))
                yes += 1 if m and m.group(1).lower().startswith("yes") else 0
            self.missing.append({"prompt": full[-80:], "option": "logprobs unsupported; sampled"})
            return (yes + 0.5) / (self.n_fallback + 1)
        floor = min([math.exp(v) for v in top.values()] or [1e-6])
        lp = {}
        for option in (" Yes", " No"):
            hit = [v for t, v in top.items() if t == option] or [v for t, v in top.items() if t.strip().lower() == option.strip().lower()]
            if hit:
                lp[option.strip()] = hit[0]
            else:
                self.missing.append({"prompt": full[-80:], "option": option}); lp[option.strip()] = math.log(floor)
        py, pn = math.exp(lp["Yes"]), math.exp(lp["No"])
        return py / (py + pn)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """2つのベクトルの角度の近さ（1＝同じ向き、0＝無関係）。地図の近さ V に使う。"""
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)
