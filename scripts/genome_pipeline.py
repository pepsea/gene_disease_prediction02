"""全遺伝子（HGNC のタンパク質コード遺伝子）から、疾患ごとにトップ500の創薬標的ランキングを作るパイプライン。
設定はすべて YAML（config/pipeline07.yaml と config/diseases.yaml）。ノートブック 07 もこのモジュールを呼ぶ。

段階0 準備（全疾患で共通）  : 全遺伝子の機能情報（UniProt。疾患に触れる記述は除去）と、その埋め込み（bge-m3）を作る
段階1 ふるい                : 疾患の説明と機能説明の埋め込みの類似度で stage1.keep 件に絞る（生成 LLM は使わない）
段階2 粗い採点              : 質問（M3s）を1順だけで聞き、順序の偏り δ を一部の遺伝子で推定して補正 → stage2.keep 件
段階3 精密な採点            : 両順 × 機能情報なし・ありの対数オッズの平均（ノートブック 06 の点数）→ stage3.keep 件＝最終順位
段階4 上位の並べ直し        : 上位 stage4.top 件を 5択 × スイス式 × Bradley-Terry で並べ直す（確認用。順位は変えない）
段階5 理由のラベル          : H3・R1・B1 を両順で聞き、最終件数の中で特に強い条件をラベルにする
段階6 裏付けの注釈          : 順位を付けたあとで Open Targets の臨床段階（ChEMBL 由来）と関連スコアを付ける

1回の実行ごとに日付フォルダ out_dir/<疾患>/<年月日-時分>/ を作り、各段階の結果（stageN.csv、top500.csv）と条件（conditions/ に
設定・疾患の定義・前置きと質問文、run_log.yaml に段階ごとの記録）をまとめる。--run-id latest で最新のフォルダから再開する。

使い方:
  python scripts/genome_pipeline.py                              # config/pipeline07.yaml の run_diseases を全段階
  python scripts/genome_pipeline.py --disease scz --stages 1 2   # 一部の段階だけ
  python scripts/genome_pipeline.py --disease ra --run-id latest       # 途中で止まった実行を、最新の日付フォルダで続きから
  python scripts/genome_pipeline.py --set stage1.keep=300 stage2.keep=100 stage3.keep=50 stage4.top=20   # 設定の上書き（試運転など）
"""
import argparse, copy, glob, json, math, os, random, re, sys, time, urllib.request
import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import fetch_function as FF                                       # 機能情報の取得と、疾患に触れる記述の除去
from yesno_question_variants import QUESTIONS as YN_QUESTIONS, STRICT_NOTE as YN_NOTE
from rank_variants import VARIANTS as CHOICE_QUESTIONS, STRICT_NOTE as CHOICE_NOTE
from rank_bt import luce
from swiss import random_groups, swiss_groups

YN_TEXT = {q: t for q, _, t in YN_QUESTIONS}
GROUP_SIZE = 5
STAGE_ORDER = ["PRECLINICAL", "IND", "EARLY_PHASE_1", "PHASE_1", "PHASE_1_2", "PHASE_2", "PHASE_2_3", "PHASE_3", "PREAPPROVAL", "APPROVAL"]
logit = lambda x: math.log(max(x, 1e-6) / max(1 - x, 1e-6))
stage_idx = lambda st: STAGE_ORDER.index(st) if st in STAGE_ORDER else -1     # UNKNOWN などは最も低い


# ---------------------------------------------------------------- 設定
def load_config(path=os.path.join(ROOT, "config", "pipeline07.yaml"), overrides=()):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    for o in overrides:                                            # "stage1.keep=300" のような上書き
        k, v = o.split("=", 1); node = cfg; parts = k.split(".")
        for p in parts[:-1]: node = node[p]
        node[parts[-1]] = yaml.safe_load(v)
    cfg["_overrides"] = list(overrides)
    dz = yaml.safe_load(open(os.path.join(ROOT, cfg["diseases_file"]), encoding="utf-8"))["diseases"]
    cfg["_diseases"] = {re.sub(r"[^a-z0-9]+", "_", d["name"].lower()).strip("_"): d for d in dz}   # フォルダ名（病名の空白を _ に）→ 定義
    return cfg


def path(cfg, key):
    return os.path.join(ROOT, os.path.expanduser(cfg["paths"][key]))


def out_dir(cfg, dkey):
    """結果を入れる日付フォルダ out_dir/<疾患>/<RUN_ID>/。RUN_ID は cfg["_run_id"]（"" = 新規、"latest" = 最新で再開、フォルダ名も可）。
    初めて作るときに、条件（config_used.yaml・disease.yaml・prompts.txt）を conditions/ に保存する。"""
    base = os.path.join(path(cfg, "out_dir"), dkey); os.makedirs(base, exist_ok=True)
    rid = cfg.setdefault("_run_ids", {}).get(dkey)
    if rid is None:
        rid = cfg.get("_run_id", "")
        if rid == "latest": rid = sorted(x for x in os.listdir(base) if re.fullmatch(r"\d{8}-\d{4}", x))[-1]
        rid = rid or time.strftime("%Y%m%d-%H%M"); cfg["_run_ids"][dkey] = rid
        d = os.path.join(base, rid); os.makedirs(os.path.join(d, "conditions"), exist_ok=True)
        keep = {k: v for k, v in cfg.items() if not k.startswith("_")}
        yaml.safe_dump({**keep, "_overrides": cfg.get("_overrides", []), "_disease_key": dkey, "_run_id": rid},
                       open(os.path.join(d, "conditions", "config_used.yaml"), "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
        yaml.safe_dump(cfg["_diseases"][dkey], open(os.path.join(d, "conditions", "disease.yaml"), "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    return os.path.join(base, rid)


def runlog(cfg, dkey, key, value):
    """out_dir/<疾患>/<RUN_ID>/run_log.yaml に段階ごとの記録を追記する。"""
    p = os.path.join(out_dir(cfg, dkey), "run_log.yaml")
    log_ = yaml.safe_load(open(p, encoding="utf-8")) if os.path.exists(p) else {"run_id": cfg["_run_ids"][dkey], "disease": dkey, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    log_[key] = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), **value}
    yaml.safe_dump(log_, open(p, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)


def universe(cfg):
    g = pd.read_csv(path(cfg, "gene_universe"), sep="\t", dtype=str).fillna("")
    g["acc"] = g["uniprot_ids"].str.split("|").str[0]
    g["gene_label"] = [f"{s} ({(p or n).split('|')[0]})" if (p or n) else s for s, p, n in zip(g["symbol"], g["protein_name_uniprot"], g["gene_name"])]
    return g.drop_duplicates("symbol").reset_index(drop=True)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- 段階0：機能情報と埋め込み（全疾患で共通）
def stage0(cfg):
    g = universe(cfg)
    cache_p = path(cfg, "function_cache")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    todo = sorted({a for a in g["acc"] if a and a not in cache})
    log(f"段階0: 機能情報 {len(g)} 遺伝子（取得済み {len(g) - len(todo)}、残り {len(todo)}）")
    b = cfg["stage0"]["uniprot_batch"]
    for i in range(0, len(todo), b):
        for e in FF.fetch(todo[i:i + b]): cache[e["primaryAccession"]] = FF.parse(e)
        for a in todo[i:i + b]: cache.setdefault(a, FF.parse({}))
        if (i // b) % 20 == 0:
            json.dump(cache, open(cache_p, "w"), ensure_ascii=False); log(f"  UniProt {min(i + b, len(todo))}/{len(todo)}")
        time.sleep(0.2)
    json.dump(cache, open(cache_p, "w"), ensure_ascii=False)
    names = [d["name"].lower() for d in cfg["_diseases"].values()]
    bad = [a for a, v in cache.items() if any(n in v["text"].lower() for n in names)]
    if bad: raise SystemExit(f"機能情報に病名が残っています: {bad[:10]}")
    log(f"  機能情報 OK（{len(cache)} 件、登録疾患の病名は含まれない）")

    edir = path(cfg, "embedding_dir"); os.makedirs(edir, exist_ok=True)
    model = cfg["stage0"]["embed_model"]; ep = os.path.join(edir, f"{model}.npz")
    old = dict(np.load(ep, allow_pickle=True)) if os.path.exists(ep) else {"symbols": np.array([]), "vectors": np.zeros((0, 1))}
    have = {s: v for s, v in zip(old["symbols"], old["vectors"])}
    texts = {s: gene_text(s, lab, cache.get(a, {})) for s, lab, a in zip(g["symbol"], g["gene_label"], g["acc"])}
    todo = [s for s in g["symbol"] if s not in have]
    log(f"段階0: 埋め込み（{model}）残り {len(todo)} 遺伝子")
    bs = cfg["stage0"]["embed_batch"]
    for i in range(0, len(todo), bs):
        chunk = todo[i:i + bs]
        for s, v in zip(chunk, embed(cfg, [texts[s] for s in chunk])): have[s] = v
        if (i // bs) % 50 == 0:
            np.savez(ep, symbols=np.array(list(have)), vectors=np.array(list(have.values()))); log(f"  埋め込み {min(i + bs, len(todo))}/{len(todo)}")
    np.savez(ep, symbols=np.array(list(have)), vectors=np.array(list(have.values())))
    log(f"  埋め込み OK（{len(have)} 件）→ {ep}")


def gene_text(symbol, label, info):
    """埋め込み用の文章：タンパク質名＋機能情報（疾患に触れる記述は除去済み）。"""
    return f"{label}. {info.get('function', '')} GO: {info.get('go', '')}. Pathways: {info.get('reactome', '')}"


def embed(cfg, texts):
    req = urllib.request.Request(cfg["stage0"]["ollama_url"] + "/api/embed", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": cfg["stage0"]["embed_model"], "input": texts}).encode())
    with urllib.request.urlopen(req, timeout=600) as r:
        v = np.array(json.load(r)["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------- 段階1：ふるい
def stage1(cfg, dkey):
    D = cfg["_diseases"][dkey]; out = os.path.join(out_dir(cfg, dkey), "stage1.csv")
    e = np.load(os.path.join(path(cfg, "embedding_dir"), f"{cfg['stage0']['embed_model']}.npz"), allow_pickle=True)
    q = embed(cfg, [f"{D['name']}. " + " ".join(D["info"])])[0]      # 疾患の説明（病名＋症状の箇条書き）
    sim = e["vectors"] @ q
    t = pd.DataFrame({"symbol": e["symbols"], "sim": sim}).sort_values("sim", ascending=False).reset_index(drop=True)
    t["rank1"] = np.arange(1, len(t) + 1); t["keep1"] = t["rank1"] <= cfg["stage1"]["keep"]
    runlog(cfg, dkey, "stage1", {"input": len(t), "kept": int(t["keep1"].sum())})
    t.to_csv(out, index=False); log(f"段階1 {dkey}: {len(t)} → {int(t['keep1'].sum())} 件 → {out}")
    return t


# ---------------------------------------------------------------- 生成 LLM（txgemma、GGUF）
class Engine:
    def __init__(self, cfg, dkey):
        from llama_cpp import Llama
        m = cfg["model"]["gguf"]
        if m == "auto":
            hits = sorted(glob.glob(os.path.join(os.path.expanduser(cfg["model"]["model_dir"]), "**", "*txgemma*.gguf"), recursive=True))
            if not hits: raise SystemExit("GGUF モデルが見つかりません（config の model.gguf を指定してください）")
            m = hits[0]
        self.model = os.path.basename(m)
        self.llm = Llama(model_path=m, n_ctx=cfg["model"]["n_ctx"], n_gpu_layers=-1, logits_all=False, verbose=False)
        self.D = cfg["_diseases"][dkey]
        ids = lambda sp: list(dict.fromkeys(t[0] for t in (self.tok(s) for s in sp) if len(t) == 1))
        self.yes, self.no = ids(["Yes", " Yes", "yes", " yes", "YES", " YES"]), ids(["No", " No", "no", " no", "NO", " NO"])
        self.digit = {n: self.tok(str(n))[0] for n in range(1, GROUP_SIZE + 2)}
        self.state = {}
        for name, text in (("yes_first", self.yn_prefix("yes_first")), ("no_first", self.yn_prefix("no_first")), ("choice", self.choice_prefix())):
            self.llm.reset(); self.llm.eval(self.tok(text, bos=True)); self.state[name] = self.llm.save_state()
        log(f"model {self.model} ready"); runlog(cfg, dkey, "model", {"path": m})
        with open(os.path.join(out_dir(cfg, dkey), "conditions", "prompts.txt"), "w", encoding="utf-8") as f:      # 条件：前置きと質問文
            for o in ("yes_first", "no_first"): f.write(f"===== Yes/No の前置き（{o}）\n{self.yn_prefix(o)}\n")
            f.write(f"===== 5択の前置き\n{self.choice_prefix()}\n")
            for q in [cfg["stage2"]["question"], cfg["stage3"]["question"], *cfg["stage5"]["questions"]]: f.write(f"===== 質問 {q}\n{YN_TEXT[q]}\n\n")
            f.write(f"===== 質問 {cfg['stage4']['question']}\n{CHOICE_QUESTIONS[cfg['stage4']['question']]}\n")

    def tok(self, text, bos=False):
        return self.llm.tokenize(text.encode("utf-8"), add_bos=bos, special=bos)

    def bullets(self):
        return "\n".join(f"- {b}" for b in self.D["info"])

    def yn_prefix(self, order):                                    # yesno_question_variants.prefix_text と同じ文面
        options = "Answer each question with Yes or No." if order == "yes_first" else "Answer each question with No or Yes."
        return ("You are an expert in drug discovery and human disease biology.\n" + YN_NOTE +
                f"Disease: {self.D['name']}\nTarget symptoms and the organ, cell and functional abnormalities behind them:\n{self.bullets()}\n{options}\n\n")

    def choice_prefix(self):                                       # rank_variants.prefix_text と同じ文面
        return ("You are an expert in drug discovery and human disease biology. You will be shown a numbered list of "
                f"{GROUP_SIZE} candidate genes for one disease, and asked which ONE number is the best answer to a question.\n" + CHOICE_NOTE +
                f"Disease: {self.D['name']}\nTarget symptoms and the organ, cell and functional abnormalities behind them:\n{self.bullets()}\n"
                f"Answer with a single number from 1 to {GROUP_SIZE + 1} only. No words, no explanation.\n\n")

    def logits(self):
        return np.ctypeslib.as_array(self.llm._ctx.get_logits(), shape=(self.llm.n_vocab(),)).astype(np.float64)

    def yes_logodds(self, label, symbol, qids, order, extra=""):
        """前置き → 遺伝子行（extra に機能情報）→ 質問ごとに巻き戻して独立に聞き、Yes の対数オッズを返す。"""
        llm = self.llm
        llm.reset(); llm.load_state(self.state[order]); llm.eval(self.tok(f"Gene: {label}\n" + extra)); base = llm.n_tokens
        out = {}
        for q in qids:
            llm.n_tokens = base
            llm.eval(self.tok(f"{q}. {YN_TEXT[q].format(disease=self.D['name'], gene=symbol)} Answer:"))
            lg = self.logits(); lg -= lg.max(); lp = lg - math.log(np.exp(lg).sum())
            lse = lambda v: max(v) + math.log(sum(math.exp(x - max(v)) for x in v))
            out[q] = logit(1 / (1 + math.exp(lse([lp[i] for i in self.no]) - lse([lp[i] for i in self.yes]))))
        return out

    def choice(self, labels, qid):
        llm = self.llm
        llm.reset(); llm.load_state(self.state["choice"])
        llm.eval(self.tok("Candidate genes:\n" + "\n".join(f"{i + 1}. {g}" for i, g in enumerate(labels)) +
                          f"\n{len(labels) + 1}. None of the above genes seem clearly relevant\n"))
        llm.eval(self.tok(f"Question: {CHOICE_QUESTIONS[qid].format(disease=self.D['name'])} Answer: "))
        lg = self.logits(); l = np.array([lg[self.digit[n]] for n in range(1, GROUP_SIZE + 2)])
        p = np.exp(l - l.max()); return p / p.sum()


def func_extra(cfg, dkey, acc, cache):
    """プロンプト用の機能情報。この疾患の病名を含む行は落とす（念のための二重チェック）。"""
    t = cache.get(acc, {}).get("text", "")
    terms = [cfg["_diseases"][dkey]["name"].lower()]
    return "".join(l + "\n" for l in t.splitlines() if l and not any(x in l.lower() for x in terms))


def resume(path_, key="symbol"):
    return pd.read_csv(path_) if os.path.exists(path_) else pd.DataFrame(columns=[key])


# ---------------------------------------------------------------- 段階2：粗い採点
def stage2(cfg, dkey, eng):
    c = cfg["stage2"]; od = out_dir(cfg, dkey); out = os.path.join(od, "stage2.csv")
    s1 = pd.read_csv(os.path.join(od, "stage1.csv")); cand = s1[s1["keep1"]]["symbol"].tolist()
    g = universe(cfg).set_index("symbol")
    done = resume(out); have = set(done["symbol"]); rows = done.to_dict("records")
    calib = set(random.Random(c["seed"]).sample(cand, min(c["calib_genes"], len(cand))))
    t0 = time.time(); todo = [s for s in cand if s not in have]
    log(f"段階2 {dkey}: {len(cand)} 件（済み {len(have)}、残り {len(todo)}、両順で聞く遺伝子 {len(calib)}）")
    for k, s in enumerate(todo, 1):
        yf = eng.yes_logodds(g.at[s, "gene_label"], s, [c["question"]], "yes_first")[c["question"]]
        nf = eng.yes_logodds(g.at[s, "gene_label"], s, [c["question"]], "no_first")[c["question"]] if s in calib else np.nan
        rows.append({"symbol": s, "lo_yes_first": yf, "lo_no_first": nf})
        if k % cfg["checkpoint_every"] == 0 or k == len(todo):
            pd.DataFrame(rows).to_csv(out, index=False)
            log(f"  {k}/{len(todo)}（{(time.time() - t0) / k:.2f} 秒/遺伝子、残り約 {(len(todo) - k) * (time.time() - t0) / k / 60:.0f} 分）")
    t = pd.DataFrame(rows)
    both = t.dropna(subset=["lo_no_first"])
    delta = float((both["lo_yes_first"] - both["lo_no_first"]).mean()) if len(both) else 0.0
    t["score2"] = np.where(t["lo_no_first"].notna(), (t["lo_yes_first"] + t["lo_no_first"]) / 2, t["lo_yes_first"] - delta / 2)
    t = t.sort_values("score2", ascending=False).reset_index(drop=True)
    t["rank2"] = np.arange(1, len(t) + 1); t["keep2"] = t["rank2"] <= c["keep"]; t["delta"] = delta
    runlog(cfg, dkey, "stage2", {"input": len(cand), "kept": int(t["keep2"].sum()), "delta_logodds": round(delta, 4), "calib_genes": len(calib)})
    t.to_csv(out, index=False); log(f"段階2 {dkey}: 順序の偏り δ={delta:+.2f}（対数オッズ）→ {int(t['keep2'].sum())} 件")
    return t


# ---------------------------------------------------------------- 段階3：精密な採点（06 の点数）
def stage3(cfg, dkey, eng):
    c = cfg["stage3"]; od = out_dir(cfg, dkey); out = os.path.join(od, "stage3.csv")
    s2 = pd.read_csv(os.path.join(od, "stage2.csv")); cand = s2[s2["keep2"]]
    g = universe(cfg).set_index("symbol"); cache = json.load(open(path(cfg, "function_cache")))
    done = resume(out); have = set(done["symbol"]); rows = done.to_dict("records"); q = c["question"]
    todo = cand[~cand["symbol"].isin(have)]; t0 = time.time()
    log(f"段階3 {dkey}: {len(cand)} 件（済み {len(have)}、残り {len(todo)}）")
    for k, r in enumerate(todo.itertuples(), 1):
        s, lab = r.symbol, g.at[r.symbol, "gene_label"]
        extra = func_extra(cfg, dkey, g.at[s, "acc"], cache)
        yf = r.lo_yes_first                                                    # 段階2の yes_first を再利用（同じプロンプト）
        nf = r.lo_no_first if not pd.isna(r.lo_no_first) else eng.yes_logodds(lab, s, [q], "no_first")[q]
        fy = eng.yes_logodds(lab, s, [q], "yes_first", extra)[q]; fn = eng.yes_logodds(lab, s, [q], "no_first", extra)[q]
        rows.append({"symbol": s, "m3s": (yf + nf) / 2, "m3sf": (fy + fn) / 2, "has_function": bool(extra)})
        if k % cfg["checkpoint_every"] == 0 or k == len(todo):
            pd.DataFrame(rows).to_csv(out, index=False)
            log(f"  {k}/{len(todo)}（{(time.time() - t0) / k:.2f} 秒/遺伝子、残り約 {(len(todo) - k) * (time.time() - t0) / k / 60:.0f} 分）")
    t = pd.DataFrame(rows); t["score"] = (t["m3s"] + t["m3sf"]) / 2
    t = t.sort_values("score", ascending=False).reset_index(drop=True); t["rank"] = np.arange(1, len(t) + 1)
    runlog(cfg, dkey, "stage3", {"input": len(cand), "kept": c["keep"], "with_function_info": int(t["has_function"].sum())})
    t.to_csv(out, index=False); log(f"段階3 {dkey}: 上位 {c['keep']} 件が最終順位")
    return t


# ---------------------------------------------------------------- 段階4：上位の並べ直し（確認用）
def stage4(cfg, dkey, eng):
    c = cfg["stage4"]; od = out_dir(cfg, dkey); out = os.path.join(od, "stage4.csv")
    s3 = pd.read_csv(os.path.join(od, "stage3.csv")); top = s3.head(c["top"])["symbol"].tolist()
    g = universe(cfg).set_index("symbol"); rng, rows, t0 = random.Random(c["seed"]), [], time.time()
    def fit():
        items = top + ["NONE"]; ix = {s: k for k, s in enumerate(items)}; groups = []
        for _, d in pd.DataFrame(rows).groupby(["round", "group"]):
            d = d.sort_values("position")
            groups.append(([ix[s] for s in d["symbol"]] + [ix["NONE"]], np.r_[d["p"].values, d["p_none"].iloc[0]]))
        return luce(groups, len(items))
    for r in range(c["rounds"]):
        groups = random_groups(top, rng) if r < c["random_rounds"] else swiss_groups(top, dict(zip(top, fit())), rng, c["jitter"])
        for gi, chunk in enumerate(groups):
            p = eng.choice([g.at[s, "gene_label"] for s in chunk], c["question"])
            for slot, s in enumerate(chunk):
                rows.append({"round": r, "group": gi, "position": slot + 1, "symbol": s, "p": round(float(p[slot]), 6), "p_none": round(float(p[-1]), 6)})
        log(f"  段階4 {dkey}: round {r + 1}/{c['rounds']}（{time.time() - t0:.0f}s）")
    s = fit()
    t = pd.DataFrame({"symbol": top, "bt": s[:-1]}); t["bt_above_none"] = t["bt"] > s[-1]
    t["rank4"] = t["bt"].rank(ascending=False, method="first").astype(int)
    runlog(cfg, dkey, "stage4", {"genes": len(top), "seconds": round(time.time() - t0)})
    t.to_csv(out, index=False); pd.DataFrame(rows).to_csv(os.path.join(od, "stage4_rounds.csv"), index=False)
    log(f"段階4 {dkey}: 上位 {len(top)} 件の並べ直し → {out}")
    return t


# ---------------------------------------------------------------- 段階5：理由のラベル
def stage5(cfg, dkey, eng):
    c = cfg["stage5"]; od = out_dir(cfg, dkey); out = os.path.join(od, "stage5.csv"); qs = list(c["questions"])
    s3 = pd.read_csv(os.path.join(od, "stage3.csv")).head(cfg["stage3"]["keep"])
    g = universe(cfg).set_index("symbol"); done = resume(out); have = set(done["symbol"]); rows = done.to_dict("records"); t0 = time.time()
    todo = [s for s in s3["symbol"] if s not in have]
    for k, s in enumerate(todo, 1):
        a = eng.yes_logodds(g.at[s, "gene_label"], s, qs, "yes_first"); b = eng.yes_logodds(g.at[s, "gene_label"], s, qs, "no_first")
        rows.append({"symbol": s, **{q: (a[q] + b[q]) / 2 for q in qs}})
        if k % cfg["checkpoint_every"] == 0 or k == len(todo):
            pd.DataFrame(rows).to_csv(out, index=False); log(f"  段階5 {dkey}: {k}/{len(todo)}（{time.time() - t0:.0f}s）")
    t = pd.DataFrame(rows); z = (t[qs] - t[qs].mean()) / t[qs].std()
    t["reasons"] = ["・".join(c["questions"][q] for q in qs if z.at[i, q] >= c["z_threshold"]) or c["questions"][z.loc[i].idxmax()] for i in t.index]
    runlog(cfg, dkey, "stage5", {"genes": len(t), "labels": {k: int(v) for k, v in t["reasons"].value_counts().head(10).items()}})
    t.to_csv(out, index=False); log(f"段階5 {dkey}: 理由のラベル → {out}")
    return t


# ---------------------------------------------------------------- 段階6：裏付けの注釈（順位付けの後）
def ot_query(cfg, query, variables=None):
    req = urllib.request.Request(cfg["stage6"]["api"], headers={"Content-Type": "application/json"},
                                 data=json.dumps({"query": query, "variables": variables or {}}).encode())
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    if "errors" in d: raise RuntimeError(d["errors"][0]["message"])
    return d["data"]


def stage6(cfg, dkey):
    od = out_dir(cfg, dkey); out = os.path.join(od, "stage6.csv"); D = cfg["_diseases"][dkey]
    did = ot_query(cfg, "query($q:String!){search(queryString:$q,entityNames:[\"disease\"],page:{index:0,size:1}){hits{id}}}",
                                               {"q": D["name"]})["search"]["hits"][0]["id"]
    d = ot_query(cfg, "query($id:String!){disease(efoId:$id){drugAndClinicalCandidates{rows{maxClinicalStage drug{name mechanismsOfAction{rows{targets{approvedSymbol}}}}}}}}",
                 {"id": did})["disease"]
    best, drugs = {}, {}
    for r in d["drugAndClinicalCandidates"]["rows"]:
        st = r["maxClinicalStage"]
        for m in ((r["drug"] or {}).get("mechanismsOfAction") or {}).get("rows", []):
            for t in m["targets"]:
                s = t["approvedSymbol"]
                if s not in best or stage_idx(st) > stage_idx(best[s]): best[s] = st
                drugs.setdefault(s, set()).add((r["drug"] or {}).get("name", ""))
    assoc, page = {}, 0
    while True:
        x = ot_query(cfg, "query($id:String!,$i:Int!){disease(efoId:$id){associatedTargets(page:{index:$i,size:500}){count rows{score target{approvedSymbol}}}}}",
                     {"id": did, "i": page})["disease"]["associatedTargets"]
        for r in x["rows"]: assoc[r["target"]["approvedSymbol"]] = r["score"]
        page += 1
        if page * 500 >= x["count"] or not x["rows"]: break
    t = pd.DataFrame({"symbol": sorted(set(best) | set(assoc))})
    t["ot_max_stage"] = t["symbol"].map(best); t["ot_drugs"] = t["symbol"].map(lambda s: "; ".join(sorted(drugs.get(s, set())))[:200])
    t["ot_assoc_score"] = t["symbol"].map(assoc); t["ot_disease_id"] = did
    runlog(cfg, dkey, "stage6", {"open_targets_id": did, "targets_with_clinical_drug": len(best), "targets_with_association": len(assoc)})
    t.to_csv(out, index=False); log(f"段階6 {dkey}: Open Targets（{did}）臨床段階あり {len(best)} 標的、関連スコア {len(assoc)} 標的 → {out}")
    return t


# ---------------------------------------------------------------- 最終表と評価
def final_table(cfg, dkey):
    od = out_dir(cfg, dkey); n = cfg["stage3"]["keep"]
    rd = lambda f: pd.read_csv(os.path.join(od, f)) if os.path.exists(os.path.join(od, f)) else None
    t = rd("stage3.csv").head(n)
    for f, cols in (("stage2.csv", ["symbol", "rank2", "score2"]), ("stage1.csv", ["symbol", "rank1", "sim"]), ("stage4.csv", ["symbol", "rank4", "bt", "bt_above_none"]),
                    ("stage5.csv", None), ("stage6.csv", ["symbol", "ot_max_stage", "ot_drugs", "ot_assoc_score"])):
        x = rd(f)
        if x is not None: t = t.merge(x if cols is None else x[cols], on="symbol", how="left")
    g = universe(cfg).set_index("symbol"); cache = json.load(open(path(cfg, "function_cache")))
    t.insert(2, "gene_name", t["symbol"].map(g["gene_name"]))
    t["function"] = [cache.get(g.at[s, "acc"], {}).get("function", "")[:160] for s in t["symbol"]]
    A, B = cfg["tiers"]["A"], cfg["tiers"]["B"]
    t["tier"] = np.where((t["rank"] <= A) & (t.get("rank4", pd.Series(np.inf, index=t.index)) <= A), "A", np.where(t["rank"] <= B, "B", "C"))
    if "ot_max_stage" in t:
        t["evidence"] = np.where(t["ot_max_stage"].isin(["APPROVAL", "PREAPPROVAL"]), "承認薬あり",
                                 np.where(t["ot_max_stage"].notna(), "臨床段階の薬あり", "なし（新規候補）"))
    t.to_csv(os.path.join(od, "top500.csv"), index=False)
    runlog(cfg, dkey, "final", {"genes": len(t), "tiers": {k: int(v) for k, v in t["tier"].value_counts().items()}})
    log(f"最終表 {dkey}: {len(t)} 件（A {int((t['tier'] == 'A').sum())}、B {int((t['tier'] == 'B').sum())}、C {int((t['tier'] == 'C').sum())}）→ {od}/top500.csv")
    return t


def evaluate(cfg, dkey, min_stage="PHASE_2"):
    """既知標的（known_file と、Open Targets で min_stage 以上の薬がある標的）が各段階でどれだけ残ったか。順位付けには使わない。"""
    od = out_dir(cfg, dkey); D = cfg["_diseases"][dkey]; pos = {}
    kf = cfg.get("evaluation", {}).get("known_files", {}).get(D["name"])
    if kf and os.path.exists(os.path.join(ROOT, kf)):
        pos["既知（known_file）"] = set(pd.read_csv(os.path.join(ROOT, kf), sep="\t")["symbol"])
    if os.path.exists(os.path.join(od, "stage6.csv")):
        s6 = pd.read_csv(os.path.join(od, "stage6.csv"))
        pos[f"Open Targets（{min_stage} 以上）"] = set(s6.loc[s6["ot_max_stage"].isin(STAGE_ORDER[STAGE_ORDER.index(min_stage):]), "symbol"])
    N = len(universe(cfg)); rows = []
    rk = {"段階1（類似度）": ("stage1.csv", "rank1"), "段階2（粗い採点）": ("stage2.csv", "rank2"), "段階3（最終）": ("stage3.csv", "rank")}
    for pname, P in pos.items():
        P = P & set(universe(cfg)["symbol"])
        for sname, (f, col) in rk.items():
            p = os.path.join(od, f)
            if not os.path.exists(p): continue
            x = pd.read_csv(p).set_index("symbol")[col]
            for K in (50, 200, 500, 2000, 5000):
                hit = len([s for s in P if s in x.index and x[s] <= K])
                rows.append({"正解": pname, "正解の数": len(P), "段階": sname, "K": K, "上位K件に入った数": hit, "recall@K": hit / len(P) if P else np.nan,
                             "濃縮率（ランダム比）": (hit / K) / (len(P) / N) if P else np.nan})
    return pd.DataFrame(rows)


def run(cfg, dkey, stages=range(0, 7)):
    stages = set(stages)
    if 0 in stages: stage0(cfg)
    if 1 in stages: stage1(cfg, dkey)
    eng = Engine(cfg, dkey) if stages & {2, 3, 4, 5} else None
    if 2 in stages: stage2(cfg, dkey, eng)
    if 3 in stages: stage3(cfg, dkey, eng)
    if 4 in stages: stage4(cfg, dkey, eng)
    if 5 in stages: stage5(cfg, dkey, eng)
    if 6 in stages and cfg["stage6"]["enabled"]: stage6(cfg, dkey)
    if os.path.exists(os.path.join(out_dir(cfg, dkey), "stage3.csv")): return final_table(cfg, dkey)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "pipeline07.yaml"))
    ap.add_argument("--disease", nargs="+", default=None)
    ap.add_argument("--stages", nargs="+", type=int, default=list(range(0, 7)))
    ap.add_argument("--set", nargs="+", default=[], help="設定の上書き（例 stage1.keep=300）")
    ap.add_argument("--run-id", default="", help='結果の日付フォルダ。"" = 新規、"latest" = 最新で再開、フォルダ名も可')
    a = ap.parse_args()
    cfg = load_config(a.config, a.set); cfg["_run_id"] = a.run_id
    for i, name in enumerate(a.disease or cfg["run_diseases"]):
        dkey = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")          # 病名 → フォルダ名（例 rheumatoid_arthritis）
        if a.stages == [0] and i > 0: break                                 # 段階0だけなら1回で十分
        run(cfg, dkey, [s for s in a.stages if s != 0 or i == 0])           # 段階0は全疾患で共通なので最初の1回だけ
        ev = evaluate(cfg, dkey)
        if len(ev): print(ev.to_string(index=False))


if __name__ == "__main__":
    main()
