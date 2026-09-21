"""ランダム重みの極小 GGUF（約 120KB）を自作する。モデルが手に入らない環境で、
llama.cpp から「はい」の確率を読む経路が動くことだけを確かめるため。出力の値に意味はない。

Write a tiny random-weight LLaMA-architecture GGUF (~300 KB) so that the
llama.cpp code path can be exercised without downloading a real model.
Outputs are random; use only for smoke tests."""
import sys, os
import numpy as np
from gguf import GGUFWriter, TokenType

out = sys.argv[1] if len(sys.argv) > 1 else "outputs/tiny_random.gguf"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
rng = np.random.default_rng(0)

n_embd, n_head, n_layer, n_ff, n_ctx = 32, 4, 1, 64, 256
# SPM-style vocab: specials + 256 byte tokens + a few words (byte fallback covers the rest)
tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)]
types = [TokenType.UNKNOWN, TokenType.CONTROL, TokenType.CONTROL] + [TokenType.BYTE] * 256
for w in ["Ye", "ye", "YE", "▁Ye", "▁ye", "▁Yes", "Yes", "▁No", "No", "▁YES", "YES", "▁NO", "NO", "▁yes", "yes", "▁no", "no", "▁", "▁is", "▁a", "▁the", "▁TNF"]:
    tokens.append(w); types.append(TokenType.NORMAL)
n_vocab = len(tokens)
scores = [0.0] * 3 + [-1000.0] * 256 + [-1.0 - 0.01 * i for i in range(n_vocab - 259)]

w = GGUFWriter(out, "llama")
w.add_name("tiny-random-smoke")
w.add_context_length(n_ctx); w.add_embedding_length(n_embd); w.add_block_count(n_layer)
w.add_feed_forward_length(n_ff); w.add_head_count(n_head); w.add_head_count_kv(n_head)
w.add_rope_dimension_count(n_embd // n_head); w.add_layer_norm_rms_eps(1e-5)
w.add_tokenizer_model("llama"); w.add_token_list(tokens); w.add_token_scores(scores); w.add_token_types(types)
w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_unk_token_id(0)

def t(name, shape, scale=0.1):
    w.add_tensor(name, (rng.standard_normal(shape) * scale).astype(np.float32))

t("token_embd.weight", (n_vocab, n_embd))
w.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
t("output.weight", (n_vocab, n_embd))
for i in range(n_layer):
    w.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(n_embd, dtype=np.float32))
    w.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
    for nm in ("attn_q", "attn_k", "attn_v", "attn_output"):
        t(f"blk.{i}.{nm}.weight", (n_embd, n_embd))
    t(f"blk.{i}.ffn_gate.weight", (n_ff, n_embd)); t(f"blk.{i}.ffn_up.weight", (n_ff, n_embd))
    t(f"blk.{i}.ffn_down.weight", (n_embd, n_ff))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {out} ({os.path.getsize(out)/1024:.0f} KB), vocab={n_vocab}")
