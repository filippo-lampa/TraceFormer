import os
import glob
import json
import sys
import pickle
import hashlib
from typing import Tuple, List, Dict, Any, Optional
import argparse

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.tokenizer import tokenizer, build_tree_from_output, build_context_from_tokens
from src.dataset import TxRandomWindowDataset, TxStrideWindowDataset
from src.model import GPTModel
from src.engine import train_network, valid_network, score_transactions

sys.set_int_max_str_digits(1_000_000)

# cache helpers

def _files_fingerprint(paths: list[str], extra_paths: Optional[list[str]] = None) -> str:
    h = hashlib.sha256()
    all_paths = list(paths)
    if extra_paths:
        all_paths.extend(extra_paths)

    for p in all_paths:
        st = os.stat(p)
        h.update(p.encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
    return h.hexdigest()[:16]


def load_token_cache(cache_path: str, expected_fingerprint: str):
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("fingerprint") != expected_fingerprint:
        return None
    return payload


def save_token_cache(
    cache_path: str,
    fingerprint: str,
    out_token_seqs: list[list[str]],
    tree_token_seqs: list[list[str]],
    ctx_token_seqs: list[list[str]],
):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "out_token_seqs": out_token_seqs,
        "tree_token_seqs": tree_token_seqs,
        "ctx_token_seqs": ctx_token_seqs,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_vocabs(path: str, out_vocab: dict, tree_vocab: dict, ctx_vocab: dict, meta: dict | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": 2,
        "meta": meta or {},
        "out_vocab": out_vocab,
        "tree_vocab": tree_vocab,
        "ctx_vocab": ctx_vocab,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_vocabs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out_vocab = payload["out_vocab"]
    tree_vocab = payload["tree_vocab"]
    ctx_vocab = payload["ctx_vocab"]

    for name, v in [("out_vocab", out_vocab), ("tree_vocab", tree_vocab), ("ctx_vocab", ctx_vocab)]:
        if "[PAD]" not in v or "[UNK]" not in v:
            raise ValueError(f"{name} missing [PAD] or [UNK].")
        if v["[PAD]"] != 0 or v["[UNK]"] != 1:
            raise ValueError(f"{name} expected [PAD]=0 and [UNK]=1.")

    return out_vocab, tree_vocab, ctx_vocab, payload.get("meta", {})


# token/vocab helpers

def load_and_merge_logs(file_paths: list[str], verbose: bool = False) -> pd.DataFrame:
    dfs = []
    for path in file_paths:
        if verbose:
            print(f"Reading {path}...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tmp = pd.DataFrame.from_records(data)
        if "_id" in tmp.columns:
            tmp = tmp.drop(columns=["_id"])
        dfs.append(tmp)
        if verbose:
            print(f" -> Loaded {len(tmp)} rows from {path}")
    return pd.concat(dfs, ignore_index=True)


def tokenize_row(row: pd.Series) -> list[str]:
    one = pd.DataFrame([row.to_dict()])
    toks = tokenizer(one)
    return [str(t) for t in toks]


def get_or_build_token_seqs(
    df: pd.DataFrame,
    file_paths: list[str],
    cache_path: str = "cache/tokenized_seqs.pkl",
    extra_fingerprint_paths: Optional[list[str]] = None,
) -> Tuple[list[list[str]], list[list[str]], list[list[str]]]:
    fp = _files_fingerprint(file_paths, extra_paths=extra_fingerprint_paths)
    cached = load_token_cache(cache_path, fp)
    if cached is not None:
        print(f"Loaded tokenization cache: {cache_path}")
        return cached["out_token_seqs"], cached["tree_token_seqs"], cached["ctx_token_seqs"]

    print("Tokenizing data (cache miss)...")
    
    seqs = [tokenize_row(r) for _, r in df.iterrows()]
    
    out_token_seqs, tree_token_seqs, ctx_token_seqs = [], [], []
    
    for seq in seqs:
        toks, tree = build_tree_from_output(seq)
        ctx = build_context_from_tokens(toks)
        out_token_seqs.append([str(t) for t in toks])
        tree_token_seqs.append([str(x) for x in tree])
        ctx_token_seqs.append([str(c) for c in ctx])
    
    save_token_cache(cache_path, fp, out_token_seqs, tree_token_seqs, ctx_token_seqs)
    print(f"Saved tokenization cache: {cache_path}")
    return out_token_seqs, tree_token_seqs, ctx_token_seqs


def flatten_token_seqs(token_seqs: list[list[str]]) -> list[str]:
    return [t for s in token_seqs for t in s]


def build_vocab(token_list: list[str], pad_token="[PAD]", unk_token="[UNK]") -> dict:
    vocab = {pad_token: 0, unk_token: 1}
    for t in token_list:
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab


def encode_token_seqs(token_seqs: list[list[str]], vocab: dict) -> list[torch.Tensor]:
    unk_id = vocab["[UNK]"]
    return [torch.tensor([vocab.get(t, unk_id) for t in seq], dtype=torch.long) for seq in token_seqs]


def make_split_indices(n: int, seed: int = 42, train_frac: float = 0.6, val_frac: float = 0.2):
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def subset(lst, idxs):
    return [lst[i] for i in idxs]


def get_or_build_vocabs_train_only(
    out_token_seqs: list[list[str]],
    tree_token_seqs: list[list[str]],
    ctx_token_seqs: list[list[str]],
    train_idx: list[int],
    file_paths: list[str],
    vocab_path: str = "models/vocabs_train_only.json",
    fingerprint: str | None = None,
    seed: int | None = None,
):
    """
    we build vocabs on train split only, to avoid leakage
    """
    if os.path.exists(vocab_path):
        out_vocab, tree_vocab, ctx_vocab, meta = load_vocabs(vocab_path)
        print(f"Loaded vocabs from {vocab_path}")
        return out_vocab, tree_vocab, ctx_vocab, meta

    print(f"No vocab file found at {vocab_path}. Building from TRAIN only and saving...")

    out_train = flatten_token_seqs(subset(out_token_seqs, train_idx))
    tree_train = flatten_token_seqs(subset(tree_token_seqs, train_idx))
    ctx_train = flatten_token_seqs(subset(ctx_token_seqs, train_idx))

    print(len(out_train))
    out_vocab = build_vocab(out_train)
    tree_vocab = build_vocab(tree_train)
    ctx_vocab = build_vocab(ctx_train)

    # remove overlapping tokens from out_vocab
    
    overlapping = (set(tree_vocab.keys()) | set(ctx_vocab.keys())) - {"[PAD]", "[UNK]"}
    
    cleaned_out_tokens = [tok for tok in out_vocab.keys() if tok not in overlapping]
    
    # re-indexing
    out_vocab = {tok: idx for idx, tok in enumerate(cleaned_out_tokens)}
    
    print(f"Cleaned out_vocab! Evicted {len(overlapping)} leaking tokens.")
    
    meta = {
        "source_files": file_paths,
        "built_from": "train_only",
        "fingerprint": fingerprint,
        "seed": seed,
        "train_size": len(train_idx),
        "total_size": len(out_token_seqs),
    }
    save_vocabs(vocab_path, out_vocab, tree_vocab, ctx_vocab, meta=meta)
    print(f"Saved vocabs to {vocab_path}")
    return out_vocab, tree_vocab, ctx_vocab, meta

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    ensure that each value within the json is tokenizable and correctly whitespaced before tokenization (e.g. multi-parameter tuples as inputValues)
    """
    def process_inputs_cell(cell_value):
        
        if isinstance(cell_value, str):
            try:
                inputs_list = ast.literal_eval(cell_value)
            except Exception:
                return cell_value
        else:
            inputs_list = cell_value

        if not isinstance(inputs_list, list):
            return cell_value

        new_inputs = []
        for item in inputs_list:
            if not isinstance(item, dict) or "inputValue" not in item:
                new_inputs.append(item)
                continue

            raw_names = str(item.get("inputName", ""))
            raw_values = str(item.get("inputValue", ""))
            
            names_split = [n.strip() for n in raw_names.split(",") if n.strip()]
            values_split = [v.strip() for v in raw_values.split(",") if v.strip()]


            if len(names_split) == len(values_split) and len(names_split) > 1:
                for name, val in zip(names_split, values_split):
                    new_inputs.append({
                        "inputName": name,
                        "type": item.get("type", "[UNKNOWN]"),
                        "inputValue": val
                    })
            else:
                # fallback: if lengths don't match, just split the values into a list
                if len(values_split) > 1:
                    item["inputValue"] = values_split
                new_inputs.append(item)

        return new_inputs

    if "inputs" in df.columns:
        df["inputs"] = df["inputs"].apply(process_inputs_cell)
        
    return df


# train

def train_main(
    seed: int = 42,
    MODEL_DIR: str = "models",
    VOCAB_PATH: str = "models/vocabs_train_only.json",
):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    EMBEDDING_DIM = 64
    BLOCK_SIZE = 128
    BATCH_SIZE = 16
    EPOCHS = 10

    file_paths = [
        "data/traceformer_train_val_test_log.json"
    ]

    print("Loading datasets...")
    df = load_and_merge_logs(file_paths, verbose=True)
    df = clean_dataframe(df)

    print(f"The dataset contains {len(df)} rows.")

    tokenizer_src = None
    fp = _files_fingerprint(file_paths, extra_paths=tokenizer_src)

    out_token_seqs, tree_token_seqs, ctx_token_seqs = get_or_build_token_seqs(
        df=df,
        file_paths=file_paths,
        cache_path="cache/tokenized_seqs.pkl",
        extra_fingerprint_paths=tokenizer_src
    )
    
    train_idx, val_idx, test_idx = make_split_indices(len(out_token_seqs), seed=seed)
    
    out_vocab, tree_vocab, ctx_vocab, vocab_meta = get_or_build_vocabs_train_only(
        out_token_seqs,
        tree_token_seqs,
        ctx_token_seqs,
        train_idx=train_idx,
        file_paths=file_paths,
        vocab_path=VOCAB_PATH,
        fingerprint=fp,
        seed=seed,
    )
    
    out_seqs = encode_token_seqs(out_token_seqs, out_vocab)
    tree_seqs = encode_token_seqs(tree_token_seqs, tree_vocab)
    ctx_seqs = encode_token_seqs(ctx_token_seqs, ctx_vocab)

    for i in range(len(out_seqs)):
        assert out_seqs[i].numel() == tree_seqs[i].numel() == ctx_seqs[i].numel()

    train_out = subset(out_seqs, train_idx)
    train_tree = subset(tree_seqs, train_idx)
    train_ctx = subset(ctx_seqs, train_idx)

    val_out = subset(out_seqs, val_idx)
    val_tree = subset(tree_seqs, val_idx)
    val_ctx = subset(ctx_seqs, val_idx)

    steps_per_epoch = 2000
    num_samples_per_epoch = steps_per_epoch * BATCH_SIZE

    train_ds = TxRandomWindowDataset(
        train_out, train_tree, train_ctx,
        block_size=BLOCK_SIZE,
        num_samples_per_epoch=num_samples_per_epoch,
    )
    val_ds = TxStrideWindowDataset(
        val_out, val_tree, val_ctx,
        block_size=BLOCK_SIZE,
        stride=BLOCK_SIZE,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

    print("Train windows per epoch:", len(train_ds))
    print("Val windows:", len(val_ds))

    model = GPTModel(
        vocab_size=len(out_vocab),
        tree_vocab_size=len(tree_vocab),
        ctx_vocab_size=len(ctx_vocab),
        embed_dim=EMBEDDING_DIM,
        feed_forward_dim=4 * EMBEDDING_DIM,
        num_heads=8,
        key_dim=EMBEDDING_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    train_loss_fn = nn.CrossEntropyLoss()
    score_loss_fn = nn.CrossEntropyLoss(reduction="none")

    os.makedirs(MODEL_DIR, exist_ok=True)

    last_ckpt_path = None
    for epoch in range(EPOCHS):
        train_network(model, optimizer, train_loss_fn, train_loader, DEVICE, epoch, EPOCHS)
        valid_network(model, score_loss_fn, val_loader, DEVICE, epoch, EPOCHS)

        last_ckpt_path = os.path.join(MODEL_DIR, f"anomaly_detection_model_{epoch}.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "seed": seed,
                "split": {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx},
                "vocab_path": VOCAB_PATH,
                "data_fingerprint": fp,
            },
            last_ckpt_path,
        )
        print("Model saved!", last_ckpt_path)

    return last_ckpt_path


# test/eval

def test_main(
    MODEL_PATH: str,
    VOCAB_PATH: str | None = None,
):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    EMBEDDING_DIM = 64
    BLOCK_SIZE = 128
    SCORE_KEY = "mean_token_nll"

    file_paths = [
        "data/traceformer_train_val_test_log.json"
    ]

    # step 1: compute threshold based on 99th percentile of scores distribution (on test data). The threshold will be used to flag txs as normal or anomalous.
    df = load_and_merge_logs(file_paths, verbose=False)
    df = clean_dataframe(df)

    tokenizer_src = None
    normal_out_token_seqs, normal_tree_token_seqs, normal_ctx_token_seqs = get_or_build_token_seqs(
        df=df,
        file_paths=file_paths,
        cache_path="cache/tokenized_seqs.pkl",
        extra_fingerprint_paths=tokenizer_src
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint must contain key 'model_state'.")

    split = checkpoint.get("split")
    if not split:
        raise ValueError("Checkpoint missing split indices. Re-train or save split in checkpoint.")
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    if VOCAB_PATH is None:
        VOCAB_PATH = checkpoint.get("vocab_path", "models/vocabs_train_only.json")

    out_vocab, tree_vocab, ctx_vocab, _meta = load_vocabs(VOCAB_PATH)
    print(f"Loaded vocabs from {VOCAB_PATH}")

    model = GPTModel(
        vocab_size=len(out_vocab),
        tree_vocab_size=len(tree_vocab),
        ctx_vocab_size=len(ctx_vocab),
        embed_dim=EMBEDDING_DIM,
        feed_forward_dim=4 * EMBEDDING_DIM,
        num_heads=8,
        key_dim=EMBEDDING_DIM,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("Model loaded from", MODEL_PATH)

    score_loss_fn = nn.CrossEntropyLoss(reduction="none")

    normal_out_seqs = encode_token_seqs(normal_out_token_seqs, out_vocab)
    normal_tree_seqs = encode_token_seqs(normal_tree_token_seqs, tree_vocab)
    normal_ctx_seqs = encode_token_seqs(normal_ctx_token_seqs, ctx_vocab)

    normal_test_out = subset(normal_out_seqs, test_idx)
    normal_test_tree = subset(normal_tree_seqs, test_idx)
    normal_test_ctx = subset(normal_ctx_seqs, test_idx)

    print("Computing test score distribution...")
    test_scores = score_transactions(
        model=model,
        loss_function_none=score_loss_fn,
        out_seqs=normal_test_out,
        tree_seqs=normal_test_tree,
        ctx_seqs=normal_test_ctx,
        block_size=BLOCK_SIZE,
        device=DEVICE,
    )
    if not test_scores:
        raise ValueError("No test scores computed.")

    test_tx_scores = [s[SCORE_KEY] for s in test_scores]
    thr = torch.quantile(torch.tensor(test_tx_scores, dtype=torch.float64), 0.99).item()
    print(f"Threshold (99th percentile test {SCORE_KEY}): {thr:.6f}")

    # step 2: evaluate the model
    ATTACK_DIR = "data/AttacksLogs"
    attack_paths = sorted(glob.glob(os.path.join(ATTACK_DIR, "*.json")))
    print(f"Found {len(attack_paths)} files in {ATTACK_DIR}")

    normal_out_s_copy, normal_tree_s_copy, normal_ctx_s_copy = get_or_build_token_seqs(
        df=df,
        file_paths=file_paths,
        cache_path="cache/tokenized_seqs.pkl",
        extra_fingerprint_paths=tokenizer_src
    )

    normal_test_out_copy = subset(normal_out_s_copy, test_idx)
    normal_test_tree_copy = subset(normal_tree_s_copy, test_idx)
    normal_test_ctx_copy = subset(normal_ctx_s_copy, test_idx)

    normal_test_out_copy = encode_token_seqs(normal_test_out_copy, out_vocab)
    normal_test_tree_copy = encode_token_seqs(normal_test_tree_copy, tree_vocab)
    normal_test_ctx_copy = encode_token_seqs(normal_test_ctx_copy, ctx_vocab)
    
    all_attacks_df = pd.DataFrame()
    attack_dfs = []
    for path in attack_paths:
        with open(path, "r", encoding="utf-8") as f:
            attack_data = json.load(f)
        if isinstance(attack_data, dict):
            attack_data = [attack_data]
            
        attack_df = pd.DataFrame.from_records(attack_data)
        
        if "_id" in attack_df.columns:
            attack_df = attack_df.drop(columns=["_id"])

        attack_dfs.append(attack_df)

    if attack_dfs:
        all_attacks_df = pd.concat(attack_dfs, ignore_index=True)
    
    attack_out_s, attack_tree_s, attack_ctx_s = get_or_build_token_seqs(
        df=all_attacks_df,
        file_paths=file_paths,
        cache_path="cache/tokenized_attacks_seqs.pkl",
    )

    attack_out_s = encode_token_seqs(attack_out_s, out_vocab)
    attack_tree_s = encode_token_seqs(attack_tree_s, tree_vocab)
    attack_ctx_s = encode_token_seqs(attack_ctx_s, ctx_vocab)

    n_attacks = len(attack_out_s)
    print("Attack transactions:", n_attacks)
    if n_attacks == 0:
        print("No attack transactions found. Exiting.")
        return

    print("Test transactions:", len(normal_test_out_copy))

    print("Scoring attack transactions...")
    attack_scores = score_transactions(
        model=model,
        loss_function_none=score_loss_fn,
        out_seqs=attack_out_s,
        tree_seqs=attack_tree_s,
        ctx_seqs=attack_ctx_s,
        block_size=BLOCK_SIZE,
        device=DEVICE,
    )

    print("Scoring normal transactions...")
    normal_scores = score_transactions(
        model=model,
        loss_function_none=score_loss_fn,
        out_seqs=normal_test_out_copy,
        tree_seqs=normal_test_tree_copy,
        ctx_seqs=normal_test_ctx_copy,
        block_size=BLOCK_SIZE,
        device=DEVICE,
    )

    atk = [s[SCORE_KEY] for s in attack_scores]
    nor = [s[SCORE_KEY] for s in normal_scores]

    def q(x, p):
        return float(torch.quantile(torch.tensor(x, dtype=torch.float64), p).item())

    print("\nScore stats (mean_token_nll):")
    print(f" Attacks: n={len(atk)}  min={min(atk):.4f}  p50={q(atk,0.5):.4f}  p90={q(atk,0.9):.4f}  max={max(atk):.4f}")
    print(f" Normals: n={len(nor)}  min={min(nor):.4f}  p50={q(nor,0.5):.4f}  p90={q(nor,0.9):.4f}  max={max(nor):.4f}")
    print(f" Threshold: {thr:.4f}")

    y_true = [1] * len(attack_scores) + [0] * len(normal_scores)
    y_score = atk + nor
    y_pred = [1 if sc > thr else 0 for sc in y_score]

    TP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    TN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    FP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    FN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = TP / max(1, TP + FP)
    recall = TP / max(1, TP + FN)
    f1 = (2 * precision * recall) / max(1e-12, (precision + recall))
    acc = (TP + TN) / max(1, (TP + TN + FP + FN))

    print("\n================== TEST RESULTS ==================")
    print(f"Score metric: {SCORE_KEY}")
    print(f"Threshold: {thr:.6f}")
    print(f"N attacks: {len(attack_scores)}  N normals: {len(normal_scores)}")
    print(f"TP={TP}  FP={FP}  TN={TN}  FN={FN}")
    print(f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}  Acc={acc:.4f}")

    missed = []
    for idx, (sc, yp) in enumerate(zip(atk, y_pred[: len(attack_scores)])):
        if yp == 0:
            fn_file, fn_local = attack_meta[idx]
            missed.append((idx, sc, fn_file, fn_local))
    missed.sort(key=lambda x: x[1])

    print("\nMissed attacks (FN), lowest score first:")
    for idx, sc, fn_file, fn_local in missed[:50]:
        print(f"  global_attack_idx={idx}  score={sc:.6f}  file={fn_file}  local_tx={fn_local}")


def main(run_train=True, run_test=True):
    ckpt = None
    if run_train:
        ckpt = train_main(seed=42)

    if run_test:
        if ckpt is not None:
            test_main(MODEL_PATH=ckpt)
        else:
            test_main(MODEL_PATH="models/anomaly_detection_model_9.pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Baseline test execution on mutants.')
    parser.add_argument('--train', action='store_true', help='Whether to run training.')
    parser.add_argument('--test', action='store_true', help='Whether to run testing.')
    args = parser.parse_args()

    main(run_train=args.train, run_test=args.test)

