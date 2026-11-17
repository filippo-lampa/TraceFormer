import os
import glob
import json
import sys
import pickle
import hashlib
from typing import Tuple, List, Dict, Any, Optional
import argparse
import ast

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.tokenizer import tokenizer, build_tree_from_output, build_context_from_tokens, RESERVED_OUTPUT_TOKENS, tokenizer_with_sources
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

import json
import pandas as pd

def load_and_merge_logs(file_paths: list[str], verbose: bool = False) -> pd.DataFrame:
    dfs = []
    source_files_per_row = []
    for path in file_paths:
        if verbose:
            print(f"Reading {path}...")
            
        try:
            # try loading as standard single-object JSON
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                data = [data]
                
            tmp = pd.DataFrame.from_records(data)
            
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                if verbose:
                    print(f"  -> Multi-object JSON detected. Parsing safely line-by-line...")
                
                # fallback, parse manually with standard json library to avoid Pandas limits on addresses or large integers
                data = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip(): 
                            data.append(json.loads(line))
                
                tmp = pd.DataFrame.from_records(data)
            else:
                raise e

        if "_id" in tmp.columns:
            tmp = tmp.drop(columns=["_id"])
            
        dfs.append(tmp)
        source_files_per_row.extend([os.path.basename(path)] * len(tmp))
        if verbose:
            print(f" -> Loaded {len(tmp)} rows from {path}")
            
    merged = pd.concat(dfs, ignore_index=True)
    # Sidecar metadata only: no extra dataframe column / tokenizer changes.
    merged.attrs["source_files_per_row"] = source_files_per_row
    return merged

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
        if fingerprint is not None and meta.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"Stale vocabulary at {vocab_path}. Archive it, choose a new "
                "VOCAB_PATH and retrain after changing the tokenizer."
            )
        print(f"Loaded vocabs from {vocab_path}")
        return out_vocab, tree_vocab, ctx_vocab, meta

    print(f"No vocab file found at {vocab_path}. Building from TRAIN only and saving...")

    out_train = flatten_token_seqs(subset(out_token_seqs, train_idx))
    tree_train = flatten_token_seqs(subset(tree_token_seqs, train_idx))
    ctx_train = flatten_token_seqs(subset(ctx_token_seqs, train_idx))

    out_vocab = build_vocab(out_train)
    tree_vocab = build_vocab(tree_train)
    ctx_vocab = build_vocab(ctx_train)

    # remove overlapping tokens from out_vocab
    
    overlapping = (set(tree_vocab.keys()) | set(ctx_vocab.keys())) - {"[PAD]", "[UNK]"}
    
    cleaned_out_tokens = [tok for tok in out_vocab.keys() if tok not in overlapping]
    # Keep the intended cross-stream separation. Reserve all normalization markers.
    for marker in RESERVED_OUTPUT_TOKENS:
        if marker not in cleaned_out_tokens:
            cleaned_out_tokens.append(marker)
    
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

def load_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        data = json.loads(content)

        if isinstance(data, dict):
            return [data]

        return data

    except json.JSONDecodeError as normal_json_error:
        records = []

        try:
            for line_number, line in enumerate(content.splitlines(), start=1):
                if not line.strip():
                    continue

                records.append(json.loads(line))

            return records

        except json.JSONDecodeError as jsonl_error:
            raise ValueError(
                f"Could not parse {path} as either JSON or JSONL.\n"
                f"Normal JSON error: {normal_json_error}\n"
                f"JSONL error on line {line_number}: {jsonl_error}"
            ) from jsonl_error

# train

def train_main(seed: int = 42, MODEL_DIR: str = "models/names_abstracted_v4_list_buckets", VOCAB_PATH: str = "models/names_abstracted_v4_list_buckets/vocabs_train_only.json",):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    EMBEDDING_DIM = 64
    BLOCK_SIZE = 128
    BATCH_SIZE = 16
    EPOCHS = 10

    file_paths = [
        "data/0x5efda50f22d34f262c29268506c5fa42cb56a1ce.json",
        "data/0x556b9306565093c855aea9ae92a594704c2cd59e.json",
        "data/0x06012c8cf97bead5deae237070f9587f8e7a266d.json",
        "data/0x323a76393544d5ecca80cd6ef2a560c6a395b7e3_from 13942537 - to 23483870.json",
        "data/0x35d1b3f3d7966a1dfe207aa4514c12a259a0492b_ from 8953736 - to 23228317.json",
        "data/0xb1690c08e213a35ed9bab7b318de14420fb57d8c.json",
        "data/0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc_from 10019997 to 10961610.json",
        "data/0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d.json",
        "data/0xff9c1b15b16263c61d017ee9f65c50e4ae0113d7_ from 13108887 - to 13124897.json",
    ]

    print("Loading datasets...")
    df = load_and_merge_logs(file_paths, verbose=True)
    df = clean_dataframe(df)

    print(f"The dataset contains {len(df)} rows.")

    tokenizer_src = ["src/tokenizer.py"]
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
    val_ds = TxStrideWindowDataset(val_out, val_tree, val_ctx, block_size=BLOCK_SIZE, stride=BLOCK_SIZE,)

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

# ============================================================
# EXTRA IMPORTS FOR ATTRIBUTION / VISUALISATION
# ============================================================

import os
import glob
import re

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# Important on headless servers / SSH / containers
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ============================================================
# ATTRIBUTION TOKEN CLASSES
# ============================================================

STRUCTURAL_TOKENS = {
    "events",
    "eventValues",
    "eventName",
    "eventFrom",
    "callId",
    "[START]",
    "[END]",
    "[CALL_START]",
    "[CALL_END]",
    "[CALLS_START]",
    "[CALLS_END]",
    "[INsSTART]",
    "[INsEND]",
    "[INXsSTART]",
    "[INXsEND]",
    "[EVsSTART]",
    "[EVsEND]",
}

def is_structural_token(token: str) -> bool:
    """
    Used ONLY for interpretability.

    Structural tokens remain part of the model input
    and remain part of the real anomaly score.
    """

    token = str(token)

    if token in STRUCTURAL_TOKENS:
        return True

    # Covers:
    # [LIST_0], [LIST_1], [LIST_55], ...
    if token.startswith("[LIST_") and token.endswith("]"):
        return True

    return False

# ============================================================
# TERMINAL TOKEN PREVIEW
# ============================================================

def preview_token(token: str, max_len: int = 100,) -> str:
    """
    Shorten long tokens ONLY for terminal / plot display.

    The full raw token remains available in CSV output.
    """

    token = str(token)

    # Prevent giant multiline proposal text from destroying
    # terminal readability.
    token = (
        token
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )

    if len(token) <= max_len:
        return token

    return (token[:max_len] + f"... <{len(token)} chars>")

def safe_file_component(text: str, max_len: int = 120,) -> str:
    """
    Convert arbitrary labels into safe filenames.
    """

    text = str(text)

    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text,)

    text = text.strip("_")

    if not text:
        text = "transaction"

    return text[:max_len]

# ============================================================
# UNK DIAGNOSTICS
# ============================================================

def raw_unk_target_fraction(raw_tokens, out_vocab,):
    """
    Fraction of autoregressive target positions whose raw
    token is outside the vocabulary.

    Position 0 is excluded because it is not predicted.
    """

    if len(raw_tokens) < 2:
        return float("nan")

    targets = raw_tokens[1:]

    unk_count = sum(1 for tok in targets if tok not in out_vocab)

    return (unk_count / len(targets))

def detailed_unk_metrics(score_result, raw_tokens, out_vocab,):
    """
    Measures how much of the transaction NLL came from
    positions represented as [UNK].
    """

    details = score_result.get("token_details", [],)

    if not details:
        return {
            "num_scored_tokens": 0,
            "unk_count": 0,
            "known_count": 0,
            "unk_target_fraction": float("nan"),
            "unk_nll": 0.0,
            "known_nll": 0.0,
            "unk_nll_fraction": float("nan"),
            "mean_unk_nll": float("nan"),
            "mean_known_nll": float("nan"),
            "known_only_score": float("nan"),
        }

    unk_nll = 0.0
    known_nll = 0.0

    unk_count = 0
    known_count = 0

    for d in details:

        pos = d["target_position"]

        if pos < 0 or pos >= len(raw_tokens):
            raise IndexError(
                f"Attribution position {pos} is outside "
                f"raw sequence length {len(raw_tokens)}. "
                f"Check token cache/index alignment."
            )

        raw_token = raw_tokens[pos]
        nll = float(d["nll"])

        if raw_token in out_vocab:

            known_count += 1
            known_nll += nll

        else:

            unk_count += 1
            unk_nll += nll

    total_count = (unk_count + known_count)

    total_nll = (unk_nll + known_nll)

    return {
        "num_scored_tokens": total_count,

        "unk_count": unk_count,
        "known_count": known_count,

        "unk_target_fraction": (
            unk_count / total_count
            if total_count > 0
            else float("nan")
        ),

        "unk_nll": unk_nll,
        "known_nll": known_nll,

        "unk_nll_fraction": (
            unk_nll / total_nll
            if total_nll > 0
            else float("nan")
        ),

        "mean_unk_nll": (
            unk_nll / unk_count
            if unk_count > 0
            else float("nan")
        ),

        "mean_known_nll": (
            known_nll / known_count
            if known_count > 0
            else float("nan")
        ),

        # DIAGNOSTIC ONLY.
        # Do not use as the main detector score yet.
        "known_only_score": (
            known_nll / known_count
            if known_count > 0
            else float("nan")
        ),
    }

# ============================================================
# RESCORING SELECTED TRANSACTIONS
# ============================================================

def rescore_indices_with_details(
    indices,
    model,
    loss_function_none,
    out_seqs,
    tree_seqs,
    ctx_seqs,
    block_size,
    device,
):
    """
    Rescore only selected transactions with token details.

    Useful for false-positive analysis without storing token
    NLL for the entire normal test set.
    """

    detailed = {}

    for original_idx in indices:

        scores = score_transactions(
            model=model,
            loss_function_none=loss_function_none,
            out_seqs=[
                out_seqs[original_idx]
            ],
            tree_seqs=[
                tree_seqs[original_idx]
            ],
            ctx_seqs=[
                ctx_seqs[original_idx]
            ],
            block_size=block_size,
            device=device,
            return_token_details=True,
        )

        if not scores:
            continue

        result = scores[0]

        # score_transactions() sees a one-element subset,
        # therefore its local index is zero.
        #
        # Restore the index relative to the original test subset.
        result["tx_index"] = original_idx

        detailed[original_idx] = result

    return detailed

# ============================================================
# TOKEN ATTRIBUTION ENRICHMENT
# ============================================================

def build_token_attribution(score_result, raw_tokens, out_vocab,):
    """
    Add raw token identity, actual model token, UNK flag,
    and structural classification.
    """

    enriched = []

    for d in score_result.get("token_details", [],):

        pos = d["target_position"]

        if pos < 0 or pos >= len(raw_tokens):
            raise IndexError(f"Position {pos} outside sequence " f"length {len(raw_tokens)}")

        raw_token = raw_tokens[pos]

        is_unk = (raw_token not in out_vocab)

        enriched.append({
            **d,
            "raw_token": raw_token,
            "model_token": (
                "[UNK]"
                if is_unk
                else raw_token
            ),
            "is_unk": is_unk,
            "is_structural": (
                is_structural_token(
                    raw_token
                )
            ),
        })

    return sorted(enriched, key=lambda x: x["nll"], reverse=True,)

# ============================================================
# GRAPH 1:
# FULL TOKEN NLL LANDSCAPE
# ============================================================

def validate_token_nll_coverage(score_result, raw_tokens):
    """Prove that every valid next-token target has exactly one scored NLL.

    With block_size=stride, engine.score_transactions scores positions 1..L-1,
    and excludes padding via real_token_losses[:valid_len].
    """
    details = score_result.get("token_details", [])
    positions = [int(d["target_position"]) for d in details]
    expected = list(range(1, len(raw_tokens)))
    if positions != expected:
        missing = sorted(set(expected) - set(positions))[:12]
        extra = sorted(set(positions) - set(expected))[:12]
        raise ValueError(
            f"NLL coverage mismatch: expected {len(expected)} targets, "
            f"got {len(positions)}; first missing={missing}, extra={extra}. "
            "Check scoring stride (use stride=block_size), token alignment, "
            "and token_details collection."
        )
    if score_result.get("num_tokens") != len(raw_tokens):
        raise ValueError("Score result sequence length differs from raw tokens")
    total_nll = sum(float(d["nll"]) for d in details)
    if not np.isclose(total_nll, float(score_result["total_nll"]), rtol=1e-5, atol=1e-3):
        raise ValueError("Per-token NLL sum disagrees with transaction total NLL")
    if details and not np.isclose(
        total_nll / len(details), float(score_result["mean_token_nll"]),
        rtol=1e-5, atol=1e-5
    ):
        raise ValueError("Mean per-token NLL disagrees with transaction score")
    return details


def plot_token_nll_sequence(
    score_result, raw_tokens, out_vocab, title, output_path,
    top_k_annotations=8, rolling_window=25,
):
    """Full NLL with linear and log10 displays, explicit coverage verification."""
    details = validate_token_nll_coverage(score_result, raw_tokens)
    if not details:
        return
    positions = np.asarray([d["target_position"] for d in details], dtype=int)
    nlls = np.asarray([d["nll"] for d in details], dtype=float)
    if not np.all(np.isfinite(nlls)) or (nlls < 0).any():
        raise ValueError("Invalid per-token NLL: negative or non-finite value")
    rolling = pd.Series(nlls).rolling(window=rolling_window, center=True, min_periods=1).mean().to_numpy()
    unk_mask = np.asarray([raw_tokens[pos] not in out_vocab for pos in positions])
    zero_count = int(np.count_nonzero(nlls == 0))
    small_count = int(np.count_nonzero((nlls > 0) & (nlls < 1e-4)))
    print(f"  NLL diagnostic: targets={len(nlls)}; exact_zero={zero_count}; "
          f"0<NLL<1e-4={small_count}; min={nlls.min():.9g}; "
          f"median={np.median(nlls):.9g}; max={nlls.max():.9g}")
    fig, (ax, axlog) = plt.subplots(2, 1, figsize=(17, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(positions, nlls, linewidth=0.65, alpha=0.6, label="All target NLLs")
    ax.plot(positions, rolling, linewidth=1.5, label=f"Rolling mean ({rolling_window})")
    if unk_mask.any():
        ax.scatter(positions[unk_mask], nlls[unk_mask], marker="x", s=12,
                   alpha=0.55, label="[UNK] target")
    for idx in np.argsort(nlls)[-min(top_k_annotations, len(nlls)):][::-1]:
        pos = int(positions[idx])
        token = preview_token(raw_tokens[pos], max_len=30)
        ax.annotate(token, xy=(pos, nlls[idx]), xytext=(0, 9),
                    textcoords="offset points", rotation=30, fontsize=7,
                    ha="left")
    # Vertical dashed lines show each new, independently scored model window.
    # The last target in a window is predicted with preceding tokens only.
    window_size = 128  # test_main BLOCK_SIZE; must stay in sync with scorer.
    for boundary in range(window_size + 1, len(raw_tokens), window_size):
        ax.axvline(boundary, color="grey", linewidth=0.35, alpha=0.25, linestyle="--")
        axlog.axvline(boundary, color="grey", linewidth=0.35, alpha=0.25, linestyle="--")
    ax.set_ylabel("NLL (linear scale)")
    ax.set_title(title + f" | scored targets: {len(nlls)}/{len(raw_tokens)-1}")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right")
    # Use a floor ONLY for visualization; exported NLLs and scores are untouched.
    log_floor = 1e-8
    axlog.plot(positions, np.maximum(nlls, log_floor), linewidth=0.8)
    axlog.set_yscale("log")
    axlog.set_ylabel("NLL (log scale)")
    axlog.set_xlabel("Original token position (position 0 has no next-token target)")
    axlog.grid(alpha=0.15)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved NLL landscape: {output_path}")

# ============================================================
# GRAPH 2:
# ZOOM AROUND STRONGEST SPIKE
# ============================================================

def plot_nll_spike_zoom(
    score_result,
    raw_tokens,
    out_vocab,
    title,
    output_path,
    radius=20,
    max_annotations=12,
    center_pos=None,
):
    """Plot +/- radius tokens around a chosen spike (default: strongest)."""

    details = score_result.get("token_details", [],)

    if not details:
        return

    strongest = max(details, key=lambda d: d["nll"],)

    if center_pos is None:
        center_pos = int(strongest["target_position"])
    else:
        center_pos = int(center_pos)

    selected = [d for d in details if abs(d["target_position"] - center_pos) <= radius]

    if not selected:
        return

    selected = sorted(selected, key=lambda d: d["target_position"],)

    positions = np.array([d["target_position"] for d in selected])

    nlls = np.array([float(d["nll"]) for d in selected])

    fig, ax = plt.subplots(figsize=(max(16, len(selected) * 0.42), 7))

    ax.plot(positions, nlls, marker="o", markersize=3, linewidth=1.0,)

    center_nll = next(float(d["nll"]) for d in details if int(d["target_position"]) == center_pos)
    ax.axvline(center_pos, linestyle="--", alpha=0.5,
               label=f"Selected spike: position {center_pos}, NLL={center_nll:.2f}")

    # Show EVERY token in this local window as an x-axis tick label.
    # Restricting text annotations to the top N produced unlabeled positions.
    # Short labels and a wide figure make a 41-token window readable.
    labels = []
    for pos in positions:
        tok = preview_token(raw_tokens[int(pos)], max_len=20)
        if raw_tokens[int(pos)] not in out_vocab:
            tok += " [UNK]"
        labels.append(tok)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.tick_params(axis="x", pad=3)

    ax.set_title(
        f"{title}\n"
        f"Local context around NLL spike at token {center_pos} "
        f"(NLL={center_nll:.2f})"
    )

    ax.set_xlabel("Token position")

    ax.set_ylabel("Token NLL")

    ax.grid(alpha=0.2)

    ax.legend()

    fig.subplots_adjust(bottom=0.29, top=0.80)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True,)

    fig.savefig(output_path, dpi=200, bbox_inches="tight",)

    plt.close(fig)

    print(f"Saved spike zoom: " f"{output_path}")

# ============================================================
# GRAPH 3:
# NLL AUTOCORRELATION / PERIODICITY
# ============================================================

def plot_nll_autocorrelation(score_result, title, output_path, max_lag=500,):
    """
    Autocorrelation of token-level NLL.

    Peaks at recurring lags suggest repeated surprisal
    structure in the serialized execution trace.

    This does NOT imply malicious behavior is periodic;
    it may reflect repeated execution grammar.
    """

    details = score_result.get("token_details", [],)

    if len(details) < 3:
        return

    nlls = np.array([float(d["nll"]) for d in details])

    x = (nlls - np.mean(nlls))

    if np.allclose(x, 0,):
        return

    n = len(x)

    # FFT autocorrelation:
    # much faster than repeatedly computing dot products
    # for long transactions.
    n_fft = 1 << ((2 * n - 1) .bit_length())

    spectrum = np.fft.rfft(x, n=n_fft,)

    autocorr = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft,)[:n]

    if autocorr[0] == 0:
        return

    autocorr = (autocorr / autocorr[0])

    max_lag = min(max_lag, n - 1,)

    lags = np.arange(1, max_lag + 1,)

    values = autocorr[1:max_lag + 1]

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(lags, values, linewidth=1.0,)

    ax.axhline(0, linewidth=0.8, alpha=0.5,)

    ax.set_title(
        f"{title}\n"
        "Autocorrelation of token-level NLL"
    )

    ax.set_xlabel("Lag (tokens)")

    ax.set_ylabel("Autocorrelation")

    ax.grid(alpha=0.2)

    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True,)

    fig.savefig(output_path, dpi=200, bbox_inches="tight",)

    plt.close(fig)

    print(f"Saved NLL autocorrelation: " f"{output_path}")

# ============================================================
# GENERATE ALL THREE PLOTS
# ============================================================

def generate_transaction_plots(score_result, raw_tokens, out_vocab, title, output_dir, filename_stem,):
    """
    Generate:
        1. full NLL sequence
        2. strongest-spike zoom
        3. autocorrelation
    """

    os.makedirs(output_dir, exist_ok=True,)

    stem = safe_file_component(filename_stem)

    plot_token_nll_sequence(
        score_result=score_result,
        raw_tokens=raw_tokens,
        out_vocab=out_vocab,
        title=title,
        output_path=os.path.join(
            output_dir,
            f"{stem}_sequence.png",
        ),
    )

    plot_nll_spike_zoom(
        score_result=score_result,
        raw_tokens=raw_tokens,
        out_vocab=out_vocab,
        title=title,
        output_path=os.path.join(
            output_dir,
            f"{stem}_spike_zoom.png",
        ),
    )

    # Additional zooms for EVERY other token with NLL > 20.
    # The strongest-spike plot above remains unchanged in filename.
    details = score_result.get("token_details", [])
    if details:
        strongest_pos = int(max(details, key=lambda d: float(d["nll"]))["target_position"])
        extra_spikes = sorted(
            (d for d in details
             if float(d["nll"]) > 20.0
             and int(d["target_position"]) != strongest_pos),
            key=lambda d: int(d["target_position"]),
        )
        for spike in extra_spikes:
            pos = int(spike["target_position"])
            plot_nll_spike_zoom(
                score_result=score_result,
                raw_tokens=raw_tokens,
                out_vocab=out_vocab,
                title=title,
                output_path=os.path.join(output_dir, f"{stem}_spike_zoom_pos_{pos}.png"),
                center_pos=pos,
            )
        print(f"Additional NLL>20 spike zooms for {stem}: {len(extra_spikes)}")

    plot_nll_autocorrelation(
        score_result=score_result,
        title=title,
        output_path=os.path.join(
            output_dir,
            f"{stem}_autocorrelation.png",
        ),
    )

# ============================================================
# FINAL TEST FUNCTION
# ============================================================

def test_main(MODEL_PATH: str, VOCAB_PATH: str | None = None,):

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", DEVICE,)

    EMBEDDING_DIM = 64
    BLOCK_SIZE = 128

    SCORE_KEY = ("mean_token_nll")

    TOP_K = 15

    TOP_ATTACKS_TO_EXPLAIN = 10
    TOP_FPS_TO_EXPLAIN = 10

    # Number for which graphical output is generated.
    # Change if you want fewer PNGs.
    TOP_ATTACKS_TO_PLOT = 10
    TOP_FPS_TO_PLOT = 5

    TN_DIAGNOSTIC_SAMPLE = 200

    # ========================================================
    # NORMAL DATA
    # ========================================================

    file_paths = [
        "data/0x5efda50f22d34f262c29268506c5fa42cb56a1ce.json",
        "data/0x556b9306565093c855aea9ae92a594704c2cd59e.json",
        "data/0x06012c8cf97bead5deae237070f9587f8e7a266d.json",
        "data/0x323a76393544d5ecca80cd6ef2a560c6a395b7e3_from 13942537 - to 23483870.json",
        "data/0x35d1b3f3d7966a1dfe207aa4514c12a259a0492b_ from 8953736 - to 23228317.json",
        "data/0xb1690c08e213a35ed9bab7b318de14420fb57d8c.json",
        "data/0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc_from 10019997 to 10961610.json",
        "data/0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d.json",
        "data/0xff9c1b15b16263c61d017ee9f65c50e4ae0113d7_ from 13108887 - to 13124897.json",
    ]

    print(
        "\nLoading normal dataset..."
    )

    df = load_and_merge_logs(file_paths, verbose=False,)

    # Preserve the original row -> JSON filename mapping before cleaning.
    normal_source_files = df.attrs["source_files_per_row"]
    df = clean_dataframe(df)

    tokenizer_src = ["src/tokenizer.py"]

    (
        normal_out_token_seqs,
        normal_tree_token_seqs,
        normal_ctx_token_seqs,
    ) = get_or_build_token_seqs(
        df=df,
        file_paths=file_paths,
        cache_path=(
            "cache/tokenized_seqs.pkl"
        ),
        extra_fingerprint_paths=(
            tokenizer_src
        ),
    )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE,)

    if "model_state" not in checkpoint:

        raise KeyError("Checkpoint must contain " "'model_state'.")

    split = checkpoint.get("split")

    if not split:

        raise ValueError("Checkpoint missing " "split indices.")

    train_idx = split["train_idx"]

    val_idx = split["val_idx"]

    test_idx = split["test_idx"]

    # ========================================================
    # VOCAB
    # ========================================================

    if VOCAB_PATH is None:

        VOCAB_PATH = checkpoint.get("vocab_path", "models/vocabs_train_only.json",)

    (out_vocab, tree_vocab, ctx_vocab, _meta,) = load_vocabs(VOCAB_PATH)

    print("Loaded vocabs from", VOCAB_PATH,)

    # ========================================================
    # MODEL
    # ========================================================

    model = GPTModel(
        vocab_size=len(
            out_vocab
        ),
        tree_vocab_size=len(
            tree_vocab
        ),
        ctx_vocab_size=len(
            ctx_vocab
        ),
        embed_dim=EMBEDDING_DIM,
        feed_forward_dim=(
            4 * EMBEDDING_DIM
        ),
        num_heads=8,
        key_dim=EMBEDDING_DIM,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    print("Model loaded from", MODEL_PATH,)

    score_loss_fn = (nn.CrossEntropyLoss(reduction="none"))

    # ========================================================
    # ENCODE NORMAL DATA
    # ========================================================

    normal_out_seqs = (encode_token_seqs(normal_out_token_seqs, out_vocab,))

    normal_tree_seqs = (encode_token_seqs(normal_tree_token_seqs, tree_vocab,))

    normal_ctx_seqs = (encode_token_seqs(normal_ctx_token_seqs, ctx_vocab,))

    # ========================================================
    # VALIDATION THRESHOLD
    # ========================================================

    normal_val_out = subset(normal_out_seqs, val_idx,)

    normal_val_tree = subset(normal_tree_seqs, val_idx,)

    normal_val_ctx = subset(normal_ctx_seqs, val_idx,)

    print(
        "\nScoring validation set..."
    )

    val_scores = score_transactions(
        model=model,
        loss_function_none=(
            score_loss_fn
        ),
        out_seqs=(
            normal_val_out
        ),
        tree_seqs=(
            normal_val_tree
        ),
        ctx_seqs=(
            normal_val_ctx
        ),
        block_size=BLOCK_SIZE,
        device=DEVICE,
    )

    val_tx_scores = [s[SCORE_KEY] for s in val_scores]

    thr = torch.quantile(torch.tensor(val_tx_scores, dtype=torch.float64,),  0.99,).item()

    print(f"Validation P99 threshold: " f"{thr:.6f}")

    # ========================================================
    # NORMAL TEST SET
    # ========================================================

    normal_test_out = subset(normal_out_seqs, test_idx,)

    normal_test_tree = subset(normal_tree_seqs, test_idx,)

    normal_test_ctx = subset(normal_ctx_seqs, test_idx,)

    # Raw tokens preserved for interpretability
    normal_test_out_tokens = subset(normal_out_token_seqs, test_idx,)

    # ========================================================
    # ATTACK DATA
    # ========================================================

    ATTACK_DIR = ("data/AttacksLogs")

    attack_paths = sorted(glob.glob(os.path.join(ATTACK_DIR, "*.json",)))

    print(
        f"\nFound "
        f"{len(attack_paths)} "
        f"files in "
        f"{ATTACK_DIR}"
    )

    attack_dfs = []
    attack_meta = []

    for path in attack_paths:

        print(path)

        attack_data = (load_json_or_jsonl(path))

        attack_df = (pd.DataFrame.from_records(attack_data))

        if "_id" in attack_df.columns:

            attack_df = (attack_df.drop(columns=["_id"]))

        attack_dfs.append(attack_df)

        for local_i in range(len(attack_df)):

            attack_meta.append((os.path.basename(path),  local_i,))

        print(f"  -> Loaded " f"{len(attack_df)} " f"transactions")

    if not attack_dfs:

        print("No attack files found.")

        return

    all_attacks_df = pd.concat(attack_dfs, ignore_index=True,)

    all_attacks_df = (clean_dataframe(all_attacks_df))

    print("Total attack transactions loaded:", len(all_attacks_df),)

    # IMPORTANT:
    # fingerprint attack cache with attack_paths
    (
        attack_out_tokens,
        attack_tree_tokens,
        attack_ctx_tokens,
    ) = get_or_build_token_seqs(
        df=all_attacks_df,
        file_paths=attack_paths,
        cache_path=(
            "cache/"
            "tokenized_attacks_seqs.pkl"
        ),
        extra_fingerprint_paths=tokenizer_src,
    )

    attack_out_s = (encode_token_seqs(attack_out_tokens, out_vocab,))

    attack_tree_s = (encode_token_seqs(attack_tree_tokens, tree_vocab,))

    attack_ctx_s = (encode_token_seqs(attack_ctx_tokens, ctx_vocab,))

    print("Raw attack transactions:", len(attack_out_s),)

    # ========================================================
    # SCORE ATTACKS
    # ========================================================

    print(
        "\nScoring attacks..."
    )

    attack_scores = (
        score_transactions(
            model=model,
            loss_function_none=(
                score_loss_fn
            ),
            out_seqs=attack_out_s,
            tree_seqs=(
                attack_tree_s
            ),
            ctx_seqs=attack_ctx_s,
            block_size=BLOCK_SIZE,
            device=DEVICE,
            return_token_details=True,
        )
    )

    scored_attack_indices = {s["tx_index"] for s in attack_scores}

    skipped_attack_indices = sorted(set(range(len(attack_out_s))) - scored_attack_indices)

    print("Scored attacks:", len(attack_scores))

    print("Skipped attack sequences:", len(skipped_attack_indices))

    if skipped_attack_indices:

        print("Skipped attack indices:", skipped_attack_indices)

    # ========================================================
    # SCORE NORMAL TEST
    # ========================================================

    print(
        "\nScoring normal test set..."
    )

    normal_scores = (
        score_transactions(
            model=model,
            loss_function_none=(
                score_loss_fn
            ),
            out_seqs=(
                normal_test_out
            ),
            tree_seqs=(
                normal_test_tree
            ),
            ctx_seqs=(
                normal_test_ctx
            ),
            block_size=BLOCK_SIZE,
            device=DEVICE,
            return_token_details=False,
        )
    )

    atk = [s[SCORE_KEY] for s in attack_scores]

    nor = [s[SCORE_KEY] for s in normal_scores]

    # ========================================================
    # STANDARD METRICS
    # ========================================================

    y_true = ([1] * len(attack_scores) + [0] * len(normal_scores))

    y_score = (atk + nor)

    y_pred = [1 if sc > thr else 0 for sc in y_score]

    TP = sum(1 for yt, yp in zip(y_true, y_pred,) if yt == 1 and yp == 1)

    TN = sum(1 for yt, yp in zip(y_true, y_pred,) if yt == 0 and yp == 0)

    FP = sum(1 for yt, yp in zip(y_true, y_pred,) if yt == 0 and yp == 1)

    FN = sum(1 for yt, yp in zip(y_true, y_pred,) if yt == 1 and yp == 0)

    precision = (TP / max(1, TP + FP,))

    recall = (TP / max(1, TP + FN,))

    f1 = (2 * precision * recall / max(1e-12, precision + recall,))

    acc = ((TP + TN) / max(1, TP + TN + FP + FN,))

    print(
        "\n================== "
        "TEST RESULTS "
        "=================="
    )

    print(f"Threshold: " f"{thr:.6f}")

    print(f"N attacks=" f"{len(attack_scores)} " f"N normals=" f"{len(normal_scores)}")

    print(f"TP={TP} " f"FP={FP} " f"TN={TN} " f"FN={FN}")

    print(
        f"Precision="
        f"{precision:.4f} "
        f"Recall="
        f"{recall:.4f} "
        f"F1="
        f"{f1:.4f} "
        f"Acc="
        f"{acc:.4f}"
    )

    # ========================================================
    # TRANSACTION-LEVEL UNK TABLE
    # ========================================================

    transaction_rows = []

    # ---------------- ATTACKS ----------------

    for score_result in attack_scores:

        tx_index = (score_result["tx_index"])

        raw_tokens = (attack_out_tokens[tx_index])

        metrics = (detailed_unk_metrics(score_result, raw_tokens, out_vocab,))

        fn_file, fn_local = (attack_meta[tx_index])

        pred = (1 if score_result[SCORE_KEY] > thr else 0)

        transaction_rows.append({
            "dataset":
                "attack",
            "class_group":
                (
                    "TP"
                    if pred == 1
                    else "FN"
                ),
            "tx_index":
                tx_index,
            "file":
                fn_file,
            "local_tx":
                fn_local,
            "score":
                score_result[
                    SCORE_KEY
                ],
            "unk_target_fraction":
                metrics[
                    "unk_target_fraction"
                ],
            "unk_nll_fraction":
                metrics[
                    "unk_nll_fraction"
                ],
            "mean_unk_nll":
                metrics[
                    "mean_unk_nll"
                ],
            "mean_known_nll":
                metrics[
                    "mean_known_nll"
                ],
            "known_only_score":
                metrics[
                    "known_only_score"
                ],
        })

    # ---------------- NORMALS ----------------

    for score_result in normal_scores:

        tx_index = (score_result["tx_index"])

        raw_tokens = (normal_test_out_tokens[tx_index])

        unk_fraction = (raw_unk_target_fraction(raw_tokens, out_vocab,))

        pred = (1 if score_result[SCORE_KEY] > thr else 0)

        transaction_rows.append({
            "dataset":
                "normal",
            "class_group":
                (
                    "FP"
                    if pred == 1
                    else "TN"
                ),
            "tx_index":
                tx_index,
            "file":
                None,
            "local_tx":
                None,
            "score":
                score_result[
                    SCORE_KEY
                ],
            "unk_target_fraction":
                unk_fraction,
            "unk_nll_fraction":
                None,
            "mean_unk_nll":
                None,
            "mean_known_nll":
                None,
            "known_only_score":
                None,
        })

    transaction_df = (pd.DataFrame(transaction_rows))

    # ========================================================
    # BASIC UNK ANALYSIS
    # ========================================================

    print(
        "\n================== "
        "UNK ANALYSIS "
        "=================="
    )

    for group in ["TP", "FP", "TN", "FN",]:

        group_df = (transaction_df[transaction_df["class_group"] == group])

        if group_df.empty:
            continue

        print(
            f"\n{group}: "
            f"n={len(group_df)}"
        )

        print("  mean score: " f"{group_df['score'].mean():.4f}")

        print("  median score: " f"{group_df['score'].median():.4f}")

        print("  mean UNK target fraction: " f"{100 * group_df['unk_target_fraction'].mean():.2f}%")

        print("  median UNK target fraction: " f"{100 * group_df['unk_target_fraction'].median():.2f}%")

    # ========================================================
    # CORRELATION
    # ========================================================

    normal_only = (transaction_df[transaction_df["dataset"] == "normal"].copy())

    if len(normal_only) > 1:

        corr = (normal_only[["score", "unk_target_fraction",]] .corr() .iloc[0, 1])

        print(
            "\nCorrelation between "
            "normal anomaly score and "
            f"UNK fraction: "
            f"{corr:.4f}"
        )

    attack_only = (transaction_df[transaction_df["dataset"] == "attack"].copy())

    if len(attack_only) > 1:

        corr = (attack_only[["score", "unk_target_fraction",]] .corr() .iloc[0, 1])

        print("Correlation between " "attack anomaly score and " f"UNK fraction: " f"{corr:.4f}")

    # ========================================================
    # FALSE POSITIVE / TRUE NEGATIVE INDICES
    # ========================================================

    fp_indices = sorted([s["tx_index"]  for s in normal_scores  if s[SCORE_KEY] > thr])

    tn_indices = sorted([s["tx_index"]  for s in normal_scores  if s[SCORE_KEY] <= thr])

    # ========================================================
    # SAMPLE TRUE NEGATIVES
    # ========================================================

    if tn_indices:

        generator = (torch.Generator() .manual_seed(42))

        perm = (torch.randperm(len(tn_indices), generator=generator,) .tolist())

        n_tn_sample = min(TN_DIAGNOSTIC_SAMPLE, len(tn_indices),)

        tn_sample_indices = [tn_indices[i] for i in perm[:n_tn_sample]]

    else:

        tn_sample_indices = []

    # ========================================================
    # RESCORE FALSE POSITIVES WITH TOKEN DETAILS
    # ========================================================

    print(
        "\nRescoring false positives "
        "with token details..."
    )

    fp_detailed = (
        rescore_indices_with_details(
            fp_indices,
            model=model,
            loss_function_none=(
                score_loss_fn
            ),
            out_seqs=(
                normal_test_out
            ),
            tree_seqs=(
                normal_test_tree
            ),
            ctx_seqs=(
                normal_test_ctx
            ),
            block_size=BLOCK_SIZE,
            device=DEVICE,
        )
    )

    print(f"Rescoring " f"{len(tn_sample_indices)} " f"true negatives...")

    tn_detailed = (
        rescore_indices_with_details(
            tn_sample_indices,
            model=model,
            loss_function_none=(
                score_loss_fn
            ),
            out_seqs=(
                normal_test_out
            ),
            tree_seqs=(
                normal_test_tree
            ),
            ctx_seqs=(
                normal_test_ctx
            ),
            block_size=BLOCK_SIZE,
            device=DEVICE,
        )
    )

    # ========================================================
    # DETAILED FP/TN UNK TABLE
    # ========================================================

    detailed_rows = []

    for (group_name, result_map,) in [("FP", fp_detailed,),  ("TN_SAMPLE", tn_detailed,),]:

        for (idx, score_result,) in result_map.items():

            raw_tokens = (normal_test_out_tokens[idx])

            metrics = (detailed_unk_metrics(score_result,  raw_tokens,  out_vocab,))

            detailed_rows.append({
                "group":
                    group_name,
                "tx_index":
                    idx,
                "score":
                    score_result[
                        SCORE_KEY
                    ],
                **metrics,
            })

    detailed_unk_df = (pd.DataFrame(detailed_rows))

    print(
        "\n================== "
        "DETAILED UNK NLL COMPARISON "
        "=================="
    )

    if not detailed_unk_df.empty:

        for group in ["FP", "TN_SAMPLE",]:

            g = (detailed_unk_df[detailed_unk_df["group"] == group])

            if g.empty:
                continue

            print(
                f"\n{group}: "
                f"n={len(g)}"
            )

            print("  Mean score: " f"{g['score'].mean():.4f}")

            print("  Mean UNK target fraction: " f"{100 * g['unk_target_fraction'].mean():.2f}%")

            print("  Mean NLL fraction from UNK: " f"{100 * g['unk_nll_fraction'].mean():.2f}%")

            print("  Mean UNK-token NLL: " f"{g['mean_unk_nll'].mean():.4f}")

            print("  Mean known-token NLL: " f"{g['mean_known_nll'].mean():.4f}")

            print("  Mean known-only score: " f"{g['known_only_score'].mean():.4f}")

    # ========================================================
    # FALSE POSITIVE RANKING
    # ========================================================

    fp_ranking = (transaction_df[transaction_df["class_group"] == "FP"] .sort_values("score", ascending=False,))

    print(
        "\n================== "
        "TOP FALSE POSITIVES "
        "=================="
    )

    for _, row in (fp_ranking .head(20) .iterrows()):

        print(
            f"tx_index="
            f"{int(row['tx_index']):5d} "
            f"score="
            f"{row['score']:.6f} "
            f"UNK_targets="
            f"{100 * row['unk_target_fraction']:.2f}%"
        )

    # ========================================================
    # NORMALS WITH MOST UNK
    # ========================================================

    high_unk_normals = (normal_only .sort_values("unk_target_fraction", ascending=False,))

    print(
        "\n================== "
        "NORMALS WITH MOST UNK "
        "=================="
    )

    for _, row in (high_unk_normals .head(20) .iterrows()):

        print(
            f"tx_index="
            f"{int(row['tx_index']):5d} "
            f"UNK_targets="
            f"{100 * row['unk_target_fraction']:.2f}% "
            f"score="
            f"{row['score']:.6f} "
            f"class="
            f"{row['class_group']}"
        )

    # ========================================================
    # ATTACK RANKING
    # ========================================================

    attack_ranking = sorted(attack_scores,  key=lambda s: s[SCORE_KEY],  reverse=True,)

    print(
        "\n================== "
        "TOP ATTACKS "
        "=================="
    )

    for score_result in (attack_ranking[:TOP_ATTACKS_TO_EXPLAIN]):

        tx_idx = (score_result["tx_index"])

        fn_file, fn_local = (attack_meta[tx_idx])

        metrics = (detailed_unk_metrics(score_result,  attack_out_tokens[tx_idx],  out_vocab,))

        print(
            f"tx_index="
            f"{tx_idx:3d} "
            f"score="
            f"{score_result[SCORE_KEY]:.6f} "
            f"UNK_targets="
            f"{100 * metrics['unk_target_fraction']:.2f}% "
            f"UNK_NLL="
            f"{100 * metrics['unk_nll_fraction']:.2f}% "
            f"file="
            f"{fn_file} "
            f"local_tx="
            f"{fn_local}"
        )

    # ========================================================
    # ATTRIBUTION OUTPUT STORAGE
    # ========================================================

    raw_attr_rows = []
    nonstructural_attr_rows = []

    # ========================================================
    # ATTRIBUTION PRINTER
    # ========================================================

    def print_and_store_attribution(
        dataset_name,
        display_name,
        score_result,
        raw_tokens,
        original_idx,
        local_tx=None,
    ):

        enriched = (build_token_attribution(score_result,  raw_tokens,  out_vocab,))

        top_raw = (enriched[:TOP_K])

        top_nonstructural = [d for d in enriched if not d["is_structural"]][:TOP_K]

        print(
            "\n----------------------------------------"
        )

        print(f"{dataset_name}: " f"{display_name}")

        print(f"score=" f"{score_result[SCORE_KEY]:.6f}")

        # ----------------------------------------------------
        # RAW
        # ----------------------------------------------------

        print(
            "\nTop raw NLL positions:"
        )

        for rank, d in enumerate(top_raw, start=1,):

            print(
                f" {rank:2d}. "
                f"pos="
                f"{d['target_position']:5d} "
                f"NLL="
                f"{d['nll']:.4f} "
                f"share="
                f"{100 * d['fraction_total_nll']:.2f}% "
                f"token="
                f"{preview_token(d['raw_token'])!r} "
                f"model="
                f"{d['model_token']!r} "
                f"UNK="
                f"{d['is_unk']} "
                f"structural="
                f"{d['is_structural']}"
            )

            raw_attr_rows.append({

                "dataset":
                    dataset_name,

                "name":
                    display_name,

                "tx_index":
                    original_idx,

                "local_tx":
                    local_tx,

                "score":
                    score_result[
                        SCORE_KEY
                    ],

                "rank":
                    rank,

                "position":
                    d[
                        "target_position"
                    ],

                # IMPORTANT:
                # Full token preserved in CSV.
                "token":
                    d[
                        "raw_token"
                    ],

                "model_token":
                    d[
                        "model_token"
                    ],

                "is_unk":
                    d[
                        "is_unk"
                    ],

                "is_structural":
                    d[
                        "is_structural"
                    ],

                "nll":
                    d[
                        "nll"
                    ],

                "fraction_total_nll":
                    d[
                        "fraction_total_nll"
                    ],
            })

        # ----------------------------------------------------
        # NON-STRUCTURAL
        # ----------------------------------------------------

        print(
            "\nTop non-structural "
            "NLL positions:"
        )

        for rank, d in enumerate(top_nonstructural, start=1,):

            print(
                f" {rank:2d}. "
                f"pos="
                f"{d['target_position']:5d} "
                f"NLL="
                f"{d['nll']:.4f} "
                f"share="
                f"{100 * d['fraction_total_nll']:.2f}% "
                f"token="
                f"{preview_token(d['raw_token'])!r} "
                f"model="
                f"{d['model_token']!r} "
                f"UNK="
                f"{d['is_unk']}"
            )

            nonstructural_attr_rows.append({
                "dataset":
                    dataset_name,
                "name":
                    display_name,
                "tx_index":
                    original_idx,
                "local_tx":
                    local_tx,
                "score":
                    score_result[
                        SCORE_KEY
                    ],
                "rank":
                    rank,
                "position":
                    d[
                        "target_position"
                    ],
                "token":
                    d[
                        "raw_token"
                    ],
                "model_token":
                    d[
                        "model_token"
                    ],
                "is_unk":
                    d[
                        "is_unk"
                    ],
                "nll":
                    d[
                        "nll"
                    ],
                "fraction_total_nll":
                    d[
                        "fraction_total_nll"
                    ],
            })

    # ========================================================
    # ALL ATTACK TOKEN LOSSES: every target, not just top 15.
    # No zeros inserted for missing positions; coverage is asserted.
    # CSV may contain hundreds of thousands of rows for long attacks.
    # ========================================================
    import csv
    token_dir = "results/attack_token_nll"
    os.makedirs(token_dir, exist_ok=True)
    full_cols = [
        "attack_file", "local_tx", "tx_index", "token_position",
        "input_position", "original_token", "normalized_token", "encoded_token", "encoded_id",
        "nll", "fraction_total_nll", "is_unk", "is_structural",
        "transaction_score",
    ]
    combined_path = os.path.join(token_dir, "ALL_ATTACKS_token_nll.csv")
    with open(combined_path, "w", newline="", encoding="utf-8") as combined_f:
        all_writer = csv.DictWriter(combined_f, fieldnames=full_cols)
        all_writer.writeheader()
        for score_result in attack_scores:
            tx_idx = score_result["tx_index"]
            raw_tokens = attack_out_tokens[tx_idx]
            validate_token_nll_coverage(score_result, raw_tokens)
            provenance = tokenizer_with_sources(
                pd.DataFrame([all_attacks_df.iloc[tx_idx].to_dict()])
            )
            if [p["normalized_token"] for p in provenance] != raw_tokens:
                raise RuntimeError(
                    f"Source-to-token alignment mismatch for attack {tx_idx}; "
                    "check tokenizer version, cache and preprocessing."
                )
            fn_file, fn_local = attack_meta[tx_idx]
            filename = safe_file_component(f"attack_{tx_idx:03d}_{fn_file}_tx_{fn_local}")
            per_attack_path = os.path.join(token_dir, filename + "_token_nll.csv")
            with open(per_attack_path, "w", newline="", encoding="utf-8") as f:
                attack_writer = csv.DictWriter(f, fieldnames=full_cols)
                attack_writer.writeheader()
                # Results are scored left-to-right; preserves original sequence order.
                for d in score_result["token_details"]:
                    pos = int(d["target_position"])
                    normalized = raw_tokens[pos]
                    source = provenance[pos]
                    token_id = out_vocab.get(normalized, out_vocab["[UNK]"])
                    encoded = normalized if normalized in out_vocab else "[UNK]"
                    source_value = source["raw_source_value"]
                    if not source["has_source_value"]:
                        source_text = normalized  # Structural marker or JSON field key.
                    elif isinstance(source_value, str):
                        source_text = source_value
                    else:
                        source_text = json.dumps(source_value, ensure_ascii=False, default=str)
                    row = {
                        "attack_file": fn_file, "local_tx": fn_local,
                        "tx_index": tx_idx, "token_position": pos,
                        "input_position": d["input_position"],
                        "original_token": source_text,
                        "normalized_token": normalized, "encoded_token": encoded,
                        "encoded_id": token_id, "nll": repr(float(d["nll"])),
                        "fraction_total_nll": repr(float(d["fraction_total_nll"])),
                        "is_unk": normalized not in out_vocab,
                        "is_structural": is_structural_token(normalized),
                        "transaction_score": repr(float(score_result[SCORE_KEY])),
                    }
                    attack_writer.writerow(row)
                    all_writer.writerow(row)
            print(f"Saved all-token attack table: {per_attack_path} "
                  f"({len(score_result['token_details'])} targets)")
    print(f"Saved combined all-attack table: {combined_path}")

    # ========================================================
    # ATTACK ATTRIBUTION + PLOTS
    # ========================================================

    print(
        "\n================== "
        "ATTACK ATTRIBUTION "
        "=================="
    )

    for rank_idx, score_result in enumerate(attack_ranking[:TOP_ATTACKS_TO_EXPLAIN], start=1,):

        tx_idx = (score_result["tx_index"])

        fn_file, fn_local = (attack_meta[tx_idx])

        raw_tokens = (attack_out_tokens[tx_idx])

        print_and_store_attribution(
            dataset_name="ATTACK",
            display_name=fn_file,
            score_result=(
                score_result
            ),
            raw_tokens=(
                raw_tokens
            ),
            original_idx=(
                tx_idx
            ),
            local_tx=(
                fn_local
            ),
        )

        # ====================================================
        # GRAPHICAL ANALYSIS
        # ====================================================

        if (rank_idx <= TOP_ATTACKS_TO_PLOT):

            plot_title = (f"{fn_file} | " f"local_tx={fn_local} | " f"score=" f"{score_result[SCORE_KEY]:.3f}")

            plot_stem = (f"rank_{rank_idx:02d}_" f"attack_{tx_idx}_" f"{fn_file}")

            generate_transaction_plots(
                score_result=(
                    score_result
                ),
                raw_tokens=(
                    raw_tokens
                ),
                out_vocab=(
                    out_vocab
                ),
                title=(
                    plot_title
                ),
                output_dir=(
                    "results/plots/"
                    "attacks"
                ),
                filename_stem=(
                    plot_stem
                ),
            )

    # ========================================================
    # FALSE POSITIVE ATTRIBUTION + PLOTS
    # ========================================================

    print(
        "\n================== "
        "FALSE POSITIVE ATTRIBUTION "
        "=================="
    )

    top_fp_indices = (fp_ranking["tx_index"]  .head(TOP_FPS_TO_EXPLAIN)  .astype(int)  .tolist())

    for fp_rank, idx in enumerate(top_fp_indices, start=1,):

        if idx not in fp_detailed:
            continue

        score_result = (fp_detailed[idx])

        raw_tokens = (normal_test_out_tokens[idx])

        # idx is the index within the TEST subset, not the original dataframe.
        original_row_idx = test_idx[idx]
        source_file = normal_source_files[original_row_idx]
        record = df.iloc[original_row_idx]
        contract_address = str(record.get("contractAddress") or "unknown")
        function_name = str(record.get("functionName") or "unknown")
        display_name = (
            f"{source_file} | contract={contract_address} | "
            f"function={function_name} | test_idx={idx}"
        )

        print_and_store_attribution(
            dataset_name=(
                "FALSE_POSITIVE"
            ),
            display_name=(
                display_name
            ),
            score_result=(
                score_result
            ),
            raw_tokens=(
                raw_tokens
            ),
            original_idx=(
                idx
            ),
        )

        # ----------------------------------------------------
        # Graphical analysis
        # ----------------------------------------------------

        if (fp_rank <= TOP_FPS_TO_PLOT):

            plot_title = (
                f"False positive | {source_file} | test_idx={idx} | "
                f"score={score_result[SCORE_KEY]:.3f}\n"
                f"contract={contract_address} | function={function_name}"
            )

            plot_stem = (f"rank_{fp_rank:02d}_" f"fp_{idx}")

            generate_transaction_plots(
                score_result=(
                    score_result
                ),
                raw_tokens=(
                    raw_tokens
                ),
                out_vocab=(
                    out_vocab
                ),
                title=(
                    plot_title
                ),
                output_dir=(
                    "results/plots/"
                    "false_positives"
                ),
                filename_stem=(
                    plot_stem
                ),
            )

    # ========================================================
    # THRESHOLD SWEEP
    # ========================================================

    threshold_rows = []

    print(
        "\n================== "
        "THRESHOLD SWEEP "
        "=================="
    )

    for percentile in [90.0, 95.0, 97.0, 99.0, 99.5, 99.9,]:

        p_thr = (torch.quantile(torch.tensor(val_tx_scores, dtype=torch.float64,),  percentile / 100.0,).item())

        attack_detected = sum(sc > p_thr for sc in atk)

        normal_flagged = sum(sc > p_thr for sc in nor)

        attack_recall = (attack_detected / max(1, len(atk),))

        test_fpr = (normal_flagged / max(1, len(nor),))

        threshold_rows.append({
            "percentile":
                percentile,
            "threshold":
                p_thr,
            "attack_recall":
                attack_recall,
            "test_false_positive_rate":
                test_fpr,
            "false_positive_count":
                normal_flagged,
        })

        print(
            f"P{percentile:5.1f} "
            f"thr="
            f"{p_thr:.4f} "
            f"attack_recall="
            f"{100 * attack_recall:6.2f}% "
            f"test_FPR="
            f"{100 * test_fpr:6.3f}% "
            f"FP="
            f"{normal_flagged}"
        )

    # ========================================================
    # SAVE CSV RESULTS
    # ========================================================

    os.makedirs("results", exist_ok=True,)

    transaction_df.to_csv("results/" "transaction_unk_diagnostics.csv",  index=False,)

    detailed_unk_df.to_csv("results/" "fp_tn_detailed_unk_diagnostics.csv",  index=False,)

    pd.DataFrame(raw_attr_rows).to_csv("results/" "top_token_attribution_raw.csv",  index=False,)

    pd.DataFrame(
        nonstructural_attr_rows
    ).to_csv(
        "results/"
        "top_token_attribution_nonstructural.csv",
        index=False,
    )

    pd.DataFrame(threshold_rows).to_csv("results/" "threshold_sweep.csv",  index=False,)

    print(
        "\n================== "
        "OUTPUT FILES "
        "=================="
    )

    print("CSV diagnostics saved in:")

    print("  results/" "transaction_unk_diagnostics.csv")

    print("  results/" "fp_tn_detailed_unk_diagnostics.csv")

    print("  results/" "top_token_attribution_raw.csv")

    print("  results/" "top_token_attribution_nonstructural.csv")

    print("  results/" "threshold_sweep.csv")

    print(
        "\nAttack plots saved in:"
    )

    print("  results/plots/attacks/")

    print(
        "\nFalse-positive plots saved in:"
    )

    print("  results/plots/" "false_positives/")

def main(run_train=True, run_test=True):
    ckpt = None
    if run_train:
        ckpt = train_main(seed=42)

    if run_test:
        if ckpt is not None:
            test_main(MODEL_PATH=ckpt)
        else:
            test_main(MODEL_PATH="models/names_abstracted_v4_list_buckets/anomaly_detection_model_9.pt")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Execute anomaly detection on smart contract execution traces.')
    parser.add_argument('--train', action='store_true', help='Whether to run training.')
    parser.add_argument('--test', action='store_true', help='Whether to run testing.')
    args = parser.parse_args()

    main(run_train=args.train, run_test=args.test)

