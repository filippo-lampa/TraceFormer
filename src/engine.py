import sys
import time
import torch
from torch.amp import autocast, GradScaler


def train_network(model, optimizer, loss_function, trainloader, device, epoch, num_epochs):
    print(f"Epoch {epoch + 1}: Training Started")
    sys.stdout.flush()

    scaler = GradScaler()

    model.train()
    running_loss = 0.0
    num_batches = 0

    tokens_seen = 0
    start_time = time.perf_counter()

    for out_ids, tree_ids, ctx_ids, y in trainloader:
        out_ids = out_ids.to(device, non_blocking=True)
        tree_ids = tree_ids.to(device, non_blocking=True)
        ctx_ids = ctx_ids.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            outputs = model(out_ids, tree_ids, ctx_ids)
            loss = loss_function(outputs.transpose(2, 1), y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        num_batches += 1
        tokens_seen += y.numel()

        if num_batches % 50 == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            tps = tokens_seen / max(1e-9, elapsed)
            print(
                f"Batch {num_batches}/{len(trainloader)} | "
                f"loss {running_loss / num_batches:.4f} | "
                f"tokens/s {tps:,.0f}"
            )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    avg_loss = running_loss / max(1, num_batches)
    tps = tokens_seen / max(1e-9, elapsed)
    print(f"Epoch {epoch + 1}/{num_epochs} | Avg train loss: {avg_loss:.4f} | tokens/s: {tps:,.0f}")


def valid_network(model, loss_function, valloader, device, epoch, num_epochs):
    print(f"Epoch {epoch + 1}: Validation Started")
    sys.stdout.flush()
    model.eval()

    total_tokens = 0
    total_nll = 0.0
    num_windows = 0
    window_nlls = []

    with torch.no_grad():
        for out_ids, tree_ids, ctx_ids, y, valid_lens in valloader:
            out_ids = out_ids.to(device, non_blocking=True)
            tree_ids = tree_ids.to(device, non_blocking=True)
            ctx_ids = ctx_ids.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = model(out_ids, tree_ids, ctx_ids)
            B, T, V = outputs.shape

            logits = outputs.reshape(B * T, V)
            targets = y.reshape(B * T)

            per_token_loss = loss_function(logits, targets)
            per_token_loss = per_token_loss.reshape(B, T)

            for b in range(B):
                valid_len = int(valid_lens[b])

                real_losses = per_token_loss[b, :valid_len]

                total_nll += real_losses.sum().item()
                total_tokens += valid_len

                window_nlls.append(real_losses.mean().item())

                num_windows += 1

    avg_nll_per_token = total_nll / max(1, total_tokens)
    avg_nll_per_window = sum(window_nlls) / max(1, len(window_nlls))

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Val NLL/token: {avg_nll_per_token:.4f} | "
        f"Val NLL/window(mean): {avg_nll_per_window:.4f} | "
        f"Val windows: {num_windows}"
    )

    return window_nlls, avg_nll_per_token, avg_nll_per_window



def score_transactions(
    model,
    loss_function_none,
    out_seqs,
    tree_seqs,
    ctx_seqs,
    block_size,
    device,
    stride=None,
):
    model.eval()
    results = []

    stride = block_size if stride is None else stride

    with torch.no_grad():
        for j in range(len(out_seqs)):
            out = out_seqs[j].to(device)
            tree = tree_seqs[j].to(device)
            ctx = ctx_seqs[j].to(device)

            L = out.numel()

            if L < 2:
                continue

            total_nll = 0.0
            total_tokens = 0
            max_window_nll = -1e30
            num_windows = 0

            for start in range(0, L - 1, stride):
                valid_len = min(block_size, L - 1 - start)

                x_out = out[start:start + valid_len]
                x_tree = tree[start:start + valid_len]
                x_ctx = ctx[start:start + valid_len]
                y = out[start + 1:start + 1 + valid_len]

                pad_len = block_size - valid_len

                if pad_len > 0:
                    pad = torch.zeros(
                        pad_len,
                        dtype=torch.long,
                        device=device
                    )

                    x_out = torch.cat([x_out, pad])
                    x_tree = torch.cat([x_tree, pad])
                    x_ctx = torch.cat([x_ctx, pad])
                    y = torch.cat([y, pad])

                x_out = x_out.unsqueeze(0)
                x_tree = x_tree.unsqueeze(0)
                x_ctx = x_ctx.unsqueeze(0)
                y = y.unsqueeze(0)

                outputs = model(x_out, x_tree, x_ctx)

                _, T, V = outputs.shape

                logits = outputs.reshape(T, V)
                targets = y.reshape(T)

                per_token = loss_function_none(logits, targets)

                real_token_losses = per_token[:valid_len]

                w_nll = real_token_losses.mean().item()

                total_nll += real_token_losses.sum().item()
                total_tokens += valid_len

                max_window_nll = max(max_window_nll, w_nll)
                num_windows += 1

            results.append({
                "tx_index": j,
                "num_tokens": int(L),
                "total_nll": float(total_nll),
                "mean_token_nll": float(
                    total_nll / max(1, total_tokens)
                ),
                "num_windows": int(num_windows),
                "max_window_nll": float(max_window_nll),
            })

    return results
