"""Training loop with early stopping, plus test-set evaluation."""

import copy
import math
import time

import torch
from torch import nn


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, top1, top5, n = 0.0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item()
        best5 = logits.topk(5, dim=-1).indices
        top1 += (best5[:, 0] == y).sum().item()
        top5 += (best5 == y.unsqueeze(1)).any(dim=1).sum().item()
        n += y.numel()
    loss = total_loss / n
    return {"loss": loss, "perplexity": math.exp(loss),
            "top1_acc": top1 / n, "top5_acc": top5 / n}


def train_model(model, train_loader, val_loader, device, epochs=20, lr=2e-3,
                weight_decay=1e-4, clip=0.5, patience=3, log=print):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_top1": [], "lr": []}
    best_val, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            running += loss.item() * y.numel()
            seen += y.numel()

        val = evaluate(model, val_loader, device)
        sched.step(val["loss"])
        history["train_loss"].append(running / seen)
        history["val_loss"].append(val["loss"])
        history["val_top1"].append(val["top1_acc"])
        history["lr"].append(opt.param_groups[0]["lr"])
        log(f"  epoch {epoch:2d}  train {running / seen:.3f}  "
            f"val {val['loss']:.3f}  ppl {val['perplexity']:.1f}")

        if val["loss"] < best_val:
            best_val, best_epoch, bad_epochs = val["loss"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"  early stop (best epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["train_seconds"] = time.time() - start
    return history
