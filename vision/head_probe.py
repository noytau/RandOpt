"""Fit a fresh N-way linear head for a dataset with its own private label
space (e.g. exdark's 12 classes) — the "center" scripts/randopt_selfcontained.py
later perturbs around. Datasets outside ImageNet-1k's label space have no
pretrained head to download the way ImageNet-1k does (Meta's released head
IS that center); this is the one-time fitting step that plays the same role.

Training loop is the plain-SGD counterpart of scripts/ft_matched_baseline.py's
fine-tuning loop (identical [CLS; mean-patch] features -> cross-entropy),
just starting from a freshly-initialized head instead of a pretrained
1000-way one, and with no comparison-run diff mechanics (delta_w etc.) since
there's no pretrained head to compare against here.
"""
import numpy as np
import torch
import torch.nn.functional as F

from data_handlers.imagenet_c import load_image_batch


def fit_linear_head(engine, train_items, handler, input_mode, lr=1e-3,
                     epochs=5, batch_size=16, scope="head", last_n_blocks=0,
                     seed=42):
    """Train engine.head (and optionally part of the backbone, per `scope` —
    same perturb_target semantics as RandOpt's perturb scope) via
    cross-entropy on train_items. Mutates `engine` in place (head ends up
    trained + in eval mode, all params frozen again). Returns final train
    accuracy.
    """
    device = engine.device
    if scope == "all":
        trainable = dict(engine._all_params())
    else:
        engine.set_perturb_scope(scope, last_n_blocks)
        trainable = dict(engine._perturb_params())
    for name, p in engine._all_params():
        p.requires_grad_(name in trainable)
    print(f"  probe scope '{scope}': {len(trainable)} trainable tensors "
          f"({sum(p.numel() for p in trainable.values()) / 1e6:.1f}M params)")

    # scope="all" on a DINOv3-scale backbone (6.72B params) does not fit a
    # single 46GB GPU under plain fp32 AdamW (~107.5GB for weights+grad+
    # optimizer moments alone) -- ft_matched_baseline.py hit and solved this
    # exact wall (see its comments for the full OOM investigation): bf16
    # weight storage, an 8-bit optimizer (halves/1-bytes the four per-param
    # tensors), AND gradient checkpointing (every one of the 40 blocks is
    # trainable here, so autograd would otherwise retain all their
    # activations for backward). Ported verbatim rather than duplicated
    # logic drifting apart. Smaller scopes (head/last_n_blocks) never
    # needed this and keep the original plain fp32 path.
    model_dtype = torch.bfloat16 if scope == "all" else torch.float32
    if model_dtype is torch.bfloat16:
        engine.backbone.to(model_dtype)
        engine.head.to(model_dtype)
        import torch.utils.checkpoint as ckpt

        def _checkpointed(fwd):
            def wrapped(*a, **kw):
                return ckpt.checkpoint(fwd, *a, use_reentrant=False, **kw)
            return wrapped

        for block in engine.backbone.blocks:
            block.forward = _checkpointed(block.forward)

    if scope == "all":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(list(trainable.values()), lr=lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(list(trainable.values()), lr=lr, weight_decay=0.0)
    steps_per_epoch = (len(train_items) + batch_size - 1) // batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs * steps_per_epoch))
    transform = engine.transforms[input_mode]
    labels_all = [int(d["class_id"]) for d in train_items]

    engine.backbone.train()
    engine.head.train()
    rng = np.random.default_rng(seed)
    order = np.arange(len(train_items))
    for epoch in range(epochs):
        rng.shuffle(order)
        losses = []
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            batch = load_image_batch([train_items[j] for j in idx],
                                      transform).to(device, dtype=model_dtype)
            labels = torch.tensor([labels_all[j] for j in idx], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f_out = engine.backbone.forward_features(batch)
                feat = torch.cat([f_out["x_norm_clstoken"],
                                   f_out["x_norm_patchtokens"].mean(dim=1)], dim=1)
                logits = engine.head(feat.to(engine.head.weight.dtype))
                loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        print(f"  probe epoch {epoch + 1}/{epochs} loss={np.mean(losses):.4f}")

    engine.backbone.eval()
    engine.head.eval()
    for _name, p in engine._all_params():
        p.requires_grad_(False)

    from scripts.randopt_imagenet_c import score
    with torch.no_grad():
        preds = engine.predict(train_items, input_mode, None)
    return score(handler, preds, train_items)
