"""Shared data, model and metric code for train.py and evaluate_test.py.

Both models see exactly the same data pipeline, so a difference in results comes from the
architecture and pretraining, not from preprocessing:
  targets   8 binary outputs, positive = grade >= threshold (default 2) on each ordinal column
  input     per-image min-max to [0, 1], zero-pad to a centred square, bilinear resize
  augment   (train only) random rotation +-10 deg, shift +-5 %, zoom 0.9-1.1; NO horizontal flip,
            because the labels are left/right specific
"""
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset

LABEL_COLS = [
    "HeartSize", "PulmonaryCongestion", "PleuralEffusion_Right", "PleuralEffusion_Left",
    "PulmonaryOpacities_Right", "PulmonaryOpacities_Left", "Atelectasis_Right", "Atelectasis_Left",
]
# The EDA's five findings: single columns, or (right, left) pairs combined with max / OR.
FINDINGS = {
    "Cardiomegaly": ["HeartSize"],
    "Congestion": ["PulmonaryCongestion"],
    "PleuralEffusion": ["PleuralEffusion_Right", "PleuralEffusion_Left"],
    "PulmonaryOpacity": ["PulmonaryOpacities_Right", "PulmonaryOpacities_Left"],
    "Atelectasis": ["Atelectasis_Right", "Atelectasis_Left"],
}

EVA_X_TINY_WEIGHTS = {
    "repo_id": "MapleF/eva_x",
    "filename": "eva_x_tiny_patch16_merged520k_mim.pt",
    "revision": "35ddcd6dab6ca99bbdb6cb45c8d1b093aefbd0ee",
}

# Per-model recipe. Each model gets its standard fine-tuning setup; the data side is shared.
#   densenet121: ImageNet weights, AdamW, the recipe of our first baseline run.
#   eva_x_tiny:  EVA-X authors' ChestX-ray14 recipe (lr 1e-3 at batch 1024, layer decay 0.55,
#                weight decay 0.05, drop path 0.2, 5 warmup epochs, mean pooling, CXR mean/std),
#                with the learning rate scaled linearly to batch 512.
MODEL_DEFAULTS = {
    "densenet121": dict(mean=0.449, std=0.226, lr=1e-4, weight_decay=1e-4, batch_size=128, epochs=20,
                        warmup_steps=500, warmup_epochs=None, layer_decay=None, drop_path=0.0, dropout=0.0),
    "eva_x_tiny": dict(mean=0.49185243, std=0.28509309, lr=5e-4, weight_decay=0.05, batch_size=512, epochs=30,
                       warmup_steps=None, warmup_epochs=5, layer_decay=0.55, drop_path=0.2, dropout=0.0),
}


# ----------------------------------------------------------------------------- data
def load_split(data_dir: Path, split: str, extra_cols=()):
    """Memory-mapped HF Arrow dataset for one split; images left undecoded (bytes)."""
    import datasets

    ds = datasets.load_dataset(
        "parquet", data_files={split: str(Path(data_dir) / "data" / f"{split}-*.parquet")}, split=split
    )
    cols = ["UID", "Image", *LABEL_COLS, *extra_cols]
    return ds.select_columns(cols).cast_column("Image", datasets.Image(decode=False))


class TaixDataset(Dataset):
    def __init__(self, hf_ds, size: int, threshold: int, augment: bool):
        self.ds, self.size, self.threshold = hf_ds, size, threshold
        self.affine = None
        if augment:
            from torchvision.transforms import v2

            self.affine = v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[int(i)]
        with Image.open(io.BytesIO(row["Image"]["bytes"])) as im:
            arr = np.asarray(im, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        h, w = arr.shape
        side = max(h, w)
        if h != w:  # zero-pad to a centred square so the anatomy is not stretched
            sq = np.zeros((side, side), dtype=np.float32)
            top, left = (side - h) // 2, (side - w) // 2
            sq[top:top + h, left:left + w] = arr
            arr = sq
        if side != self.size:
            arr = np.asarray(Image.fromarray(arr, mode="F").resize((self.size, self.size), Image.BILINEAR))
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        if self.affine is not None:
            x = self.affine(x)
        y = torch.tensor([float(row[c] >= self.threshold) for c in LABEL_COLS], dtype=torch.float32)
        return x.to(torch.float16), y


def to_input(x, device, mean: float, std: float):
    """[0, 1] grey batch -> normalised 3-channel channels-last tensor on the GPU."""
    x = x.to(device, non_blocking=True).float()
    x = ((x - mean) / std).expand(-1, 3, -1, -1)
    return x.contiguous(memory_format=torch.channels_last)


# ----------------------------------------------------------------------------- models
def _eva_x_tiny(drop_path: float, num_classes: int):
    from timm.models.eva import Eva

    # Same architecture as eva_x_tiny_patch16 in the EVA-X repo (eva_x.py), with mean pooling.
    return Eva(
        img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True, use_rot_pos_emb=True, ref_feat_shape=(14, 14),
        num_classes=num_classes, global_pool="avg", drop_path_rate=drop_path,
    )


def load_eva_x_checkpoint(path: Path):
    """Load the EVA-X MIM checkpoint with torch's safe (weights-only) unpickler.

    The DeepSpeed file also stores numpy scalars, so exactly those types are allow-listed;
    arbitrary code in the pickle still cannot run.
    """
    import codecs

    from numpy._core.multiarray import scalar

    safe = [(scalar, "numpy.core.multiarray.scalar"), np.dtype, codecs.encode, set, type(np.dtype("float64"))]
    with torch.serialization.safe_globals(safe):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt.get("module", ckpt.get("model", ckpt))


def eva_x_state_dict(src: dict) -> dict:
    """EVA-X MIM keys -> timm Eva keys (mirrors checkpoint_filter_fn in EVA-X's eva_x.py)."""
    out = {}
    for k, v in src.items():
        if "rope" in k or k in ("mask_token", "lm_head.weight", "lm_head.bias"):
            continue  # fixed RoPE buffers are rebuilt; MIM-only tokens/heads are dropped
        if k in ("norm.weight", "norm.bias"):
            k = k.replace("norm", "fc_norm")  # mean pooling uses fc_norm; start it from the MIM norm
        k = (k.replace("mlp.ffn_ln", "mlp.norm").replace("attn.inner_attn_ln", "attn.norm")
              .replace("mlp.w12", "mlp.fc1").replace("mlp.w1", "mlp.fc1_g")
              .replace("mlp.w2", "mlp.fc1_x").replace("mlp.w3", "mlp.fc2"))
        out[k] = v.float()
    return out


def build_model(name: str, device, pretrained: bool = True, drop_path: float = 0.0, dropout: float = 0.0,
                weights_dir="data/pretrained"):
    """dropout: DenseNet-121 only, applied to the pooled features before the classifier.
    With dropout=0 the classifier stays a plain Linear, so older checkpoints still load."""
    n = len(LABEL_COLS)
    if name == "densenet121":
        from torchvision.models import DenseNet121_Weights, densenet121

        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        head = nn.Linear(m.classifier.in_features, n)
        m.classifier = nn.Sequential(nn.Dropout(dropout), head) if dropout > 0 else head
    elif name == "eva_x_tiny":
        m = _eva_x_tiny(drop_path, n)
        if pretrained:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(**EVA_X_TINY_WEIGHTS, local_dir=weights_dir)
            msg = m.load_state_dict(eva_x_state_dict(load_eva_x_checkpoint(path)), strict=False)
            unexpected = list(msg.unexpected_keys)
            missing = [k for k in msg.missing_keys if not k.startswith("head.")]
            if unexpected or missing:
                raise RuntimeError(f"EVA-X weights did not match: missing={missing} unexpected={unexpected}")
            print(f"Loaded EVA-X-Tiny MIM weights ({EVA_X_TINY_WEIGHTS['revision'][:8]}); new head: {msg.missing_keys}")
    else:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODEL_DEFAULTS)}")
    return m.to(device, memory_format=torch.channels_last)


def param_groups(model, name: str, weight_decay: float, layer_decay=None):
    """Optimizer parameter groups.

    Without layer decay (DenseNet-121): one group, weight decay on everything, as in the baseline run.
    With layer decay (EVA-X): BEiT/EVA-X style, as in EVA-X's utils/lr_decay.py, i.e. no weight decay
    on 1-D params and tokens, and lr scaled by layer_decay ** (depth from the top).
    """
    if not layer_decay:
        return [{"params": list(model.parameters()), "weight_decay": weight_decay, "lr_scale": 1.0}]

    skip = set(getattr(model, "no_weight_decay", lambda: set())())
    num_layers = len(model.blocks) + 1
    scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in ("cls_token", "pos_embed") or n.startswith("patch_embed"):
            layer = 0
        elif n.startswith("blocks."):
            layer = int(n.split(".")[1]) + 1
        else:
            layer = num_layers  # fc_norm, head
        wd = 0.0 if (p.ndim <= 1 or n in skip) else weight_decay
        g = groups.setdefault((layer, wd), {"params": [], "weight_decay": wd, "lr_scale": scales[layer]})
        g["params"].append(p)
    return list(groups.values())


# ----------------------------------------------------------------------------- metrics
def auroc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def auprc(y, p):
    """Area under the precision-recall curve (average precision). Chance level = prevalence."""
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def finding_arrays(probs: np.ndarray, targets: np.ndarray):
    """(probs, targets) for the 5 findings: bilateral = max(left, right) score, OR of labels."""
    idx = {c: j for j, c in enumerate(LABEL_COLS)}
    p5, t5 = [], []
    for cols in FINDINGS.values():
        js = [idx[c] for c in cols]
        p5.append(probs[:, js].max(axis=1))
        t5.append(targets[:, js].max(axis=1))
    return np.stack(p5, 1), np.stack(t5, 1)


def metrics(probs: np.ndarray, targets: np.ndarray) -> dict:
    """Primary metric: mean AUROC. Secondary: mean AUPRC (average precision), which, unlike AUROC,
    depends on prevalence and puts weight on how clean the top-ranked positives are."""
    p5, t5 = finding_arrays(probs, targets)
    roc8 = {c: auroc(targets[:, j], probs[:, j]) for j, c in enumerate(LABEL_COLS)}
    roc5 = {name: auroc(t5[:, j], p5[:, j]) for j, name in enumerate(FINDINGS)}
    pr8 = {c: auprc(targets[:, j], probs[:, j]) for j, c in enumerate(LABEL_COLS)}
    pr5 = {name: auprc(t5[:, j], p5[:, j]) for j, name in enumerate(FINDINGS)}
    mean = lambda d: float(np.nanmean(list(d.values())))  # noqa: E731
    return {
        "mean_auroc_8": mean(roc8), "mean_auroc_5": mean(roc5),
        "mean_auprc_8": mean(pr8), "mean_auprc_5": mean(pr5),
        "auroc_8": roc8, "auroc_5": roc5, "auprc_8": pr8, "auprc_5": pr5,
        "prevalence_8": {c: float(targets[:, j].mean()) for j, c in enumerate(LABEL_COLS)},
        "prevalence_5": {name: float(t5[:, j].mean()) for j, name in enumerate(FINDINGS)},
    }


@torch.no_grad()
def predict(model, loader, device, amp_dtype, mean: float, std: float):
    model.eval()
    probs, targets = [], []
    with torch.autocast("cuda", dtype=amp_dtype):
        for x, y in loader:
            probs.append(torch.sigmoid(model(to_input(x, device, mean, std)).float()).cpu())
            targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()
