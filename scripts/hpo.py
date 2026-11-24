"""Hyperparameter optimisation (HPO) for DenseNet-121 and EVA-X-Tiny on the TAIX-Ray validation split.

Strategy: BOHB-style (a Bayesian sampler with a multi-fidelity scheduler), in three stages.

  Stage 1  `search`  Bayesian optimisation with Optuna's TPE sampler (multivariate), with Hyperband
                     pruning (successive halving across brackets). Fidelity is training length: every
                     trial gets a complete short schedule (FIDELITY_EPOCHS epochs: 1 warmup epoch, then
                     cosine decay), and Hyperband stops weak trials after MIN_RESOURCE epochs.
                     Trial 0 is the model's default recipe, so the search has to beat it.
  Stage 2  `refine`  The TOP_K best stage-1 configurations are retrained on the short (6-epoch) schedule
                     and on the model's full schedule with early stopping. Schedule length interacts
                     with overfitting, so it is selected here rather than assumed.
  Stage 3  `local`   The best stage-1 configuration with only the learning rate scaled (e.g. x0.5 ... x6),
                     unpruned, to check that the optimum is a real peak along the most important
                     dimension. Hyperband can be biased against high learning rates, which start slow.
  `report`           Figures (search history, validation curves of all trials, score against each
                     hyperparameter, stage-2 curves, local check) and a check that the best values are
                     not at the edge of the search space.

The final configuration is the stage-2 or stage-3 run with the highest validation mean AUROC.

Objective: mean validation AUROC over the 8 labels (primary metric). Mean validation AUPRC is recorded
for every trial as the secondary metric. The test split is never used here.

Usage:
    python scripts/hpo.py search --model densenet121 --n-trials 20
    python scripts/hpo.py refine --model densenet121 --top-k 3
    python scripts/hpo.py local --model densenet121 --lr-factors 0.5,2,3,4.5,6
    python scripts/hpo.py report --model densenet121
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

FIDELITY_EPOCHS = 6     # stage-1 schedule length (complete warmup + cosine in 6 epochs)
MIN_RESOURCE = 2        # Hyperband: first point where a trial can be stopped
REDUCTION_FACTOR = 3    # Hyperband/successive halving: keep about the top 1/3 at each rung
TOP_K = 3               # stage-2 configurations retrained at full fidelity

# Search spaces. Each entry: (type, low, high, log-scale?). Rationale in docs/hpo/README_HPO.md.
#   densenet121: the baseline overfit after epoch 6, so besides the learning rate the space covers the
#                two regularisers that act on a pretrained CNN: AdamW weight decay and dropout before
#                the new classifier.
#   eva_x_tiny:  the four knobs the EVA-X authors tune when fine-tuning: learning rate, layer-wise lr
#                decay, weight decay and stochastic depth (drop path).
SPACES = {
    "densenet121": {
        "lr": ("float", 1e-5, 1e-3, True),
        "weight_decay": ("float", 1e-6, 1e-1, True),
        "dropout": ("float", 0.0, 0.5, False),
    },
    "eva_x_tiny": {
        "lr": ("float", 5e-5, 2e-3, True),
        "layer_decay": ("float", 0.40, 0.95, False),
        "weight_decay": ("float", 1e-3, 3e-1, True),
        "drop_path": ("float", 0.0, 0.4, False),
    },
}


HPO_ROOT = Path("outputs/hpo")  # --smoke switches this to outputs/hpo_smoke


def out_dir(model: str) -> Path:
    d = HPO_ROOT.resolve() / model
    (d / "trials").mkdir(parents=True, exist_ok=True)
    return d


def load_study(model: str, create: bool = False):
    import optuna

    storage = f"sqlite:///{(out_dir(model) / 'study.db').as_posix()}"
    if not create:
        return optuna.load_study(study_name=f"{model}_stage1", storage=storage)
    return optuna.create_study(
        study_name=f"{model}_stage1", storage=storage, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, n_startup_trials=8),
        pruner=optuna.pruners.HyperbandPruner(min_resource=MIN_RESOURCE, max_resource=FIDELITY_EPOCHS,
                                              reduction_factor=REDUCTION_FACTOR),
    )


def suggest(trial, model: str) -> dict:
    out = {}
    for name, (kind, lo, hi, log) in SPACES[model].items():
        out[name] = trial.suggest_float(name, lo, hi, log=log)
    return out


# ----------------------------------------------------------------------------- stage 1
def cmd_search(args):
    import optuna

    from common import MODEL_DEFAULTS
    from train import make_loaders, resolve_config, train_model

    study = load_study(args.model, create=True)
    d = MODEL_DEFAULTS[args.model]
    if len(study.trials) == 0:
        study.enqueue_trial({k: d[k] for k in SPACES[args.model]})  # trial 0 = default recipe
        study.set_user_attr("search_space", {k: list(v) for k, v in SPACES[args.model].items()})
        study.set_user_attr("fidelity", {"epochs": FIDELITY_EPOCHS, "warmup_epochs": 1,
                                         "min_resource": MIN_RESOURCE, "reduction_factor": REDUCTION_FACTOR})

    finished = [t for t in study.trials if t.state.is_finished()]
    remaining = args.n_trials - len(finished)
    print(f"{args.model}: {len(finished)} trials done, {remaining} to go", flush=True)
    if remaining <= 0:
        return

    base = {"epochs": FIDELITY_EPOCHS, "warmup_epochs": 1, "patience": FIDELITY_EPOCHS + 1,
            "workers": args.workers, "seed": 0, "max_train_steps": args.max_train_steps}
    cfg0 = resolve_config(args.model, base)
    t0 = time.time()
    _, _, train_loader, val_loader = make_loaders(cfg0, Path(args.data_dir).resolve())
    print(f"loaders ready in {time.time() - t0:.0f}s", flush=True)
    trials_dir = out_dir(args.model) / "trials"

    def objective(trial):
        params = suggest(trial, args.model)
        cfg = resolve_config(args.model, {**base, **params})
        print(f"\n--- trial {trial.number}: " + ", ".join(f"{k}={v:.3g}" for k, v in params.items()), flush=True)
        rows = []

        def report(epoch, row):
            rows.append({"trial": trial.number, **params, **row})
            pd.DataFrame(rows).to_csv(trials_dir / f"trial_{trial.number:03d}.csv", index=False)
            trial.report(row["val_mean_auroc_8"], epoch)
            if epoch < FIDELITY_EPOCHS and trial.should_prune():
                trial.set_user_attr("pruned_at_epoch", epoch)
                best = max(rows, key=lambda r: r["val_mean_auroc_8"])
                trial.set_user_attr("val_mean_auprc_8", best["val_mean_auprc_8"])
                print(f"    pruned after epoch {epoch}", flush=True)
                raise optuna.TrialPruned()

        t = time.time()
        try:
            out = train_model(cfg, train_loader, val_loader, report=report, log=lambda s: print("   ", s, flush=True))
        finally:
            import torch
            torch.cuda.empty_cache()
        b = out["best"]
        trial.set_user_attr("best_epoch", b["epoch"])
        trial.set_user_attr("val_mean_auprc_8", b["val_mean_auprc_8"])
        trial.set_user_attr("val_mean_auroc_5", b["val_mean_auroc_5"])
        trial.set_user_attr("minutes", round((time.time() - t) / 60, 1))
        return b["val_mean_auroc_8"]

    study.optimize(objective, n_trials=remaining, catch=(FloatingPointError,), gc_after_trial=True)
    export_trials(args.model)


def export_trials(model: str) -> pd.DataFrame:
    study = load_study(model)
    rows = []
    for t in study.trials:
        rows.append({"trial": t.number, "state": t.state.name, "value": t.value, **t.params,
                     "last_epoch": max(t.intermediate_values) if t.intermediate_values else None,
                     **{k: v for k, v in t.user_attrs.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir(model) / "stage1_trials.csv", index=False)
    return df


# ----------------------------------------------------------------------------- stage 2
def cmd_refine(args):
    from train import make_loaders, resolve_config, train_model

    df = export_trials(args.model)
    done = df[df.state == "COMPLETE"].sort_values("value", ascending=False).head(args.top_k)
    print(f"{args.model}: refining top {len(done)} stage-1 trials at full fidelity:\n"
          f"{done[['trial', 'value', *SPACES[args.model]]].to_string(index=False)}", flush=True)

    cfg0 = resolve_config(args.model, {"workers": args.workers})
    train_ds, val_ds, train_loader, val_loader = make_loaders(cfg0, Path(args.data_dir).resolve())
    uids = list(val_ds["UID"])
    schedules = [x.strip() for x in args.schedules.split(",") if x.strip()]
    results = []
    for _, r in done.iterrows():
        params = {k: float(r[k]) for k in SPACES[args.model]}
        for sched in schedules:
            # "full":  the model's full schedule with early stopping (DenseNet-121 20, EVA-X-Tiny 30 epochs).
            # "short": the stage-1 schedule itself (6 epochs, 1 warmup epoch). Schedule length interacts
            #          with overfitting, so it is treated as part of the configuration being selected.
            name = f"hpo_{args.model}_trial{int(r.trial):03d}" + ("" if sched == "full" else "_short")
            run_dir = Path(args.runs_dir).resolve() / name
            if (run_dir / "summary.json").exists():
                print(f"{name}: already done, skipping", flush=True)
                s = json.loads((run_dir / "summary.json").read_text())
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                extra = ({"epochs": FIDELITY_EPOCHS, "warmup_epochs": 1, "patience": FIDELITY_EPOCHS + 1}
                         if sched == "short" else {"epochs": args.epochs_override})
                cfg = resolve_config(args.model, {**params, **extra, "workers": args.workers,
                                                  "max_train_steps": args.max_train_steps})
                (run_dir / "config.json").write_text(json.dumps({"args": cfg, "stage1_trial": int(r.trial),
                                                                 "schedule": sched}, indent=2))
                print(f"=== {name} ({sched} schedule, {cfg['epochs']} epochs): "
                      + ", ".join(f"{k}={v:.3g}" for k, v in params.items()), flush=True)
                t = time.time()
                out = train_model(cfg, train_loader, val_loader, val_uids=uids, run_dir=run_dir,
                                  log=lambda s: print("   ", s, flush=True))
                b = out["best"]
                s = {"model": args.model, "stage1_trial": int(r.trial), "schedule": sched, "params": params,
                     "best_epoch": b["epoch"], "epochs_run": out["epochs_run"],
                     "minutes": round((time.time() - t) / 60, 1), "val": b["val_metrics"]}
                (run_dir / "summary.json").write_text(json.dumps(s, indent=2))
            results.append({"run": name, "path": str(run_dir), "schedule": s.get("schedule", sched),
                            "stage1_trial": int(r.trial), "stage1_value": float(r.value), **params,
                            "val_mean_auroc_8": s["val"]["mean_auroc_8"], "val_mean_auprc_8": s["val"]["mean_auprc_8"],
                            "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"], "minutes": s["minutes"]})
    res = pd.DataFrame(results).sort_values("val_mean_auroc_8", ascending=False)
    res.to_csv(out_dir(args.model) / "stage2_runs.csv", index=False)
    print("\n" + res.to_string(index=False))
    print(f"\nSelected (best validation mean AUROC at full fidelity): {res.iloc[0]['run']}")


# ----------------------------------------------------------------------------- local check
def cmd_local(args):
    """One-dimensional check around the stage-1 optimum along its most important axis, the learning rate.

    The best stage-1 configuration is retrained on the stage-1 (6-epoch) schedule with only the learning
    rate multiplied by each factor, without pruning. If the optimum is real, the unscaled learning rate
    (factor 1, the stage-1 result) scores at least as well as its neighbours on both sides.
    """
    from train import make_loaders, resolve_config, train_model

    df = export_trials(args.model)
    best = df[df.state == "COMPLETE"].sort_values("value").iloc[-1]
    params = {k: float(best[k]) for k in SPACES[args.model]}
    factors = [float(x) for x in args.lr_factors.split(",") if x.strip()]
    print(f"{args.model}: local learning-rate check around trial {int(best.trial)} "
          f"(lr={params['lr']:.3g}, value={best.value:.4f}), factors {factors}", flush=True)

    cfg0 = resolve_config(args.model, {"workers": args.workers})
    _, val_ds, train_loader, val_loader = make_loaders(cfg0, Path(args.data_dir).resolve())
    uids = list(val_ds["UID"])
    rows = [{"factor": 1.0, "lr": params["lr"], "val_mean_auroc_8": float(best.value),
             "val_mean_auprc_8": float(best.get("val_mean_auprc_8", np.nan)), "run": f"stage-1 trial {int(best.trial)}"}]
    for f in factors:
        name = f"hpo_{args.model}_trial{int(best.trial):03d}_lrx{f:g}"
        run_dir = Path(args.runs_dir).resolve() / name
        if (run_dir / "summary.json").exists():
            print(f"{name}: already done, skipping", flush=True)
            s = json.loads((run_dir / "summary.json").read_text())
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = resolve_config(args.model, {**params, "lr": params["lr"] * f, "epochs": FIDELITY_EPOCHS,
                                              "warmup_epochs": 1, "patience": FIDELITY_EPOCHS + 1,
                                              "workers": args.workers, "max_train_steps": args.max_train_steps})
            (run_dir / "config.json").write_text(json.dumps({"args": cfg, "stage1_trial": int(best.trial),
                                                             "lr_factor": f}, indent=2))
            print(f"=== {name}: lr={cfg['lr']:.3g}", flush=True)
            t = time.time()
            out = train_model(cfg, train_loader, val_loader, val_uids=uids, run_dir=run_dir,
                              log=lambda s: print("   ", s, flush=True))
            b = out["best"]
            s = {"model": args.model, "stage1_trial": int(best.trial), "lr_factor": f, "lr": cfg["lr"],
                 "best_epoch": b["epoch"], "epochs_run": out["epochs_run"],
                 "minutes": round((time.time() - t) / 60, 1), "val": b["val_metrics"]}
            (run_dir / "summary.json").write_text(json.dumps(s, indent=2))
        rows.append({"factor": f, "lr": params["lr"] * f, "val_mean_auroc_8": s["val"]["mean_auroc_8"],
                     "val_mean_auprc_8": s["val"]["mean_auprc_8"], "run": name})
    res = pd.DataFrame(rows).sort_values("lr")
    res.to_csv(out_dir(args.model) / "local_check.csv", index=False)
    print("\n" + res.to_string(index=False))
    top = res.sort_values("val_mean_auroc_8").iloc[-1]
    print(f"\nHighest score at lr factor {top.factor:g}"
          + (" (the stage-1 optimum itself)" if top.factor == 1.0 else " (NOT the stage-1 optimum)"))


# ----------------------------------------------------------------------------- report
PRETTY = {"lr": "learning rate", "weight_decay": "weight decay", "dropout": "classifier dropout",
          "layer_decay": "layer-wise lr decay", "drop_path": "drop-path rate"}


def near_edge(value, lo, hi, log, tol=0.05) -> str:
    a, b, v = (math.log10(lo), math.log10(hi), math.log10(value)) if log else (lo, hi, value)
    pos = (v - a) / (b - a)
    return "low edge" if pos < tol else "high edge" if pos > 1 - tol else "inside"


def cmd_report(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    df = export_trials(args.model)
    fig_dir = Path(args.fig_dir).resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    space = SPACES[args.model]
    label = {"densenet121": "DenseNet-121", "eva_x_tiny": "EVA-X-Tiny"}[args.model]
    BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#9aa3ad", "#1d232a"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e3e7eb", "grid.linewidth": 0.8,
                         "axes.edgecolor": "#8a949f", "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "figure.dpi": 140})

    # Each trial's best validation mAUROC within stage 1 (pruned trials: best before they were stopped).
    curves = {int(p.stem.split("_")[1]): pd.read_csv(p) for p in (out_dir(args.model) / "trials").glob("trial_*.csv")}
    df["score"] = [curves[t].val_mean_auroc_8.max() if t in curves else np.nan for t in df.trial]
    complete = df[df.state == "COMPLETE"]

    # 1) Optimisation history: every trial, best-so-far among full-fidelity (complete) trials.
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for state, color, mk, lab in [("COMPLETE", BLUE, "o", "completed (6 epochs)"),
                                  ("PRUNED", GREY, "x", "stopped early by Hyperband")]:
        s = df[df.state == state]
        ax.scatter(s.trial, s.score, c=color, marker=mk, s=36, label=lab, zorder=3)
    best_so_far = complete.set_index("trial")["value"].reindex(df.trial).ffill().cummax()
    ax.step(df.trial, best_so_far, where="post", color=INK, lw=1.6, label="best so far (completed)", zorder=2)
    ax.scatter([0], [float(df.loc[df.trial == 0, "score"].iloc[0])], s=120, facecolors="none", edgecolors=ORANGE, lw=1.6,
               label="trial 0 = default recipe", zorder=4)
    ax.set(xlabel="Trial", ylabel="Validation mean AUROC (8 labels)", title=f"{label}: stage-1 search history")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.model}_history.png")
    plt.close(fig)

    # 2) Learning curves of all stage-1 trials (the "validation accuracy" graph).
    fig, ax = plt.subplots(figsize=(7, 3.6))
    best_trial = int(complete.sort_values("value").iloc[-1].trial)
    for t, c in sorted(curves.items()):
        st = df.loc[df.trial == t, "state"].item()
        if t == best_trial:
            continue
        ax.plot(c.epoch, c.val_mean_auroc_8, color=BLUE if st == "COMPLETE" else GREY,
                alpha=0.55 if st == "COMPLETE" else 0.45, lw=1.2, marker="o", ms=2.5)
    c = curves[best_trial]
    ax.plot(c.epoch, c.val_mean_auroc_8, color=ORANGE, lw=2.4, marker="o", ms=4, label=f"best trial ({best_trial})")
    ax.plot([], [], color=BLUE, lw=1.2, label="completed trials")
    ax.plot([], [], color=GREY, lw=1.2, label="stopped early")
    ax.axvline(MIN_RESOURCE, color=GREY, ls=":", lw=1)
    lo = np.nanpercentile([v for c in curves.values() for v in c.val_mean_auroc_8], 10)
    ax.set_ylim(lo - 0.005, None)
    ax.set(xlabel="Epoch (stage-1 schedule)", ylabel="Validation mean AUROC (8 labels)",
           title=f"{label}: validation curves of every stage-1 trial")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.model}_curves.png")
    plt.close(fig)

    # 3) Score against each hyperparameter (slice plots), with the search range as the x-axis.
    fig, axes = plt.subplots(1, len(space), figsize=(3.1 * len(space), 3.2), sharey=True)
    for ax, (name, (_, lo_, hi_, log)) in zip(np.atleast_1d(axes), space.items()):
        for state, color, mk in [("PRUNED", GREY, "x"), ("COMPLETE", BLUE, "o")]:
            s = df[df.state == state]
            ax.scatter(s[name], s.score, c=color, marker=mk, s=28, zorder=3)
        b = df[df.trial == best_trial]
        ax.scatter(b[name], b.score, s=110, facecolors="none", edgecolors=ORANGE, lw=1.8, zorder=4)
        ax.set_xlim(lo_ / 1.4 if log else lo_ - 0.03 * (hi_ - lo_), hi_ * 1.4 if log else hi_ + 0.03 * (hi_ - lo_))
        if log:
            ax.set_xscale("log")
        ax.axvspan(lo_, hi_, color="#f2f5f8", zorder=0)
        ax.set_xlabel(PRETTY.get(name, name))
    np.atleast_1d(axes)[0].set_ylabel("Validation mean AUROC")
    ymin = complete["score"].min() - 0.004
    np.atleast_1d(axes)[0].set_ylim(ymin, complete["score"].max() + 0.002)
    fig.legend(
        handles=[plt.Line2D([], [], ls="", marker="o", color=BLUE, label="completed (6 epochs)"),
                 plt.Line2D([], [], ls="", marker="x", color=GREY, label="stopped early (score at <= 2 epochs; low ones below the axis)"),
                 plt.Line2D([], [], ls="", marker="o", mfc="none", mec=ORANGE, mew=1.8, ms=9, label="best trial")],
        fontsize=8, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{label}: score against each hyperparameter (orange = best trial)", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(fig_dir / f"{args.model}_slices.png")
    plt.close(fig)

    # 4) Hyperparameter importance (fANOVA over completed trials), when enough trials exist.
    importance = {}
    try:
        import optuna

        importance = optuna.importance.get_param_importances(load_study(args.model))
    except Exception as e:  # noqa: BLE001 - optional diagnostic
        print("importance skipped:", e)

    # 5) Stage 2: validation curves of the refined configs (both schedule lengths) vs the default-recipe run.
    stage2 = out_dir(args.model) / "stage2_runs.csv"
    s2 = pd.read_csv(stage2) if stage2.exists() else pd.DataFrame()
    if len(s2):
        if "schedule" not in s2:
            s2["schedule"] = "full"
        selected_run = s2.sort_values("val_mean_auroc_8").iloc[-1].run
        order = s2.groupby("stage1_trial").val_mean_auroc_8.max().sort_values(ascending=False).index
        trial_colors = dict(zip(order, [ORANGE, BLUE, "#1baf7a", "#6250d6"]))
        lines = [(f"trial {int(r.stage1_trial)}, {'6-epoch' if r.schedule == 'short' else 'full'} schedule",
                  Path(r.path), trial_colors.get(r.stage1_trial, GREY), "--" if r.schedule == "short" else "-",
                  r.run == selected_run) for r in s2.itertuples()]
        default_run = sorted(Path("outputs/runs").glob(f"{args.model}_s224_*"))
        if default_run:
            lines.append(("default recipe, full schedule (baseline)", default_run[0], GREY, ":", False))
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
        for name, path, color, ls, sel in lines:
            h = pd.read_csv(path / "history.csv")
            for ax, col in zip(axes, ["val_mean_auroc_8", "val_mean_auprc_8"]):
                if col in h:
                    ax.plot(h.epoch, h[col], color=color, ls=ls, lw=2.6 if sel else 1.3, marker="o", ms=3,
                            label=name + (" (selected)" if sel else ""), zorder=3 if sel else 2)
        axes[0].set(xlabel="Epoch", ylabel="Validation mean AUROC (8 labels)", title="Primary: mAUROC")
        axes[1].set(xlabel="Epoch", ylabel="Validation mean AUPRC (8 labels)", title="Secondary: mAUPRC")
        for ax in axes:  # zoom on where the runs separate; epochs 1-2 sit below the axis
            ys = [y for ln in ax.get_lines() for x, y in zip(ln.get_xdata(), ln.get_ydata()) if x >= 3]
            if ys:
                ax.set_ylim(min(ys) - 0.003, max(ys) + 0.002)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].legend(fontsize=7.5, loc="lower right", frameon=False)
        fig.suptitle(f"{label}: stage-2 validation curves (solid = full schedule, dashed = 6-epoch schedule)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{args.model}_stage2.png")
        plt.close(fig)

    # 6) Local learning-rate check around the optimum (hpo.py local).
    local_path = out_dir(args.model) / "local_check.csv"
    local = pd.read_csv(local_path) if local_path.exists() else pd.DataFrame()
    if len(local):
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(local.lr, local.val_mean_auroc_8, color=BLUE, lw=1.6, marker="o", ms=6, zorder=3)
        one = local[local.factor == 1.0]
        ax.scatter(one.lr, one.val_mean_auroc_8, s=140, facecolors="none", edgecolors=ORANGE, lw=2, zorder=4,
                   label="stage-1 optimum (factor 1)")
        for r in local.itertuples():
            ax.annotate(f"x{r.factor:g}", (r.lr, r.val_mean_auroc_8), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=INK)
        lo_, hi_ = space["lr"][1], space["lr"][2]
        ax.axvspan(lo_, hi_, facecolor="#e6ebf0", edgecolor="#c5ced8", zorder=0, label="range searched in stage 1")
        ax.set_xscale("log")
        ax.margins(x=0.08, y=0.25)
        ax.set(xlabel="Learning rate (other hyperparameters fixed at the optimum)",
               ylabel="Validation mean AUROC", title=f"{label}: local check along the learning rate")
        ax.legend(fontsize=8, loc="lower left", frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{args.model}_local_lr.png")
        plt.close(fig)

    best = complete.sort_values("value").iloc[-1]
    edges = {k: near_edge(float(best[k]), lo_, hi_, log) for k, (_, lo_, hi_, log) in space.items()}
    rank_corr = None
    if len(s2):
        per = s2.groupby("stage1_trial").agg(stage1=("stage1_value", "first"), stage2=("val_mean_auroc_8", "max"))
        if len(per) >= 3:
            from scipy.stats import spearmanr

            rank_corr = float(spearmanr(per.stage1, per.stage2).correlation)
    summary = {
        "model": args.model, "n_trials": int(len(df)), "completed": int((df.state == "COMPLETE").sum()),
        "pruned": int((df.state == "PRUNED").sum()), "failed": int((df.state == "FAIL").sum()),
        "stage1_epochs_total": int(sum(len(c) for c in curves.values())),
        "default_trial0": float(df.loc[df.trial == 0, "score"].iloc[0]),
        "best_trial": int(best.trial), "best_value": float(best.value),
        "best_params": {k: float(best[k]) for k in space}, "best_param_position": edges,
        "importance": {k: float(v) for k, v in importance.items()},
        "stage2": s2.to_dict(orient="records") if len(s2) else None,
        "stage1_vs_stage2_spearman": rank_corr,
        "local_lr_check": local.to_dict(orient="records") if len(local) else None,
    }
    (out_dir(args.model) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"figures -> {fig_dir}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("search", "refine", "local", "report"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, choices=list(SPACES))
        p.add_argument("--data-dir", default=os.environ.get("TAIX_DATA_DIR", "data/TAIX-Ray"))
        p.add_argument("--workers", type=int, default=16)
        p.add_argument("--max-train-steps", type=int, default=0, help="debug: cap steps per epoch")
        p.add_argument("--smoke", action="store_true", help="write to outputs/hpo_smoke (pipeline test)")
    sub.choices["search"].add_argument("--n-trials", type=int, default=20)
    sub.choices["refine"].add_argument("--top-k", type=int, default=TOP_K)
    sub.choices["refine"].add_argument("--runs-dir", default="outputs/runs")
    sub.choices["refine"].add_argument("--epochs-override", type=int, default=None, help="debug only")
    sub.choices["refine"].add_argument("--schedules", default="full,short",
                                       help="schedule lengths per top config: full and/or short")
    sub.choices["report"].add_argument("--fig-dir", default="docs/hpo/figures")
    sub.choices["local"].add_argument("--lr-factors", default="0.5,2,3")
    sub.choices["local"].add_argument("--runs-dir", default="outputs/runs")
    args = ap.parse_args()
    if args.smoke:
        global HPO_ROOT
        HPO_ROOT = Path("outputs/hpo_smoke")
    {"search": cmd_search, "refine": cmd_refine, "local": cmd_local, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
