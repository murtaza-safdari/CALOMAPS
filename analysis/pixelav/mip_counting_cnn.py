#!/usr/bin/env python3
"""Reticles -> MIP counts with a small convolutional network on the hit map.

    python analysis/pixelav/mip_counting_cnn.py <reticles.npz> [more reticles.npz ...] --out-dir <dir>
                                                [--threshold 1000] [--noise 100] [--seed 1]
                                                [--folds 5] [--extrap-max-k 2] [--skip-extrapolation]
                                                [--sizes tiny,small] [--input binary,tot2,analog]
                                                [--targets n_frac] [--steps 3000 | --epochs E]
                                                [--batch 512] [--lr 0.003] [--device auto]
                                                [--layer-weights L:W[,L:W ...]]
                                                [--out-name cnn] [--compare <preds.npz> ...]

The readout, truths, folds and file tags are those of analysis/pixelav/mip_counting.py (its
prepare, on the files sorted by overlay, with the same seed: the same hit maps); the reticle shape
comes from the files. For each fold a model is trained on the reticles of all files whose events
lie in the other folds and predicts that fold's reticles of every file; EXTRAPOLATION (unless
--skip-extrapolation) trains on the files of overlay <= --extrap-max-k only and predicts the files
of larger overlay. Reticles without a hit are left out of training and predicted 0, as in
mip_counting.py.

INPUT (--input, comma-separated): binary, the hit map; tot2, a 2-bit charge per pixel, the number
of the levels 1, 4 and 16 times the threshold that the noisy charge exceeds (0..3, scaled to
0..1); analog, log1p(charge / threshold) of the noisy charge above the threshold, 0 below. The
estimators are cnn_<size> for binary and cnn_<size>_<input> otherwise.

MODELS (MODELS): 3 x 3 convolutions with ReLU and zero padding (the border pixels see the reticle
edge), a 1 x 1 convolution to one channel, softplus per pixel and the sum over the map: the count
is a sum of non-negative local contributions, so the same weights serve any density and any
reticle shape. tiny: 1 -> 8 -> 8 -> 8 channels; small: 1 -> 16 -> 32 -> 32 -> 32. The parameter
counts and multiply-adds per reticle are printed and stored.

TRAINING: Poisson negative log-likelihood of the count, Adam, learning rate --lr with cosine decay,
batches of --batch, whole epochs: --epochs if given, else the fewest epochs holding --steps
batches. The training maps and counts are held on the device for the whole training and indexed
there; the order of each epoch is numpy default_rng((seed, fold, model, target, training files,
input)).permutation, torch seeded with the same numbers. Device auto: cuda, else mps, else cpu
(mps and cuda results are reproducible to float rounding only). The wall time of every training
is printed and stored.

OUTPUT in --out-dir: <out-name>_preds.npz in the layout of mip_counting.py's preds.npz (readable
by its --include-preds), results_<out-name>.json (the metrics of mip_counting.py for these
estimators, the parameter counts, losses and wall times). With --compare <preds.npz> ... (of
mip_counting.py or of this script, the same files and readout, checked): the comparison table of
every estimator and the figures fig_<out-name>_*.png with all of them.
"""
import argparse, json, math, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import mip_counting as mc                     # noqa: E402

MODELS = {"tiny": (8, 8, 8), "small": (16, 32, 32, 32)}
INPUTS = ("binary", "tot2", "analog")
SCALE = {"binary": 1.0, "tot2": 1.0 / 3.0, "analog": 1.0}
STEPS = 3000
BATCH = 512
LR = 3e-3
PRED_BATCH = 8192
HEAD_BIAS = -5.0      # softplus(-5) * 273 pixels ~ 2 particles per 13 x 21 map at the start of training


def pick_device(name):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build(channels):
    import torch.nn as nn
    layers, cin = [], 1
    for c in channels:
        layers += [nn.Conv2d(cin, c, 3, padding=1), nn.ReLU()]
        cin = c
    head = nn.Conv2d(cin, 1, 1)
    nn.init.constant_(head.bias, HEAD_BIAS)
    layers += [head, nn.Softplus()]
    return nn.Sequential(*layers)


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def macs_per_reticle(channels, ny, nx):
    cin, m = 1, 0
    for c in channels:
        m += cin * c * 9
        cin = c
    return int((m + cin) * ny * nx)


def est_name(size, inp):
    return f"cnn_{size}" if inp == "binary" else f"cnn_{size}_{inp}"


def count(model, x):
    """the predicted count of each map of x [b, 1, ny, nx]"""
    return model(x).sum(dim=(1, 2, 3))


def train(X, y, scale, channels, args, device, seed_key, log=None):
    """A model trained on the device maps X ([n, ny, nx], no empty map) times scale and the device
    counts y; the epoch losses, the epochs and the steps."""
    import torch
    torch.manual_seed(int(np.random.SeedSequence(seed_key).generate_state(1)[0]))
    rng = np.random.default_rng(seed_key)
    n = len(y)
    per_epoch = math.ceil(n / args.batch)
    epochs = args.epochs if args.epochs else max(1, math.ceil(args.steps / per_epoch))
    model = build(channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * per_epoch)
    # a count of mean mu fluctuates by sqrt(mu), so reticles of 1 and of 300 objects weigh alike in
    # units of their own fluctuation
    loss_fn = torch.nn.PoissonNLLLoss(log_input=False, eps=1e-6)
    losses = []
    model.train()
    for ep in range(epochs):
        order = torch.from_numpy(rng.permutation(n)).to(device)
        tot = torch.zeros((), device=device)
        for a in range(0, n, args.batch):
            idx = order[a:a + args.batch]
            xb = (X[idx].float() * scale).unsqueeze(1)
            loss = loss_fn(count(model, xb), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.detach()
        if log or ep == epochs - 1:
            losses.append(float(tot) / per_epoch)
            if log:
                log(f"      epoch {ep}: loss {losses[-1]:.4f}")
    return model, losses, epochs, epochs * per_epoch


def predict(model, X, scale):
    """the predicted counts of the device maps X, on the host"""
    import torch
    model.eval()
    with torch.no_grad():
        out = [count(model, (X[a:a + PRED_BATCH].float() * scale).unsqueeze(1))
               for a in range(0, len(X), PRED_BATCH)]
    return torch.cat(out).cpu().numpy().astype(np.float64) if out else np.zeros(0)


def crossfit_cnn(Ds, Xd, Yd, target, size, inp, args, device, train_tags, eval_tags, n_events, log):
    """{tag: out-of-fold prediction} of model size on input inp, and per fold the losses and wall
    time; Xd and Yd hold each file's maps and target counts on the device."""
    import torch
    tr = [D for D in Ds if D["tag"] in train_tags]
    ev = [D for D in Ds if D["tag"] in eval_tags]
    P = {D["tag"]: np.full(D["n"], np.nan) for D in ev}
    folds = []
    key0 = (args.seed, list(MODELS).index(size), mc.TRUTHS.index(target), len(train_tags), INPUTS.index(inp))
    for f in range(args.folds):
        seen = np.zeros(n_events, bool)
        for D in tr:
            seen[D["events"][D["fold"] != f].ravel()] = True
        for D in ev:
            mc._require(not seen[D["events"][D["fold"] == f].ravel()].any(),
                        f"fold {f}: an event of {D['tag']} is also in the training set")
        t0 = time.time()
        rows = [torch.from_numpy(np.flatnonzero((D["fold"] != f) & (D["n_hits"] > 0))).to(device) for D in tr]
        X = torch.cat([Xd[D["tag"]][r] for D, r in zip(tr, rows)])
        y = torch.cat([Yd[D["tag"]][target][r] for D, r in zip(tr, rows)])
        model, ls, epochs, steps = train(X, y, SCALE[inp], MODELS[size], args, device, (key0[0], f) + key0[1:],
                                         log if (f == 0 and args.verbose) else None)
        del X, y
        for D in ev:
            s = np.flatnonzero(D["fold"] == f)
            p = predict(model, Xd[D["tag"]][torch.from_numpy(s).to(device)], SCALE[inp])
            P[D["tag"]][s] = np.where(D["n_hits"][s] > 0, p, 0.0)
        dt = time.time() - t0
        n_train = int(sum(len(r) for r in rows))
        folds.append({"training_maps": n_train, "epochs": epochs, "steps": steps, "final_loss": ls[-1],
                      "wall_s": dt})
        log(f"    {est_name(size, inp)} {target} fold {f}: {n_train} training maps, {epochs} epochs "
            f"({steps} steps), final loss {ls[-1]:.4f}, {dt:.1f} s")
    for D in ev:
        mc._require(np.all(np.isfinite(P[D["tag"]])), f"{D['tag']}: a reticle without a prediction")
    return P, folds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reticles", nargs="+", help="reticles.npz of analysis/pixelav/reticles.py, the same events")
    ap.add_argument("--out-dir", required=True, metavar="DIR")
    ap.add_argument("--threshold", type=float, default=mc.THRESHOLD, metavar="E")
    ap.add_argument("--noise", type=float, default=mc.NOISE, metavar="E")
    ap.add_argument("--seed", type=int, default=mc.SEED)
    ap.add_argument("--folds", type=int, default=mc.N_FOLDS)
    ap.add_argument("--extrap-max-k", type=int, default=mc.EXTRAP_MAX_K, metavar="K")
    ap.add_argument("--skip-extrapolation", action="store_true")
    ap.add_argument("--sizes", default="tiny,small", help=f"comma-separated, of {', '.join(MODELS)}")
    ap.add_argument("--input", default="binary", help=f"comma-separated, of {', '.join(INPUTS)}")
    ap.add_argument("--targets", default="n_frac", help=f"comma-separated, of {', '.join(mc.TRUTHS)}")
    ap.add_argument("--steps", type=int, default=STEPS, help="training batches per model (rounded up to whole epochs)")
    ap.add_argument("--epochs", type=int, help="training epochs per model, in place of --steps")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    ap.add_argument("--layer-weights", metavar="L:W[,L:W ...]",
                    help="also give the event totals with each reticle weighed W from layer L on (mip_counting.py)")
    ap.add_argument("--out-name", default="cnn", metavar="NAME", help="name of the output files")
    ap.add_argument("--verbose", action="store_true", help="the epoch losses of the first fold")
    ap.add_argument("--compare", nargs="+", default=[], metavar="NPZ",
                    help="preds.npz of mip_counting.py or of this script, the same files and readout")
    args = ap.parse_args(argv)
    sizes = [s for s in args.sizes.split(",") if s]
    inputs = [s for s in args.input.split(",") if s]
    targets = [t for t in args.targets.split(",") if t]
    if not sizes or any(s not in MODELS for s in sizes) or len(set(sizes)) != len(sizes):
        ap.error(f"--sizes: distinct, of {', '.join(MODELS)}")
    if not inputs or any(s not in INPUTS for s in inputs) or len(set(inputs)) != len(inputs):
        ap.error(f"--input: distinct, of {', '.join(INPUTS)}")
    if not targets or any(t not in mc.TRUTHS for t in targets) or len(set(targets)) != len(targets):
        ap.error(f"--targets: distinct, of {', '.join(mc.TRUTHS)}")
    if args.steps < 1 or args.batch < 1 or (args.epochs is not None and args.epochs < 1):
        ap.error("--steps, --epochs and --batch must be positive")
    if args.folds < 2:
        ap.error("--folds must be at least 2")
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = pick_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    def log(s):
        print(s, flush=True)

    raws = [mc.load_reticles(p) for p in args.reticles]
    order = np.argsort([R["overlay"] for R in raws], kind="stable")
    raws = [raws[i] for i in order]
    tags = mc.file_tags(raws)
    try:
        Ds, ev_fold = mc.prepare(raws, tags, args.threshold, args.noise, args.seed, args.folds,
                                 keep_hits="binary" in inputs, encode=tuple(i for i in inputs if i != "binary"))
        if args.layer_weights:
            steps = mc.parse_layer_weights(args.layer_weights)
            for D in Ds:
                D["layer_w"] = mc.layer_weights(steps, D["layer"])
    except mc.Inconsistent as err:
        raise SystemExit(f"mip_counting_cnn: {err}") from None
    del raws
    n_events = int(len(ev_fold))
    ny, nx = Ds[0]["ny"], Ds[0]["nx"]
    settings = {"threshold": args.threshold, "noise": args.noise, "seed": args.seed,
                "folds": args.folds, "event_fold": ev_fold}
    info = {}
    for s in sizes:
        info[s] = {"channels": list(MODELS[s]), "parameters": n_params(build(MODELS[s])),
                   "multiply_adds_per_reticle": macs_per_reticle(MODELS[s], ny, nx)}
    log(f"mip_counting_cnn: {len(Ds)} file(s), {n_events} events, {ny} x {nx} pixels, device {device}; "
        + "; ".join(f"{s}: {v['parameters']} parameters, {v['multiply_adds_per_reticle']} multiply-adds per reticle"
                    for s, v in info.items()))

    preds, xpreds, runs = {}, {}, {}
    small = [D["tag"] for D in Ds if D["overlay"] <= args.extrap_max_k]
    large = [D["tag"] for D in Ds if D["overlay"] > args.extrap_max_k]
    t_all = time.time()
    Yd = {D["tag"]: {t: torch.from_numpy(D[t].astype(np.float32)).to(device) for t in targets} for D in Ds}
    try:
        for inp in inputs:
            Xd = {D["tag"]: torch.from_numpy(D["hits"].view(np.uint8) if inp == "binary" else D["enc"][inp]).to(device)
                  for D in Ds}
            for t in targets:
                for s in sizes:
                    nm = est_name(s, inp)
                    t0 = time.time()
                    P, fl = crossfit_cnn(Ds, Xd, Yd, t, s, inp, args, device, tags, tags, n_events, log)
                    for g, p in P.items():
                        preds.setdefault(g, {}).setdefault(t, {})[nm] = p
                    runs[f"{nm}__{t}"] = {"folds": fl, "wall_s": time.time() - t0}
                    log(f"  {nm} {t}: {time.time() - t0:.1f} s for {args.folds} trainings")
                    if small and large and not args.skip_extrapolation:
                        log(f"  extrapolation {nm} {t}: train on {small}, evaluate {large}")
                        t0 = time.time()
                        X, fl = crossfit_cnn(Ds, Xd, Yd, t, s, inp, args, device, small, large, n_events, log)
                        for g, p in X.items():
                            xpreds.setdefault(g, {}).setdefault(t, {})[nm] = p
                        runs[f"{nm}__{t}__extrapolation"] = {"folds": fl, "wall_s": time.time() - t0}
            del Xd
    except mc.Inconsistent as err:
        raise SystemExit(f"mip_counting_cnn: {err}") from None
    log(f"  training and prediction: {time.time() - t_all:.0f} s")
    mc.save_preds(os.path.join(args.out_dir, f"{args.out_name}_preds.npz"), Ds, preds, xpreds, settings)
    results = mc.evaluate(Ds, preds, xpreds)
    names = [est_name(s, i) for i in inputs for s in sizes]
    out = {"settings": {"files": [D["path"] for D in Ds], "tags": tags, "threshold": args.threshold,
                        "noise": args.noise, "seed": args.seed, "folds": args.folds,
                        "extrap_max_k": args.extrap_max_k, "skip_extrapolation": args.skip_extrapolation,
                        "steps": args.steps, "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                        "device": str(device), "targets": targets, "inputs": inputs,
                        "layer_weights": args.layer_weights,
                        "reticle_shape": [ny, nx], "estimators": names},
           "models": info, "trainings": runs, "results": results}

    lines = ["", "rms/bias of estimate - truth per truth bin (n_frac rounded to an integer); sum ratio = "
             "mean+-std over super-events of sum(estimate)/sum(truth); spread = std/mean of sum(estimate)"]
    if args.compare:
        allp, allx = {}, {}
        try:
            for path in args.compare:
                p2, x2 = mc.load_preds(path, Ds, settings)
                mc.merge_preds(allp, p2)
                mc.merge_preds(allx, x2)
            mc.merge_preds(allp, preds)
            mc.merge_preds(allx, xpreds)
        except mc.Inconsistent as err:
            raise SystemExit(f"mip_counting_cnn: {err}") from None
        both = mc.evaluate(Ds, allp, allx)
        out["comparison"] = both
        for t in targets:
            lines += mc.table(both, t)
            lines += mc.table(both, t, kind="extrapolation")
        fig_names = list(dict.fromkeys(list(mc.FIG_ESTIMATORS) + [nm for P in allp.values() for by in P.values()
                                                                   for nm in by if nm.startswith("cnn_")]))
        figs = []
        for t in targets:
            figs += mc.figures(Ds, allp, both, args.out_dir, truth=t, prefix=f"fig_{args.out_name}", names=fig_names)
        figs.append(mc.example_maps(Ds, allp, args.out_dir, truth=targets[0], seed=args.seed,
                                    prefix=f"fig_{args.out_name}"))
        out["figures"] = figs
    else:
        for t in targets:
            lines += mc.table(results, t)
            lines += mc.table(results, t, kind="extrapolation")
    with open(os.path.join(args.out_dir, f"results_{args.out_name}.json"), "w") as f:
        json.dump(mc._jsonable(out), f, indent=1)
    lines.append(f"  written: {args.out_dir}/{args.out_name}_preds.npz, results_{args.out_name}.json"
                 + (f", {len(out['figures'])} figure(s)" if args.compare else ""))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
