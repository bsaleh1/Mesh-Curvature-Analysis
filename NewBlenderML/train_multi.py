#!/usr/bin/env python3
# train_multi.py
# Usage: python train_multi.py --in_dir /path/to/folder --target worn_target
import argparse, os, glob
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def load_all(in_dir):
    paths = sorted(glob.glob(os.path.join(in_dir, "curvature_features_*.csv")))
    if not paths:
        raise SystemExit("No curvature_features_*.csv found.")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "object" not in df.columns:
            # derive object name from file name as fallback
            base = os.path.basename(p)
            name = base.replace("curvature_features_","").replace(".csv","")
            df["object"] = name
        df["__path__"] = p
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    return big, paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with curvature_features_*.csv")
    ap.add_argument("--target", default="worn_target", help="Target column to learn; if missing, a synthetic one is created")
    args = ap.parse_args()

    big, paths = load_all(args.in_dir)
    feat = ["x","y","z","nx","ny","nz","H","K","k1","k2"]
    for c in feat:
        if c not in big.columns:
            raise ValueError(f"Missing feature {c} in input CSVs.")

    # Prepare target
    if args.target in big.columns:
        y = big[args.target].astype(np.float32).to_numpy()
        print(f"Using provided target: {args.target}")
    else:
        # Synthetic wear proxy (tweakable)
        y = (
            np.clip(big["H"].to_numpy(), 0, None) * 0.6
            + np.clip(big["k1"].to_numpy(), 0, None) * 0.3
            + np.maximum(0, big["ny"].to_numpy()) * 0.4
        ).astype(np.float32)
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        print("No target column found; generated synthetic wear proxy.")

    X = big[feat].astype(np.float32).to_numpy()
    n = len(big)
    print(f"Total rows (all meshes): {n}")

    # Choose model based on data size
    if n < 1000:
        model = make_pipeline(StandardScaler(), LinearRegression())
        print("Model: LinearRegression (small dataset)")
    else:
        model = make_pipeline(StandardScaler(),
                              MLPRegressor(hidden_layer_sizes=(64,64), activation="relu",
                                           learning_rate_init=1e-3, max_iter=400, random_state=3))
        print("Model: MLPRegressor (64,64)")

    # quick score
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=3)
    model.fit(Xtr, ytr)
    try:
        pred_te = model.predict(Xte)
        print("R^2 (holdout):", round(r2_score(yte, pred_te), 4))
    except Exception:
        pass

    # Fit on all, predict all
    model.fit(X, y)
    big["ml_value"] = model.predict(X).astype(np.float32)

    # Write one predictions file per object
    out_dir = args.in_dir
    for obj_name, grp in big.groupby("object"):
        # keep vertex order (index column is 0..n-1 per mesh because we exported per-object)
        if "index" in grp.columns:
            grp_sorted = grp.sort_values("index")
        else:
            grp_sorted = grp.copy()
            grp_sorted["index"] = np.arange(len(grp_sorted))
        vals = grp_sorted["ml_value"].to_numpy()
        # normalize per-mesh for nicer visualization
        pmin, pmax = float(vals.min()), float(vals.max())
        norm = (vals - pmin) / (pmax - pmin + 1e-8)
        out = pd.DataFrame({"index": grp_sorted["index"], "ml_value": vals, "ml_value_norm01": norm})
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in obj_name)
        out_path = os.path.join(out_dir, f"predictions_{safe}.csv")
        out.to_csv(out_path, index=False)
        print(f"✓ Wrote {out_path} ({len(out)} rows)")

if __name__ == "__main__":
    main()
