"""Remaining Useful Life (RUL) prediction on NASA C-MAPSS FD001.
Compares Ridge, Random Forest, MLP and LSTM on the same windows of data.

Usage:  python predictive-maintenance.py            (expects the 3 FD001 files in data/)
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

DATA = "data/"
OUT = "results/"
CAP, W = 125, 30                     # RUL cap and window length (flights)
FEATS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
         "s12", "s13", "s14", "s15", "s17", "s20", "s21"]   # sensors that vary
dev = "cuda" if torch.cuda.is_available() else "cpu"
T = lambda a: torch.tensor(a, dtype=torch.float32)


# ---------------- data ----------------
def load():
    cols = ["unit", "cycle"] + [f"set{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
    rd = lambda f: pd.read_csv(DATA + f, sep=r"\s+", header=None, names=cols)
    tr, te = rd("train_FD001.txt"), rd("test_FD001.txt")
    rt = pd.read_csv(DATA + "RUL_FD001.txt", header=None, names=["rul"])
    # label = flights left until failure, capped at CAP
    tr["rul"] = (tr.groupby("unit").cycle.transform("max") - tr.cycle).clip(upper=CAP)
    sc = MinMaxScaler().fit(tr[FEATS])           # fit on training data only
    tr[FEATS] = sc.transform(tr[FEATS])
    te[FEATS] = sc.transform(te[FEATS])
    return tr, te, rt.rul.values


def windows(df):
    """Sliding windows inside each engine (never across engines)."""
    X, y, u = [], [], []
    for k, g in df.groupby("unit"):
        d, r = g[FEATS].values, g["rul"].values
        for e in range(W, len(g) + 1):
            X.append(d[e - W:e]); y.append(r[e - 1]); u.append(k)
    return np.array(X), np.array(y), np.array(u)


# ---------------- metrics ----------------
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

def nasa(p, t):
    """NASA asymmetric score: late predictions are penalised more."""
    d = p - t
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


# ---------------- models ----------------
class MLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Flatten(), nn.Linear(W * len(FEATS), 128), nn.ReLU(),
                              nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


class LSTM(nn.Module):
    def __init__(s, hidden=64, layers=2):
        super().__init__()
        s.lstm = nn.LSTM(len(FEATS), hidden, layers, batch_first=True, dropout=0.2)
        s.head = nn.Linear(hidden, 1)
    def forward(s, x):
        out, _ = s.lstm(x)
        return s.head(out[:, -1]).squeeze(-1)


def fit(model, Xtr, ytr, Xv, yv, epochs=150, lr=1e-3, bs=128,
        patience=20, min_epochs=30, seed=0):
    torch.manual_seed(seed)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = T(Xtr).to(dev), T(ytr / CAP).to(dev)       # targets scaled to 0-1
    Xvv, yvv = T(Xv).to(dev), T(yv / CAP).to(dev)
    best, bad, state = 1e9, 0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = ((model(Xt[idx]) - yt[idx]) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = ((model(Xvv) - yvv) ** 2).mean().sqrt().item() * CAP
        if ep % 5 == 0:
            print(f"  epoch {ep:3d}  val RMSE {v:.2f}")
        if v < best:
            best, bad = v, 0
            state = {k: x.clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience and ep >= min_epochs:
                break
    model.load_state_dict(state)
    return model


def predict(model, A):
    model.eval()
    with torch.no_grad():
        return model(T(A).to(dev)).cpu().numpy() * CAP


# ---------------- main ----------------
def main(seed=0):
    tr, te, yte = load()
    X, y, u = windows(tr)
    # validation = 20 whole engines (never split one engine across sets)
    val = np.random.RandomState(0).choice(np.arange(1, 101), 20, replace=False)
    mv = np.isin(u, val)
    Xtr, ytr, Xv, yv = X[~mv], y[~mv], X[mv], y[mv]
    # test input = last W flights of each test engine
    Xte = np.array([g[FEATS].values[-W:] for _, g in te.groupby("unit")])
    print("windows:", X.shape, "train", Xtr.shape, "val", Xv.shape, "test", Xte.shape)

    results, preds = [], {}
    def add(name, pv, pt, sec):
        pt = np.clip(pt, 0, CAP)
        preds[name] = pt
        results.append(dict(model=name, val_rmse=rmse(pv, yv), test_rmse=rmse(pt, yte),
                            test_rmse_capped=rmse(pt, np.minimum(yte, CAP)),
                            nasa_score=nasa(pt, yte), seconds=sec))

    flat = lambda a: a.reshape(len(a), -1)

    t = time.time()
    m = Ridge(alpha=1.0).fit(flat(Xtr), ytr)
    add("Ridge", m.predict(flat(Xv)), m.predict(flat(Xte)), time.time() - t)

    t = time.time()
    m = RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=5,
                              max_features=0.15, n_jobs=-1, random_state=0
                              ).fit(flat(Xtr)[::3], ytr[::3])      # every 3rd window, for speed
    add("RandomForest", m.predict(flat(Xv)), m.predict(flat(Xte)), time.time() - t)

    for name, mdl in [("MLP", MLP()), ("LSTM", LSTM())]:
        print(name)
        t = time.time()
        mdl = fit(mdl, Xtr, ytr, Xv, yv, seed=seed)
        add(name, predict(mdl, Xv), predict(mdl, Xte), time.time() - t)

    res = pd.DataFrame(results).round(2)
    res.to_csv(OUT + "results.csv", index=False)
    print(res.to_string(index=False))

    fig, ax = plt.subplots(1, len(preds), figsize=(4 * len(preds), 4), sharey=True)
    for a, (n, p) in zip(ax, preds.items()):
        a.scatter(yte, p, s=12)
        a.plot([0, 140], [0, 140], "r--")
        a.set_title(n); a.set_xlabel("true RUL")
    ax[0].set_ylabel("predicted RUL")
    plt.tight_layout()
    plt.savefig(OUT + "pred_vs_true.png", dpi=150)


if __name__ == "__main__":
    main()
