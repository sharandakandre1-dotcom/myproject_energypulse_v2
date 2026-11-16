"""Компактна реалізація LSTM на NumPy (прямий і зворотний прохід, оптимізатор Adam).
Мережа: шар LSTM по вікну попередніх значень + повнозв'язний вихід, що отримує стан LSTM і
зовнішні ознаки цільового кроку. Реалізовано без важких залежностей, щоб працювати в онлайн-середовищі."""
import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyLSTM:
    def __init__(self, n_exog, hidden=16, lr=0.01, epochs=25, batch=64, patience=4,
                 max_samples=3000, clip=1.0, seed=28):
        self.E, self.H = n_exog, hidden
        self.lr, self.epochs, self.batch = lr, epochs, batch
        self.patience, self.max_samples, self.clip = patience, max_samples, clip
        self.rng = np.random.default_rng(seed)
        H, E = hidden, n_exog
        self.p = {
            "Wx": self.rng.normal(0, 0.3, (1, 4 * H)),
            "Wh": self.rng.normal(0, 1 / np.sqrt(H), (H, 4 * H)),
            "b": np.zeros(4 * H),
            "Wo": self.rng.normal(0, 1 / np.sqrt(H + E), (H + E, 1)),
            "bo": np.zeros(1),
        }
        self.p["b"][H:2 * H] = 1.0            # зсув воріт забування
        self.history = []

    # ---- прямий прохід ----
    def _forward(self, X, E, keep=False):
        p, H = self.p, self.H
        N, L = X.shape
        h = np.zeros((N, H))
        c = np.zeros((N, H))
        cache = []
        for t in range(L):
            x = X[:, t:t + 1]
            z = x @ p["Wx"] + h @ p["Wh"] + p["b"]
            i, f = _sig(z[:, :H]), _sig(z[:, H:2 * H])
            g, o = np.tanh(z[:, 2 * H:3 * H]), _sig(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            if keep:
                cache.append((x, h, c, i, f, g, o, tc))
            h, c = o * tc, c_new
        feat = np.concatenate([h, E], axis=1)
        out = (feat @ p["Wo"] + p["bo"])[:, 0]
        return out, feat, cache

    # ---- зворотний прохід (BPTT) ----
    def _backward(self, y, out, feat, cache):
        p, H = self.p, self.H
        N = len(y)
        dout = 2.0 * (out - y) / N
        g = {"Wo": feat.T @ dout[:, None], "bo": np.array([dout.sum()])}
        dh = (dout[:, None] @ p["Wo"].T)[:, :H]
        dc = np.zeros_like(dh)
        g["Wx"], g["Wh"], g["b"] = np.zeros_like(p["Wx"]), np.zeros_like(p["Wh"]), np.zeros_like(p["b"])
        for x, h_prev, c_prev, i, f, gg, o, tc in reversed(cache):
            do = dh * tc
            dct = dc + dh * o * (1 - tc ** 2)
            di, dg, df = dct * gg, dct * i, dct * c_prev
            dc = dct * f
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - gg ** 2), do * o * (1 - o)], axis=1)
            g["Wx"] += x.T @ dz
            g["Wh"] += h_prev.T @ dz
            g["b"] += dz.sum(0)
            dh = dz @ p["Wh"].T
        return g

    def loss_and_grads(self, X, E, y):
        out, feat, cache = self._forward(X, E, keep=True)
        return float(np.mean((out - y) ** 2)), self._backward(y, out, feat, cache)

    def predict(self, X, E):
        res = []
        for s in range(0, len(X), 512):
            res.append(self._forward(X[s:s + 512], E[s:s + 512])[0])
        return np.concatenate(res) if res else np.array([])

    # ---- навчання ----
    def fit(self, X, E, y):
        if len(X) > self.max_samples:
            X, E, y = X[-self.max_samples:], E[-self.max_samples:], y[-self.max_samples:]
        nv = max(int(len(X) * 0.1), 1)
        Xt, Et, yt, Xv, Ev, yv = X[:-nv], E[:-nv], y[:-nv], X[-nv:], E[-nv:], y[-nv:]
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v2 = {k: np.zeros_like(v) for k, v in self.p.items()}
        best, best_p, bad, step = np.inf, None, 0, 0
        for ep in range(self.epochs):
            order = self.rng.permutation(len(Xt))
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                _, grads = self.loss_and_grads(Xt[idx], Et[idx], yt[idx])
                norm = np.sqrt(sum(float((gr ** 2).sum()) for gr in grads.values()))
                scale = min(1.0, self.clip / (norm + 1e-12))
                step += 1
                for k in self.p:
                    gr = grads[k] * scale
                    m[k] = 0.9 * m[k] + 0.1 * gr
                    v2[k] = 0.999 * v2[k] + 0.001 * gr ** 2
                    mh, vh = m[k] / (1 - 0.9 ** step), v2[k] / (1 - 0.999 ** step)
                    self.p[k] = self.p[k] - self.lr * mh / (np.sqrt(vh) + 1e-8)
            tr = float(np.mean((self.predict(Xt, Et) - yt) ** 2))
            va = float(np.mean((self.predict(Xv, Ev) - yv) ** 2))
            self.history.append((tr, va))
            if va < best - 1e-7:
                best, bad = va, 0
                best_p = {k: v.copy() for k, v in self.p.items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_p is not None:
            self.p = best_p
        return self
