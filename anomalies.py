"""Виявлення нетипових відхилень: (1) z-оцінка добових сум, (2) залишки прогнозу,
(3) Isolation Forest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from calendar_utils import calendar_frame
from features import feature_columns, make_features


def zscore_anomalies(series, threshold=2.5):
    """Формула (3.3): z = (y - mu) / sigma, відхилення якщо |z| >= threshold."""
    y = series["consumption"].to_numpy(float)
    sigma = y.std()
    z = (y - y.mean()) / sigma if sigma > 0 else np.zeros_like(y)
    out = series[["timestamp", "consumption"]].copy()
    out["score"] = z
    out["is_anomaly"] = np.abs(z) >= threshold
    return out


def residual_anomalies(series, gran, use_temp, k=3.0):
    """Формула (1.4): |y - y_pred| > delta, де delta = k * sigma залишків (робастна оцінка за MAD).
    Передбачення одного кроку отримують на вибірках, що не входили до навчання (часові блоки)."""
    f = make_features(series, gran, use_temp)
    if len(f) < 120:
        raise ValueError("Замало точок для методу залишків (потрібно щонайменше ~120).")
    cols = feature_columns(gran, use_temp)
    X, y = f[cols].to_numpy(float), f["y"].to_numpy(float)
    pred = np.full(len(f), np.nan)
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        m = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=28, n_jobs=1)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    mask = ~np.isnan(pred)
    res = (y - pred)[mask]
    sigma = 1.4826 * np.median(np.abs(res - np.median(res)))
    if sigma <= 0:
        sigma = res.std() or 1.0
    delta = k * sigma
    out = pd.DataFrame({"timestamp": f["timestamp"].to_numpy()[mask], "consumption": y[mask],
                        "predicted": pred[mask], "score": res})
    out["is_anomaly"] = np.abs(res) > delta
    out.attrs["delta"] = float(delta)
    return out


def isolation_forest_anomalies(series, gran, use_temp, contamination=0.01):
    cal = calendar_frame(series["timestamp"])
    X = pd.DataFrame({"consumption": series["consumption"].to_numpy(float), "dayofweek": cal["dayofweek"]})
    if gran == "h":
        X["hour"] = cal["hour"]
    if use_temp:
        X["temperature"] = series["temperature"].to_numpy(float)
    Xs = StandardScaler().fit_transform(X.to_numpy(float))
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=28)
    lab = iso.fit_predict(Xs)
    out = series[["timestamp", "consumption"]].copy()
    out["score"] = -iso.decision_function(Xs)          # більше значення = більш аномально
    out["is_anomaly"] = lab == -1
    return out
