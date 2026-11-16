"""Модулі прогнозування: базовий метод, Random Forest, XGBoost, LSTM. Єдиний інтерфейс
fit / forecast, рекурсивний багатокрокований прогноз і оцінка якості на ковзних початках."""
import importlib.util

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from calendar_utils import calendar_frame
from features import (build_row, exog_matrix, expected_temp, feature_columns, make_features)
from lstm_np import NumpyLSTM

SEED = 28


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    mask = np.abs(y_true) > 1e-6
    mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100) if mask.any() else float("nan")
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))), "mape": mape}


class Forecaster:
    name = "base"

    def fit(self, train, gran, use_temp):
        return self

    def forecast(self, history, future_ts, gran, use_temp):
        raise NotImplementedError


class BaselineForecaster(Forecaster):
    """Середнє за останні 7 діб (для погодинних даних: та сама година за 7 діб)."""
    name = "Базовий (середнє за 7 діб)"

    def forecast(self, history, future_ts, gran, use_temp):
        vals = list(history["consumption"].to_numpy(float))
        period = 24 if gran == "h" else 1
        out = []
        for _ in range(len(future_ts)):
            pts = [vals[-period * k] for k in range(1, 8) if len(vals) >= period * k]
            p = float(np.mean(pts))
            vals.append(p)
            out.append(p)
        return np.array(out)


class TreeForecaster(Forecaster):
    def __init__(self, name, factory):
        self.name, self.factory, self.model = name, factory, None

    def fit(self, train, gran, use_temp):
        f = make_features(train, gran, use_temp)
        cols = feature_columns(gran, use_temp)
        self.model = self.factory()
        self.model.fit(f[cols].to_numpy(float), f["y"].to_numpy(float))
        return self

    def forecast(self, history, future_ts, gran, use_temp):
        vals = list(history["consumption"].to_numpy(float))
        temps = list(history["temperature"].to_numpy(float)) if use_temp else None
        cal = calendar_frame(future_ts)
        cal = {c: cal[c].to_numpy() for c in cal.columns}
        cols = feature_columns(gran, use_temp)
        out = []
        for k in range(len(future_ts)):
            t_exp = expected_temp(temps, gran) if use_temp else None
            row = build_row(vals, cal, k, t_exp, gran, use_temp)
            p = max(0.0, float(self.model.predict(np.array([[row[c] for c in cols]], dtype=float))[0]))
            vals.append(p)
            if use_temp:
                temps.append(t_exp)
            out.append(p)
        return np.array(out)


class LSTMForecaster(Forecaster):
    name = "LSTM"

    def __init__(self, hidden=16, epochs=25):
        self.hidden, self.epochs, self.net = hidden, epochs, None

    def _window(self, gran):
        return 24 if gran == "h" else 14

    def fit(self, train, gran, use_temp):
        L = self._window(gran)
        y = train["consumption"].to_numpy(float)
        self.ymin, self.ymax = float(y.min()), float(y.max())
        self.den = max(self.ymax - self.ymin, 1e-9)
        ys = (y - self.ymin) / self.den                                   # формула (3.1)
        t = train["temperature"].to_numpy(float) if use_temp else None
        if use_temp:
            self.tmin, self.tmax = float(np.nanmin(t)), float(np.nanmax(t))
            t = (t - self.tmin) / max(self.tmax - self.tmin, 1e-9)
        E = exog_matrix(calendar_frame(train["timestamp"]), t, gran, use_temp)
        n = len(ys)
        win = np.lib.stride_tricks.sliding_window_view(ys, L)[: n - L]     # (n-L, L)
        self.net = NumpyLSTM(n_exog=E.shape[1], hidden=self.hidden, epochs=self.epochs, seed=SEED)
        self.net.fit(win, E[L:], ys[L:])
        return self

    def forecast(self, history, future_ts, gran, use_temp):
        L = self._window(gran)
        ys = list((history["consumption"].to_numpy(float) - self.ymin) / self.den)
        temps = list(history["temperature"].to_numpy(float)) if use_temp else None
        cal = calendar_frame(future_ts)
        out = []
        for k in range(len(future_ts)):
            if use_temp:
                t_exp = expected_temp(temps, gran)
                temps.append(t_exp)
                ts = (t_exp - self.tmin) / max(self.tmax - self.tmin, 1e-9)
            else:
                ts = None
            row = cal.iloc[[k]]
            E = exog_matrix({c: row[c].to_numpy() for c in row.columns}, None if ts is None else np.array([ts]),
                            gran, use_temp)
            p_s = float(self.net.predict(np.array([ys[-L:]]), E)[0])
            ys.append(p_s)
            out.append(max(0.0, p_s * self.den + self.ymin))
        return np.array(out)


def xgboost_available():
    return importlib.util.find_spec("xgboost") is not None


def _rf():
    return RandomForestRegressor(n_estimators=160, max_depth=8, random_state=SEED, n_jobs=1)


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
                        colsample_bytree=0.9, random_state=SEED, n_jobs=1, verbosity=0)


def model_registry():
    reg = {
        "Базовий (середнє за 7 діб)": lambda: BaselineForecaster(),
        "Random Forest": lambda: TreeForecaster("Random Forest", _rf),
        "LSTM": lambda: LSTMForecaster(),
    }
    if xgboost_available():
        reg["XGBoost"] = lambda: TreeForecaster("XGBoost", _xgb)
    return dict(sorted(reg.items(), key=lambda kv: ["Базовий (середнє за 7 діб)", "Random Forest", "XGBoost", "LSTM"].index(kv[0])))


def split_sizes(n, horizon, gran, test_frac=0.2):
    test_n = min(max(int(n * test_frac), 2 * horizon), n // 2)
    train_n = n - test_n
    min_train = 24 * 21 if gran == "h" else 40
    if train_n < min_train or test_n < horizon + 1:
        need = (min_train + 2 * horizon)
        raise ValueError(f"Замало даних для горизонту {horizon}: потрібно щонайменше {need} точок ряду, є {n}. "
                         "Розширте період у фільтрах або зменшіть горизонт.")
    return train_n, test_n


def evaluate_forecaster(make, series, gran, use_temp, horizon, n_origins=6):
    """Хронологічне розбиття 80/20 і багатокрокова оцінка на кількох початках прогнозу."""
    n = len(series)
    train_n, _ = split_sizes(n, horizon, gran)
    fc = make().fit(series.iloc[:train_n], gran, use_temp)
    last = n - horizon
    origins = np.unique(np.linspace(train_n, last, num=max(1, min(n_origins, last - train_n + 1))).astype(int))
    rows = []
    for o in origins:
        fut = series.iloc[o:o + horizon]
        pred = fc.forecast(series.iloc[:o], pd.DatetimeIndex(fut["timestamp"]), gran, use_temp)
        rows.append(pd.DataFrame({"timestamp": fut["timestamp"].to_numpy(), "actual": fut["consumption"].to_numpy(),
                                  "predicted": pred, "origin": o}))
    det = pd.concat(rows, ignore_index=True)
    m = regression_metrics(det["actual"], det["predicted"])
    m["n_origins"] = len(origins)
    return m, det


def final_forecast(make, series, gran, use_temp, horizon):
    fc = make().fit(series, gran, use_temp)
    step = pd.Timedelta(hours=1) if gran == "h" else pd.Timedelta(days=1)
    fut = pd.date_range(series["timestamp"].iloc[-1] + step, periods=horizon, freq=gran)
    return pd.DataFrame({"timestamp": fut, "forecast": fc.forecast(series, fut, gran, use_temp)})
