"""Швидка самоперевірка ядра (без інтерфейсу): python selftest.py"""
import os
import tempfile
import time

import numpy as np
import pandas as pd

import db
import demo
import etl
import features as F
import models as M
from lstm_np import NumpyLSTM
from calendar_utils import calendar_frame
from anomalies import isolation_forest_anomalies, residual_anomalies, zscore_anomalies
from recommend import build_recommendations


def ok(msg):
    print("OK  ", msg)


def test_lstm_gradients():
    net = NumpyLSTM(n_exog=3, hidden=4, seed=1)
    rng = np.random.default_rng(0)
    X, E, y = rng.normal(size=(5, 6)), rng.normal(size=(5, 3)), rng.normal(size=5)
    _, g = net.loss_and_grads(X, E, y)
    worst = 0.0
    for k in net.p:
        flat = net.p[k].reshape(-1)
        for i in rng.choice(len(flat), min(6, len(flat)), replace=False):
            old = flat[i]
            flat[i] = old + 1e-5; lp = net.loss_and_grads(X, E, y)[0]
            flat[i] = old - 1e-5; lm = net.loss_and_grads(X, E, y)[0]
            flat[i] = old
            num = (lp - lm) / 2e-5
            ana = g[k].reshape(-1)[i]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    assert worst < 1e-4, worst
    ok(f"градієнти LSTM збігаються з числовими (похибка {worst:.1e})")


def test_features_equal(gran):
    d = demo.make_demo_data(days=40)
    s, info = F.build_series(d, ["Корпус А"], gran)
    f = F.make_features(s, gran, True)
    cols = F.feature_columns(gran, True)
    row = f.iloc[len(f) // 2]
    pos = int(np.where(s["timestamp"].to_numpy() == row["timestamp"])[0][0])
    vals = list(s["consumption"].iloc[:pos])
    cal = calendar_frame(s["timestamp"].iloc[[pos]])
    cal = {c: cal[c].to_numpy() for c in cal.columns}
    r = F.build_row(vals, cal, 0, float(s["temperature"].iloc[pos]), gran, True)
    assert np.allclose([r[c] for c in cols], row[cols].to_numpy(float)), (gran,)
    ok(f"векторизовані та рядкові ознаки збігаються ({gran})")


def test_db_and_models():
    tmp = tempfile.mkdtemp()
    db.DB_PATH = os.path.join(tmp, "t.db")
    d = demo.make_demo_data(days=60)
    n = db.save_consumption(d, replace=True)
    assert n == len(d)
    c = db.table_counts()
    assert c["Fact_Consumption"] == len(d) and c["Dim_Object"] == 3
    back = db.load_consumption()
    assert len(back) == len(d)
    ok(f"БД: записано {n} рядків, таблиці {c}")
    gran = F.detect_granularity(back)
    assert gran == "h"
    series, info = F.build_series(back, ["Корпус А", "Корпус Б"], "h")
    series = series.iloc[-24 * 45:].reset_index(drop=True)
    res = []
    for name, make in M.model_registry().items():
        t0 = time.time()
        m, det = M.evaluate_forecaster(make, series, "h", info["has_temp"], 24)
        res.append({"model_name": name, **{k: m[k] for k in ("mae", "rmse", "mape")}})
        print(f"     {name:28s} MAE={m['mae']:.2f} RMSE={m['rmse']:.2f} MAPE={m['mape']:.1f}%  ({time.time()-t0:.1f} с)")
    ids = db.save_models(res, "2026-01-01 00:00:00")
    fc = M.final_forecast(M.model_registry()["Random Forest"], series, "h", info["has_temp"], 24)
    db.save_forecast("Сума: A+B", ids["Random Forest"], fc, 1)
    assert db.table_counts()["Fact_Forecast"] == 24
    assert len(db.load_forecasts()) == 24
    ok("моделі, оцінка на ковзних початках, запис прогнозу в БД")
    # денний режим
    sd, infod = F.build_series(back, ["Корпус А"], "D")
    m, _ = M.evaluate_forecaster(M.model_registry()["Random Forest"], sd, "D", infod["has_temp"], 7)
    ok(f"денний режим (7 діб): MAE={m['mae']:.2f}")
    # аномалії
    s1, i1 = F.build_series(back, ["Корпус А"], "h")
    r = residual_anomalies(s1, "h", i1["has_temp"])
    iso = isolation_forest_anomalies(s1, "h", i1["has_temp"])
    z = zscore_anomalies(sd)
    print(f"     аномалій: залишки={int(r.is_anomaly.sum())}, IF={int(iso.is_anomaly.sum())}, z-score(діб)={int(z.is_anomaly.sum())}")
    assert r.is_anomaly.sum() >= 1 and iso.is_anomaly.sum() >= 1
    n = db.set_anomaly_flags("Корпус А", s1["timestamp"].min(), s1["timestamp"].max(),
                             r.loc[r.is_anomaly, "timestamp"], "h")
    assert n == int(r.is_anomaly.sum())
    ok("три методи виявлення аномалій, запис позначок у БД")
    recs = build_recommendations(s1, "h", fc, int(r.is_anomaly.sum()))
    assert recs
    ok(f"рекомендації ({len(recs)})")


def test_etl():
    ts = pd.date_range("2026-01-01", periods=48, freq="h")
    wide = pd.DataFrame({"timestamp": ts, "Panther_office_A": np.arange(48.0), "Panther_lodging_B": np.arange(48.0) + 5})
    w = etl.normalize_columns(wide)
    assert etl.detect_format(w) == "wide"
    long = etl.wide_to_long(w, etl.wide_columns(w))
    weather = pd.DataFrame({"timestamp": ts, "site_id": "Panther", "airTemperature": 5.0})
    long = etl.attach_weather(long, weather)
    clean, rep = etl.validate_and_clean(long)
    assert rep["коректних рядків"] == 96 and clean["temperature"].notna().all()
    ok("ETL: широкий формат, погода, перевірка якості")


if __name__ == "__main__":
    import sys, types
    if not M.xgboost_available():           # імітація, лише щоб перевірити інтеграцію без бібліотеки xgboost
        from sklearn.ensemble import GradientBoostingRegressor
        stub = types.ModuleType("xgboost")
        stub.XGBRegressor = lambda **kw: GradientBoostingRegressor(n_estimators=60, random_state=28)
        sys.modules["xgboost"] = stub
        M.xgboost_available = lambda: True
        print("УВАГА: xgboost не встановлено, для перевірки використано заглушку")
    test_lstm_gradients()
    test_features_equal("h")
    test_features_equal("D")
    test_etl()
    test_db_and_models()
    print("\nУсі перевірки пройдено")
