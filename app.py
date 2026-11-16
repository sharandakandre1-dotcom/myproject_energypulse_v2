"""EnergyPulse: програмний засіб моніторингу та прогнозування енергоспоживання (прототип).
Запуск: streamlit run app.py"""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import anomalies as an
import auth
import db
import demo
import etl
import features as F
import models as M
import recommend as R

st.set_page_config(page_title="EnergyPulse", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp { background: #0b1220; color: #e6edf7; }
[data-testid="stSidebar"] { background: #0f1a2e; }
h1, h2, h3 { color: #f1f5fb; }
.card { background:#12203a; border:1px solid #1f3256; border-radius:12px; padding:14px 18px; margin-bottom:10px; }
.card b { color:#7cc4ff; }
.rec { background:#12203a; border-left:4px solid #f5b942; border-radius:8px; padding:10px 14px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")


# ---------- допоміжні функції сумісності з різними версіями Streamlit ----------
def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def show_fig(fig):
    fig.update_layout(**PLOT, margin=dict(l=10, r=10, t=40, b=10))
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def fmt(x, d=1):
    return f"{x:,.{d}f}".replace(",", " ")


@st.cache_data(show_spinner=False)
def cached_load(version):
    return db.load_consumption()


def ensure_demo():
    if db.table_counts()["Fact_Consumption"] == 0:
        db.save_consumption(demo.make_demo_data(), replace=True)


# ---------- автентифікація ----------
auth.ensure_default_users()
if "user" not in st.session_state:
    st.title("⚡ EnergyPulse")
    st.caption("Моніторинг та прогнозування енергоспоживання із застосуванням штучного інтелекту")
    with st.form("login"):
        u = st.text_input("Логін")
        p = st.text_input("Пароль", type="password")
        go_login = st.form_submit_button("Увійти")
    st.info("Демонстраційні облікові записи: admin / admin123 (адміністратор), manager / manager123 (енергоменеджер). "
            "Змінити пароль можна змінними середовища ENERGYPULSE_ADMIN_PASSWORD і ENERGYPULSE_MANAGER_PASSWORD "
            "до першого запуску.")
    if go_login:
        role = auth.verify(u, p)
        if role:
            st.session_state["user"], st.session_state["role"] = u.strip(), role
            st.rerun()
        else:
            st.error("Невірний логін або пароль.")
    st.stop()

role = st.session_state["role"]
ensure_demo()
data = cached_load(db.data_version())

# ---------- бічна панель ----------
PAGES = ["Огляд", "Моніторинг", "Аналітика", "Прогнозування", "Аномалії", "Звіти та рекомендації"]
if role == "admin":
    PAGES.append("Керування даними")

with st.sidebar:
    st.markdown("## ⚡ EnergyPulse")
    st.caption(f"{st.session_state['user']} · {auth.ROLES[role]}")
    page = st.radio("Розділ", PAGES, label_visibility="collapsed")
    if st.button("Вийти"):
        for k in ("user", "role", "forecast_run"):
            st.session_state.pop(k, None)
        st.rerun()
    st.divider()
    counts = db.table_counts()
    st.caption(f"База даних: SQLite (структура «зірка»)  \nЗаписів вимірювань: {fmt(counts['Fact_Consumption'], 0)}  \n"
               f"Об'єктів: {data['object_name'].nunique() if len(data) else 0}")

# ---------- сторінка керування даними (лише адміністратор) ----------
if page == "Керування даними":
    st.title("Керування даними")
    tab_imp, tab_obj, tab_usr, tab_db = st.tabs(["Імпорт даних", "Об'єкти", "Користувачі", "База даних"])

    with tab_imp:
        st.markdown("Файл CSV або Excel. Довгий формат: `timestamp`, `object_name`, `consumption` (кВт·год), "
                    "необов'язково `temperature`, `power_kw`. Підтримується також широкий формат "
                    "(час + по стовпцю на об'єкт, як у BDG2).")
        up = st.file_uploader("Файл вимірювань", type=["csv", "xlsx", "xls"])
        weather_up = st.file_uploader("Файл погоди (необов'язково)", type=["csv", "xlsx", "xls"])
        if up is not None:
            key = (up.name, up.size)
            if st.session_state.get("imp_key") != key:
                try:
                    st.session_state["imp_raw"] = etl.normalize_columns(etl.read_table(up.name, up.getvalue()))
                    st.session_state["imp_key"] = key
                except Exception as exc:
                    st.error(f"Не вдалося прочитати файл: {exc}")
                    st.stop()
            raw = st.session_state["imp_raw"]
            fmt_kind = etl.detect_format(raw)
            if fmt_kind == "unknown":
                st.error("Не вдалося визначити формат: потрібні стовпці timestamp та consumption (+ object_name) "
                         "або час і числові стовпці об'єктів.")
            else:
                long = raw
                if fmt_kind == "wide":
                    cols = etl.wide_columns(raw)
                    chosen = st.multiselect("Об'єкти з файлу (широкий формат; обмежте кількість для швидкості)",
                                            cols, default=cols[:5])
                    long = etl.wide_to_long(raw, chosen) if chosen else None
                if long is not None:
                    if weather_up is not None:
                        try:
                            wdf = etl.read_table(weather_up.name, weather_up.getvalue())
                            long = etl.attach_weather(long, wdf)
                        except Exception as exc:
                            st.warning(f"Погоду не додано: {exc}")
                    clean, rep = etl.validate_and_clean(long)
                    st.markdown("**Звіт про перевірку якості**")
                    show_df(pd.DataFrame({"Показник": list(rep.keys()), "Значення": list(rep.values())}))
                    st.markdown("**Попередній перегляд**")
                    show_df(clean.head(20))
                    mode = st.radio("Режим", ["Додати до наявних даних", "Замінити всі дані"], horizontal=True)
                    if st.button("Імпортувати", disabled=clean.empty):
                        with st.spinner("Запис у базу..."):
                            n = db.save_consumption(clean, replace=(mode == "Замінити всі дані"))
                        st.success(f"Збережено {fmt(n, 0)} записів.")
                        st.cache_data.clear()
                        st.rerun()

    with tab_obj:
        objs = db.load_objects()
        if objs.empty:
            st.info("Об'єктів немає.")
        else:
            edited = st.data_editor(objs, disabled=["object_id", "name"], hide_index=True, key="obj_editor")
            if st.button("Зберегти зміни об'єктів"):
                db.update_objects(edited)
                st.cache_data.clear()
                st.success("Збережено.")

    with tab_usr:
        show_df(pd.DataFrame(auth.list_users(), columns=["Логін", "Роль"]))
        with st.form("new_user"):
            nu = st.text_input("Новий логін")
            npw = st.text_input("Пароль (від 6 символів)", type="password")
            nr = st.selectbox("Роль", list(auth.ROLES.keys()), format_func=lambda r: auth.ROLES[r])
            if st.form_submit_button("Додати / оновити"):
                try:
                    auth.create_user(nu, npw, nr)
                    st.success("Користувача збережено.")
                except ValueError as exc:
                    st.error(str(exc))

    with tab_db:
        st.markdown("**Кількість записів у таблицях**")
        show_df(pd.DataFrame({"Таблиця": list(counts.keys()), "Записів": list(counts.values())}))
        st.download_button("Завантажити всі вимірювання (CSV)", data.to_csv(index=False).encode("utf-8-sig"),
                           "energy_consumption.csv", "text/csv")
        st.markdown("---")
        if st.checkbox("Підтверджую: відновити демонстраційні (синтетичні) дані, поточні дані буде видалено"):
            if st.button("Відновити демо-дані"):
                db.save_consumption(demo.make_demo_data(), replace=True)
                st.cache_data.clear()
                st.session_state.pop("forecast_run", None)
                st.rerun()
    st.stop()

# ---------- спільні фільтри ----------
if data.empty:
    st.warning("У базі немає даних. Адміністратор може імпортувати файл у розділі «Керування даними».")
    st.stop()

all_objects = sorted(data["object_name"].unique())
dmin, dmax = data["timestamp"].min().date(), data["timestamp"].max().date()
f1, f2 = st.columns([2, 2])
sel_objects = f1.multiselect("Об'єкти", all_objects, default=all_objects)
default_from = max(dmin, dmax - dt.timedelta(days=120))
period = f2.date_input("Період", value=(default_from, dmax), min_value=dmin, max_value=dmax)
if isinstance(period, (list, tuple)):
    d_from, d_to = (period[0], period[-1]) if len(period) else (dmin, dmax)
else:
    d_from = d_to = period
if not sel_objects:
    st.warning("Оберіть хоча б один об'єкт.")
    st.stop()

mask = (data["object_name"].isin(sel_objects) & (data["timestamp"].dt.date >= d_from) & (data["timestamp"].dt.date <= d_to))
flt = data[mask].copy()
if flt.empty:
    st.warning("Для вибраних фільтрів даних немає.")
    st.stop()

gran_data = F.detect_granularity(data)
flt["date"] = flt["timestamp"].dt.floor("D")
daily = flt.groupby("date", as_index=False)["consumption"].sum()
unit_peak = "година" if gran_data == "h" else "доба"


def kpis():
    tot = flt["consumption"].sum()
    avg_day = daily["consumption"].mean()
    if gran_data == "h":
        peak_series = flt.groupby("timestamp")["consumption"].sum()
    else:
        peak_series = daily.set_index("date")["consumption"]
    return {"Сумарне споживання, кВт·год": fmt(tot, 0),
            "Середнє за добу, кВт·год": fmt(avg_day, 1),
            f"Пік ({unit_peak}), кВт·год": f"{fmt(peak_series.max(), 1)} ({peak_series.idxmax():%d.%m %H:%M})",
            "Остання доба, кВт·год": fmt(daily['consumption'].iloc[-1], 1)}


# ---------- Огляд ----------
if page == "Огляд":
    st.title("Огляд")
    k = kpis()
    cols = st.columns(4)
    for c, (name, val) in zip(cols, k.items()):
        c.markdown(f"<div class='card'><b>{name}</b><br><span style='font-size:1.4rem'>{val}</span></div>", unsafe_allow_html=True)
    fig = px.area(daily, x="date", y="consumption", title="Динаміка добового споживання, кВт·год",
                  labels={"date": "Дата", "consumption": "кВт·год"})
    show_fig(fig)
    by_obj = flt.groupby("object_name", as_index=False)["consumption"].sum()
    show_fig(px.bar(by_obj, x="object_name", y="consumption", title="Споживання за об'єктами, кВт·год",
                    labels={"object_name": "Об'єкт", "consumption": "кВт·год"}))
    st.markdown("**Останні вимірювання**")
    show_df(flt.sort_values("timestamp", ascending=False).head(15)[["timestamp", "object_name", "consumption", "temperature", "is_anomaly"]])

# ---------- Моніторинг ----------
elif page == "Моніторинг":
    st.title("Моніторинг")
    t1, t2 = st.tabs(["Графік", "Таблиця"])
    with t1:
        step = st.radio("Крок", ["Година", "Доба"] if gran_data == "h" else ["Доба"], horizontal=True)
        d = flt.copy()
        d["x"] = d["timestamp"].dt.floor("h" if step == "Година" else "D")
        g = d.groupby(["x", "object_name"], as_index=False)["consumption"].sum()
        fig = px.line(g, x="x", y="consumption", color="object_name", labels={"x": "Час", "consumption": "кВт·год", "object_name": "Об'єкт"})
        show_fig(fig)
    with t2:
        show_df(flt.drop(columns=["date"]).sort_values("timestamp", ascending=False))
        st.download_button("Експорт CSV", flt.drop(columns=["date"]).to_csv(index=False).encode("utf-8-sig"), "monitoring.csv", "text/csv")

# ---------- Аналітика ----------
elif page == "Аналітика":
    st.title("Аналітика")
    c1, c2 = st.columns(2)
    wk = daily.set_index("date")["consumption"].resample("W").sum().reset_index()
    with c1:
        show_fig(px.bar(wk, x="date", y="consumption", title="Тижневі підсумки, кВт·год", labels={"date": "Тиждень", "consumption": "кВт·год"}))
    mo = daily.set_index("date")["consumption"].resample("MS").sum().reset_index()
    with c2:
        show_fig(px.line(mo, x="date", y="consumption", markers=True, title="Місячні підсумки, кВт·год", labels={"date": "Місяць", "consumption": "кВт·год"}))
    if gran_data == "h":
        hp = flt.assign(hour=flt["timestamp"].dt.hour).groupby("hour", as_index=False)["consumption"].mean()
        show_fig(px.bar(hp, x="hour", y="consumption", title="Середній добовий профіль (піки навантаження за годинами)",
                        labels={"hour": "Година", "consumption": "кВт·год (середнє)"}))
    wd = flt.assign(dow=flt["timestamp"].dt.dayofweek).groupby("dow", as_index=False)["consumption"].mean()
    wd["День"] = wd["dow"].map(dict(enumerate(["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"])))
    show_fig(px.bar(wd, x="День", y="consumption", title="Середнє споживання за днями тижня", labels={"consumption": "кВт·год"}))
    show_fig(px.histogram(daily, x="consumption", nbins=30, title="Розподіл добового споживання", labels={"consumption": "кВт·год"}))
    cmp_ = flt.groupby("object_name").agg(Сума=("consumption", "sum"), Середнє=("consumption", "mean"),
                                          Максимум=("consumption", "max"), Вимірювань=("consumption", "size")).reset_index()
    show_df(cmp_.rename(columns={"object_name": "Об'єкт"}).round(1))

# ---------- Прогнозування ----------
elif page == "Прогнозування":
    st.title("Прогнозування")
    reg = M.model_registry()
    options = ["Погодинна (до 24 год)", "Добова (7-30 діб)"] if gran_data == "h" else ["Добова (7-30 діб)"]
    gsel = st.radio("Режим прогнозу", options, horizontal=True)
    gran = "h" if gsel.startswith("Погодинна") else "D"
    horizon = st.slider("Горизонт, годин" if gran == "h" else "Горизонт, діб",
                        *((1, 24, 24) if gran == "h" else (7, 30, 14)))
    chosen = st.multiselect("Моделі", list(reg.keys()), default=list(reg.keys()))
    if not M.xgboost_available():
        st.warning("Бібліотеку xgboost не встановлено: модель XGBoost недоступна (pip install xgboost).")
    if st.button("Побудувати прогноз") and chosen:
        try:
            series, info = F.build_series(flt.drop(columns=["date"]), sel_objects, gran)
            use_temp = info["has_temp"]
            if info["truncated"]:
                st.info("У ряді є великі прогалини, використано останній безперервний фрагмент.")
            M.split_sizes(len(series), horizon, gran)
            results, fcs, details = [], {}, {}
            prog = st.progress(0.0, text="Навчання та оцінка моделей...")
            for i, name in enumerate(chosen):
                prog.progress(i / len(chosen), text=f"Модель: {name}")
                met, det = M.evaluate_forecaster(reg[name], series, gran, use_temp, horizon)
                fcs[name] = M.final_forecast(reg[name], series, gran, use_temp, horizon)
                details[name] = det
                results.append({"model_name": name, "mae": met["mae"], "rmse": met["rmse"], "mape": met["mape"],
                                "origins": met["n_origins"]})
            prog.progress(1.0, text="Готово")
            res = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
            main = res.iloc[0]["model_name"]
            trained_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ids = db.save_models(results, trained_at)
            scope = sel_objects[0] if len(sel_objects) == 1 else "Сума: " + ", ".join(sorted(sel_objects))[:180]
            for name in chosen:
                db.save_forecast(scope, ids[name], fcs[name], 1 if gran == "h" else 24)
            st.session_state["forecast_run"] = {"res": res, "main": main, "fcs": fcs, "series": series, "gran": gran,
                                                "horizon": horizon, "scope": scope, "trained_at": trained_at}
        except ValueError as exc:
            st.error(str(exc))

    run = st.session_state.get("forecast_run")
    if run:
        res, main, fcs, series = run["res"], run["main"], run["fcs"], run["series"]
        st.success(f"Основна модель за найменшим RMSE: {main}. Прогноз збережено в таблицях Dim_Model та Fact_Forecast.")
        cc = st.columns(3)
        best = res.iloc[0]
        for c, (n, v) in zip(cc, [("MAE", best["mae"]), ("RMSE", best["rmse"]), ("MAPE, %", best["mape"])]):
            c.markdown(f"<div class='card'><b>{n} ({main})</b><br><span style='font-size:1.4rem'>{v:.2f}</span></div>", unsafe_allow_html=True)
        tab = res.copy()
        tab["Основна"] = np.where(tab["model_name"] == main, "✔", "")
        show_df(tab.rename(columns={"model_name": "Модель", "mae": "MAE", "rmse": "RMSE", "mape": "MAPE, %", "origins": "Початків прогнозу"}).round(2))
        st.caption(f"Метрики розраховано для багатокрокового прогнозу на горизонт {run['horizon']} "
                   f"{'год' if run['gran'] == 'h' else 'діб'} (хронологічне розбиття 80/20, ковзні початки прогнозу на тестовій частині).")
        tail = series.iloc[-(24 * 7 if run["gran"] == "h" else 60):]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tail["timestamp"], y=tail["consumption"], name="Факт", line=dict(color="#e6edf7")))
        for name, fc in fcs.items():
            fig.add_trace(go.Scatter(x=fc["timestamp"], y=fc["forecast"], name=name, line=dict(dash="dash", width=3 if name == main else 1.5)))
        fig.update_layout(title=f"Факт і прогноз: {run['scope']}", yaxis_title="кВт·год")
        show_fig(fig)
        out = fcs[main].rename(columns={"timestamp": "Час", "forecast": "Прогноз, кВт·год"}).round({"Прогноз, кВт·год": 2})
        show_df(out)
        st.download_button("Експорт прогнозу (CSV)", out.to_csv(index=False).encode("utf-8-sig"), "forecast.csv", "text/csv")
    with st.expander("Історія прогнозів у базі даних"):
        hist = db.load_forecasts(300)
        if hist.empty:
            st.caption("Збережених прогнозів ще немає.")
        else:
            show_df(hist)
            show_df(db.load_models())

# ---------- Аномалії ----------
elif page == "Аномалії":
    st.title("Нетипові відхилення")
    methods = ["z-оцінка (добові суми)", "Залишки прогнозу", "Isolation Forest"]
    method = st.radio("Метод", methods, horizontal=True)
    try:
        agg_gran = "D" if method == methods[0] else gran_data
        series, info = F.build_series(flt.drop(columns=["date"]), sel_objects, agg_gran)
        use_temp = info["has_temp"]
        if method == methods[0]:
            thr = st.slider("Поріг |z|", 1.5, 4.0, 2.5, 0.1)
            res = an.zscore_anomalies(series, thr)
            note = f"поріг |z| ≥ {thr}"
        elif method == methods[1]:
            kk = st.slider("Коефіцієнт k у δ = k·σ", 2.0, 5.0, 3.0, 0.5)
            res = an.residual_anomalies(series, agg_gran, use_temp, kk)
            note = f"δ = {res.attrs['delta']:.2f} кВт·год"
        else:
            cont = st.slider("Частка аномалій (contamination)", 0.005, 0.05, 0.01, 0.005)
            res = an.isolation_forest_anomalies(series, agg_gran, use_temp, cont)
            note = f"очікувана частка {cont:.3f}"
        flagged = res[res["is_anomaly"]]
        cc = st.columns(3)
        cc[0].markdown(f"<div class='card'><b>Перевірено точок</b><br><span style='font-size:1.4rem'>{len(res)}</span></div>", unsafe_allow_html=True)
        cc[1].markdown(f"<div class='card'><b>Відхилень</b><br><span style='font-size:1.4rem'>{len(flagged)}</span></div>", unsafe_allow_html=True)
        cc[2].markdown(f"<div class='card'><b>Параметр</b><br><span style='font-size:1.1rem'>{note}</span></div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["timestamp"], y=res["consumption"], name="Споживання", line=dict(color="#7cc4ff")))
        fig.add_trace(go.Scatter(x=flagged["timestamp"], y=flagged["consumption"], mode="markers", name="Відхилення",
                                 marker=dict(color="#ff5c5c", size=9)))
        fig.update_layout(title="Споживання та позначені відхилення", yaxis_title="кВт·год")
        show_fig(fig)
        show_df(flagged.sort_values("timestamp", ascending=False).round({"consumption": 2, "predicted": 2, "score": 2}))
        st.caption("Статистичне відхилення не завжди означає несправність: остаточне рішення приймає енергоменеджер.")
        if len(sel_objects) == 1:
            if st.button("Зберегти позначки у базі (Fact_Consumption.is_anomaly)"):
                n = db.set_anomaly_flags(sel_objects[0], res["timestamp"].min(), res["timestamp"].max(),
                                         flagged["timestamp"], agg_gran)
                st.cache_data.clear()
                st.success(f"Оновлено записів: {n}.")
        else:
            st.caption("Щоб записати позначки в базу, оберіть один об'єкт.")
    except ValueError as exc:
        st.error(str(exc))

# ---------- Звіти та рекомендації ----------
elif page == "Звіти та рекомендації":
    st.title("Звіти та рекомендації")
    try:
        series, info = F.build_series(flt.drop(columns=["date"]), sel_objects, gran_data)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    run = st.session_state.get("forecast_run")
    fc = run["fcs"][run["main"]] if run and run["gran"] == gran_data else None
    n_anom = int(flt["is_anomaly"].sum())
    recs = R.build_recommendations(series, gran_data, fc, n_anom)
    if fc is None:
        st.caption("Щоб рекомендації враховували прогноз, спочатку побудуйте його у розділі «Прогнозування».")
    for t, d in recs:
        st.markdown(f"<div class='rec'><b>{t}.</b> {d}</div>", unsafe_allow_html=True)
    metrics = run["res"] if run else None
    title = "Звіт EnergyPulse"
    per = f"{d_from:%d.%m.%Y} - {d_to:%d.%m.%Y}"
    k = kpis()
    st.download_button("Завантажити звіт (HTML)", R.report_html(title, per, sel_objects, k, recs, metrics).encode("utf-8"),
                       "energypulse_report.html", "text/html")
    st.download_button("Завантажити звіт (TXT)", R.report_text(title, per, sel_objects, k, recs, metrics).encode("utf-8"),
                       "energypulse_report.txt", "text/plain")
