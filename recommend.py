"""Правила формування рекомендацій з оптимізації споживання та текстовий звіт.
Порогові значення евристичні й задаються константами нижче."""
import html

import numpy as np
import pandas as pd

PEAK_RATIO = 1.3        # піковий час перевищує середнє більш ніж у 1,3 раза
NIGHT_SHARE = 0.6       # нічне навантаження більше 60 % від середнього
WEEKEND_SHARE = 0.85    # вихідні майже як будні
TREND_PCT = 5.0         # зміна за 30 діб, відсотки


def build_recommendations(series, gran, forecast=None, anomaly_count=0):
    recs = []
    s = series.copy()
    s["hour"] = s["timestamp"].dt.hour
    s["dow"] = s["timestamp"].dt.dayofweek
    mean_all = s["consumption"].mean()

    if gran == "h" and len(s) >= 48:
        prof = s.groupby("hour")["consumption"].mean()
        ph = int(prof.idxmax())
        ratio = prof.max() / prof.mean()
        if ratio >= PEAK_RATIO:
            recs.append(("Пікове навантаження",
                         f"Найвище середнє споживання припадає на {ph:02d}:00 (у {ratio:.2f} раза більше за середнє "
                         "по годинах). Варто перевірити можливість перенесення частини навантаження на менш завантажені години."))
        night = prof.loc[[h for h in range(0, 6) if h in prof.index]].mean()
        if night / prof.mean() >= NIGHT_SHARE:
            recs.append(("Нічне фонове навантаження",
                         f"У нічні години (00:00-05:59) споживання становить {night / prof.mean() * 100:.0f} % від середнього. "
                         "Варто перевірити обладнання, що залишається ввімкненим уночі."))
    if len(s) >= 14:
        wk = s.loc[s["dow"] < 5, "consumption"].mean()
        we = s.loc[s["dow"] >= 5, "consumption"].mean()
        if wk > 0 and we / wk >= WEEKEND_SHARE:
            recs.append(("Вихідні дні",
                         f"У вихідні споживання становить {we / wk * 100:.0f} % від будніх днів. "
                         "Варто перевірити, чи потрібне повне навантаження у ці дні."))
    per_day = 24 if gran == "h" else 1
    if len(s) >= 60 * per_day:
        last = s["consumption"].iloc[-30 * per_day:].mean()
        prev = s["consumption"].iloc[-60 * per_day:-30 * per_day].mean()
        if prev > 0:
            ch = (last / prev - 1) * 100
            if ch >= TREND_PCT:
                recs.append(("Зростання споживання", f"За останні 30 діб споживання зросло на {ch:.1f} % порівняно з попередніми 30 добами."))
            elif ch <= -TREND_PCT:
                recs.append(("Зменшення споживання", f"За останні 30 діб споживання зменшилось на {abs(ch):.1f} % порівняно з попередніми 30 добами."))
    if forecast is not None and len(forecast):
        hist_q95 = float(np.percentile(s["consumption"], 95))
        fmax = float(forecast["forecast"].max())
        when = forecast.loc[forecast["forecast"].idxmax(), "timestamp"]
        if fmax >= hist_q95:
            recs.append(("Прогноз піку",
                         f"Прогнозований максимум {fmax:,.1f} кВт·год ({pd.Timestamp(when):%d.%m.%Y %H:%M}) не нижчий за "
                         f"95-й процентиль історії ({hist_q95:,.1f}). Варто заздалегідь підготуватися до підвищеного навантаження.".replace(",", " ")))
    if anomaly_count > 0:
        recs.append(("Нетипові відхилення",
                     f"За вибраний період позначено {anomaly_count} нетипових значень. Їх потрібно перевірити: причиною "
                     "можуть бути збій вимірювання, несправність обладнання або позаплановий режим роботи."))
    if not recs:
        recs.append(("Загальна оцінка", "За вибраний період суттєвих відхилень чи помітних можливостей оптимізації за "
                                         "заданими правилами не виявлено."))
    return recs


def report_text(title, period, objects, kpis, recs, metrics=None):
    lines = [title, "=" * len(title), f"Період: {period}", f"Об'єкти: {', '.join(objects)}", "", "Ключові показники:"]
    lines += [f"  - {k}: {v}" for k, v in kpis.items()]
    if metrics is not None and len(metrics):
        lines += ["", "Якість моделей (MAE, RMSE, MAPE):"]
        lines += [f"  - {r.model_name}: MAE={r.mae:.2f}, RMSE={r.rmse:.2f}, MAPE={r.mape:.1f} %" for r in metrics.itertuples()]
    lines += ["", "Рекомендації:"]
    lines += [f"  {i}. {t}. {d}" for i, (t, d) in enumerate(recs, 1)]
    return "\n".join(lines)


def report_html(title, period, objects, kpis, recs, metrics=None):
    e = html.escape
    parts = [f"<html><head><meta charset='utf-8'><title>{e(title)}</title>"
             "<style>body{font-family:Arial,sans-serif;max-width:860px;margin:30px auto;line-height:1.5}"
             "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px 10px}</style></head><body>",
             f"<h1>{e(title)}</h1><p><b>Період:</b> {e(period)}<br><b>Об'єкти:</b> {e(', '.join(objects))}</p>",
             "<h2>Ключові показники</h2><table>"]
    parts += [f"<tr><td>{e(str(k))}</td><td>{e(str(v))}</td></tr>" for k, v in kpis.items()]
    parts.append("</table>")
    if metrics is not None and len(metrics):
        parts.append("<h2>Якість моделей</h2><table><tr><th>Модель</th><th>MAE</th><th>RMSE</th><th>MAPE, %</th></tr>")
        parts += [f"<tr><td>{e(r.model_name)}</td><td>{r.mae:.2f}</td><td>{r.rmse:.2f}</td><td>{r.mape:.1f}</td></tr>"
                  for r in metrics.itertuples()]
        parts.append("</table>")
    parts.append("<h2>Рекомендації</h2><ol>")
    parts += [f"<li><b>{e(t)}.</b> {e(d)}</li>" for t, d in recs]
    parts.append("</ol></body></html>")
    return "".join(parts)
