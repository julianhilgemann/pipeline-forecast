from __future__ import annotations

import numpy as np
import pandas as pd


def draw_deal_size(segment: str, rng: np.random.Generator) -> float:
    params = {
        "SMB": (8.9, 0.55),
        "MidMarket": (10.1, 0.60),
        "Enterprise": (11.2, 0.70),
    }
    mu, sigma = params[segment]
    return float(np.exp(rng.normal(mu, sigma)))


def simulate_opportunities(
    cal: pd.DataFrame, regime_switch_date: pd.Timestamp, rng: np.random.Generator
) -> pd.DataFrame:
    seg_values = np.array(["SMB", "MidMarket", "Enterprise"])
    seg_probs = np.array([0.60, 0.30, 0.10])
    ch_values = np.array(["Inbound", "Outbound", "Partner"])
    ch_probs = np.array([0.48, 0.37, 0.15])
    dow_mult = {0: 1.25, 1: 1.15, 2: 1.00, 3: 0.95, 4: 0.80, 5: 0.45, 6: 0.35}

    rows: list[dict] = []
    opp_id = 1
    for r in cal.itertuples(index=False):
        date_ts = pd.Timestamp(r.date)
        base = 12.0 if r.is_business_day else 2.2
        lam = base * dow_mult[int(r.dow)]
        if r.is_holiday:
            lam *= 0.25
        if date_ts >= regime_switch_date:
            lam *= 1.20
        lam = max(lam, 0.05)
        n = int(rng.poisson(lam=lam))
        if n == 0:
            continue

        segs = rng.choice(seg_values, size=n, p=seg_probs)
        chs = rng.choice(ch_values, size=n, p=ch_probs)
        for i in range(n):
            segment = str(segs[i])
            rows.append(
                {
                    "opp_id": opp_id,
                    "created_date": r.date,
                    "created_biz_day_index": int(r.biz_day_index),
                    "segment": segment,
                    "channel": str(chs[i]),
                    "deal_size": draw_deal_size(segment, rng),
                }
            )
            opp_id += 1
    opp = pd.DataFrame(rows)
    opp["deal_size"] = opp["deal_size"].round(2)
    return opp


def _clip_prob(v: float, low: float = 0.01, high: float = 0.99) -> float:
    return float(min(high, max(low, v)))


def simulate_events(
    opp: pd.DataFrame, cal: pd.DataFrame, regime_switch_biz_idx: int, rng: np.random.Generator
) -> pd.DataFrame:
    biz_map = (
        cal[cal["is_business_day"]]
        .drop_duplicates("biz_day_index")
        .set_index("biz_day_index")["date"]
        .to_dict()
    )
    max_biz_idx = int(cal["biz_day_index"].max())

    seg_close_adj = {"SMB": 0.03, "MidMarket": 0.00, "Enterprise": -0.06}
    ch_close_adj = {"Inbound": 0.04, "Outbound": -0.03, "Partner": 0.01}
    seg_win_adj = {"SMB": -0.02, "MidMarket": 0.03, "Enterprise": 0.08}
    ch_win_adj = {"Inbound": 0.06, "Outbound": -0.04, "Partner": 0.02}
    seg_age_shift = {"SMB": 0.0, "MidMarket": 3.0, "Enterprise": 8.0}
    ch_age_shift = {"Inbound": -1.0, "Outbound": 2.0, "Partner": 1.0}

    rows: list[dict] = []
    for r in opp.itertuples(index=False):
        created_idx = int(r.created_biz_day_index)
        post_regime = created_idx >= regime_switch_biz_idx

        p_close = 0.86 + seg_close_adj[r.segment] + ch_close_adj[r.channel]
        if post_regime:
            p_close -= 0.08
        p_close = _clip_prob(p_close, 0.30, 0.98)

        if rng.random() > p_close:
            rows.append(
                {
                    "opp_id": int(r.opp_id),
                    "outcome": "censored",
                    "closed_date": None,
                    "closed_biz_day_index": pd.NA,
                    "time_to_close_biz_days": pd.NA,
                }
            )
            continue

        raw_ttc = rng.gamma(shape=2.4, scale=6.0)
        ttc = int(round(raw_ttc + seg_age_shift[r.segment] + ch_age_shift[r.channel]))
        ttc = max(1, ttc)
        if post_regime:
            ttc = int(round(ttc * 1.35 + rng.integers(1, 5)))
            ttc = max(1, ttc)

        closed_idx = created_idx + ttc
        if closed_idx > max_biz_idx or closed_idx not in biz_map:
            rows.append(
                {
                    "opp_id": int(r.opp_id),
                    "outcome": "censored",
                    "closed_date": None,
                    "closed_biz_day_index": pd.NA,
                    "time_to_close_biz_days": pd.NA,
                }
            )
            continue

        p_win = 0.29 + seg_win_adj[r.segment] + ch_win_adj[r.channel] + (0.0016 * ttc)
        if post_regime:
            p_win -= 0.12
        p_win = _clip_prob(p_win, 0.05, 0.90)

        outcome = "won" if rng.random() < p_win else "lost"
        rows.append(
            {
                "opp_id": int(r.opp_id),
                "outcome": outcome,
                "closed_date": biz_map[closed_idx],
                "closed_biz_day_index": int(closed_idx),
                "time_to_close_biz_days": int(ttc),
            }
        )

    ev = pd.DataFrame(rows)
    ev["closed_biz_day_index"] = ev["closed_biz_day_index"].astype("Int64")
    ev["time_to_close_biz_days"] = ev["time_to_close_biz_days"].astype("Int64")
    return ev


def build_scd_status(opp: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    merged = opp.merge(ev, on="opp_id", how="left")
    rows: list[dict] = []
    for r in merged.itertuples(index=False):
        rows.append(
            {
                "opp_id": int(r.opp_id),
                "valid_from_date": r.created_date,
                "valid_to_date": r.closed_date if r.outcome in ("won", "lost") else None,
                "status": "active",
            }
        )
        if r.outcome in ("won", "lost"):
            rows.append(
                {
                    "opp_id": int(r.opp_id),
                    "valid_from_date": r.closed_date,
                    "valid_to_date": None,
                    "status": r.outcome,
                }
            )
    return pd.DataFrame(rows)
