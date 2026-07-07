"""
GA4 Enterprise Agent Architecture -- BigQuery & Vertex AI Pipeline
Steps 1-6: Fetch -> Enrich -> Classify -> Clean -> Summarize -> Meridian MMM
"""
import argparse
import json
import os
import random
import string
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

GCP_PROJECT_ID    = os.getenv("GCP_PROJECT_ID", "YOUR_GCP_PROJECT_ID")
BQ_PUBLIC_DATASET = "bigquery-public-data.ga4_obfuscated_sample_ecommerce"
BQ_DATE_START     = "20201101"
BQ_DATE_END       = "20210131"
VERTEX_AI_ENDPOINT = "YOUR_VERTEX_AI_ENDPOINT_ID"
VERTEX_AI_REGION   = "us-central1"
OUTPUT_DIR  = "data"
RAW_CSV     = os.path.join(OUTPUT_DIR, "raw_ga4_events.csv")
CLEANED_CSV = os.path.join(OUTPUT_DIR, "cleaned_ga4_events.csv")
N_SESSIONS  = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

GA4_SESSION_QUERY = """
WITH sessions AS (
  SELECT
    user_pseudo_id AS client_id,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS session_id,
    MIN(event_timestamp) AS session_start_us,
    MAX(event_timestamp) AS session_end_us,
    TIMESTAMP_DIFF(
      TIMESTAMP_MICROS(MAX(event_timestamp)),
      TIMESTAMP_MICROS(MIN(event_timestamp)),
      SECOND
    ) AS session_duration_sec,
    COUNT(*) AS total_events,
    COUNTIF(event_name IN ('click', 'user_engagement')) AS click_events,
    COUNTIF(event_name = 'scroll') AS scroll_events,
    MAX(COALESCE(
      (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'percent_scrolled'),
      0
    )) AS max_scroll_pct,
    ANY_VALUE(CONCAT(
      COALESCE(traffic_source.source, '(direct)'), ' / ',
      COALESCE(traffic_source.medium, '(none)')
    )) AS source_medium,
    ANY_VALUE(COALESCE(geo.country, 'Unknown')) AS geo_country,
    ANY_VALUE(device.category) AS device_category,
    ANY_VALUE(COALESCE(
      (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location'),
      '/'
    )) AS landing_page
  FROM `{dataset}.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '{date_start}' AND '{date_end}'
    AND event_name IN (
      'page_view','session_start','scroll','click',
      'user_engagement','purchase','add_to_cart','view_item'
    )
  GROUP BY 1, 2
)
SELECT
  client_id,
  FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M:%S', TIMESTAMP_MICROS(session_start_us)) AS event_timestamp,
  GREATEST(session_duration_sec, 0) AS session_duration_sec,
  total_events,
  click_events,
  scroll_events,
  COALESCE(max_scroll_pct, 0) AS page_scroll_depth_pct,
  ROUND(SAFE_DIVIDE(total_events, GREATEST(session_duration_sec, 1)), 2) AS event_velocity_per_sec,
  source_medium,
  geo_country,
  COALESCE(device_category, 'desktop') AS device_category,
  landing_page
FROM sessions
WHERE session_id IS NOT NULL
ORDER BY RAND()
LIMIT {n_sessions}
"""


def fetch_ga4_public_data(n_sessions=N_SESSIONS):
    try:
        from google.cloud import bigquery
    except ImportError:
        print("\n[ERROR] google-cloud-bigquery not installed.")
        print("        Run: pip install google-cloud-bigquery db-dtypes")
        print("        Or use --mock for offline mode.\n")
        sys.exit(1)
    if GCP_PROJECT_ID == "YOUR_GCP_PROJECT_ID":
        print("\n[ERROR] GCP_PROJECT_ID is not set.")
        print("        Set it in the script or: export GCP_PROJECT_ID=my-project")
        print("        Or run: python scripts/bq_vertex_pipeline.py --mock\n")
        sys.exit(1)
    print(f"\n[Step 1] Querying GA4 public dataset on BigQuery...")
    print(f"         Dataset : {BQ_PUBLIC_DATASET}.events_*")
    print(f"         Dates   : {BQ_DATE_START} to {BQ_DATE_END}")
    print(f"         Project : {GCP_PROJECT_ID}")
    client = bigquery.Client(project=GCP_PROJECT_ID)
    query  = GA4_SESSION_QUERY.format(
        dataset=BQ_PUBLIC_DATASET, date_start=BQ_DATE_START,
        date_end=BQ_DATE_END, n_sessions=n_sessions,
    )
    df = client.query(query).to_dataframe()
    if df.empty:
        print("[ERROR] Query returned 0 rows.")
        sys.exit(1)
    print(f"         Fetched {len(df):,} real GA4 sessions")
    df["session_duration_sec"]   = df["session_duration_sec"].fillna(0).astype(int)
    df["click_events"]           = df["click_events"].fillna(0).astype(int)
    df["scroll_events"]          = df["scroll_events"].fillna(0).astype(int)
    df["page_scroll_depth_pct"]  = df["page_scroll_depth_pct"].fillna(0).astype(int)
    df["event_velocity_per_sec"] = df["event_velocity_per_sec"].fillna(1.0).astype(float)
    top_sources = df["source_medium"].value_counts().head(5)
    print(f"\n         Top sources:")
    for src, cnt in top_sources.items():
        print(f"           {src:<35} {cnt:>5} sessions")
    return df


def generate_mock_ga4_data(n=N_SESSIONS):
    """[MOCK] Generates n synthetic GA4 session rows.
    Traffic split: 55% Human, 25% LLM_Scraper, 20% Ad_Fraud"""
    print(f"\n[Step 1] Generating {n:,} synthetic GA4 sessions (offline mock mode)...")
    traffic_types = np.random.choice(
        ["Human", "LLM_Scraper", "Ad_Fraud"], size=n, p=[0.55, 0.25, 0.20]
    )
    base_time = datetime(2025, 1, 6, 8, 0, 0)
    sources   = ["google / cpc","meta / paid_social","direct / (none)","bing / cpc","programmatic / display"]
    countries = ["US","GB","CA","AU","DE","BR","IN","FR","NL","SG"]
    pages     = ["/","/pricing","/about","/blog/ai-detection","/contact","/demo"]
    devices   = ["desktop","mobile","tablet"]
    rows    = []
    elapsed = 0
    for t_type in traffic_types:
        is_human   = (t_type == "Human")
        is_scraper = (t_type == "LLM_Scraper")
        session_duration = (
            random.randint(45,600) if is_human else
            random.randint(0,6)    if is_scraper else
            random.randint(0,3)
        )
        click_events  = random.randint(1,12) if is_human else 0
        scroll_events = random.randint(2,15) if is_human else 0
        scroll_depth  = random.randint(35,95) if is_human else random.randint(0,8)
        total_events  = (
            random.randint(4,20) if is_human else
            random.randint(1,3)  if is_scraper else
            random.randint(1,2)
        )
        velocity = round(total_events / max(session_duration,1), 2)
        elapsed += random.uniform(0.3, 6.0)
        rows.append({
            "client_id"             : "".join(random.choices(string.ascii_uppercase+string.digits, k=12)),
            "event_timestamp"       : (base_time + timedelta(seconds=elapsed)).isoformat(),
            "session_duration_sec"  : session_duration,
            "total_events"          : total_events,
            "click_events"          : click_events,
            "scroll_events"         : scroll_events,
            "page_scroll_depth_pct" : scroll_depth,
            "event_velocity_per_sec": velocity,
            "source_medium"         : random.choice(sources),
            "geo_country"           : random.choice(countries),
            "device_category"       : random.choice(devices),
            "landing_page"          : random.choice(pages),
            "raw_traffic_label"     : t_type,
        })
    df   = pd.DataFrame(rows)
    dist = dict(pd.Series(traffic_types).value_counts())
    print(f"         Distribution: {dist}")
    return df


def enrich_with_edge_signals(df, is_real_data=True):
    """Adds edge_score and mouse_move_events.
    Both are [SIMULATED] -- not part of the GA4 BigQuery schema."""
    if not is_real_data:
        def _edge(row):
            if row["raw_traffic_label"] == "Human":
                return round(float(np.random.beta(8,2)),3)
            elif row["raw_traffic_label"] == "LLM_Scraper":
                return round(float(np.random.beta(2,6)),3)
            else:
                return round(float(np.random.beta(1,9)),3)
        df["edge_score"] = df.apply(_edge, axis=1)
        df["mouse_move_events"] = df.apply(
            lambda r: random.randint(20,250) if r["raw_traffic_label"]=="Human" else 0, axis=1
        )
        return df

    print(f"\n[Step 2] Enriching {len(df):,} real sessions with simulated edge signals...")
    print("         edge_score        -- [SIMULATED] Cloud Armor not in GA4 schema")
    print("         mouse_move_events -- [SIMULATED] client-side JS not in GA4 schema")
    edge_scores = []
    mouse_list  = []
    for _, row in df.iterrows():
        duration = row["session_duration_sec"]
        velocity = row["event_velocity_per_sec"]
        source   = str(row.get("source_medium","")).lower()
        if duration >= 30 and velocity <= 5 and row["click_events"] > 0:
            score = round(float(np.random.beta(8,2)),3)
        elif duration <= 2 or velocity > 15:
            if "programmatic" in source or "display" in source:
                score = round(float(np.random.beta(1,8)),3)
            else:
                score = round(float(np.random.beta(2,6)),3)
        else:
            score = round(float(np.random.beta(5,4)),3)
        if row["click_events"] > 0 or row["scroll_events"] > 0:
            lo    = max(5, int(duration * 0.3))
            hi    = max(lo, min(300, int(duration * 0.8)))
            mouse = random.randint(lo, hi)
        else:
            mouse = 0
        edge_scores.append(score)
        mouse_list.append(mouse)
    df = df.copy()
    df["edge_score"]               = edge_scores
    df["mouse_move_events"]        = mouse_list
    df["edge_score_source"]        = "SIMULATED"
    df["mouse_move_events_source"] = "SIMULATED"
    print(f"         Score range: {df['edge_score'].min():.3f} to {df['edge_score'].max():.3f}")
    return df


def vertex_ai_clustering(df, has_ground_truth=False):
    """Rule-based classifier mirroring Vertex AI Isolation Forest output.
    [PRODUCTION] Replace with: endpoint.predict(instances=df[FEATURES].tolist())"""
    print(f"\n[Step 3] Running Vertex AI classification on {len(df):,} sessions...")
    def classify_row(row):
        score    = row["edge_score"]
        velocity = row["event_velocity_per_sec"]
        duration = row["session_duration_sec"]
        mouse    = row["mouse_move_events"]
        if score < 0.20 and velocity > 20:
            return "Ad_Fraud"
        if score < 0.48 and velocity > 8 and duration < 10 and mouse == 0:
            return "LLM_Scraper"
        if score >= 0.55 and velocity <= 5 and duration >= 30 and mouse > 0:
            return "Human"
        if score >= 0.62:
            return "Human"
        elif score >= 0.38:
            return "LLM_Scraper"
        else:
            return "Ad_Fraud"
    df = df.copy()
    df["vertex_ai_classification"] = df.apply(classify_row, axis=1)
    pred_dist = dict(df["vertex_ai_classification"].value_counts())
    print(f"         Result: {pred_dist}")
    if has_ground_truth and "raw_traffic_label" in df.columns:
        accuracy = (df["vertex_ai_classification"] == df["raw_traffic_label"]).mean()
        print(f"         Accuracy: {accuracy:.1%}")
        for cls in ["Human","LLM_Scraper","Ad_Fraud"]:
            sub = df[df["raw_traffic_label"]==cls]
            acc = (sub["vertex_ai_classification"]==cls).mean()
            print(f"           {cls:<15}: {acc:.1%}  (n={len(sub)})")
    return df


CONV_VALUES = {
    "google / cpc"           : 150.00,
    "meta / paid_social"     : 120.00,
    "bing / cpc"             :  90.00,
    "direct / (none)"        : 200.00,
    "programmatic / display" :  60.00,
    "(direct) / (none)"      : 200.00,
    "google / organic"       : 180.00,
}


def export_cleaned_data(df):
    """Filters to Human rows, assigns conversion_value_usd, exports CSV."""
    print(f"\n[Step 4] Exporting cleaned data payload...")
    df_human = df[df["vertex_ai_classification"]=="Human"].copy()
    df_human["conversion_value_usd"] = df_human["source_medium"].map(CONV_VALUES).fillna(100.00)
    export_cols = [c for c in [
        "client_id","event_timestamp","source_medium","geo_country","device_category",
        "landing_page","edge_score","session_duration_sec","event_velocity_per_sec",
        "page_scroll_depth_pct","click_events","scroll_events","mouse_move_events",
        "vertex_ai_classification","conversion_value_usd",
    ] if c in df_human.columns]
    df_export = df_human[export_cols].reset_index(drop=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_export.to_csv(CLEANED_CSV, index=False)
    total_raw     = len(df)
    total_cleaned = len(df_export)
    bots_removed  = total_raw - total_cleaned
    total_value   = df_export["conversion_value_usd"].sum()
    avg_value     = df_export["conversion_value_usd"].mean()
    print(f"\n  {'='*50}")
    print(f"  Pipeline Summary  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'='*50}")
    print(f"  Raw sessions   : {total_raw:>6,}")
    print(f"  Human (cleaned): {total_cleaned:>6,}  ({total_cleaned/total_raw:.1%})")
    print(f"  Bots removed   : {bots_removed:>6,}  ({bots_removed/total_raw:.1%})")
    print(f"  Total conv val : ${total_value:>10,.2f}")
    print(f"  Avg conv val   : ${avg_value:>10.2f}")
    print(f"  Spend saved    : ${bots_removed*2.40:>10.2f}  ({bots_removed} bots x $2.40 CPC)")
    print(f"  Cleaned CSV    : {CLEANED_CSV}")
    return df_export


# =========================================================
# STEP 6: MERIDIAN MMM -- Channel Attribution
# =========================================================
# Maps GA4 source_medium values to clean display names.
CHANNEL_DISPLAY_NAMES = {
    "google / cpc":           "Google Ads",
    "meta / paid_social":     "Meta Ads",
    "bing / cpc":             "Bing Ads",
    "programmatic / display": "Display",
    "direct / (none)":        "Direct",
    "(direct) / (none)":      "Direct",
    "google / organic":       "Organic",
    "google / organic search":"Organic",
}

# Paid channels included in MMM (excludes zero-spend channels)
PAID_CHANNELS = ["Google Ads", "Meta Ads", "Bing Ads", "Display"]

# Assumed target ROAS per channel (revenue per $1 spend).
# Used to synthesize media spend from conversion values.
# [SYNTHETIC] -- real ad spend is not in the GA4 BigQuery schema.
CHANNEL_ROAS = {
    "Google Ads": 4.0,
    "Meta Ads":   3.2,
    "Bing Ads":   3.5,
    "Display":    2.5,
}


def _map_channel(source_medium):
    """Maps GA4 source_medium string to a clean channel display name."""
    s = str(source_medium).lower()
    for key, name in CHANNEL_DISPLAY_NAMES.items():
        if key in s:
            return name
    return "Other"


def _compute_channel_attribution(df_raw, df_cleaned):
    """Computes raw and calibrated channel contribution from session data.

    Raw contribution = estimated KPI if all sessions were human
                       (raw_sessions * avg_conv_value_per_channel)
    Calibrated contribution = actual KPI from human-only cleaned sessions

    Returns (channel_results_list, total_clean_kpi, total_raw_kpi_estimate)
    """
    dr = df_raw.copy()
    dr["channel"] = dr["source_medium"].apply(_map_channel)
    dc = df_cleaned.copy()
    dc["channel"] = dc["source_medium"].apply(_map_channel)

    # Calibrated KPI (real human conversion value)
    clean_kpi = dc.groupby("channel")["conversion_value_usd"].sum().fillna(0.0)

    # Session counts for bot inflation estimate
    raw_sessions   = dr.groupby("channel").size()
    clean_sessions = dc.groupby("channel").size()

    # Avg conversion value per channel (from cleaned data)
    avg_conv = dc.groupby("channel")["conversion_value_usd"].mean().fillna(100.0)

    # Use intersection of PAID_CHANNELS and observed data
    observed = set(raw_sessions.index) | set(clean_kpi.index)
    channels = [c for c in PAID_CHANNELS if c in observed]
    if not channels:
        channels = PAID_CHANNELS

    data = {}
    for ch in channels:
        r  = int(raw_sessions.get(ch, 0))
        c  = int(clean_sessions.get(ch, 0))
        ck = float(clean_kpi.get(ch, 0.0))
        av = float(avg_conv.get(ch, 100.0))
        data[ch] = {
            "raw_sessions":   r,
            "clean_sessions": c,
            "clean_kpi":      ck,
            "raw_kpi":        r * av,   # estimated -- assumes same avg_conv for bots
        }

    total_clean = sum(v["clean_kpi"] for v in data.values()) or 1.0
    total_raw   = sum(v["raw_kpi"]   for v in data.values()) or 1.0

    results = []
    for ch, d in data.items():
        raw_pct  = round(d["raw_kpi"]   / total_raw   * 100, 1)
        cal_pct  = round(d["clean_kpi"] / total_clean * 100, 1)
        bot_rate = 1.0 - (d["clean_sessions"] / max(d["raw_sessions"], 1))
        bot_pct  = round(bot_rate * 100, 1)
        roi      = CHANNEL_ROAS.get(ch, 3.0)

        if bot_pct >= 40:
            insight = "High bot inflation -- calibrated lift significantly below raw"
        elif bot_pct >= 20:
            insight = "Moderate bot inflation -- calibrated view recommended for bidding"
        else:
            insight = "Low bot contamination -- raw signal relatively reliable"

        results.append({
            "name":                        ch,
            "raw_contribution_pct":        raw_pct,
            "calibrated_contribution_pct": cal_pct,
            "bot_inflation_pct":           bot_pct,
            "roi":                         roi,
            "insight":                     insight,
        })

    return results, total_clean, total_raw


def _try_meridian(kpi_arr, media_arr, spend_total, channel_names, channel_results):
    """Attempts to run real google-meridian Bayesian MMM.

    Requires: pip install google-meridian
    Minimum recommended: 26 weeks of data, GPU for practical run times.

    Returns meridian version string on success, None on failure (falls back
    to the simple attribution model in _compute_channel_attribution).
    """
    try:
        import importlib.metadata
        ver = importlib.metadata.version("google-meridian")
    except Exception:
        ver = "unknown"

    try:
        import xarray as xr
        from meridian.data import input_data as data_lib
        from meridian.model import model as model_lib

        print(f"         google-meridian {ver} found -- building xarray InputData...")

        n_geos, n_weeks, n_ch = media_arr.shape
        from datetime import date, timedelta as td
        today = date.today()
        dates = [
            (today - td(weeks=(n_weeks - 1 - w))).isoformat()
            for w in range(n_weeks)
        ]

        kpi_da = xr.DataArray(
            kpi_arr,
            dims=["geo", "time"],
            coords={"geo": ["national"], "time": dates},
            name="kpi",
        )
        pop_da = xr.DataArray(
            np.array([1.0]),
            dims=["geo"],
            coords={"geo": ["national"]},
            name="population",
        )
        media_da = xr.DataArray(
            media_arr,
            dims=["geo", "media_time", "media_channel"],
            coords={
                "geo":           ["national"],
                "media_time":    dates,
                "media_channel": channel_names,
            },
            name="media",
        )
        spend_da = xr.DataArray(
            spend_total,
            dims=["media_channel"],
            coords={"media_channel": channel_names},
            name="media_spend",
        )

        input_data = data_lib.InputData(
            kpi=kpi_da,
            kpi_type="revenue",
            population=pop_da,
            media=media_da,
            media_spend=spend_da,
        )

        print("         Fitting Bayesian NUTS-MCMC (n_chains=1, n_samples=500)...")
        print("         Note: GPU recommended. CPU run may take several minutes.")

        mmm = model_lib.Meridian(input_data=input_data)
        mmm.sample_posterior(
            n_chains=1,
            n_adapt=200,
            n_burnin=0,
            n_samples=500,
            seed=RANDOM_SEED,
        )

        # Extract posterior ROI estimates
        try:
            from meridian.analysis import analyzer as az_lib
            ana      = az_lib.Analyzer(mmm)
            roi_vals = ana.roi().mean(dim=["chain", "draw"]).values
            for i, ch_r in enumerate(channel_results):
                if i < len(roi_vals):
                    ch_r["roi"]          = round(float(roi_vals[i]), 2)
                    ch_r["meridian_roi"] = True
            print(f"         Posterior ROI: {[round(float(v),2) for v in roi_vals]}")
        except Exception as ex:
            print(f"         ROI extraction skipped ({ex}) -- using ROAS priors")

        print(f"         Meridian MCMC complete (google-meridian v{ver})")
        return ver

    except ImportError as e:
        print(f"         google-meridian not installed: {e}")
        print("         Install: pip install google-meridian")
        print("         Falling back to channel attribution model.")
        return None
    except Exception as e:
        print(f"         Meridian error: {e}")
        print("         Falling back to channel attribution model.")
        return None


def run_meridian_mmm(df_raw, df_cleaned):
    """Step 6: Bayesian MMM via google-meridian (attribution fallback if not installed).

    Computes channel lift before and after bot removal.
    Media spend is synthesized from conversion values + ROAS assumptions [SYNTHETIC]
    because real ad account spend data is not part of the GA4 BigQuery schema.

    Writes results under the 'meridian' key in docs/data/summary.json.
    """
    print(f"\n[Step 6] Running Meridian MMM on {len(df_cleaned):,} cleaned sessions...")
    print("         Computing channel attribution (raw vs calibrated)...")

    channel_results, total_clean, total_raw = _compute_channel_attribution(df_raw, df_cleaned)
    channel_names = [c["name"] for c in channel_results]
    n_channels    = len(channel_names)

    print(f"         Paid channels : {channel_names}")
    print(f"         Clean KPI     : ${total_clean:,.2f}")
    print(f"         Raw KPI est.  : ${total_raw:,.2f}")
    print("         Synthesizing 52-week media time series [SYNTHETIC media spend]...")

    # Build synthetic 52-week panel data for Meridian.
    # Channel shares are grounded in real cleaned KPI proportions.
    # Media spend is derived via assumed ROAS (labelled SYNTHETIC).
    n_weeks = 52
    np.random.seed(RANDOM_SEED)

    shares = [c["calibrated_contribution_pct"] / 100.0 for c in channel_results]
    s_sum  = sum(shares) or 1.0
    shares = [s / s_sum for s in shares]

    kpi_arr   = np.zeros((1, n_weeks))
    media_arr = np.zeros((1, n_weeks, n_channels))

    for w in range(n_weeks):
        # Modest e-commerce holiday uplift in weeks 44-52 (Nov-Dec)
        season = 1.0 + 0.5 * np.exp(-((w - 49.0) ** 2) / 15.0) if w > 42 else 1.0
        noise  = max(0.5, np.random.normal(1.0, 0.10))
        wk_kpi = (total_clean / n_weeks) * season * noise
        kpi_arr[0, w] = wk_kpi
        for ci, ch_r in enumerate(channel_results):
            roas   = CHANNEL_ROAS.get(ch_r["name"], 3.0)
            ch_kpi = wk_kpi * shares[ci]
            media_arr[0, w, ci] = max(
                0.0, (ch_kpi / roas) * np.random.normal(1.0, 0.05)
            )

    spend_total = np.array([
        total_clean * shares[ci] / CHANNEL_ROAS.get(channel_results[ci]["name"], 3.0)
        for ci in range(n_channels)
    ])

    meridian_ver = _try_meridian(
        kpi_arr, media_arr, spend_total, channel_names, channel_results
    )

    if meridian_ver:
        model_label = f"google-meridian v{meridian_ver} (Bayesian NUTS-MCMC)"
        status      = "fitted"
    else:
        model_label = (
            "Attribution model (numpy) -- install google-meridian for Bayesian MMM"
        )
        status = "fallback"

    print(f"\n  Meridian Step 6 Results ({status}):")
    header = f"  {'Channel':<16} {'Raw':>8} {'Calibrated':>12} {'Bot Infl':>10} {'ROI':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ch_r in channel_results:
        print(
            f"  {ch_r['name']:<16}"
            f" {ch_r['raw_contribution_pct']:>7.1f}%"
            f" {ch_r['calibrated_contribution_pct']:>11.1f}%"
            f" {ch_r['bot_inflation_pct']:>9.1f}%"
            f" {ch_r['roi']:>5.1f}x"
        )

    return {
        "status":             status,
        "model":              model_label,
        "n_weeks_data":       n_weeks,
        "media_spend_source": (
            "SYNTHETIC -- derived from channel KPI proportions and ROAS assumptions. "
            "Real ad spend is not part of the GA4 BigQuery schema."
        ),
        "channels":           channel_results,
        "total_clean_value":  round(total_clean, 2),
        "total_raw_value":    round(total_raw, 2),
    }


def export_summary_json(df_raw, df_cleaned, mode, meridian_results=None):
    """Writes docs/data/summary.json for the live GitHub Pages dashboard.
    Commit this file after each run to push real data to the live site."""
    print(f"\n[Step 5] Exporting summary JSON for live dashboard...")
    df_raw_copy = df_raw.copy()
    df_raw_copy["_date"] = pd.to_datetime(df_raw_copy["event_timestamp"]).dt.date
    daily_raw = df_raw_copy.groupby("_date").size().reset_index(name="raw")
    df_cln    = df_cleaned.copy()
    df_cln["_date"] = pd.to_datetime(df_cln["event_timestamp"]).dt.date
    daily_cln = df_cln.groupby("_date").size().reset_index(name="cleaned")
    daily = daily_raw.merge(daily_cln, on="_date", how="left").fillna(0).tail(7).reset_index(drop=True)
    total_raw     = len(df_raw)
    total_cleaned = len(df_cleaned)
    bots_removed  = total_raw - total_cleaned
    summary = {
        "generated_at"  : datetime.now().isoformat(),
        "pipeline_mode" : mode,
        "data_source"   : (
            "bigquery-public-data.ga4_obfuscated_sample_ecommerce (Google Merchandise Store, Nov 2020 to Jan 2021)"
            if mode == "bigquery" else
            "Synthetic mock data -- run: python scripts/bq_vertex_pipeline.py to refresh"
        ),
        "sessions": {
            "total_raw"        : total_raw,
            "total_cleaned"    : total_cleaned,
            "bots_removed"     : bots_removed,
            "bots_removed_pct" : round(bots_removed / total_raw * 100, 1),
            "human_pct"        : round(total_cleaned / total_raw * 100, 1),
        },
        "daily": {
            "labels"  : [d.strftime("%a %d %b") for d in daily["_date"]],
            "raw"     : daily["raw"].astype(int).tolist(),
            "cleaned" : daily["cleaned"].astype(int).tolist(),
        },
        "vbb": {
            "human_sessions"        : total_cleaned,
            "bot_sessions"          : bots_removed,
            "total_conv_value"      : round(float(df_cleaned["conversion_value_usd"].sum()), 2),
            "avg_conv_value"        : round(float(df_cleaned["conversion_value_usd"].mean()), 2),
            "estimated_spend_saved" : round(bots_removed * 2.40, 2),
        },
    }
    if meridian_results is not None:
        summary["meridian"] = meridian_results

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(script_dir)
    out_dir    = os.path.join(repo_root, "docs", "data")
    out_path   = os.path.join(out_dir, "summary.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"         Written to: {out_path}")
    print(f"         Commit docs/data/summary.json to update the live dashboard.")
    return summary


def main():
    parser = argparse.ArgumentParser(description="GA4 Enterprise Agent Architecture Pipeline")
    parser.add_argument("--mock", action="store_true", help="Offline mock mode (no GCP needed)")
    parser.add_argument("--sessions", type=int, default=N_SESSIONS)
    args = parser.parse_args()
    use_mock = args.mock or os.getenv("PIPELINE_MODE","").lower() == "mock"
    mode     = "mock" if use_mock else "bigquery"
    print("="*58)
    print("  GA4 Enterprise Agent Architecture -- Pipeline")
    print(f"  Mode: {'MOCK (offline)' if use_mock else 'BIGQUERY (real GA4 public dataset)'}")
    print("="*58)
    if use_mock:
        df_raw = generate_mock_ga4_data(args.sessions)
        df_raw = enrich_with_edge_signals(df_raw, is_real_data=False)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_raw.to_csv(RAW_CSV, index=False)
        df_classified = vertex_ai_clustering(df_raw, has_ground_truth=True)
        df_cleaned    = export_cleaned_data(df_classified)
    else:
        df_raw = fetch_ga4_public_data(args.sessions)
        df_raw = enrich_with_edge_signals(df_raw, is_real_data=True)
        df_raw["raw_traffic_label"] = "UNKNOWN (real data)"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_raw.to_csv(RAW_CSV, index=False)
        df_classified = vertex_ai_clustering(df_raw, has_ground_truth=False)
        df_cleaned    = export_cleaned_data(df_classified)
    meridian_results = run_meridian_mmm(df_raw, df_cleaned)
    export_summary_json(df_raw, df_cleaned, mode, meridian_results=meridian_results)
    print("\n  [DONE] Pipeline complete.\n")


if __name__ == "__main__":
    main()
