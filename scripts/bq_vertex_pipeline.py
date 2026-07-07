"""
GA4 Enterprise Agent Architecture — BigQuery & Vertex AI Pipeline
=================================================================
Queries the publicly available GA4 Obfuscated Sample Ecommerce dataset on
Google BigQuery, then runs bot classification and exports the cleaned signal.

  Dataset : bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*
            Google Merchandise Store · Nov 2020 – Jan 2021 · privacy-obfuscated
            Free to query: first 1 TB/month at no cost under GCP free tier.

  Pipeline:
    Step 1  fetch_ga4_public_data()   — real session-level rows from BigQuery
    Step 2  enrich_with_edge_signals() — simulate Cloud Armor edge score (not in GA4)
    Step 3  vertex_ai_clustering()    — classify Human / LLM_Scraper / Ad_Fraud
    Step 4  export_cleaned_data()     — human-only CSV → VBB + Meridian MMM payload

  Modes:
    bigquery  (default)  queries real GA4 data — requires GCP project + credentials
    mock      (--mock)   generates synthetic data — no GCP needed, works offline

  Run:
    python scripts/bq_vertex_pipeline.py                   # BigQuery mode
    python scripts/bq_vertex_pipeline.py --mock            # Offline / no-GCP mode
    PIPELINE_MODE=mock python scripts/bq_vertex_pipeline.py

  GCP Setup (one-time, takes ~5 minutes):
    1. Create a free GCP project → https://console.cloud.google.com
    2. Run: gcloud auth application-default login
    3. Set GCP_PROJECT_ID below (or export GCP_PROJECT_ID=your-project-id)
    4. pip install google-cloud-bigquery db-dtypes pandas numpy

  Note on edge_score and mouse_move_events:
    These signals come from Cloud Armor / reCAPTCHA Enterprise and client-side
    JS, respectively — neither is exported to the GA4 BigQuery schema. In
    BigQuery mode they are simulated from real session features (duration,
    device, source) and are clearly labelled SIMULATED in the output.
"""

import argparse
import os
import random
import string
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

# [REQUIRED for BigQuery mode] Your GCP project ID (billing must be enabled,
# but the first 1 TB/month queried is free under the GCP free tier).
# Override at runtime: export GCP_PROJECT_ID=my-gcp-project
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "YOUR_GCP_PROJECT_ID")

# Public dataset — no changes needed
BQ_PUBLIC_DATASET = "bigquery-public-data.ga4_obfuscated_sample_ecommerce"
BQ_DATE_START     = "20201101"   # Nov 2020
BQ_DATE_END       = "20210131"   # Jan 2021

# [PRODUCTION] Your own GA4 BigQuery export — replace the public dataset above
# with these when pointing at a real property:
PROD_PROJECT_ID    = "YOUR_GCP_PROJECT_ID"
PROD_BQ_DATASET    = "YOUR_BIGQUERY_DATASET"   # e.g. "analytics_123456789"
PROD_BQ_TABLE      = "events_*"

# [PRODUCTION] Vertex AI endpoint (replace rule-based classifier in Step 3)
VERTEX_AI_ENDPOINT = "YOUR_VERTEX_AI_ENDPOINT_ID"
VERTEX_AI_REGION   = "us-central1"

OUTPUT_DIR  = "data"
RAW_CSV     = os.path.join(OUTPUT_DIR, "raw_ga4_events.csv")
CLEANED_CSV = os.path.join(OUTPUT_DIR, "cleaned_ga4_events.csv")

N_SESSIONS  = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════
# STEP 1A: FETCH REAL GA4 DATA FROM BIGQUERY (BigQuery mode)
#
# Queries the public GA4 Obfuscated Sample Ecommerce dataset.
# Aggregates event-level rows into one row per session with
# behavioral features directly derivable from the GA4 schema:
#   session_duration_sec  — real (from MIN/MAX event_timestamp)
#   event_velocity_per_sec — real (total_events / session_duration)
#   click_events          — real (COUNTIF event_name = 'click')
#   page_scroll_depth_pct — real (max percent_scrolled param)
#   source_medium         — real (traffic_source.source / medium)
#   geo_country           — real (geo.country)
#
# [PRODUCTION] Swap BQ_PUBLIC_DATASET for your own GA4 export:
#   `{PROD_PROJECT_ID}.{PROD_BQ_DATASET}.{PROD_BQ_TABLE}`
# ══════════════════════════════════════════════════════════════

GA4_SESSION_QUERY = """
WITH sessions AS (
  SELECT
    user_pseudo_id                                                    AS client_id,
    (SELECT value.int_value  FROM UNNEST(event_params) WHERE key = 'ga_session_id')
                                                                      AS session_id,
    MIN(event_timestamp)                                              AS session_start_us,
    MAX(event_timestamp)                                              AS session_end_us,
    TIMESTAMP_DIFF(
      TIMESTAMP_MICROS(MAX(event_timestamp)),
      TIMESTAMP_MICROS(MIN(event_timestamp)),
      SECOND
    )                                                                 AS session_duration_sec,
    COUNT(*)                                                          AS total_events,
    COUNTIF(event_name IN ('click', 'user_engagement'))               AS click_events,
    COUNTIF(event_name = 'scroll')                                    AS scroll_events,
    MAX(COALESCE(
      (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'percent_scrolled'),
      0
    ))                                                                AS max_scroll_pct,
    ANY_VALUE(CONCAT(
      COALESCE(traffic_source.source, '(direct)'), ' / ',
      COALESCE(traffic_source.medium, '(none)')
    ))                                                                AS source_medium,
    ANY_VALUE(COALESCE(geo.country, 'Unknown'))                       AS geo_country,
    ANY_VALUE(device.category)                                        AS device_category,
    ANY_VALUE(COALESCE(
      (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location'),
      '/'
    ))                                                                AS landing_page
  FROM `{dataset}.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '{date_start}' AND '{date_end}'
    AND event_name IN (
      'page_view', 'session_start', 'scroll', 'click',
      'user_engagement', 'purchase', 'add_to_cart', 'view_item'
    )
  GROUP BY 1, 2
)
SELECT
  client_id,
  FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M:%S',
    TIMESTAMP_MICROS(session_start_us))                               AS event_timestamp,
  GREATEST(session_duration_sec, 0)                                   AS session_duration_sec,
  total_events,
  click_events,
  scroll_events,
  COALESCE(max_scroll_pct, 0)                                         AS page_scroll_depth_pct,
  ROUND(SAFE_DIVIDE(total_events, GREATEST(session_duration_sec, 1)), 2)
                                                                      AS event_velocity_per_sec,
  source_medium,
  geo_country,
  COALESCE(device_category, 'desktop')                                AS device_category,
  landing_page
FROM sessions
WHERE session_id IS NOT NULL
ORDER BY RAND()
LIMIT {n_sessions}
"""


def fetch_ga4_public_data(n_sessions: int = N_SESSIONS) -> pd.DataFrame:
    """
    Queries the GA4 Obfuscated Sample Ecommerce public dataset on BigQuery.
    Returns one row per session with real behavioral features.

    Requires:
      - GCP project with billing enabled (first 1TB/month free)
      - Application Default Credentials: gcloud auth application-default login
      - pip install google-cloud-bigquery db-dtypes
    """
    try:
        from google.cloud import bigquery   # type: ignore
    except ImportError:
        print("\n[ERROR] google-cloud-bigquery not installed.")
        print("        Run: pip install google-cloud-bigquery db-dtypes")
        print("        Then re-run this script, or use --mock for offline mode.\n")
        sys.exit(1)

    if GCP_PROJECT_ID == "YOUR_GCP_PROJECT_ID":
        print("\n[ERROR] GCP_PROJECT_ID is not set.")
        print("        Set it at the top of this script, or:")
        print("        export GCP_PROJECT_ID=my-gcp-project")
        print("        Or run in offline mode: python scripts/bq_vertex_pipeline.py --mock\n")
        sys.exit(1)

    print(f"\n[Step 1] Querying GA4 public dataset on BigQuery...")
    print(f"         Dataset  : {BQ_PUBLIC_DATASET}.events_*")
    print(f"         Dates    : {BQ_DATE_START} → {BQ_DATE_END}  (Nov 2020 – Jan 2021)")
    print(f"         Sessions : {n_sessions:,}")
    print(f"         Project  : {GCP_PROJECT_ID}  (billing project for query costs)")
    print(f"         Est. cost: < 0.001 TB → well within 1TB free tier\n")

    client = bigquery.Client(project=GCP_PROJECT_ID)

    query = GA4_SESSION_QUERY.format(
        dataset    = BQ_PUBLIC_DATASET,
        date_start = BQ_DATE_START,
        date_end   = BQ_DATE_END,
        n_sessions = n_sessions,
    )

    print("         Running query...")
    df = client.query(query).to_dataframe()

    if df.empty:
        print("[ERROR] Query returned 0 rows. Check dataset access and date range.")
        sys.exit(1)

    print(f"         ✓ Fetched {len(df):,} real GA4 sessions from BigQuery")
    print(f"           Columns: {list(df.columns)}")

    # Normalise types
    df["session_duration_sec"]    = df["session_duration_sec"].fillna(0).astype(int)
    df["click_events"]            = df["click_events"].fillna(0).astype(int)
    df["scroll_events"]           = df["scroll_events"].fillna(0).astype(int)
    df["page_scroll_depth_pct"]   = df["page_scroll_depth_pct"].fillna(0).astype(int)
    df["event_velocity_per_sec"]  = df["event_velocity_per_sec"].fillna(1.0).astype(float)

    # Show source distribution
    top_sources = df["source_medium"].value_counts().head(5)
    print(f"\n         Top traffic sources in real data:")
    for src, cnt in top_sources.items():
        print(f"           {src:<35} {cnt:>5} sessions")

    return df


# ══════════════════════════════════════════════════════════════
# STEP 1B: SYNTHETIC DATA GENERATOR (--mock / offline mode)
#
# Used when no GCP credentials are available.
# Generates 1,000 rows with distributions across three traffic
# types to demonstrate bot detection. Kept as fallback.
# ══════════════════════════════════════════════════════════════

def _random_ip(is_bot: bool) -> str:
    if is_bot:
        prefixes = ["66.249", "40.77", "157.55", "207.46", "54.173", "34.86"]
        return f"{random.choice(prefixes)}.{random.randint(1,254)}.{random.randint(1,254)}"
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def generate_mock_ga4_data(n: int = N_SESSIONS) -> pd.DataFrame:
    """
    [MOCK / OFFLINE] Generates n synthetic GA4 session rows.
    Used when --mock flag is passed or no GCP credentials are available.
    Traffic split: 55% Human · 25% LLM_Scraper · 20% Ad_Fraud
    """
    print(f"\n[Step 1] Generating {n:,} synthetic GA4 sessions (offline mock mode)...")
    print("         [NOTE] No BigQuery connection — all data is fabricated.")

    traffic_types = np.random.choice(
        ["Human", "LLM_Scraper", "Ad_Fraud"], size=n, p=[0.55, 0.25, 0.20]
    )

    base_time = datetime(2025, 1, 6, 8, 0, 0)
    sources   = ["google / cpc", "meta / paid_social", "direct / (none)",
                 "bing / cpc", "programmatic / display"]
    countries = ["US", "GB", "CA", "AU", "DE", "BR", "IN", "FR", "NL", "SG"]
    pages     = ["/", "/pricing", "/about", "/blog/ai-detection", "/contact", "/demo"]
    devices   = ["desktop", "mobile", "tablet"]

    rows = []
    elapsed = 0

    for t_type in traffic_types:
        is_human   = (t_type == "Human")
        is_scraper = (t_type == "LLM_Scraper")

        session_duration = (
            random.randint(45, 600) if is_human  else
            random.randint(0, 6)    if is_scraper else
            random.randint(0, 3)
        )
        click_events   = random.randint(1, 12) if is_human else 0
        scroll_events  = random.randint(2, 15) if is_human else 0
        scroll_depth   = random.randint(35, 95) if is_human else random.randint(0, 8)
        total_events   = (
            random.randint(4, 20)  if is_human  else
            random.randint(1, 3)   if is_scraper else
            random.randint(1, 2)
        )
        velocity = round(total_events / max(session_duration, 1), 2)

        elapsed += random.uniform(0.3, 6.0)
        rows.append({
            "client_id"              : ''.join(random.choices(string.ascii_uppercase + string.digits, k=12)),
            "event_timestamp"        : (base_time + timedelta(seconds=elapsed)).isoformat(),
            "session_duration_sec"   : session_duration,
            "total_events"           : total_events,
            "click_events"           : click_events,
            "scroll_events"          : scroll_events,
            "page_scroll_depth_pct"  : scroll_depth,
            "event_velocity_per_sec" : velocity,
            "source_medium"          : random.choice(sources),
            "geo_country"            : random.choice(countries),
            "device_category"        : random.choice(devices),
            "landing_page"           : random.choice(pages),
            "raw_traffic_label"      : t_type,   # ground truth for validation
        })

    df = pd.DataFrame(rows)
    dist = dict(pd.Series(traffic_types).value_counts())
    print(f"         Distribution : {dist}")
    return df


# ══════════════════════════════════════════════════════════════
# STEP 2: ENRICH WITH SIMULATED EDGE SIGNALS
#
# Two features cannot be derived from the GA4 BigQuery schema:
#
#   edge_score        — set by Cloud Armor / reCAPTCHA Enterprise
#                       at the network edge, before hits reach GA4.
#                       Simulated here from device + source signals.
#
#   mouse_move_events — client-side JS behavioral signal, not a
#                       standard GA4 event. Simulated from session
#                       duration and click count.
#
# Both are clearly marked SIMULATED in the exported CSV.
# In production they would come from Server-Side GTM custom variables.
# ══════════════════════════════════════════════════════════════

def enrich_with_edge_signals(df: pd.DataFrame, is_real_data: bool = True) -> pd.DataFrame:
    """
    Adds edge_score and mouse_move_events to real BigQuery rows.
    For mock data, these were already set to realistic values by the generator.
    Labels both columns as SIMULATED so downstream consumers know their origin.
    """
    if not is_real_data:
        # Mock data: derive edge scores from raw_traffic_label ground truth
        def _edge_from_label(row):
            if row["raw_traffic_label"] == "Human":
                return round(float(np.random.beta(8, 2)), 3)
            elif row["raw_traffic_label"] == "LLM_Scraper":
                return round(float(np.random.beta(2, 6)), 3)
            else:
                return round(float(np.random.beta(1, 9)), 3)
        df["edge_score"]        = df.apply(_edge_from_label, axis=1)
        df["mouse_move_events"] = df.apply(
            lambda r: random.randint(20, 250) if r["raw_traffic_label"] == "Human" else 0, axis=1
        )
        return df

    print(f"\n[Step 2] Enriching {len(df):,} real GA4 sessions with simulated edge signals...")
    print("         edge_score        — [SIMULATED] Cloud Armor / reCAPTCHA not in GA4 schema")
    print("         mouse_move_events — [SIMULATED] client-side JS signal not in GA4 schema")

    edge_scores      = []
    mouse_move_list  = []

    for _, row in df.iterrows():
        duration = row["session_duration_sec"]
        velocity = row["event_velocity_per_sec"]
        device   = str(row.get("device_category", "desktop")).lower()
        source   = str(row.get("source_medium", "")).lower()

        # ── Simulate edge_score from available session signals ────────
        # Real humans from search / direct on desktop/mobile → high scores.
        # High-velocity, zero-duration sessions → low scores (suspicious).
        if duration >= 30 and velocity <= 5 and row["click_events"] > 0:
            # Looks like a genuine engaged session
            score = round(float(np.random.beta(8, 2)), 3)   # 0.65–0.98 range
        elif duration <= 2 or velocity > 15:
            # Very short or very fast — bot signal
            if "programmatic" in source or "display" in source:
                score = round(float(np.random.beta(1, 8)), 3)  # 0.02–0.25
            else:
                score = round(float(np.random.beta(2, 6)), 3)  # 0.10–0.45
        else:
            # Ambiguous — mid-range score
            score = round(float(np.random.beta(5, 4)), 3)      # 0.35–0.75

        # ── Simulate mouse_move_events from engagement proxies ────────
        # Real GA4 has scroll % and click events — use these as proxies.
        if row["click_events"] > 0 or row["scroll_events"] > 0:
            # Engaged session: scale mouse moves with duration
            mouse = random.randint(
                max(5, int(duration * 0.3)),
                min(300, int(duration * 0.8))
            )
        else:
            mouse = 0   # No engagement signals → likely no mouse movement

        edge_scores.append(score)
        mouse_move_list.append(mouse)

    df = df.copy()
    df["edge_score"]        = edge_scores
    df["mouse_move_events"] = mouse_move_list

    # Add provenance flag so downstream knows which signals are real vs simulated
    df["edge_score_source"]        = "SIMULATED (Cloud Armor not in GA4 schema)"
    df["mouse_move_events_source"] = "SIMULATED (client-side signal not in GA4 schema)"

    print(f"         ✓ Edge score range   : {df['edge_score'].min():.3f} – {df['edge_score'].max():.3f}")
    print(f"         ✓ Mean edge score    : {df['edge_score'].mean():.3f}  (real data → mostly high)")

    return df


# ══════════════════════════════════════════════════════════════
# STEP 3: VERTEX AI CLUSTERING
#
# [MOCK]       Rule-based classifier that mirrors what a trained
#              Isolation Forest / K-Means model learns from the
#              four primary features.
#
# [PRODUCTION] Replace classify_row() with a Vertex AI call:
#
#   from google.cloud import aiplatform
#   aiplatform.init(project=GCP_PROJECT_ID, location=VERTEX_AI_REGION)
#   endpoint  = aiplatform.Endpoint(VERTEX_AI_ENDPOINT)
#   FEATURES  = ["edge_score", "event_velocity_per_sec",
#                "session_duration_sec", "mouse_move_events",
#                "page_scroll_depth_pct"]
#   instances = df[FEATURES].values.tolist()
#   response  = endpoint.predict(instances=instances)
#   df["vertex_ai_classification"] = [p["label"] for p in response.predictions]
# ══════════════════════════════════════════════════════════════

def vertex_ai_clustering(df: pd.DataFrame, has_ground_truth: bool = False) -> pd.DataFrame:
    """
    Classifies each session as Human / LLM_Scraper / Ad_Fraud using a
    rule hierarchy that mirrors what a trained Vertex AI model learns.

    For real BigQuery data: no ground-truth labels exist, so accuracy
    metrics are skipped. The distribution itself is the output.
    """
    print(f"\n[Step 3] Running Vertex AI classification on {len(df):,} sessions...")
    print("         [PRODUCTION] This would call your trained Vertex AI endpoint.")

    def classify_row(row) -> str:
        score    = row["edge_score"]
        velocity = row["event_velocity_per_sec"]
        duration = row["session_duration_sec"]
        mouse    = row["mouse_move_events"]

        # Rule 1: Ad Fraud — very low score + extreme event velocity
        if score < 0.20 and velocity > 20:
            return "Ad_Fraud"

        # Rule 2: LLM Scraper — low score + high velocity + no behavioral signals
        if score < 0.48 and velocity > 8 and duration < 10 and mouse == 0:
            return "LLM_Scraper"

        # Rule 3: Confirmed Human — high score + normal velocity + engagement
        if score >= 0.55 and velocity <= 5 and duration >= 30 and mouse > 0:
            return "Human"

        # Rule 4: Edge-score tiebreaker for ambiguous sessions
        if score >= 0.62:
            return "Human"
        elif score >= 0.38:
            return "LLM_Scraper"
        else:
            return "Ad_Fraud"

    df = df.copy()
    df["vertex_ai_classification"] = df.apply(classify_row, axis=1)
    pred_dist = dict(df["vertex_ai_classification"].value_counts())
    print(f"         Classification result : {pred_dist}")

    if has_ground_truth and "raw_traffic_label" in df.columns:
        accuracy = (df["vertex_ai_classification"] == df["raw_traffic_label"]).mean()
        print(f"         Accuracy vs labels   : {accuracy:.1%}")
        for cls in ["Human", "LLM_Scraper", "Ad_Fraud"]:
            subset  = df[df["raw_traffic_label"] == cls]
            cls_acc = (subset["vertex_ai_classification"] == cls).mean()
            print(f"           {cls:<15} precision : {cls_acc:.1%}  (n={len(subset)})")

    return df


# ══════════════════════════════════════════════════════════════
# STEP 4: EXPORT CLEANED DATA
#
# [MOCK]       Writes human-only rows to local CSV.
# [PRODUCTION] Write to BigQuery then trigger downstream:
#
#   df_cleaned.to_gbq(
#     destination_table=f"{GCP_PROJECT_ID}.{PROD_BQ_DATASET}.cleaned_events",
#     project_id=GCP_PROJECT_ID, if_exists="replace"
#   )
#   → Trigger Google Ads offline conversion import (VBB)
#   → Trigger Meridian MMM Vertex AI Pipeline
# ══════════════════════════════════════════════════════════════

CONV_VALUES = {
    "google / cpc"           : 150.00,
    "meta / paid_social"     : 120.00,
    "bing / cpc"             :  90.00,
    "direct / (none)"        : 200.00,
    "programmatic / display" :  60.00,
    "(direct) / (none)"      : 200.00,   # GA4 public dataset format
    "google / organic"       : 180.00,
}

def export_cleaned_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters to Human-classified rows, assigns conversion_value_usd,
    and exports the cleaned payload for VBB upload and Meridian MMM.
    """
    print(f"\n[Step 4] Exporting cleaned human-only data payload...")
    print("         [PRODUCTION] Would write to BigQuery → trigger Meridian pipeline.")

    df_human = df[df["vertex_ai_classification"] == "Human"].copy()

    df_human["conversion_value_usd"] = (
        df_human["source_medium"].map(CONV_VALUES).fillna(100.00)
    )

    export_cols = [
        "client_id", "event_timestamp", "source_medium",
        "geo_country", "device_category", "landing_page",
        "edge_score", "session_duration_sec",
        "event_velocity_per_sec", "page_scroll_depth_pct",
        "click_events", "scroll_events", "mouse_move_events",
        "vertex_ai_classification", "conversion_value_usd",
    ]

    # Keep only columns that exist (real data may not have all mock columns)
    export_cols = [c for c in export_cols if c in df_human.columns]
    df_export   = df_human[export_cols].reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_export.to_csv(CLEANED_CSV, index=False)

    total_raw     = len(df)
    total_cleaned = len(df_export)
    bots_removed  = total_raw - total_cleaned
    total_value   = df_export["conversion_value_usd"].sum()
    avg_value     = df_export["conversion_value_usd"].mean()

    print(f"\n{'═'*58}")
    print(f"  Pipeline Summary — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*58}")
    print(f"  Raw GA4 sessions         : {total_raw:>6,}")
    print(f"  Human (cleaned signal)   : {total_cleaned:>6,}  ({total_cleaned/total_raw:.1%})")
    print(f"  Bots / Scrapers removed  : {bots_removed:>6,}  ({bots_removed/total_raw:.1%})")
    print(f"  {'─'*48}")
    print(f"  Total Conversion Value   : ${total_value:>10,.2f}")
    print(f"  Avg Conv Value / Session : ${avg_value:>10.2f}")
    print(f"  {'─'*48}")
    print(f"  Cleaned CSV              : {CLEANED_CSV}")
    print(f"{'═'*58}")
    print(f"\n  Downstream targets:")
    print(f"    → Google Ads VBB  : upload {total_cleaned:,} conversions at avg ${avg_value:.2f}")
    print(f"    → Meridian MMM    : {bots_removed:,} bot sessions removed from attribution model")
    print(f"    → Looker Studio   : refreshed dataset ready for executive reporting")

    # Source breakdown
    if total_cleaned > 0:
        print(f"\n  Conversion value by source:")
        src_summary = df_export.groupby("source_medium")["conversion_value_usd"].agg(["count", "sum"])
        for src, row in src_summary.iterrows():
            print(f"    {src:<35} {int(row['count']):>5} sessions  ${row['sum']:>10,.2f}")

    return df_export


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GA4 Enterprise Agent Architecture — BigQuery & Vertex AI Pipeline"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Run in offline mock mode (no GCP credentials required)"
    )
    parser.add_argument(
        "--sessions", type=int, default=N_SESSIONS,
        help=f"Number of sessions to fetch/generate (default: {N_SESSIONS})"
    )
    args = parser.parse_args()

    use_mock = args.mock or os.getenv("PIPELINE_MODE", "").lower() == "mock"

    print("═" * 58)
    print("  GA4 Enterprise Agent Architecture")
    print("  BigQuery & Vertex AI Pipeline")
    mode_str = "MOCK (offline)" if use_mock else "BIGQUERY (real GA4 public dataset)"
    print(f"  Mode    : {mode_str}")
    if not use_mock:
        print(f"  Project : {GCP_PROJECT_ID}")
        print(f"  Dataset : {BQ_PUBLIC_DATASET}")
    print("═" * 58)

    if use_mock:
        # ── Offline / mock mode ──────────────────────────────────────
        df_raw = generate_mock_ga4_data(args.sessions)
        df_raw = enrich_with_edge_signals(df_raw, is_real_data=False)
        df_raw.to_csv(RAW_CSV, index=False)
        print(f"         Saved to : {RAW_CSV}")

        df_classified = vertex_ai_clustering(df_raw, has_ground_truth=True)
        df_cleaned    = export_cleaned_data(df_classified)

    else:
        # ── BigQuery mode — real GA4 public data ────────────────────
        df_raw = fetch_ga4_public_data(args.sessions)
        df_raw = enrich_with_edge_signals(df_raw, is_real_data=True)

        # No ground-truth labels in real data — skip accuracy metrics
        df_raw["raw_traffic_label"] = "UNKNOWN (real data)"
        df_raw.to_csv(RAW_CSV, index=False)
        print(f"         Saved raw sessions to : {RAW_CSV}")

        df_classified = vertex_ai_clustering(df_raw, has_ground_truth=False)
        df_cleaned    = export_cleaned_data(df_classified)

    print("\n  [DONE] Pipeline complete.\n")


if __name__ == "__main__":
    main()
