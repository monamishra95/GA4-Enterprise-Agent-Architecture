"""
GA4 Enterprise Agent Architecture — BigQuery & Vertex AI Pipeline Simulator
============================================================================
Simulates the full backend data pipeline that runs in production on GCP:

  Step 1: Generate 1,000 mock GA4 raw event rows → data/raw_ga4_events.csv
  Step 2: vertex_ai_clustering() — classify each row as Human / LLM_Scraper / Ad_Fraud
  Step 3: Export cleaned (human-only) data → data/cleaned_ga4_events.csv
           (this CSV simulates the payload sent to Meridian MMM and Google Ads VBB)

Each section is clearly marked [MOCK] vs [PRODUCTION] so you know exactly
where real GCP API calls replace the simulation logic.

Dependencies: pip install pandas numpy scikit-learn
Run with:     python scripts/bq_vertex_pipeline.py
"""

import os
import random
import string
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────
# CONFIG  — Replace ALL caps values in production
# ─────────────────────────────────────────────────────────────

GCP_PROJECT_ID       = "YOUR_GCP_PROJECT_ID"        # [PRODUCTION] e.g. "my-analytics-prod"
BIGQUERY_DATASET     = "YOUR_BIGQUERY_DATASET"       # [PRODUCTION] e.g. "ga4_raw_events"
BIGQUERY_TABLE       = "YOUR_BIGQUERY_TABLE"         # [PRODUCTION] e.g. "events_*"
VERTEX_AI_ENDPOINT   = "YOUR_VERTEX_AI_ENDPOINT_ID"  # [PRODUCTION] Vertex AI model endpoint ID
VERTEX_AI_REGION     = "us-central1"                 # [PRODUCTION] GCP region

OUTPUT_DIR       = "data"
RAW_CSV          = os.path.join(OUTPUT_DIR, "raw_ga4_events.csv")
CLEANED_CSV      = os.path.join(OUTPUT_DIR, "cleaned_ga4_events.csv")

N_ROWS      = 1000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════
# STEP 1: GENERATE MOCK GA4 RAW EVENT DATA
#
# [MOCK]       Uses numpy/random to synthesize realistic GA4 rows.
# [PRODUCTION] Replace the body of generate_mock_ga4_data() with:
#
#   from google.cloud import bigquery
#   client = bigquery.Client(project=GCP_PROJECT_ID)
#   query  = f"""
#     SELECT
#       event_timestamp,
#       user_pseudo_id          AS client_id,
#       geo.country             AS geo_country,
#       traffic_source.source   AS source,
#       traffic_source.medium   AS medium,
#       device.web_info.browser AS browser,
#       (SELECT value.string_value FROM UNNEST(event_params)
#        WHERE key = 'page_location') AS page_location
#     FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}`
#     WHERE _TABLE_SUFFIX BETWEEN '20250101' AND '20250107'
#       AND event_name = 'page_view'
#     LIMIT {N_ROWS}
#   """
#   df = client.query(query).to_dataframe()
# ══════════════════════════════════════════════════════════════

def _random_ip(is_bot: bool) -> str:
    """[MOCK] Generate realistic IPs. Bots use known crawler ranges."""
    if is_bot:
        # Known bot / datacenter IP prefixes (Googlebot, GPTBot, ad fraud farms)
        prefixes = ["66.249", "40.77", "157.55", "207.46", "54.173", "34.86"]
        return f"{random.choice(prefixes)}.{random.randint(1,254)}.{random.randint(1,254)}"
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def _random_user_agent(traffic_type: str) -> str:
    """[MOCK] Assign user agents matching each traffic profile."""
    human_uas = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Firefox/121.0",
        "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
    ]
    scraper_uas = [
        "GPTBot/1.0 (+https://openai.com/gptbot)",
        "ClaudeBot/1.0 (+https://www.anthropic.com/claude)",
        "python-requests/2.31.0",
        "curl/8.4.0",
        "Wget/1.21.3",
        "Go-http-client/2.0",
    ]
    fraud_uas = [
        "AdsBot-Google (+http://www.google.com/adsbot.html)",
        "Mozilla/5.0 (compatible; BLEXBot/1.0)",
        "Mozilla/5.0 (compatible; SemrushBot/7~bl)",
        "Googlebot/2.1 (+http://www.google.com/bot.html)",
    ]
    if traffic_type == "Human":      return random.choice(human_uas)
    if traffic_type == "LLM_Scraper": return random.choice(scraper_uas)
    return random.choice(fraud_uas)


def generate_mock_ga4_data(n: int = N_ROWS) -> pd.DataFrame:
    """
    [MOCK] Generates n synthetic GA4 raw event rows with realistic
    distributions across three traffic types:
      55% Human · 25% LLM_Scraper · 20% Ad_Fraud

    Each row contains the features used by vertex_ai_clustering():
      edge_score, event_velocity_per_sec, session_duration_sec,
      mouse_move_events, page_scroll_depth_pct
    """
    print(f"\n[Step 1] Generating {n:,} mock GA4 raw event rows...")

    traffic_types = np.random.choice(
        ["Human", "LLM_Scraper", "Ad_Fraud"],
        size=n,
        p=[0.55, 0.25, 0.20]
    )

    base_time = datetime(2025, 1, 6, 8, 0, 0)
    sources   = ["google / cpc", "meta / paid_social", "direct / (none)", "bing / cpc", "programmatic / display"]
    countries = ["US", "GB", "CA", "AU", "DE", "BR", "IN", "FR", "NL", "SG"]
    pages     = ["/", "/pricing", "/about", "/blog/ai-detection", "/contact", "/demo"]

    rows = []
    elapsed = 0

    for i, t_type in enumerate(traffic_types):
        is_human  = (t_type == "Human")
        is_fraud  = (t_type == "Ad_Fraud")
        is_scraper = (t_type == "LLM_Scraper")

        # ── Edge Score (primary detection signal from Cloud Armor) ──
        # [MOCK] In production this comes from reCAPTCHA Enterprise / Cloud Armor
        # and is attached to the hit via a Server-Side GTM custom variable.
        if is_human:
            edge_score = round(float(np.random.beta(8, 2)), 3)    # 0.65–1.0 range
        elif is_scraper:
            edge_score = round(float(np.random.beta(2, 6)), 3)    # 0.10–0.45 range
        else:
            edge_score = round(float(np.random.beta(1, 9)), 3)    # 0.02–0.20 range

        # ── Event Velocity (hits/sec sent to GA4 collection endpoint) ──
        event_velocity = (
            random.randint(1, 4)   if is_human  else
            random.randint(10, 35) if is_scraper else
            random.randint(30, 80)                    # Ad fraud: very high
        )

        # ── Session Duration ──
        session_duration = (
            random.randint(45, 600) if is_human  else
            random.randint(0, 6)    if is_scraper else
            random.randint(0, 3)
        )

        # ── Behavioral Signals ──
        mouse_moves     = random.randint(20, 250) if is_human else 0
        scroll_depth    = random.randint(35, 95)  if is_human else random.randint(0, 8)
        click_events    = random.randint(1, 12)   if is_human else 0

        elapsed += random.uniform(0.3, 6.0)

        rows.append({
            "event_timestamp"        : (base_time + timedelta(seconds=elapsed)).isoformat(),
            "client_id"              : ''.join(random.choices(string.ascii_uppercase + string.digits, k=12)),
            "ip_address"             : _random_ip(not is_human),
            "user_agent"             : _random_user_agent(t_type),
            "source_medium"          : random.choice(sources),
            "geo_country"            : random.choice(countries),
            "landing_page"           : random.choice(pages),
            "edge_score"             : edge_score,
            "event_velocity_per_sec" : event_velocity,
            "session_duration_sec"   : session_duration,
            "mouse_move_events"      : mouse_moves,
            "page_scroll_depth_pct"  : scroll_depth,
            "click_events"           : click_events,
            "raw_traffic_label"      : t_type,   # Ground truth for validation only
        })

    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)

    dist = dict(pd.Series(traffic_types).value_counts())
    print(f"         Distribution : {dist}")
    print(f"         Saved to     : {RAW_CSV}")
    return df


# ══════════════════════════════════════════════════════════════
# STEP 2: VERTEX AI CLUSTERING
#
# [MOCK]       Uses interpretable rule-based logic that mirrors
#              what a trained Isolation Forest or K-Means model
#              would produce on these features.
#
# [PRODUCTION] Replace the classify_row() logic with a call to
#              your trained Vertex AI endpoint:
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

def vertex_ai_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies each GA4 hit into ['Human', 'LLM_Scraper', 'Ad_Fraud']
    using a rule hierarchy that mirrors a trained Vertex AI model.

    Feature priority:
      1. edge_score              — strongest single signal
      2. event_velocity_per_sec  — bots fire events much faster than humans
      3. session_duration_sec    — humans stay; bots leave instantly
      4. mouse_move_events       — humans move mice; bots do not
    """
    print(f"\n[Step 2] Running Vertex AI clustering simulation on {len(df):,} rows...")
    print("         [PRODUCTION] This would call your trained Vertex AI endpoint.")

    def classify_row(row) -> str:
        score    = row["edge_score"]
        velocity = row["event_velocity_per_sec"]
        duration = row["session_duration_sec"]
        mouse    = row["mouse_move_events"]
        scroll   = row["page_scroll_depth_pct"]

        # ── Rule 1: Ad Fraud ──────────────────────────────────────────
        # Very low edge score + extreme event velocity = click fraud bot
        if score < 0.20 and velocity > 20:
            return "Ad_Fraud"

        # ── Rule 2: LLM Scraper ───────────────────────────────────────
        # Low-medium score + high velocity + near-zero behavioral signals
        if score < 0.48 and velocity > 8 and duration < 10 and mouse == 0:
            return "LLM_Scraper"

        # ── Rule 3: Confirmed Human ───────────────────────────────────
        # High edge score + normal velocity + meaningful engagement
        if score >= 0.55 and velocity <= 5 and duration >= 30 and mouse > 0:
            return "Human"

        # ── Rule 4: Edge-score tiebreaker ────────────────────────────
        # Catches ambiguous sessions (suspicious range 0.4–0.6)
        if score >= 0.62:
            return "Human"
        elif score >= 0.38:
            return "LLM_Scraper"
        else:
            return "Ad_Fraud"

    df = df.copy()
    df["vertex_ai_classification"] = df.apply(classify_row, axis=1)

    # ── Validation vs ground truth ───────────────────────────────────
    accuracy = (df["vertex_ai_classification"] == df["raw_traffic_label"]).mean()
    pred_dist = dict(df["vertex_ai_classification"].value_counts())

    print(f"         Predicted distribution : {pred_dist}")
    print(f"         Accuracy vs labels     : {accuracy:.1%}")

    # Per-class accuracy breakdown
    for cls in ["Human", "LLM_Scraper", "Ad_Fraud"]:
        subset = df[df["raw_traffic_label"] == cls]
        cls_acc = (subset["vertex_ai_classification"] == cls).mean()
        print(f"           {cls:<15} precision : {cls_acc:.1%}  (n={len(subset)})")

    return df


# ══════════════════════════════════════════════════════════════
# STEP 3: EXPORT CLEANED DATA
#
# [MOCK]       Writes filtered rows to a local CSV.
# [PRODUCTION] Replace df.to_csv() with BigQuery write:
#
#   df_cleaned.to_gbq(
#     destination_table=f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.cleaned_events",
#     project_id=GCP_PROJECT_ID,
#     if_exists="replace"
#   )
#
#   Then trigger:
#   - Google Ads offline conversion import (for VBB)
#   - Meridian MMM pipeline via Vertex AI Pipelines
# ══════════════════════════════════════════════════════════════

# Conversion values by source — [MOCK] In production, pull from CRM / Salesforce
CONV_VALUES = {
    "google / cpc"           : 150.00,
    "meta / paid_social"     : 120.00,
    "bing / cpc"             :  90.00,
    "direct / (none)"        : 200.00,
    "programmatic / display" :  60.00,
}

def export_cleaned_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters to Human-classified rows only, assigns conversion_value_usd
    for Google Ads VBB upload, and exports to CSV.

    The resulting CSV represents the "clean signal" payload sent to:
      - Google Ads Enhanced Conversions (Value-Based Bidding)
      - Meridian MMM for incrementality calibration
      - Looker Studio for executive reporting
    """
    print(f"\n[Step 3] Exporting cleaned data payload...")
    print("         [PRODUCTION] This would write to BigQuery → trigger Meridian pipeline.")

    df_human = df[df["vertex_ai_classification"] == "Human"].copy()

    # Assign conversion values for VBB
    df_human["conversion_value_usd"] = (
        df_human["source_medium"].map(CONV_VALUES).fillna(100.00)
    )

    # Columns exported (matches GA4 Enhanced Conversions schema)
    export_cols = [
        "event_timestamp", "client_id", "source_medium",
        "geo_country", "landing_page",
        "edge_score", "session_duration_sec",
        "page_scroll_depth_pct", "mouse_move_events",
        "vertex_ai_classification", "conversion_value_usd",
    ]
    df_export = df_human[export_cols].reset_index(drop=True)
    df_export.to_csv(CLEANED_CSV, index=False)

    # ── Pipeline Summary ─────────────────────────────────────────────
    total_raw      = len(df)
    total_cleaned  = len(df_export)
    bots_removed   = total_raw - total_cleaned
    total_value    = df_export["conversion_value_usd"].sum()
    avg_value      = df_export["conversion_value_usd"].mean()
    clean_pct      = total_cleaned / total_raw
    bot_pct        = bots_removed  / total_raw

    print(f"\n{'═'*58}")
    print(f"  Pipeline Summary — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*58}")
    print(f"  Raw GA4 hits             : {total_raw:>6,}")
    print(f"  Human (cleaned signal)   : {total_cleaned:>6,}  ({clean_pct:.1%})")
    print(f"  Bots / Scrapers removed  : {bots_removed:>6,}  ({bot_pct:.1%})")
    print(f"  {'─'*48}")
    print(f"  Total Conversion Value   : ${total_value:>10,.2f}")
    print(f"  Avg Conv Value / Session : ${avg_value:>10.2f}")
    print(f"  {'─'*48}")
    print(f"  Cleaned CSV              : {CLEANED_CSV}")
    print(f"{'═'*58}")
    print(f"\n  Downstream targets:")
    print(f"    → Google Ads VBB  : upload {total_cleaned:,} conversions at avg ${avg_value:.2f}")
    print(f"    → Meridian MMM    : {bots_removed:,} bot events removed from attribution model")
    print(f"    → Looker Studio   : refreshed dataset available at BigQuery destination")

    return df_export


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 58)
    print("  GA4 Enterprise Agent Architecture")
    print("  BigQuery & Vertex AI Pipeline Simulator")
    print(f"  GCP Project: {GCP_PROJECT_ID}")
    print("═" * 58)

    # Step 1: Generate raw mock data (replace with BigQuery in production)
    df_raw = generate_mock_ga4_data(N_ROWS)

    # Step 2: Vertex AI clustering (replace with endpoint.predict() in production)
    df_classified = vertex_ai_clustering(df_raw)

    # Step 3: Export cleaned signal payload
    df_cleaned = export_cleaned_data(df_classified)

    print("\n  [DONE] Pipeline complete.\n")
