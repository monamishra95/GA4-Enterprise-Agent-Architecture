# GA4 Enterprise Agent Architecture - Measuring how agents & bots engage with your business

** Product Architecture Brief**

> Simulating a server-side GA4 data pipeline for AI agent detection, BigQuery/Vertex AI signal cleaning, Value-Based Bidding integration, and Meridian MMM incrementality calibration.

---

## The Problem This Solves

LLM scrapers, ad fraud bots, and AI agents now account for a significant share of measured web traffic. They trigger GA4 tags, inflate conversion counts, and corrupt the signals that bidding algorithms rely on. The result: Google Ads Smart Bidding model could be learning from noise, the advertiser's Meridian MMM is attributing lift to bots, and advertiser CPA reports could be made more accurate & complete.

This architecture demonstrates how enterprise teams can fix this at the infrastructure layer, not the reporting layer.

---

## Architecture Overview

```
User / Bot / LLM Agent
        │
        ▼
┌─────────────────────────────────────┐
│  Cloud Armor + reCAPTCHA Enterprise │  ← Server-side edge scoring (0.0–1.0)
│  (Edge Score assigned per request)  │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│        Server-Side GTM              │  ← Edge Score injected as custom variable
│  (No client-side JS tag execution)  │  ← GA4 Measurement Protocol fires server-side
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         GA4 → BigQuery Export       │  ← Raw event stream (1:1 hit-level data)
│     (all hits, including bots)      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│    Vertex AI Clustering Pipeline    │  ← ML model classifies: Human / LLM_Scraper / Ad_Fraud
│    (Isolation Forest / K-Means)     │
└──────────┬──────────────────────────┘
           │
     ┌─────┴──────────────────┐
     ▼                        ▼
┌──────────────┐    ┌──────────────────────┐
│  Google Ads  │    │    Meridian MMM       │
│  VBB Upload  │    │  Incrementality       │
│  $0 for bots │    │  Calibration          │
└──────────────┘    └──────────────────────┘
```

---

## The V2 Strategy: Gaps in Client-side detection

### The Old Approach and its Deficiencies

Traditional bot detection relied on client-side JavaScript: loading a detection library in the browser, checking mouse movement, fingerprinting the device, and flagging suspicious sessions after the fact. This approach has three deficiencies:

**Core Web Vitals penalty.** Every third-party JS tag adds render-blocking weight to your page. Detection libraries routinely add 80–200ms to Time to Interactive — which leads to a direct hit to an advertiser's Google Search ranking and Quality Score.

**Ad blocker bypass rate exceeds 40%.** The broswer can block any detection that runs in it. uBlock Origin, Privacy Badger, and enterprise network proxies strip client-side detection tags before they execute. 

**LLM agents don't run JavaScript.** GPTBot, ClaudeBot, and most production AI scrapers use `domcontentloaded`-only page fetches. They never execute GA4 gtag.js, so they never appear in your client-side analytics at all. This means they are absent in the data. That's worse than seeing them as bots. You can't exclude what you can't measure.

### The V2 Approach: Server-Side Edge Detection

The correct architecture moves detection upstream, to the network edge, before the request ever reaches your application:

**Cloud Armor** evaluates each request against behavioral signatures (IP reputation, request cadence, header anomalies) and attaches a risk score to the request header.

**reCAPTCHA Enterprise** provides a 0.0–1.0 token score that can be validated server-side on every page load — no client JS required for the scoring itself.

**Server-Side GTM** reads this score as a custom variable and fires GA4 Measurement Protocol events server-to-server. The GA4 hit is sent with `traffic_type: internal` for bots (filtered in GA4 UI) and `conversion_value: 0` for the VBB signal.

The result: 100% hit coverage regardless of ad blockers, zero Core Web Vitals impact, and detection signals that LLM agents cannot evade because the scoring happens before the TCP connection is established.

---

## The BigQuery / Vertex AI Moat

### Why GA4's Raw BigQuery Export Is a Strategic Asset

Most analytics teams currently use GA4 as a reporting tool. The V2 architecture treats it as a data warehouse input. GA4's BigQuery export gives an advertiser the hit-level, unsampled, real-time event data — every page_view, scroll_depth, session_start, and conversion, with all event parameters intact. This raw stream is the foundation of your ML moat:

**Feature engineering that GA4 UI never exposes.** Session-level features like `event_velocity_per_sec`, `time_between_events`, and `scroll_depth_vs_session_duration` are trivially computed in BigQuery SQL but invisible in the GA4 interface. These features are your strongest bot detection signals.

**Your own labeled training data.** After running this pipeline for 30 days, you have a labeled dataset of Human / LLM_Scraper / Ad_Fraud events tied to real business outcomes. No vendor can replicate this. It is specific to your traffic patterns, your customer profiles, and your conversion funnel.

**Vertex AI Pipelines for retraining.** Schedule a weekly pipeline that pulls the latest BigQuery export, retrains the clustering model on fresh data, and redeploys the endpoint — fully automated. Your detection model improves as the bot landscape evolves.

### The Clustering Model

The simulation uses an interpretable rule hierarchy that mirrors what a production Isolation Forest or K-Means model learns from these four primary features:

| Feature | Human | LLM Scraper | Ad Fraud |
|---|---|---|---|
| Edge Score | > 0.65 | 0.15 – 0.48 | < 0.20 |
| Event Velocity (hits/sec) | 1 – 5 | 8 – 35 | 30 – 80 |
| Session Duration | 45 – 600s | 0 – 6s | 0 – 3s |
| Mouse Move Events | 20 – 250 | 0 | 0 |

In production, replace the rule-based `classify_row()` function in `bq_vertex_pipeline.py` with a call to your Vertex AI endpoint. The labeled output of this simulation serves as your initial training set.

---

## Value-Based Bidding Integration

### The Strategic Shift: From Manual Exclusions to Algorithmic $0 Bidding

The conventional approach to bot traffic in paid search is audience exclusion: identify bot IPs, create exclusion lists, upload them to Google Ads, and hope the lists stay current. This is manual, reactive, and fundamentally incomplete — by the time you've identified a bot IP range, it has rotated.

The V2 approach is different: **let the bots click, but assign them $0 conversion value.**

Google Ads Smart Bidding optimizes toward conversion value, not conversion volume. When a bot click results in a `conversion_value = $0` upload, the algorithm learns that traffic pattern is worthless and reduces its bid for similar future traffic — automatically, continuously, without any exclusion list management.

**The mechanism:**

1. Vertex AI classifies the GA4 hit as Human / Bot.
2. For human hits: upload an Enhanced Conversion with `conversion_value = $150` (or your actual LTV).
3. For bot hits: upload the same Enhanced Conversion event with `conversion_value = $0.00`.
4. Smart Bidding ingests both signals and adjusts Target ROAS bids accordingly.

**The compounding benefit:** as your model improves accuracy over time, the bidding signal gets cleaner. Your CPA falls not because you're excluding traffic, but because the algorithm is learning the true value of each traffic source at a granularity no human-managed exclusion list could match.

---

## Meridian MMM & Incrementality Calibration


Media Mix Modeling (MMM) measures the incremental lift each channel contributes to business outcomes. The problem: if bots are triggering your GA4 conversion tags, they inflate the measured lift for every channel they interact with.

In the simulation, Meta Ads raw lift reads at **45%** — a compelling number. After Meridian calibration (removing bot-inflated conversion signals), it drops to **17%**. Google Ads holds steady at **36%** because its traffic profile contains proportionally fewer bot interactions.

The implication for budget allocation is significant: the raw data suggests Meta Ads is your highest-performing channel. The calibrated data suggests the opposite.

**Meridian receives the cleaned BigQuery export** — human-only sessions with verified conversion values — rather than the raw GA4 stream. This ensures the MMM model is estimating true human response to advertising, not a mix of human behavior and automated crawling.

---

## Repository Structure

```
GA4-Enterprise-Agent-Architecture/
├── docs/
│   └── index.html              # Command Center dashboard (zero-build, open directly)
├── scripts/
│   ├── traffic_generator.py    # Playwright traffic simulator (human + bot)
│   └── bq_vertex_pipeline.py   # BigQuery / Vertex AI pipeline simulator
├── data/                       # Auto-created by bq_vertex_pipeline.py
│   ├── raw_ga4_events.csv      # 1,000 synthetic GA4 hits
│   └── cleaned_ga4_events.csv  # Human-only payload for VBB / Meridian
└── README.md
```

---

## How to Run the Simulation

### 1. Command Center Dashboard (No build required)

Simply open `docs/index.html` in any modern browser:

```bash
# macOS / Linux
open docs/index.html

# Windows
start docs/index.html
```

The dashboard will:
- Auto-run the Edge Detection Simulator on load (check the browser console for raw GA4 MP payloads)
- Stream live VBB click events every 2 seconds
- Show the Vertex AI cleaning chart with this week's mock traffic data
- Allow toggling between Raw and Calibrated Meridian views

### 2. BigQuery / Vertex AI Pipeline

The pipeline supports two modes:

**BigQuery mode** (default) — queries the publicly available GA4 Obfuscated Sample Ecommerce dataset (`bigquery-public-data.ga4_obfuscated_sample_ecommerce`). Real session data from the Google Merchandise Store, Nov 2020 – Jan 2021. First 1 TB/month is free under the GCP free tier.

**Mock mode** (`--mock`) — generates synthetic data with no GCP credentials required.

```bash
# Install dependencies
pip install pandas numpy google-cloud-bigquery db-dtypes

# One-time GCP setup (BigQuery mode only)
# 1. Create a free GCP project at https://console.cloud.google.com
# 2. Set your project ID at the top of bq_vertex_pipeline.py, or:
export GCP_PROJECT_ID=your-gcp-project-id
# 3. Authenticate
gcloud auth application-default login

# Run with real GA4 public data (BigQuery mode)
python scripts/bq_vertex_pipeline.py

# Run offline with synthetic data (no GCP needed)
python scripts/bq_vertex_pipeline.py --mock
```

Output: `data/raw_ga4_events.csv` and `data/cleaned_ga4_events.csv`

> **Note:** `edge_score` (Cloud Armor) and `mouse_move_events` (client-side JS) are not part of the GA4 BigQuery schema — they are simulated from real session signals in both modes and labelled `SIMULATED` in the CSV output.

### 3. Playwright Traffic Spawner

```bash
# Install dependencies
pip install playwright
playwright install chromium

# Start a local file server first (required for Playwright to load the HTML)
python -m http.server 8080

# In a new terminal, run the traffic spawner
python scripts/traffic_generator.py
```

A Chromium window will open and run 50 sessions visibly — alternating between human browsing behavior and instant LLM scraper patterns. Intended for live demo use.

---

## Production Deployment Notes

| Component | This Repo | Production Replacement |
|---|---|---|
| Edge Scoring | `Math.random()` in JS dashboard | Cloud Armor + reCAPTCHA Enterprise API |
| GA4 Data Source | **Real** — `bigquery-public-data.ga4_obfuscated_sample_ecommerce` (or `--mock` for synthetic) | Your GA4 BigQuery export (`events_*`) |
| ML Classification | Rule-based Python (mirrors Isolation Forest logic) | Vertex AI endpoint (`endpoint.predict()`) |
| Conversion Upload | Console log | Google Ads Enhanced Conversions API |
| MMM Input | Cleaned CSV (human-only sessions) | Meridian via Vertex AI Pipelines |
| Scheduler | Manual script run | Cloud Scheduler + Cloud Run |

No real GCP credentials, GA4 Measurement IDs, or API keys are used anywhere in this repository. All placeholders follow the `YOUR_GCP_PROJECT_ID` convention.

---

## Skills Demonstrated

- **Google Analytics 4** — Measurement Protocol, BigQuery export schema, Enhanced Conversions
- **Google Cloud Platform** — Cloud Armor, reCAPTCHA Enterprise, Vertex AI, BigQuery, Server-Side GTM
- **Python** — pandas, numpy, scikit-learn, Playwright async automation
- **Marketing Science** — Media Mix Modeling (Meridian), Value-Based Bidding, incrementality testing
- **Frontend** — Zero-dependency dark-mode dashboard (HTML + Tailwind CDN + Chart.js)
