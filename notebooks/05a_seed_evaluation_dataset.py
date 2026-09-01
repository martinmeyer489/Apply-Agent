# Databricks notebook source
# MAGIC %md
# MAGIC # 05a — Seed the Agent Evaluation Dataset
# MAGIC
# MAGIC Populates `job_agent.ops.evaluation_dataset` with at least 20 labelled
# MAGIC matching cases used by the evaluation runner
# MAGIC (`notebooks/05_run_evaluation.py`) to score the Matching Agent with
# MAGIC MLflow LLM-as-judge scorers (Requirement 9 AC2, AC3).
# MAGIC
# MAGIC Each labelled case pairs a synthetic `User_Profile` (skills, resolved
# MAGIC home location, commute radius) with the `expected_listing_ids` a
# MAGIC well-behaved Matching Agent should surface for that profile. Because
# MAGIC there is no live matching history to mine "real" expected results
# MAGIC from, the expected listing IDs reference plausible matches from the
# MAGIC bundled fallback dataset (`data/bundled_fallback/bundled_listings.csv`)
# MAGIC whose job title and location line up with the profile — derived with
# MAGIC the same `derive_listing_id(source_url)` helper used by the ingestion
# MAGIC pipeline, so the IDs are stable and reproducible.
# MAGIC
# MAGIC Cases intentionally vary across:
# MAGIC - **Skill sets**: data engineering, product management, frontend,
# MAGIC   backend/full-stack, ML/data science, DevOps/SRE, security, mobile,
# MAGIC   QA, and design.
# MAGIC - **Locations**: a mix of European cities (Berlin, Munich, London,
# MAGIC   Manchester, Lyon, Madrid, Amsterdam, Dublin, Zurich, Warsaw) and US
# MAGIC   cities (New York, San Francisco, Austin, Seattle, Boston, Chicago).
# MAGIC - **Commute radii**: from narrow (10-15 km) to wide (150-200 km).
# MAGIC
# MAGIC This notebook is idempotent: it always overwrites
# MAGIC `job_agent.ops.evaluation_dataset` with the canonical seed set defined
# MAGIC below, so re-running it resets the table rather than accumulating
# MAGIC duplicate or drifted rows.
# MAGIC
# MAGIC **Prerequisite**: `notebooks/00_setup_catalog.py` must have already
# MAGIC created the `job_agent` catalog, the `ops` schema, and the empty
# MAGIC `ops.evaluation_dataset` Delta table.
# MAGIC
# MAGIC Requirements: 9.2

# COMMAND ----------

dbutils.widgets.text("catalog", "job_agent", "Catalog name")
dbutils.widgets.text("schema", "ops", "Schema name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TABLE_NAME = f"{CATALOG}.{SCHEMA}.evaluation_dataset"

print(f"Target table: {TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Define the labelled evaluation cases
# MAGIC
# MAGIC Each case is a dict with `case_id`, `user_profile` (a dict serialized
# MAGIC to JSON for the `user_profile` STRING column), `expected_listing_ids`,
# MAGIC and a human-readable `description`. Expected listing IDs are derived
# MAGIC from `source_url` values in the bundled fallback dataset via
# MAGIC `derive_listing_id`, picking listings whose title/location match the
# MAGIC profile's intent.

# COMMAND ----------

import sys

sys.path.append("../src")

from utils.listing_id import derive_listing_id  # noqa: E402

# Source URLs pulled from data/bundled_fallback/bundled_listings.csv whose
# job title and location_text are a good semantic/geographic match for the
# corresponding profile below. Referencing the URLs (rather than hardcoding
# the pre-computed hex digests) keeps this notebook self-documenting and
# resilient to any future re-generation of the bundled dataset.
BUNDLED_URLS = {
    "senior_data_engineer_berlin": "https://bundled-fallback.local/jobs/0000",
    "data_engineer_munich": "https://bundled-fallback.local/jobs/0001",
    "data_engineer_munich_2": "https://bundled-fallback.local/jobs/0041",
    "product_manager_hamburg": "https://bundled-fallback.local/jobs/0002",
    "senior_product_manager_london": "https://bundled-fallback.local/jobs/0003",
    "senior_product_manager_london_2": "https://bundled-fallback.local/jobs/0043",
    "frontend_developer_manchester": "https://bundled-fallback.local/jobs/0004",
    "frontend_developer_manchester_2": "https://bundled-fallback.local/jobs/0044",
    "backend_developer_edinburgh": "https://bundled-fallback.local/jobs/0005",
    "backend_developer_edinburgh_2": "https://bundled-fallback.local/jobs/0045",
    "full_stack_developer_paris": "https://bundled-fallback.local/jobs/0006",
    "ml_engineer_lyon": "https://bundled-fallback.local/jobs/0007",
    "senior_ml_engineer_madrid": "https://bundled-fallback.local/jobs/0008",
    "devops_engineer_barcelona": "https://bundled-fallback.local/jobs/0009",
    "sre_amsterdam": "https://bundled-fallback.local/jobs/0010",
    "data_scientist_rotterdam": "https://bundled-fallback.local/jobs/0011",
    "senior_data_scientist_dublin": "https://bundled-fallback.local/jobs/0012",
    "cloud_architect_zurich": "https://bundled-fallback.local/jobs/0014",
    "ux_designer_warsaw": "https://bundled-fallback.local/jobs/0020",
    "ui_ux_designer_krakow": "https://bundled-fallback.local/jobs/0021",
    "security_engineer_new_york": "https://bundled-fallback.local/jobs/0028",
    "mobile_ios_san_francisco": "https://bundled-fallback.local/jobs/0029",
    "mobile_android_austin": "https://bundled-fallback.local/jobs/0030",
    "dba_seattle": "https://bundled-fallback.local/jobs/0031",
    "solutions_consultant_boston": "https://bundled-fallback.local/jobs/0032",
    "customer_success_chicago": "https://bundled-fallback.local/jobs/0033",
}

EXPECTED = {key: derive_listing_id(url) for key, url in BUNDLED_URLS.items()}


def _profile(skills, location, commute_radius_km):
    return {
        "skills": skills,
        "location": location,
        "commute_radius_km": commute_radius_km,
    }


CASES = [
    {
        "case_id": "case_001",
        "user_profile": _profile(
            ["Python", "SQL", "Apache Spark", "Delta Lake", "Airflow"],
            "Berlin",
            30,
        ),
        "expected_listing_ids": [EXPECTED["senior_data_engineer_berlin"]],
        "description": "Berlin-based senior data engineer profile matching relevant local listings.",
    },
    {
        "case_id": "case_002",
        "user_profile": _profile(
            ["Python", "distributed systems", "Kafka", "Delta Lake"],
            "Munich",
            50,
        ),
        "expected_listing_ids": [
            EXPECTED["data_engineer_munich"],
            EXPECTED["data_engineer_munich_2"],
        ],
        "description": "Munich data engineer profile with a wide-enough radius to catch duplicate postings.",
    },
    {
        "case_id": "case_003",
        "user_profile": _profile(
            ["product strategy", "roadmap planning", "stakeholder management", "cloud infrastructure"],
            "Hamburg",
            10,
        ),
        "expected_listing_ids": [EXPECTED["product_manager_hamburg"]],
        "description": "Hamburg product manager profile with a narrow 10km commute radius.",
    },
    {
        "case_id": "case_004",
        "user_profile": _profile(
            ["product management", "machine learning models", "on-call leadership", "technical roadmapping"],
            "London",
            25,
        ),
        "expected_listing_ids": [
            EXPECTED["senior_product_manager_london"],
            EXPECTED["senior_product_manager_london_2"],
        ],
        "description": "London senior product manager profile matching two similar postings.",
    },
    {
        "case_id": "case_005",
        "user_profile": _profile(
            ["JavaScript", "React", "TypeScript", "modern JavaScript frameworks", "CSS"],
            "Manchester",
            15,
        ),
        "expected_listing_ids": [
            EXPECTED["frontend_developer_manchester"],
            EXPECTED["frontend_developer_manchester_2"],
        ],
        "description": "Manchester frontend developer profile with a tight 15km commute radius.",
    },
    {
        "case_id": "case_006",
        "user_profile": _profile(
            ["Node.js", "REST APIs", "CI/CD automation", "PostgreSQL"],
            "Edinburgh",
            40,
        ),
        "expected_listing_ids": [
            EXPECTED["backend_developer_edinburgh"],
            EXPECTED["backend_developer_edinburgh_2"],
        ],
        "description": "Edinburgh backend developer profile covering CI/CD and API skills.",
    },
    {
        "case_id": "case_007",
        "user_profile": _profile(
            ["full stack development", "data warehousing", "SQL", "React", "Node.js"],
            "Paris",
            60,
        ),
        "expected_listing_ids": [EXPECTED["full_stack_developer_paris"]],
        "description": "Paris full-stack developer profile with a moderate 60km commute radius.",
    },
    {
        "case_id": "case_008",
        "user_profile": _profile(
            ["machine learning", "REST and GraphQL APIs", "Python", "MLOps"],
            "Lyon",
            20,
        ),
        "expected_listing_ids": [EXPECTED["ml_engineer_lyon"]],
        "description": "Lyon machine learning engineer profile with a narrow 20km commute radius.",
    },
    {
        "case_id": "case_009",
        "user_profile": _profile(
            ["deep learning", "containerized microservices", "Kubernetes", "PyTorch"],
            "Madrid",
            35,
        ),
        "expected_listing_ids": [EXPECTED["senior_ml_engineer_madrid"]],
        "description": "Madrid senior ML engineer profile emphasising containerized deployment skills.",
    },
    {
        "case_id": "case_010",
        "user_profile": _profile(
            ["DevOps", "stakeholder communication", "Terraform", "CI/CD"],
            "Barcelona",
            45,
        ),
        "expected_listing_ids": [EXPECTED["devops_engineer_barcelona"]],
        "description": "Barcelona DevOps engineer profile with cross-team communication focus.",
    },
    {
        "case_id": "case_011",
        "user_profile": _profile(
            ["site reliability", "agile delivery practices", "observability", "incident response"],
            "Amsterdam",
            150,
        ),
        "expected_listing_ids": [EXPECTED["sre_amsterdam"]],
        "description": "Amsterdam SRE profile with a wide 150km commute radius to test broad recall.",
    },
    {
        "case_id": "case_012",
        "user_profile": _profile(
            ["data science", "data visualization and reporting", "Python", "statistics"],
            "Rotterdam",
            25,
        ),
        "expected_listing_ids": [EXPECTED["data_scientist_rotterdam"]],
        "description": "Rotterdam data scientist profile focused on reporting and visualization.",
    },
    {
        "case_id": "case_013",
        "user_profile": _profile(
            ["Spark", "Delta Lake", "advanced statistics", "experiment design"],
            "Dublin",
            50,
        ),
        "expected_listing_ids": [EXPECTED["senior_data_scientist_dublin"]],
        "description": "Dublin senior data scientist profile with Spark/Delta Lake emphasis.",
    },
    {
        "case_id": "case_014",
        "user_profile": _profile(
            ["cloud architecture", "A/B testing and experimentation", "AWS", "system design"],
            "Zurich",
            30,
        ),
        "expected_listing_ids": [EXPECTED["cloud_architect_zurich"]],
        "description": "Zurich cloud solutions architect profile with experimentation platform experience.",
    },
    {
        "case_id": "case_015",
        "user_profile": _profile(
            ["UX research", "CI/CD automation", "Figma", "usability testing"],
            "Warsaw",
            15,
        ),
        "expected_listing_ids": [EXPECTED["ux_designer_warsaw"]],
        "description": "Warsaw UX designer profile with a narrow 15km commute radius.",
    },
    {
        "case_id": "case_016",
        "user_profile": _profile(
            ["UI design", "large-scale data warehousing", "prototyping", "design systems"],
            "Krakow",
            20,
        ),
        "expected_listing_ids": [EXPECTED["ui_ux_designer_krakow"]],
        "description": "Krakow UI/UX designer profile emphasising design systems work.",
    },
    {
        "case_id": "case_017",
        "user_profile": _profile(
            ["application security", "Kubernetes orchestration", "threat modeling", "Python"],
            "New York",
            40,
        ),
        "expected_listing_ids": [EXPECTED["security_engineer_new_york"]],
        "description": "New York security engineer profile matching a local security-focused listing.",
    },
    {
        "case_id": "case_018",
        "user_profile": _profile(
            ["iOS development", "Swift", "A/B testing and experimentation", "mobile architecture"],
            "San Francisco",
            25,
        ),
        "expected_listing_ids": [EXPECTED["mobile_ios_san_francisco"]],
        "description": "San Francisco iOS developer profile with a moderate commute radius.",
    },
    {
        "case_id": "case_019",
        "user_profile": _profile(
            ["Android development", "Kotlin", "Python and SQL", "mobile CI/CD"],
            "Austin",
            200,
        ),
        "expected_listing_ids": [EXPECTED["mobile_android_austin"]],
        "description": "Austin Android developer profile with the maximum 200km commute radius.",
    },
    {
        "case_id": "case_020",
        "user_profile": _profile(
            ["database administration", "distributed data pipelines", "SQL tuning", "backup and recovery"],
            "Seattle",
            35,
        ),
        "expected_listing_ids": [EXPECTED["dba_seattle"]],
        "description": "Seattle database administrator profile matching a local DBA listing.",
    },
    {
        "case_id": "case_021",
        "user_profile": _profile(
            ["solutions consulting", "cloud infrastructure", "customer-facing technical sales", "SaaS"],
            "Boston",
            180,
        ),
        "expected_listing_ids": [EXPECTED["solutions_consultant_boston"]],
        "description": "Boston solutions consultant profile with a wide 180km commute radius.",
    },
    {
        "case_id": "case_022",
        "user_profile": _profile(
            ["customer success", "machine learning models", "account management", "onboarding"],
            "Chicago",
            1,
        ),
        "expected_listing_ids": [EXPECTED["customer_success_chicago"]],
        "description": "Chicago customer success manager profile with the minimum 1km commute radius.",
    },
]

print(f"Defined {len(CASES)} labelled evaluation cases")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Validate the cases before writing
# MAGIC
# MAGIC Sanity-checks case count, required keys, JSON-serializability of the
# MAGIC user profile, commute radius bounds (Requirement 6.3: 1-200 km
# MAGIC inclusive), and `case_id` uniqueness ahead of the Delta write.

# COMMAND ----------

import json

REQUIRED_KEYS = {"case_id", "user_profile", "expected_listing_ids", "description"}

assert len(CASES) >= 20, f"Expected at least 20 evaluation cases, found {len(CASES)}"

case_ids = [case["case_id"] for case in CASES]
assert len(case_ids) == len(set(case_ids)), "Duplicate case_id values found in CASES"

for case in CASES:
    missing = REQUIRED_KEYS - set(case.keys())
    assert not missing, f"Case '{case.get('case_id')}' is missing keys: {missing}"

    profile = case["user_profile"]
    assert isinstance(profile, dict), f"Case '{case['case_id']}' user_profile must be a dict before serialization"

    radius = profile["commute_radius_km"]
    assert 1 <= radius <= 200, (
        f"Case '{case['case_id']}' commute_radius_km={radius} is outside the 1-200 km range"
    )

    assert profile["skills"], f"Case '{case['case_id']}' must have a non-empty skills list"
    assert profile["location"], f"Case '{case['case_id']}' must have a non-empty location"
    assert case["expected_listing_ids"], f"Case '{case['case_id']}' must have at least 1 expected listing ID"

print(f"Validated {len(CASES)} cases: unique case_ids, required keys present, radii in [1, 200]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Build rows and write to `ops.evaluation_dataset`
# MAGIC
# MAGIC Serializes each `user_profile` dict to a JSON string (matching the
# MAGIC `user_profile` STRING column in `OPS_EVALUATION_DATASET_SCHEMA`) and
# MAGIC overwrites the target Delta table so re-running this notebook resets
# MAGIC the table to the canonical seed set rather than accumulating rows.

# COMMAND ----------

from models.schemas import OPS_EVALUATION_DATASET_SCHEMA  # noqa: E402

rows = [
    (
        case["case_id"],
        json.dumps(case["user_profile"]),
        case["expected_listing_ids"],
        case["description"],
    )
    for case in CASES
]

# Re-validate that the serialized JSON round-trips cleanly before writing.
for case_id, user_profile_json, _, _ in rows:
    parsed = json.loads(user_profile_json)
    assert 1 <= parsed["commute_radius_km"] <= 200, (
        f"Case '{case_id}' serialized commute_radius_km out of range after round-trip"
    )

eval_df = spark.createDataFrame(rows, schema=OPS_EVALUATION_DATASET_SCHEMA)

row_count = eval_df.count()
print(f"Built {row_count} rows for {TABLE_NAME}")

# COMMAND ----------

(
    eval_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

print(f"Loaded {row_count} rows into {TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Validate the loaded table

# COMMAND ----------

loaded_df = spark.table(TABLE_NAME)
loaded_count = loaded_df.count()

distinct_case_ids = loaded_df.select("case_id").distinct().count()

print(f"{TABLE_NAME}: {loaded_count} rows, {distinct_case_ids} distinct case_ids")

assert loaded_count >= 20, f"Expected >= 20 rows in {TABLE_NAME}, found {loaded_count}"
assert distinct_case_ids == loaded_count, "Found duplicate case_id values in the loaded table"

display(loaded_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

summary = {
    "table": TABLE_NAME,
    "cases_loaded": loaded_count,
}
print(summary)
