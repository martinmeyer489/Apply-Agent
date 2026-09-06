# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — UC Function Tool Definitions
# MAGIC
# MAGIC Registers the 4 Unity Catalog Functions used as tools by the
# MAGIC Matching Agent (Req 7.2):
# MAGIC
# MAGIC - `job_agent.gold.search_listings` — semantic retrieval over the
# MAGIC   Vector Search index, capped at `min(max_results, 200)` (Req 4 AC7,
# MAGIC   7.4).
# MAGIC - `job_agent.gold.compute_commute_distance` — haversine great-circle
# MAGIC   distance in km, matching `src/utils/haversine.py` (Req 7.6).
# MAGIC - `job_agent.gold.get_user_profile` — reads `gold.user_profiles` by
# MAGIC   `profile_id` (Req 7.2).
# MAGIC - `job_agent.gold.draft_application` — generates a tailored cover
# MAGIC   letter via Foundation Model APIs (Req 8.1, 8.2, 8.3, 8.4, 8.6,
# MAGIC   8.8).
# MAGIC
# MAGIC Each function body is defined in this notebook as a Python string
# MAGIC and executed via `spark.sql(...)`, so re-running the notebook is
# MAGIC idempotent (`CREATE OR REPLACE FUNCTION`).
# MAGIC
# MAGIC **Prerequisite**: `notebooks/00_setup_catalog.py` (creates the
# MAGIC `gold` schema and `gold.user_profiles`/`silver.enriched_listings`
# MAGIC tables) and `notebooks/04_sync_vector_index.py` (creates the Vector
# MAGIC Search endpoint/index queried by `search_listings`) must have
# MAGIC already run.
# MAGIC
# MAGIC Requirements: 7.2, 7.4, 8.1, 8.2, 8.3, 8.4, 8.6, 8.8

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets

# COMMAND ----------

dbutils.widgets.text("catalog", "job_agent", "Catalog name")
dbutils.widgets.text("vs_endpoint_name", "job_agent_vs_endpoint", "Vector Search endpoint name")
dbutils.widgets.text("vs_index_name", "silver.enriched_listings_index", "Vector Search index name (schema.table, catalog prepended)")
dbutils.widgets.text("warehouse_id", "", "SQL warehouse ID used by draft_application to read listing details")
dbutils.widgets.text("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct", "Foundation Model APIs chat endpoint")

CATALOG = dbutils.widgets.get("catalog")
VS_ENDPOINT_NAME = dbutils.widgets.get("vs_endpoint_name")
VS_INDEX_NAME = f"{CATALOG}.{dbutils.widgets.get('vs_index_name')}"
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id").strip()
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")

if not WAREHOUSE_ID:
    raise ValueError(
        "The 'warehouse_id' widget must be set to the Free Edition 2X-Small SQL "
        "warehouse ID. draft_application uses it to read listing details via the "
        "Databricks SDK's statement execution API from inside the UC Function body."
    )

print(f"Catalog:            {CATALOG}")
print(f"Vector Search endpoint: {VS_ENDPOINT_NAME}")
print(f"Vector Search index:    {VS_INDEX_NAME}")
print(f"Warehouse ID:            {WAREHOUSE_ID}")
print(f"LLM endpoint:            {LLM_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. `search_listings` (Req 4.7, 7.4)
# MAGIC
# MAGIC Queries the Vector Search index via
# MAGIC `VectorSearchClient().get_index(...).similarity_search(...)` using
# MAGIC `query_text` (the agent supplies the user's skills, job title
# MAGIC history, and qualifications summary concatenated). `num_results` is
# MAGIC clamped to `min(max_results, 200)` so a caller can never request more
# MAGIC than the 200-candidate ceiling mandated for the index (Req 4 AC7).

# COMMAND ----------

search_listings_sql = f"""
CREATE OR REPLACE FUNCTION {CATALOG}.gold.search_listings(
    query_text STRING COMMENT 'Concatenated skills, job titles, and qualifications to search for',
    max_results INT DEFAULT 100 COMMENT 'Maximum number of candidates to return (capped at 100)'
)
RETURNS TABLE (
    listing_id STRING,
    job_title STRING,
    company_name STRING,
    latitude DOUBLE,
    longitude DOUBLE,
    enrichment_state STRING,
    similarity_score DOUBLE
)
COMMENT 'Searches the Vector Search index for job listings semantically similar to query_text. Returns at most min(max_results, 100) candidates ranked by similarity in descending order.'
RETURN
    SELECT
        listing_id,
        job_title,
        company_name,
        latitude,
        longitude,
        enrichment_state,
        search_score AS similarity_score
    FROM vector_search(
        index => '{VS_INDEX_NAME}',
        query_text => search_listings.query_text,
        -- Req 4 AC7 / vector_search() preview cap: the SQL vector_search
        -- function does not support num_results > 100, so clamp here.
        num_results => least(coalesce(search_listings.max_results, 100), 100)
    )
"""

spark.sql(search_listings_sql)
print(f"Registered {CATALOG}.gold.search_listings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. `compute_commute_distance` (Req 7.6)
# MAGIC
# MAGIC Haversine great-circle distance in kilometres, rounded to 1 decimal
# MAGIC place. The formula and rounding mirror `src/utils/haversine.py`
# MAGIC exactly (`R = 6371.0` km, `round(..., 1)`) so the UC Function and the
# MAGIC Python utility used in property tests (Property 18) always agree.

# COMMAND ----------

compute_commute_distance_sql = f"""
CREATE OR REPLACE FUNCTION {CATALOG}.gold.compute_commute_distance(
    lat1 DOUBLE COMMENT 'Latitude of the user home location in decimal degrees',
    lon1 DOUBLE COMMENT 'Longitude of the user home location in decimal degrees',
    lat2 DOUBLE COMMENT 'Latitude of the job listing location in decimal degrees',
    lon2 DOUBLE COMMENT 'Longitude of the job listing location in decimal degrees'
)
RETURNS DOUBLE
COMMENT 'Computes the great-circle (haversine) distance in kilometres between two geographic coordinate pairs, rounded to 1 decimal place. Mirrors src/utils/haversine.py.'
LANGUAGE PYTHON
AS $$
    import math

    R = 6371.0  # Earth radius in km — matches src/utils/haversine.py
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 1)
$$;
"""

spark.sql(compute_commute_distance_sql)
print(f"Registered {CATALOG}.gold.compute_commute_distance")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `get_user_profile` (Req 7.2)
# MAGIC
# MAGIC Reads `job_agent.gold.user_profiles` filtered by the supplied
# MAGIC `profile_id`. Note the design document's initial draft used
# MAGIC `WHERE profile_id = profile_id`, which is a self-comparison that
# MAGIC always evaluates true regardless of the argument — this refined
# MAGIC version qualifies the table column (`t.profile_id`) against the
# MAGIC function parameter (`profile_id`) so the filter is actually applied.

# COMMAND ----------

get_user_profile_sql = f"""
CREATE OR REPLACE FUNCTION {CATALOG}.gold.get_user_profile(
    profile_id STRING COMMENT 'The session-scoped user profile identifier'
)
RETURNS TABLE (
    skills ARRAY<STRING>,
    years_of_experience INT,
    job_title_history ARRAY<STRING>,
    qualifications_summary STRING,
    home_latitude DOUBLE,
    home_longitude DOUBLE,
    home_location_name STRING,
    commute_radius_km INT
)
COMMENT 'Retrieves the user profile including parsed CV fields, resolved home coordinates, and commute radius, filtered by profile_id.'
LANGUAGE SQL
AS $$
    SELECT t.skills, t.years_of_experience, t.job_title_history, t.qualifications_summary,
           t.home_latitude, t.home_longitude, t.home_location_name, t.commute_radius_km
    FROM {CATALOG}.gold.user_profiles AS t
    WHERE t.profile_id = get_user_profile.profile_id
$$;
"""

spark.sql(get_user_profile_sql)
print(f"Registered {CATALOG}.gold.get_user_profile")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. `draft_application` (Req 8.1, 8.2, 8.3, 8.4, 8.6, 8.8)
# MAGIC
# MAGIC 1. **Profile validation (Req 8 AC8)** — if `user_skills` and
# MAGIC    `user_job_titles` are both empty/null, returns an error string
# MAGIC    stating the profile lacks sufficient content and generates no
# MAGIC    cover letter. No listing lookup or LLM call is made in that case.
# MAGIC 2. Otherwise, fetches the listing's `job_title`, `company_name`,
# MAGIC    `job_description`, and `required_skills_text` from
# MAGIC    `silver.enriched_listings` via
# MAGIC    `WorkspaceClient().sql.execute_statement(...)` against the
# MAGIC    Free Edition SQL warehouse.
# MAGIC 3. Calls Foundation Model APIs
# MAGIC    (`databricks-meta-llama-3-3-70b-instruct`) via
# MAGIC    `WorkspaceClient().serving_endpoints.query(...)` with a prompt
# MAGIC    requesting a 200-500 word cover letter that references the
# MAGIC    listing and references at least one skill/title/qualification
# MAGIC    from the profile (Req 8 AC3, AC4).
# MAGIC 4. Counts words in the response; if the count falls outside
# MAGIC    [200, 500], re-prompts **once** with an explicit instruction to
# MAGIC    hit the target length (Req 8 AC6 / design "re-prompt once if out
# MAGIC    of range"), then returns whichever response resulted — the
# MAGIC    original if it was in range, otherwise the response from the
# MAGIC    single re-prompt attempt, regardless of its length.
# MAGIC
# MAGIC **30-second completion note (Req 8 AC6):** Unity Catalog Functions
# MAGIC execute in a sandboxed, single-invocation Python runtime that does
# MAGIC not expose OS-level signal/alarm primitives or a supported way to
# MAGIC hard-abort a call already in flight to `serving_endpoints.query`.
# MAGIC Enforcing the 30-second budget inside the function body is
# MAGIC therefore **best-effort**: the two Foundation Model API calls (the
# MAGIC initial draft and, in the worst case, a single re-prompt) rely on the
# MAGIC Databricks SDK's own client-side request timeout, and the function
# MAGIC issues at most one such re-prompt so that in practice two LLM round
# MAGIC trips is the maximum amount of work performed per invocation. A hard,
# MAGIC OS-level deadline across the whole function body is not achievable
# MAGIC from within a UC Function's Python body with the SDK alone; achieving
# MAGIC a hard timeout would require an external caller-side deadline (e.g.
# MAGIC the Matching Agent enforcing its own 60s tool-call budget per Req
# MAGIC 7 AC11).

# COMMAND ----------

draft_application_sql = f"""
CREATE OR REPLACE FUNCTION {CATALOG}.gold.draft_application(
    listing_id STRING COMMENT 'The stable listing identifier of the target job',
    user_skills ARRAY<STRING> COMMENT 'The skills extracted from the user CV',
    user_job_titles ARRAY<STRING> COMMENT 'The job title history from the user CV',
    user_qualifications_summary STRING COMMENT 'The qualifications summary from the user CV',
    user_years_of_experience INT COMMENT 'Years of experience from the user CV'
)
RETURNS STRING
COMMENT 'Generates a tailored cover letter (200-500 words) for the specified job listing using the user profile. Returns the cover letter text, or an error message if the profile lacks sufficient content (Req 8 AC8). Best-effort 30s budget: see notebook 06 comments.'
LANGUAGE PYTHON
AS $$
    # Req 8 AC8: reject before any listing lookup or LLM call if the
    # profile has zero skills AND zero job title history entries.
    if not user_skills and not user_job_titles:
        return (
            "ERROR: Profile lacks sufficient content for drafting. "
            "Please upload a CV with at least one skill or job title before requesting a draft."
        )

    import re
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    warehouse_id = "{WAREHOUSE_ID}"

    # Fetch listing details (Req 8 AC2). Values are escaped for the
    # single-quoted SQL literal to guard against embedded apostrophes in
    # listing_id (listing_id is a SHA-256 hex digest, but escape defensively).
    safe_listing_id = listing_id.replace("'", "''")
    listing_statement = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=(
            "SELECT job_title, company_name, job_description, required_skills_text "
            "FROM {CATALOG}.silver.enriched_listings "
            f"WHERE listing_id = '{{safe_listing_id}}'"
        ),
        wait_timeout="30s",
    )

    result_rows = (listing_statement.result.data_array or []) if listing_statement.result else []
    if not result_rows:
        return f"ERROR: Listing '{{listing_id}}' was not found."

    job_title, company_name, job_description, required_skills_text = result_rows[0]

    def build_prompt(extra_instruction: str = "") -> str:
        return (
            "Write a cover letter (200-500 words) for the following job. "
            "Include at least 1 skill, prior job title, or qualification from the "
            "applicant's profile that matches the job requirements.\\n\\n"
            f"Job: {{job_title}} at {{company_name}}\\n"
            f"Description: {{job_description or ''}}\\n"
            f"Required skills: {{required_skills_text or ''}}\\n\\n"
            "Applicant profile:\\n"
            f"Skills: {{', '.join(user_skills or [])}}\\n"
            f"Experience: {{user_years_of_experience if user_years_of_experience is not None else 'unknown'}} years\\n"
            f"Recent roles: {{', '.join(user_job_titles or [])}}\\n"
            f"Summary: {{user_qualifications_summary or ''}}\\n"
            f"{{extra_instruction}}"
        )

    def call_llm(prompt: str) -> str:
        response = w.serving_endpoints.query(
            name="{LLM_ENDPOINT}",
            messages=[{{"role": "user", "content": prompt}}],
        )
        return response.choices[0].message.content

    def word_count(text: str) -> int:
        return len(re.findall(r"\\S+", text or ""))

    # First attempt (Req 8 AC3, AC4).
    cover_letter = call_llm(build_prompt())
    count = word_count(cover_letter)

    # Req 8 AC6 / design: re-prompt at most once if outside [200, 500] words.
    if not (200 <= count <= 500):
        retry_instruction = (
            "\\nIMPORTANT: Your previous draft had "
            f"{{count}} words. Rewrite the cover letter so its total length is "
            "strictly between 200 and 500 words inclusive."
        )
        cover_letter = call_llm(build_prompt(retry_instruction))

    return cover_letter
$$;
"""

spark.sql(draft_application_sql)
print(f"Registered {CATALOG}.gold.draft_application")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification

# COMMAND ----------

registered_functions = [
    f"{CATALOG}.gold.search_listings",
    f"{CATALOG}.gold.compute_commute_distance",
    f"{CATALOG}.gold.get_user_profile",
    f"{CATALOG}.gold.draft_application",
]

print("Registered UC Functions:")
for fn in registered_functions:
    print(f"  - {fn}")
    display(spark.sql(f"DESCRIBE FUNCTION EXTENDED {fn}"))
