"""In-memory Python dataclass equivalents of the Unity Catalog table schemas.

These mirror `src/models/schemas.py` field-for-field but are plain Python
dataclasses (no PySpark dependency) for use in pipeline notebooks, UC
Function tools, the agent, and the Gradio app where a lightweight in-memory
representation is more convenient than a Spark Row/DataFrame.

Requirements: 2.2, 3.1, 4.4, 11.2
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# ---------------------------------------------------------------------------
# Bronze tier
# ---------------------------------------------------------------------------


@dataclass
class JobListing:
    """A single job posting record in `bronze.job_listings`."""

    listing_id: str
    job_title: str
    company_name: str
    source_url: str
    ingestion_timestamp: datetime
    ingestion_mode: str
    source_domain: str
    job_description: Optional[str] = None
    location_text: Optional[str] = None
    enrichment_state: str = "unenriched"


@dataclass
class IngestionError:
    """A record in `bronze.ingestion_errors`."""

    error_id: str
    error_type: str
    error_message: str
    error_timestamp: datetime
    source_domain: Optional[str] = None
    source_url: Optional[str] = None
    missing_fields: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Silver tier
# ---------------------------------------------------------------------------


@dataclass
class EnrichedListing:
    """A record in `silver.enriched_listings`."""

    listing_id: str
    job_title: str
    company_name: str
    source_url: str
    enrichment_state: str
    enrichment_timestamp: datetime
    job_description: Optional[str] = None
    location_text: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    required_skills: Optional[List[str]] = None
    required_skills_text: Optional[str] = None
    seniority_level: Optional[str] = None
    employment_type: Optional[str] = None
    industry: Optional[str] = None
    company_size_band: Optional[str] = None
    unresolved_attributes: Optional[List[str]] = None
    failure_reason: Optional[str] = None
    embedding_text: Optional[str] = None


@dataclass
class EnrichedListingChunk:
    """A record in `silver.enriched_listings_chunks`."""

    chunk_id: str
    listing_id: str
    chunk_index: int
    embedding_text: str
    job_title: str
    company_name: str
    enrichment_state: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


# ---------------------------------------------------------------------------
# Gold tier
# ---------------------------------------------------------------------------


@dataclass
class UserProfile:
    """A record in `gold.user_profiles`."""

    profile_id: str
    commute_radius_km: int
    created_at: datetime
    updated_at: datetime
    skills: List[str] = field(default_factory=list)
    years_of_experience: Optional[int] = None
    education_history: Optional[str] = None
    job_title_history: List[str] = field(default_factory=list)
    qualifications_summary: Optional[str] = None
    home_latitude: Optional[float] = None
    home_longitude: Optional[float] = None
    home_location_name: Optional[str] = None
    cv_file_path: Optional[str] = None


@dataclass
class PipelineSummary:
    """A record in `gold.pipeline_summary`."""

    run_id: str
    pipeline_name: str
    records_added: int
    records_updated: int
    records_skipped: int
    source_errors: int
    run_timestamp: datetime
    ingestion_mode: Optional[str] = None
    enriched_count: Optional[int] = None
    partially_enriched_count: Optional[int] = None
    failed_count: Optional[int] = None
    unenriched_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Ops tier
# ---------------------------------------------------------------------------


@dataclass
class ReachabilityReportEntry:
    """A record in `ops.reachability_report`."""

    domain: str
    probe_timestamp: datetime
    outcome: str
    http_status_code: Optional[int] = None
    error_message: Optional[str] = None
    reachable_count: Optional[int] = None
    blocked_count: Optional[int] = None


@dataclass
class GeocodeLookupEntry:
    """A record in `ops.geocode_lookup`."""

    city_name: str
    country: str
    latitude: float
    longitude: float
    postal_code: Optional[str] = None


@dataclass
class BatchCheckpoint:
    """A record in `ops.batch_checkpoints`."""

    pipeline_name: str
    run_id: str
    last_successful_batch: int
    checkpoint_timestamp: datetime


@dataclass
class IndexingError:
    """A record in `ops.indexing_errors`."""

    error_id: str
    error_message: str
    error_timestamp: datetime


@dataclass
class EvaluationCase:
    """A record in `ops.evaluation_dataset`."""

    case_id: str
    user_profile: str
    expected_listing_ids: List[str]
    description: Optional[str] = None


@dataclass
class EvaluationResult:
    """A record in `ops.evaluation_results`."""

    eval_run_id: str
    model_version: int
    match_relevance_mean: float
    groundedness_mean: float
    eval_timestamp: datetime
