"""One-off generator for the bundled fallback job listings dataset.

Produces `data/bundled_fallback/bundled_listings.csv` with >=200 synthetic
but realistic job listing rows, covering every required field:
`listing_id`, `job_title`, `company_name`, `job_description`,
`location_text`, `source_url`, `ingestion_timestamp`.

`listing_id` is derived via the same SHA-256-based algorithm as
`src/utils/listing_id.py::derive_listing_id`, imported directly to keep the
two in sync.

Usage:
    python scripts/generate_bundled_listings.py
"""

import csv
import itertools
import os
import sys
from datetime import datetime, timedelta, timezone

# Make src/ importable regardless of CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from utils.listing_id import derive_listing_id  # noqa: E402

OUTPUT_PATH = os.path.join(
    _REPO_ROOT, "data", "bundled_fallback", "bundled_listings.csv"
)

JOB_TITLES = [
    "Senior Data Engineer",
    "Data Engineer",
    "Product Manager",
    "Senior Product Manager",
    "Frontend Developer",
    "Backend Developer",
    "Full Stack Developer",
    "Machine Learning Engineer",
    "Senior Machine Learning Engineer",
    "DevOps Engineer",
    "Site Reliability Engineer",
    "Data Scientist",
    "Senior Data Scientist",
    "Analytics Engineer",
    "Cloud Solutions Architect",
    "Software Engineer",
    "Senior Software Engineer",
    "Staff Software Engineer",
    "QA Engineer",
    "Engineering Manager",
    "UX Designer",
    "UI/UX Designer",
    "Business Analyst",
    "Data Analyst",
    "Marketing Manager",
    "Sales Engineer",
    "Technical Program Manager",
    "Platform Engineer",
    "Security Engineer",
    "Mobile Developer (iOS)",
    "Mobile Developer (Android)",
    "Database Administrator",
    "Solutions Consultant",
    "Customer Success Manager",
    "IT Support Specialist",
    "Scrum Master",
    "Growth Marketing Lead",
    "Content Strategist",
    "HR Business Partner",
    "Financial Analyst",
]

COMPANY_NAMES = [
    "Northwind Analytics", "Bluepeak Systems", "Cascade Robotics",
    "Ironleaf Technologies", "Solstice Data Group", "Harbor & Vine Software",
    "Vertex Cloud Labs", "Meridian Digital", "Quartz Innovations",
    "Amberfield Technologies", "Riverstone AI", "Northstar Analytics",
    "Glasswing Software", "Copperline Systems", "Falcon Ridge Technologies",
    "Everview Data Co.", "Brightwell Solutions", "Loom & Ledger Tech",
    "Kestrel Software Group", "Pinecone Digital", "Silverbrook Analytics",
    "Aurora Compute", "Thistle & Thorne Tech", "Redwood Cloud Systems",
    "Junction Point Labs", "Lighthouse Data Partners", "Marlowe Technologies",
    "Cobalt Stream Systems", "Fennec Software", "Windmere Analytics",
    "Basalt Digital Works", "Cinderwood Technologies", "Palisade Software",
    "Driftwood Analytics", "Nightingale Data Systems", "Foxglove Tech",
    "Granite Peak Software", "Meadowlark Digital", "Sable Cloud Group",
    "Timberline Systems",
]

LOCATIONS = [
    "Berlin, Germany", "Munich, Germany", "Hamburg, Germany",
    "London, UK", "Manchester, UK", "Edinburgh, UK",
    "Paris, France", "Lyon, France",
    "Madrid, Spain", "Barcelona, Spain",
    "Amsterdam, Netherlands", "Rotterdam, Netherlands",
    "Dublin, Ireland",
    "Vienna, Austria",
    "Zurich, Switzerland", "Geneva, Switzerland",
    "Stockholm, Sweden", "Copenhagen, Denmark", "Oslo, Norway",
    "Helsinki, Finland",
    "Warsaw, Poland", "Krakow, Poland",
    "Prague, Czech Republic",
    "Lisbon, Portugal", "Porto, Portugal",
    "Brussels, Belgium",
    "Milan, Italy", "Rome, Italy",
    "New York, NY", "San Francisco, CA", "Austin, TX", "Seattle, WA",
    "Boston, MA", "Chicago, IL", "Denver, CO", "Los Angeles, CA",
    "Atlanta, GA", "Portland, OR", "Raleigh, NC", "Remote",
]

SKILL_PHRASES = [
    "Python and SQL", "distributed data pipelines", "cloud infrastructure",
    "machine learning models", "modern JavaScript frameworks",
    "CI/CD automation", "large-scale data warehousing",
    "REST and GraphQL APIs", "containerized microservices",
    "stakeholder communication", "agile delivery practices",
    "data visualization and reporting", "Spark and Delta Lake",
    "Kubernetes orchestration", "A/B testing and experimentation",
]

DESCRIPTION_TEMPLATES = [
    (
        "We are looking for a {title} to join {company}. You will design and "
        "build scalable systems, collaborate closely with cross-functional "
        "teams, and take ownership of {skill}. The ideal candidate has "
        "several years of relevant experience and thrives in a fast-paced, "
        "collaborative environment."
    ),
    (
        "{company} is hiring a {title} to help us scale our platform. In "
        "this role you will work on {skill}, partner with product and "
        "engineering leadership, and mentor junior teammates. We value "
        "curiosity, clear communication, and a bias toward shipping."
    ),
    (
        "As a {title} at {company}, you will drive initiatives involving "
        "{skill}. You will work in a small, autonomous team responsible for "
        "an end-to-end product area, with a strong emphasis on quality, "
        "measurable outcomes, and continuous improvement."
    ),
    (
        "{company} seeks an experienced {title} to strengthen our growing "
        "team. Responsibilities include {skill}, participating in on-call "
        "rotations, and contributing to technical roadmap planning. Remote "
        "and hybrid arrangements are available for the right candidate."
    ),
]

TARGET_COUNT = 220


def build_rows(n: int):
    rows = []
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    title_cycle = itertools.cycle(JOB_TITLES)
    company_cycle = itertools.cycle(COMPANY_NAMES)
    location_cycle = itertools.cycle(LOCATIONS)
    skill_cycle = itertools.cycle(SKILL_PHRASES)
    template_cycle = itertools.cycle(DESCRIPTION_TEMPLATES)

    for i in range(n):
        title = next(title_cycle)
        company = next(company_cycle)
        location = next(location_cycle)
        skill = next(skill_cycle)
        template = next(template_cycle)

        source_url = f"https://bundled-fallback.local/jobs/{i:04d}"
        listing_id = derive_listing_id(source_url)
        description = template.format(title=title, company=company, skill=skill)
        ingestion_timestamp = (
            (base_time + timedelta(minutes=i * 7))
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        rows.append(
            {
                "listing_id": listing_id,
                "job_title": title,
                "company_name": company,
                "job_description": description,
                "location_text": location,
                "source_url": source_url,
                "ingestion_timestamp": ingestion_timestamp,
            }
        )
    return rows


def main():
    rows = build_rows(TARGET_COUNT)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fieldnames = [
        "listing_id",
        "job_title",
        "company_name",
        "job_description",
        "location_text",
        "source_url",
        "ingestion_timestamp",
    ]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
