"""
naukri_resume_matcher_app.py

A Streamlit app that:
  1. Lets you upload a resume (PDF) and extracts keywords from it
     IN MEMORY ONLY — the file is never saved to disk and is discarded
     as soon as keywords are pulled out.
  2. Lets you pick job keyword + location, plus three multi-select
     filters: Salary range, Experience range, Posted-within (freshness).
  3. Scrapes public Naukri.com search results (no login/API) with
     Playwright, then filters + ranks jobs by how many resume keywords
     match, so the most relevant postings float to the top.

Setup:
    pip install streamlit playwright pdfplumber pandas
    playwright install chromium

Run:
    streamlit run naukri_resume_matcher_app.py

Note on selectors: Naukri periodically changes its page markup. If the
scraper returns 0 results, open a Naukri search page, right-click a job
card -> Inspect, and update the CSS selectors marked below.
"""

import io
import re
import time
import random
from datetime import datetime

import pandas as pd
import streamlit as st
from playwright.sync_api import sync_playwright

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

import subprocess
subprocess.run(["playwright", "install", "chromium"], check=False)
# ----------------------------------------------------------------------
# Resume handling — strictly in-memory, discarded after keyword extraction
# ----------------------------------------------------------------------

# A small stopword-ish list so we don't turn common resume boilerplate
# into "keywords". Extend as needed.
GENERIC_WORDS = {
    "and", "or", "the", "a", "an", "with", "of", "in", "on", "to", "for",
    "years", "year", "experience", "skills", "professional", "summary",
    "education", "certifications", "projects", "additional", "information",
}


def extract_resume_keywords(uploaded_file) -> list[str]:
    """Extract a keyword list from an uploaded PDF resume.

    Reads the file into memory, pulls text, extracts likely skill/keyword
    terms, and returns just the keyword list. The raw bytes and text are
    never written to disk and are dropped as soon as this function returns.
    """
    if pdfplumber is None:
        st.error("pdfplumber is not installed. Run: pip install pdfplumber")
        return []

    raw_bytes = uploaded_file.read()
    text_chunks = []
    try:
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for p in pdf.pages:
                page_text = p.extract_text() or ""
                text_chunks.append(page_text)
    finally:
        # Drop references to the raw bytes as soon as we're done with them.
        del raw_bytes

    full_text = "\n".join(text_chunks)
    del text_chunks

    keywords = _parse_keywords(full_text)

    # Drop the extracted text now that keywords are pulled out.
    del full_text

    return keywords


def _parse_keywords(text: str) -> list[str]:
    """Pull a skills/keyword list out of resume text.

    Looks for a "CORE SKILLS" / "SKILLS" style section first (bullet- or
    delimiter-separated); falls back to frequent capitalized/technical
    tokens elsewhere in the resume if no clear skills section is found.
    """
    keywords = set()

    # 1. Try to find an explicit skills section.
    skills_match = re.search(
        r"(?:CORE SKILLS|SKILLS|TECHNICAL SKILLS)\s*[:\n]?(.*?)(?:\n[A-Z ]{4,}\n|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if skills_match:
        chunk = skills_match.group(1)
        parts = re.split(r"[•●,|\n]", chunk)
        for part in parts:
            term = part.strip(" .-")
            if 2 <= len(term) <= 40 and term.lower() not in GENERIC_WORDS:
                keywords.add(term)

    # 2. Fallback / supplement: pick out likely tech/tool tokens
    #    (mixed-case or all-caps words, 3+ chars) from the whole resume.
    if len(keywords) < 5:
        tokens = re.findall(r"\b[A-Z][A-Za-z0-9+#./]{2,}\b", text)
        for t in tokens:
            if t.lower() not in GENERIC_WORDS:
                keywords.add(t.strip())

    # Clean up and cap the list to a reasonable size.
    cleaned = sorted({k for k in keywords if k}, key=str.lower)
    return cleaned[:40]


# ----------------------------------------------------------------------
# Filter option definitions
# ----------------------------------------------------------------------

SALARY_BUCKETS = {
    "0-3 Lacs": (0, 3),
    "3-6 Lacs": (3, 6),
    "6-10 Lacs": (6, 10),
    "10-15 Lacs": (10, 15),
    "15-25 Lacs": (15, 25),
    "25-50 Lacs": (25, 50),
    "50+ Lacs": (50, 999),
    "Not Disclosed": None,
}

EXPERIENCE_BUCKETS = {
    "0-1 Yrs": (0, 1),
    "1-3 Yrs": (1, 3),
    "3-5 Yrs": (3, 5),
    "5-10 Yrs": (5, 10),
    "10-15 Yrs": (10, 15),
    "15+ Yrs": (15, 99),
}

FRESHNESS_BUCKETS = {
    "Today": 0,
    "Last 3 days": 3,
    "Last 7 days": 7,
    "Last 15 days": 15,
    "Last 30 days": 30,
}


def parse_salary_to_range(text: str):
    """Parse a salary string like '6-10 Lacs PA' into (min, max) in Lacs."""
    if not text or "not disclosed" in text.lower():
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        return float(nums[0]), float(nums[0])
    return None


def parse_experience_to_range(text: str):
    """Parse an experience string like '2-5 Yrs' into (min, max) years."""
    if not text:
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        return float(nums[0]), float(nums[0])
    return None


def parse_posted_days_ago(text: str):
    """Parse a posted-date string like '3 Days Ago' / 'Today' into an int."""
    if not text:
        return None
    t = text.strip().lower()
    if "today" in t or "just now" in t:
        return 0
    match = re.search(r"(\d+)\s*\+?\s*day", t)
    if match:
        return int(match.group(1))
    return None


def ranges_overlap(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


# ----------------------------------------------------------------------
# Scraper
# ----------------------------------------------------------------------

def build_search_url(keyword: str, location: str) -> str:
    kw_slug = keyword.strip().lower().replace(" ", "-")
    url = f"https://www.naukri.com/{kw_slug}-jobs"
    if location:
        loc_slug = location.strip().lower().replace(" ", "-")
        url = f"https://www.naukri.com/{kw_slug}-jobs-in-{loc_slug}"
    return url


def scrape_naukri(keyword: str, location: str, pages: int, status_cb=None) -> list[dict]:
    all_jobs = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        for i in range(1, pages + 1):
            url = build_search_url(keyword, location)
            if i > 1:
                url += f"-{i}"
            if status_cb:
                status_cb(f"Scraping page {i}: {url}")

            try:
                page.goto(url, timeout=30000)
                page.wait_for_selector("div.cust-job-tuple", timeout=15000)
                cards = page.query_selector_all("div.cust-job-tuple")

                for card in cards:
                    title_el = card.query_selector("a.title")
                    company_el = card.query_selector("a.comp-name")
                    loc_el = card.query_selector("span.locWdth")
                    exp_el = card.query_selector("span.expwdth")
                    sal_el = card.query_selector("span.sal-wrap") or card.query_selector("span.sal")
                    posted_el = card.query_selector("span.job-post-day")

                    all_jobs.append({
                        "title": title_el.inner_text().strip() if title_el else "",
                        "company": company_el.inner_text().strip() if company_el else "",
                        "location": loc_el.inner_text().strip() if loc_el else "",
                        "experience": exp_el.inner_text().strip() if exp_el else "",
                        "salary": sal_el.inner_text().strip() if sal_el else "Not disclosed",
                        "posted": posted_el.inner_text().strip() if posted_el else "",
                        "link": title_el.get_attribute("href") if title_el else "",
                    })
            except Exception as e:
                if status_cb:
                    status_cb(f"  page {i} failed: {e}")

            time.sleep(random.uniform(3, 6))

        browser.close()
    return all_jobs


def score_job(job: dict, keywords: list[str]) -> int:
    haystack = f"{job['title']} {job['company']}".lower()
    return sum(1 for kw in keywords if kw.lower() in haystack)


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------

st.set_page_config(page_title="Naukri Resume Job Matcher", layout="wide")
st.title("📄 Resume-Matched Naukri Job Search")
st.caption(
    "Upload your resume, pick your filters, and get Naukri job listings "
    "ranked by how well they match your resume. Nothing from your resume "
    "is saved to disk — it's read into memory, keywords are pulled out, "
    "and the file is discarded."
)

# --- Resume upload ---
resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if resume_file is not None:
    if "resume_keywords" not in st.session_state or st.session_state.get("resume_name") != resume_file.name:
        with st.spinner("Reading resume and extracting keywords (in memory only)..."):
            st.session_state["resume_keywords"] = extract_resume_keywords(resume_file)
            st.session_state["resume_name"] = resume_file.name
    # The uploaded_file object and its bytes are not referenced again after
    # extract_resume_keywords() returns — only the derived keyword list
    # is kept, in session memory, for this browser session.

resume_keywords = st.session_state.get("resume_keywords", [])
if resume_keywords:
    st.success(f"Extracted {len(resume_keywords)} keywords from resume.")
    with st.expander("View extracted keywords"):
        st.write(", ".join(resume_keywords))

st.divider()

# --- Search inputs ---
col1, col2 = st.columns(2)
with col1:
    job_keyword = st.text_input("Job title / keyword", value="data analyst")
with col2:
    job_location = st.text_input("Location (optional)", value="")

st.subheader("Filters")
f1, f2, f3 = st.columns(3)
with f1:
    selected_salary = st.multiselect("Salary range", list(SALARY_BUCKETS.keys()))
with f2:
    selected_experience = st.multiselect("Experience range", list(EXPERIENCE_BUCKETS.keys()))
with f3:
    selected_freshness = st.multiselect("Posted within", list(FRESHNESS_BUCKETS.keys()))

pages_to_scrape = st.slider("Pages to scrape", min_value=1, max_value=5, value=2)

run = st.button("🔍 Search Naukri", type="primary")

if run:
    if not job_keyword.strip():
        st.warning("Enter a job keyword first.")
    else:
        status_box = st.empty()

        def status_cb(msg):
            status_box.text(msg)

        with st.spinner("Scraping Naukri..."):
            jobs = scrape_naukri(job_keyword, job_location, pages_to_scrape, status_cb)
        status_box.empty()

        if not jobs:
            st.error(
                "No jobs found. Naukri may have changed its page structure — "
                "inspect the site and update the CSS selectors in the script."
            )
        else:
            df = pd.DataFrame(jobs)

            # --- Apply salary filter ---
            if selected_salary:
                def salary_ok(row):
                    parsed = parse_salary_to_range(row["salary"])
                    for label in selected_salary:
                        bucket = SALARY_BUCKETS[label]
                        if bucket is None:
                            if parsed is None:
                                return True
                        elif parsed is not None and ranges_overlap(parsed, bucket):
                            return True
                    return False
                df = df[df.apply(salary_ok, axis=1)]

            # --- Apply experience filter ---
            if selected_experience:
                def exp_ok(row):
                    parsed = parse_experience_to_range(row["experience"])
                    if parsed is None:
                        return False
                    return any(
                        ranges_overlap(parsed, EXPERIENCE_BUCKETS[label])
                        for label in selected_experience
                    )
                df = df[df.apply(exp_ok, axis=1)]

            # --- Apply freshness filter ---
            if selected_freshness:
                max_days = max(FRESHNESS_BUCKETS[label] for label in selected_freshness)
                def fresh_ok(row):
                    days = parse_posted_days_ago(row["posted"])
                    return days is not None and days <= max_days
                df = df[df.apply(fresh_ok, axis=1)]

            # --- Score by resume keyword match ---
            if resume_keywords:
                df["match_score"] = df.apply(lambda r: score_job(r, resume_keywords), axis=1)
                df = df.sort_values("match_score", ascending=False)
            else:
                df["match_score"] = 0

            st.success(f"Found {len(df)} matching jobs.")
            st.dataframe(df, use_container_width=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                data=csv_bytes,
                file_name=f"naukri_jobs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
