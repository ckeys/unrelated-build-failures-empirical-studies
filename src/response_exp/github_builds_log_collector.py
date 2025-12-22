"""
GitHub Builds Log Collector for Unrelated Build Failures Study.

This module collects ALL build logs from GitHub Actions for the TEAMMATES project
directly from the GitHub API, without depending on teammates_pr_data_v4.json.

The output includes build_id which can be used for mapping with other data sources.

GitHub API Reference:
- List workflow runs: GET /repos/{owner}/{repo}/actions/runs
- List jobs for a run: GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs
- Get job logs: GET /repos/{owner}/{repo}/actions/jobs/{job_id}/logs
- Rate limit: 5000 requests/hour for authenticated requests
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rebuttal_data" / "github"

# GitHub API configuration
GITHUB_API_BASE = "https://api.github.com"

# Default repository (can be overridden via command line)
DEFAULT_REPO_OWNER = "keycloak"
DEFAULT_REPO_NAME = "keycloak"

# These will be set based on command line arguments
REPO_OWNER = DEFAULT_REPO_OWNER
REPO_NAME = DEFAULT_REPO_NAME
BUILD_LOGS_DIR = DATA_DIR / f"{DEFAULT_REPO_NAME}_logs"
CHECKPOINT_FILE = DATA_DIR / f"{DEFAULT_REPO_NAME}_build_logs_checkpoint.json"
WORKFLOW_RUNS_CACHE_FILE = DATA_DIR / f"{DEFAULT_REPO_NAME}_workflow_runs_cache.json"

# GitHub Token - 认证后: 5000 requests/hour
# 使用环境变量或默认 token
GITHUB_TOKEN = os.getenv(
    "GITHUB_TOKEN",
    ""
)

# Rate limiting configuration
REQUEST_DELAY_SECONDS = 0.5  # To stay within rate limits
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
RUNS_PER_PAGE = 100  # Max allowed by GitHub API


def get_github_token() -> str:
    """Get GitHub token.
    
    Returns:
        GitHub personal access token.
        
    Raises:
        ValueError: If token is not available.
    """
    if not GITHUB_TOKEN:
        raise ValueError(
            "GitHub token is not set. "
            "Please set GITHUB_TOKEN environment variable or update the default token."
        )
    return GITHUB_TOKEN


def load_checkpoint() -> dict:
    """Load the checkpoint file to resume from previous progress.
    
    Returns:
        Dictionary with collected job IDs and their status.
    """
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'collected_job_ids': [],
        'failed_job_ids': [],
        'processed_run_ids': [],
        'last_updated': None,
        'total_runs_fetched': 0,
    }


def save_checkpoint(checkpoint: dict) -> None:
    """Save the checkpoint file.
    
    Args:
        checkpoint: Dictionary with collected job IDs and their status.
    """
    checkpoint['last_updated'] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2)


def ensure_logs_dir() -> None:
    """Ensure the build logs directory exists."""
    BUILD_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_log_file_path(job_id: int) -> Path:
    """Get the file path for a specific job's log.
    
    Args:
        job_id: The job ID.
        
    Returns:
        Path to the log file.
    """
    return BUILD_LOGS_DIR / f"{job_id}.json"


def load_existing_log_ids() -> set[str]:
    """Load IDs of existing build logs from the logs directory.
    
    Returns:
        Set of job IDs that have already been collected.
    """
    if not BUILD_LOGS_DIR.exists():
        return set()
    
    existing_ids = set()
    for log_file in BUILD_LOGS_DIR.glob("*.json"):
        # Extract job_id from filename (e.g., "12345.json" -> "12345")
        job_id = log_file.stem
        existing_ids.add(job_id)
    
    return existing_ids


def save_single_log(job_id: int, log_data: dict) -> None:
    """Save a single build log to its own file.
    
    Args:
        job_id: The job ID.
        log_data: Dictionary containing the log data.
    """
    ensure_logs_dir()
    log_file = get_log_file_path(job_id)
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)


def load_single_log(job_id: int) -> Optional[dict]:
    """Load a single build log from its file.
    
    Args:
        job_id: The job ID.
        
    Returns:
        Dictionary containing the log data, or None if not found.
    """
    log_file = get_log_file_path(job_id)
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_workflow_runs_cache() -> list[dict]:
    """Load cached workflow runs.
    
    Returns:
        List of workflow run dictionaries.
    """
    if WORKFLOW_RUNS_CACHE_FILE.exists():
        with open(WORKFLOW_RUNS_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_workflow_runs_cache(runs: list[dict]) -> None:
    """Save workflow runs to cache file.
    
    Args:
        runs: List of workflow run dictionaries.
    """
    with open(WORKFLOW_RUNS_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(runs, f, indent=2)


def make_api_request(
    url: str,
    token: str,
    params: Optional[dict] = None
) -> Optional[dict | str]:
    """Make a GitHub API request with retry logic.
    
    Args:
        url: The API endpoint URL.
        token: GitHub personal access token.
        params: Optional query parameters.
        
    Returns:
        The response JSON or text, or None if failed.
    """
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=60)
            
            if response.status_code == 200:
                # Check if response is JSON or text (logs are returned as text)
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    return response.json()
                return response.text
            
            elif response.status_code == 404:
                return None
            
            elif response.status_code == 403:
                # Rate limited
                reset_time = response.headers.get('X-RateLimit-Reset')
                remaining = response.headers.get('X-RateLimit-Remaining', 'unknown')
                print(f"  Rate limit remaining: {remaining}")
                
                if reset_time:
                    wait_time = int(reset_time) - int(time.time()) + 1
                    if wait_time > 0:
                        print(f"  Rate limited. Waiting {wait_time} seconds...")
                        time.sleep(min(wait_time, 3600))  # Cap at 1 hour
                else:
                    time.sleep(60)
                continue
            
            elif response.status_code >= 500:
                print(f"  Server error {response.status_code}, retrying...")
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            
            else:
                print(f"  Unexpected status code {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            return None
    
    return None


def fetch_all_workflow_runs(
    token: str,
    checkpoint: dict,
    max_runs: Optional[int] = None,
) -> list[dict]:
    """Fetch all completed workflow runs from GitHub Actions.
    
    Args:
        token: GitHub personal access token.
        checkpoint: Checkpoint dictionary.
        max_runs: Maximum number of runs to fetch (None for all).
        
    Returns:
        List of workflow run dictionaries.
    """
    # Try to load from cache first
    cached_runs = load_workflow_runs_cache()
    if cached_runs:
        print(f"  Found {len(cached_runs)} cached workflow runs")
        return cached_runs
    
    all_runs = []
    page = 1
    
    print("  Fetching workflow runs from GitHub API...")
    
    while True:
        url = f"{GITHUB_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs"
        params = {
            'per_page': RUNS_PER_PAGE,
            'page': page,
            'status': 'completed',  # Only get completed runs
        }
        
        print(f"  Fetching page {page}...")
        response = make_api_request(url, token, params)
        
        if not response or not isinstance(response, dict):
            print(f"  Failed to fetch page {page}")
            break
        
        runs = response.get('workflow_runs', [])
        if not runs:
            print(f"  No more runs found")
            break
        
        all_runs.extend(runs)
        print(f"  Fetched {len(runs)} runs (total: {len(all_runs)})")
        
        if max_runs and len(all_runs) >= max_runs:
            all_runs = all_runs[:max_runs]
            break
        
        # Check if there are more pages
        total_count = response.get('total_count', 0)
        fetched_so_far = page * RUNS_PER_PAGE
        if fetched_so_far >= total_count:
            break
        
        page += 1
        time.sleep(REQUEST_DELAY_SECONDS)
    
    # Save to cache
    save_workflow_runs_cache(all_runs)
    print(f"  Cached {len(all_runs)} workflow runs")
    
    return all_runs


def fetch_jobs_for_run(run_id: int, token: str) -> list[dict]:
    """Fetch all jobs for a specific workflow run.
    
    Args:
        run_id: The workflow run ID.
        token: GitHub personal access token.
        
    Returns:
        List of job dictionaries.
    """
    url = f"{GITHUB_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs/{run_id}/jobs"
    params = {'per_page': 100}
    
    response = make_api_request(url, token, params)
    
    if response and isinstance(response, dict):
        return response.get('jobs', [])
    return []


def fetch_job_log(
    job_id: int,
    token: str,
    max_log_size: int = 100000
) -> Optional[str]:
    """Fetch the log for a specific GitHub Actions job.
    
    Args:
        job_id: The GitHub Actions job ID.
        token: GitHub personal access token.
        max_log_size: Maximum number of characters to store.
        
    Returns:
        The job log content as a string, or None if failed.
    """
    url = f"{GITHUB_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/actions/jobs/{job_id}/logs"
    
    log_content = make_api_request(url, token)
    
    if log_content and isinstance(log_content, str):
        # Truncate if too long, keeping the end (usually contains error info)
        if len(log_content) > max_log_size:
            log_content = "... [truncated] ...\n" + log_content[-max_log_size:]
        return log_content
    
    return None


def extract_error_summary(log_content: str) -> Optional[str]:
    """Extract a summary of errors from the log content.
    
    Args:
        log_content: The full log content.
        
    Returns:
        A summary of error-related lines, or None if no errors found.
    """
    if not log_content:
        return None
    
    error_keywords = [
        'error', 'Error', 'ERROR',
        'failed', 'Failed', 'FAILED',
        'failure', 'Failure', 'FAILURE',
        'exception', 'Exception', 'EXCEPTION',
        'AssertionError', 'TypeError', 'ValueError',
        'NullPointerException', 'RuntimeException',
        'FAIL:', 'FAILED:', '✗', '✘',
        'Process completed with exit code 1',
    ]
    
    lines = log_content.split('\n')
    error_lines = []
    
    for i, line in enumerate(lines):
        if any(keyword in line for keyword in error_keywords):
            # Include some context (2 lines before and after)
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            context_lines = lines[start:end]
            error_lines.extend(context_lines)
            error_lines.append('---')
    
    if error_lines:
        # Deduplicate while preserving order
        seen = set()
        unique_lines = []
        for line in error_lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        # Limit the summary size
        summary = '\n'.join(unique_lines[:150])
        return summary[:15000] if len(summary) > 15000 else summary
    
    return None


def collect_all_build_logs(
    token: str,
    checkpoint: dict,
    max_runs: Optional[int] = None,
    only_failed: bool = True,
    save_interval: int = 20
) -> int:
    """Collect logs for all builds from GitHub Actions.
    
    Each log is saved as a separate file: {BUILD_LOGS_DIR}/{job_id}.json
    
    Args:
        token: GitHub personal access token.
        checkpoint: Checkpoint dictionary for resuming.
        max_runs: Maximum number of workflow runs to process.
        only_failed: If True, only collect logs for failed jobs.
        save_interval: How often to save checkpoint.
        
    Returns:
        Total number of logs collected.
    """
    # Ensure logs directory exists
    ensure_logs_dir()
    
    collected_ids = set(checkpoint.get('collected_job_ids', []))
    failed_ids = set(checkpoint.get('failed_job_ids', []))
    processed_run_ids = set(checkpoint.get('processed_run_ids', []))
    
    # Also check existing files in the logs directory
    existing_log_ids = load_existing_log_ids()
    collected_ids.update(existing_log_ids)
    
    # Fetch all completed workflow runs
    print("\nFetching workflow runs...")
    workflow_runs = fetch_all_workflow_runs(token, checkpoint, max_runs)
    
    print(f"\nTotal workflow runs: {len(workflow_runs)}")
    print(f"Already processed runs: {len(processed_run_ids)}")
    print(f"Already collected logs: {len(collected_ids)}")
    
    # Filter runs that need processing
    runs_to_process = [r for r in workflow_runs if r['id'] not in processed_run_ids]
    print(f"Runs to process: {len(runs_to_process)}")
    
    jobs_processed = 0
    logs_collected_this_session = 0
    
    for run_idx, run in enumerate(runs_to_process):
        run_id = run['id']
        run_name = run.get('name', 'unknown')
        run_conclusion = run.get('conclusion', 'unknown')
        run_created_at = run.get('created_at', '')
        head_sha = run.get('head_sha', '')
        
        # Get PR number if available
        pr_number = None
        pull_requests = run.get('pull_requests', [])
        if pull_requests:
            pr_number = pull_requests[0].get('number')
        
        print(f"\n[Run {run_idx + 1}/{len(runs_to_process)}] Run ID: {run_id}, "
              f"Conclusion: {run_conclusion}, PR: {pr_number}")
        
        # Fetch jobs for this run
        jobs = fetch_jobs_for_run(run_id, token)
        time.sleep(REQUEST_DELAY_SECONDS)
        
        if not jobs:
            print(f"  No jobs found for run {run_id}")
            processed_run_ids.add(run_id)
            continue
        
        # Filter jobs - only collect failed jobs
        if only_failed:
            jobs_to_collect = [j for j in jobs if j.get('conclusion') == 'failure']
        else:
            jobs_to_collect = jobs
        
        print(f"  Found {len(jobs)} jobs, {len(jobs_to_collect)} failed to collect")
        
        for job in jobs_to_collect:
            job_id = job['id']
            job_name = job.get('name', '')
            job_conclusion = job.get('conclusion', '')
            job_started_at = job.get('started_at', '')
            job_completed_at = job.get('completed_at', '')
            
            # Skip if already collected
            if str(job_id) in collected_ids or str(job_id) in failed_ids:
                print(f"    Skipping job {job_id} (already collected)")
                continue
            
            print(f"    Collecting job {job_id} ({job_name})...")
            
            log_content = fetch_job_log(job_id, token)
            time.sleep(REQUEST_DELAY_SECONDS)
            
            if log_content:
                error_summary = extract_error_summary(log_content)
                
                log_data = {
                    'job_id': job_id,
                    'job_name': job_name,
                    'job_conclusion': job_conclusion,
                    'job_started_at': job_started_at,
                    'job_completed_at': job_completed_at,
                    'job_url': job.get('html_url', ''),
                    'run_id': run_id,
                    'run_name': run_name,
                    'run_conclusion': run_conclusion,
                    'run_created_at': run_created_at,
                    'pr_number': pr_number,
                    'commit_sha': head_sha,
                    'log_content': log_content,
                    'error_summary': error_summary,
                    'collected_at': datetime.now().isoformat(),
                }
                
                # Save each log to its own file
                save_single_log(job_id, log_data)
                collected_ids.add(str(job_id))
                logs_collected_this_session += 1
                print(f"      ✓ Saved to {job_id}.json ({len(log_content)} chars)")
            else:
                failed_ids.add(str(job_id))
                print(f"      ✗ Failed to collect")
            
            jobs_processed += 1
            
            # Save checkpoint periodically
            if jobs_processed % save_interval == 0:
                checkpoint['collected_job_ids'] = list(collected_ids)
                checkpoint['failed_job_ids'] = list(failed_ids)
                checkpoint['processed_run_ids'] = list(processed_run_ids)
                save_checkpoint(checkpoint)
                print(f"      [Checkpoint saved: {len(collected_ids)} total logs]")
        
        processed_run_ids.add(run_id)
    
    # Final checkpoint save
    checkpoint['collected_job_ids'] = list(collected_ids)
    checkpoint['failed_job_ids'] = list(failed_ids)
    checkpoint['processed_run_ids'] = list(processed_run_ids)
    checkpoint['total_runs_fetched'] = len(workflow_runs)
    save_checkpoint(checkpoint)
    
    return len(collected_ids)


def main() -> None:
    """Main entry point for build log collection."""
    global REPO_OWNER, REPO_NAME, BUILD_LOGS_DIR, CHECKPOINT_FILE, WORKFLOW_RUNS_CACHE_FILE
    
    parser = argparse.ArgumentParser(
        description='Collect GitHub Actions build logs for a GitHub repository'
    )
    parser.add_argument(
        '--owner',
        type=str,
        default=DEFAULT_REPO_OWNER,
        help=f'Repository owner (default: {DEFAULT_REPO_OWNER})'
    )
    parser.add_argument(
        '--repo',
        type=str,
        default=DEFAULT_REPO_NAME,
        help=f'Repository name (default: {DEFAULT_REPO_NAME})'
    )
    parser.add_argument(
        '--max-runs',
        type=int,
        default=None,
        help='Maximum number of workflow runs to process (default: all)'
    )
    parser.add_argument(
        '--all-jobs',
        action='store_true',
        help='Collect logs for all jobs, not just failed ones'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear the workflow runs cache and fetch fresh data'
    )
    parser.add_argument(
        '--clear-checkpoint',
        action='store_true',
        help='Clear the checkpoint file and start fresh'
    )
    
    args = parser.parse_args()
    
    # Update global variables based on command line arguments
    REPO_OWNER = args.owner
    REPO_NAME = args.repo
    BUILD_LOGS_DIR = DATA_DIR / f"{REPO_NAME}_logs"
    CHECKPOINT_FILE = DATA_DIR / f"{REPO_NAME}_build_logs_checkpoint.json"
    WORKFLOW_RUNS_CACHE_FILE = DATA_DIR / f"{REPO_NAME}_workflow_runs_cache.json"
    
    print("=" * 70)
    print(f"GitHub Builds Log Collector for {REPO_OWNER}/{REPO_NAME}")
    print("=" * 70)
    
    # Get GitHub token
    try:
        token = get_github_token()
        print("✓ GitHub token loaded")
    except ValueError as e:
        print(f"✗ {e}")
        print("\nTo set the token, run:")
        print("  export GITHUB_TOKEN='your_github_personal_access_token'")
        return
    
    # Clear cache if requested
    if args.clear_cache and WORKFLOW_RUNS_CACHE_FILE.exists():
        WORKFLOW_RUNS_CACHE_FILE.unlink()
        print("✓ Workflow runs cache cleared")
    
    # Clear checkpoint if requested
    if args.clear_checkpoint and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("✓ Checkpoint file cleared")
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = load_checkpoint()
    existing_log_ids = load_existing_log_ids()
    print(f"✓ Checkpoint loaded (last updated: {checkpoint.get('last_updated', 'N/A')})")
    print(f"✓ Existing logs in {BUILD_LOGS_DIR.name}/: {len(existing_log_ids)}")
    
    # Collect logs
    print("\n" + "=" * 70)
    print("Starting log collection...")
    print(f"  Output directory: {BUILD_LOGS_DIR}")
    print(f"  Max runs: {args.max_runs or 'all'}")
    print(f"  Only failed jobs: {not args.all_jobs}")
    print("=" * 70)
    
    total_logs = collect_all_build_logs(
        token=token,
        checkpoint=checkpoint,
        max_runs=args.max_runs,
        only_failed=not args.all_jobs,
    )
    
    print("\n" + "=" * 70)
    print("Collection complete!")
    print("=" * 70)
    print(f"Total logs collected: {total_logs}")
    print(f"Output directory: {BUILD_LOGS_DIR}")
    
    # Print summary statistics from files
    if BUILD_LOGS_DIR.exists():
        log_files = list(BUILD_LOGS_DIR.glob("*.json"))
        print(f"\nLog files in directory: {len(log_files)}")
        
        # Sample a few files to show conclusions
        conclusions = {}
        for log_file in log_files[:100]:  # Sample first 100
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                    conclusion = log_data.get('job_conclusion', 'unknown')
                    conclusions[conclusion] = conclusions.get(conclusion, 0) + 1
            except Exception:
                pass
        
        if conclusions:
            print("\nSample logs by conclusion (first 100):")
            for conclusion, count in sorted(conclusions.items()):
                print(f"  {conclusion}: {count}")


if __name__ == "__main__":
    main()
