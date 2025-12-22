"""
GitHub Commit Files Collector for Unrelated Build Failures Study.

This module collects changed files information for each COMMIT (not PR) from GitHub API.
Since different builds within the same PR may correspond to different commits,
we need to collect file changes at the commit level for accurate feature extraction.

Features collected:
- Has Config Files
- Config Lines Added/Deleted/Modified
- Has Source Code
- Source Code Lines Added/Deleted/Modified
- Modified Source Code Files

GitHub API Reference:
- Get a commit: GET /repos/{owner}/{repo}/commits/{ref}
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
TEAMMATES_DATA_FILE = DATA_DIR / "teammates_pr_data_v4.json"
COMMIT_FILES_OUTPUT_FILE = DATA_DIR / "teammates_commit_files.json"
COMMIT_FILES_CHECKPOINT_FILE = DATA_DIR / "commit_files_checkpoint.json"

# GitHub API configuration
GITHUB_API_BASE = "https://api.github.com"
REPO_OWNER = "TEAMMATES"
REPO_NAME = "teammates"

# GitHub Token
GITHUB_TOKEN = os.getenv(
    "GITHUB_TOKEN",
    ""
)

# Rate limiting configuration
REQUEST_DELAY_SECONDS = 0.5
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# File type definitions
# Config files: build configs, settings, properties
CONFIG_EXTENSIONS = {
    '.yaml', '.yml', '.xml', '.properties', '.json', '.config', 
    '.ini', '.toml', '.env', '.conf', '.gradle'
}

# Source code files: programming logic files (NOT templates/styles)
# This follows the paper's definition of "source code" as files containing program logic
SOURCE_CODE_EXTENSIONS = {
    # Backend
    '.java', '.py', '.go', '.rb', '.php', '.cs', '.kt', '.scala',
    # Frontend (logic code only, NOT html/css)
    '.ts', '.tsx', '.js', '.jsx', '.vue', '.svelte',
    # Other logic
    '.sql', '.sh', '.bash', '.ps1',
}

# Note: HTML (.html) and CSS (.css, .scss) are NOT considered source code
# as they are templates/styles, not program logic


def get_github_token() -> str:
    """Get GitHub token."""
    if not GITHUB_TOKEN:
        raise ValueError("GitHub token is not set.")
    return GITHUB_TOKEN


def load_teammates_data() -> list[dict]:
    """Load the TEAMMATES PR data from JSON file."""
    with open(TEAMMATES_DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_checkpoint() -> dict:
    """Load checkpoint file."""
    if COMMIT_FILES_CHECKPOINT_FILE.exists():
        with open(COMMIT_FILES_CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'collected_commits': [], 'failed_commits': [], 'last_updated': None}


def save_checkpoint(checkpoint: dict) -> None:
    """Save checkpoint file."""
    checkpoint['last_updated'] = datetime.now().isoformat()
    with open(COMMIT_FILES_CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2)


def load_existing_commit_files() -> dict[str, dict]:
    """Load existing commit files data."""
    if COMMIT_FILES_OUTPUT_FILE.exists():
        with open(COMMIT_FILES_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {item['commit_sha']: item for item in data}
    return {}


def save_commit_files(commit_files: dict[str, dict]) -> None:
    """Save commit files data."""
    data = list(commit_files.values())
    with open(COMMIT_FILES_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def make_api_request(url: str, token: str, params: Optional[dict] = None) -> Optional[dict | list]:
    """Make a GitHub API request with retry logic."""
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=60)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            elif response.status_code == 403:
                reset_time = response.headers.get('X-RateLimit-Reset')
                if reset_time:
                    wait_time = int(reset_time) - int(time.time()) + 1
                    if wait_time > 0:
                        print(f"  Rate limited. Waiting {wait_time} seconds...")
                        time.sleep(min(wait_time, 3600))
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


def get_file_extension(filename: str) -> str:
    """Get the file extension from a filename."""
    if '.' in filename:
        return '.' + filename.rsplit('.', 1)[-1].lower()
    return ''


def fetch_commit_data(commit_sha: str, token: str) -> Optional[dict]:
    """Fetch commit data including files and timestamp.
    
    Args:
        commit_sha: The commit SHA.
        token: GitHub token.
        
    Returns:
        Dictionary with commit data (files, timestamps, message), or None if failed.
    """
    url = f"{GITHUB_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/commits/{commit_sha}"
    
    response = make_api_request(url, token)
    
    if response is None:
        return None
    
    # Extract relevant data from commit response
    commit_info = response.get('commit', {})
    author_info = commit_info.get('author', {})
    committer_info = commit_info.get('committer', {})
    
    return {
        'files': response.get('files', []),
        'commit_message': commit_info.get('message', ''),
        'author_date': author_info.get('date'),  # When the commit was authored
        'committer_date': committer_info.get('date'),  # When the commit was committed
        'author_name': author_info.get('name'),
        'author_email': author_info.get('email'),
    }


def analyze_commit_files(files: list[dict]) -> dict:
    """Analyze the changed files and extract metrics.
    
    Args:
        files: List of file dictionaries from GitHub API.
        
    Returns:
        Dictionary with file analysis metrics.
    """
    # Initialize metrics
    metrics = {
        # Config files metrics
        'has_config_files': False,
        'config_files_count': 0,
        'config_lines_added': 0,
        'config_lines_deleted': 0,
        'config_lines_modified': 0,  # additions + deletions as proxy for modifications
        
        # Source code metrics (program logic files: .java, .ts, .js, etc.)
        # Note: HTML/CSS are NOT included as they are templates/styles, not logic
        'has_source_code': False,
        'source_code_files_count': 0,
        'source_code_lines_added': 0,
        'source_code_lines_deleted': 0,
        'source_code_lines_modified': 0,
        
        # Total metrics
        'total_files_changed': len(files),
        'total_additions': 0,
        'total_deletions': 0,
        
        # File lists for reference
        'config_files': [],
        'source_files': [],
    }
    
    for file_info in files:
        filename = file_info.get('filename', '')
        additions = file_info.get('additions', 0)
        deletions = file_info.get('deletions', 0)
        
        ext = get_file_extension(filename)
        
        # Total metrics
        metrics['total_additions'] += additions
        metrics['total_deletions'] += deletions
        
        # Config files (.yaml, .xml, .properties, .json, etc.)
        if ext in CONFIG_EXTENSIONS:
            metrics['has_config_files'] = True
            metrics['config_files_count'] += 1
            metrics['config_lines_added'] += additions
            metrics['config_lines_deleted'] += deletions
            metrics['config_lines_modified'] += additions + deletions
            metrics['config_files'].append(filename)
        
        # Source code files (.java, .ts, .js, etc. - NOT html/css)
        if ext in SOURCE_CODE_EXTENSIONS:
            metrics['has_source_code'] = True
            metrics['source_code_files_count'] += 1
            metrics['source_code_lines_added'] += additions
            metrics['source_code_lines_deleted'] += deletions
            metrics['source_code_lines_modified'] += additions + deletions
            metrics['source_files'].append(filename)
    
    return metrics


def extract_unique_commits_from_failed_builds(pr_data: list[dict]) -> list[dict]:
    """Extract unique commits that have failed builds.
    
    Args:
        pr_data: List of PR dictionaries.
        
    Returns:
        List of dictionaries with commit_sha and pr_number.
    """
    commits = {}  # commit_sha -> pr_number
    
    for pr in pr_data:
        pr_number = pr.get('pr_number')
        build_associations = pr.get('build_comment_associations', [])
        
        for assoc in build_associations:
            build = assoc.get('build', {})
            conclusion = build.get('conclusion')
            commit_sha = build.get('commit_sha')
            
            if conclusion == 'failure' and commit_sha:
                # Store commit with its PR number
                if commit_sha not in commits:
                    commits[commit_sha] = pr_number
    
    # Convert to list of dicts
    result = [
        {'commit_sha': sha, 'pr_number': pr_num}
        for sha, pr_num in commits.items()
    ]
    
    return result


def collect_commit_files(
    commits: list[dict],
    token: str,
    checkpoint: dict,
    existing_data: dict[str, dict],
    save_interval: int = 50
) -> dict[str, dict]:
    """Collect files information for all commits.
    
    Args:
        commits: List of commit dictionaries with commit_sha and pr_number.
        token: GitHub token.
        checkpoint: Checkpoint dictionary.
        existing_data: Already collected data.
        save_interval: How often to save progress.
        
    Returns:
        Dictionary mapping commit SHA to file metrics.
    """
    collected_commits = set(checkpoint.get('collected_commits', []))
    failed_commits = set(checkpoint.get('failed_commits', []))
    
    # Filter commits that need collection
    commits_to_collect = [
        c for c in commits 
        if c['commit_sha'] not in collected_commits and c['commit_sha'] not in failed_commits
    ]
    
    print(f"Total unique commits: {len(commits)}")
    print(f"Already collected: {len(collected_commits)}")
    print(f"Failed: {len(failed_commits)}")
    print(f"To collect: {len(commits_to_collect)}")
    
    for i, commit_info in enumerate(commits_to_collect):
        commit_sha = commit_info['commit_sha']
        pr_number = commit_info['pr_number']
        
        print(f"[{i+1}/{len(commits_to_collect)}] Collecting files for commit {commit_sha[:8]}... (PR #{pr_number})")
        
        commit_data = fetch_commit_data(commit_sha, token)
        time.sleep(REQUEST_DELAY_SECONDS)
        
        if commit_data is not None:
            files = commit_data['files']
            metrics = analyze_commit_files(files)
            
            # Add commit metadata
            metrics['commit_sha'] = commit_sha
            metrics['pr_number'] = pr_number
            metrics['commit_message'] = commit_data['commit_message']
            metrics['author_date'] = commit_data['author_date']  # When commit was authored
            metrics['committer_date'] = commit_data['committer_date']  # When commit was committed
            metrics['author_name'] = commit_data['author_name']
            metrics['collected_at'] = datetime.now().isoformat()
            
            existing_data[commit_sha] = metrics
            collected_commits.add(commit_sha)
            
            print(f"  ✓ {len(files)} files, {metrics['total_additions']}+/{metrics['total_deletions']}-")
        else:
            failed_commits.add(commit_sha)
            print(f"  ✗ Failed to collect")
        
        # Save progress periodically
        if (i + 1) % save_interval == 0:
            checkpoint['collected_commits'] = list(collected_commits)
            checkpoint['failed_commits'] = list(failed_commits)
            save_checkpoint(checkpoint)
            save_commit_files(existing_data)
            print(f"  [Checkpoint saved: {len(collected_commits)} commits]")
    
    # Final save
    checkpoint['collected_commits'] = list(collected_commits)
    checkpoint['failed_commits'] = list(failed_commits)
    save_checkpoint(checkpoint)
    save_commit_files(existing_data)
    
    return existing_data


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Collect commit changed files information from GitHub API'
    )
    parser.add_argument(
        '--max-commits',
        type=int,
        default=None,
        help='Maximum number of commits to process (default: all)'
    )
    parser.add_argument(
        '--clear-checkpoint',
        action='store_true',
        help='Clear checkpoint and start fresh'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GitHub Commit Files Collector")
    print("=" * 70)
    
    # Get token
    try:
        token = get_github_token()
        print("✓ GitHub token loaded")
    except ValueError as e:
        print(f"✗ {e}")
        return
    
    # Clear checkpoint if requested
    if args.clear_checkpoint:
        if COMMIT_FILES_CHECKPOINT_FILE.exists():
            COMMIT_FILES_CHECKPOINT_FILE.unlink()
        if COMMIT_FILES_OUTPUT_FILE.exists():
            COMMIT_FILES_OUTPUT_FILE.unlink()
        print("✓ Checkpoint and data cleared")
    
    # Load PR data
    print("\nLoading TEAMMATES PR data...")
    pr_data = load_teammates_data()
    print(f"✓ Loaded {len(pr_data)} PRs")
    
    # Extract unique commits with failed builds
    commits = extract_unique_commits_from_failed_builds(pr_data)
    print(f"✓ Found {len(commits)} unique commits with failed builds")
    
    if args.max_commits:
        commits = commits[:args.max_commits]
        print(f"  Limited to {len(commits)} commits")
    
    # Load checkpoint and existing data
    checkpoint = load_checkpoint()
    existing_data = load_existing_commit_files()
    print(f"✓ Checkpoint loaded (last updated: {checkpoint.get('last_updated', 'N/A')})")
    print(f"✓ Existing data: {len(existing_data)} commits")
    
    # Collect files
    print("\n" + "=" * 70)
    print("Starting collection...")
    print("=" * 70)
    
    result = collect_commit_files(
        commits=commits,
        token=token,
        checkpoint=checkpoint,
        existing_data=existing_data,
    )
    
    print("\n" + "=" * 70)
    print("Collection complete!")
    print("=" * 70)
    print(f"Total commits collected: {len(result)}")
    print(f"Output file: {COMMIT_FILES_OUTPUT_FILE}")
    
    # Print summary statistics
    if result:
        has_config = sum(1 for c in result.values() if c.get('has_config_files'))
        has_source = sum(1 for c in result.values() if c.get('has_source_code'))
        
        print(f"\nSummary:")
        print(f"  Commits with config files: {has_config} ({has_config/len(result)*100:.1f}%)")
        print(f"  Commits with source code: {has_source} ({has_source/len(result)*100:.1f}%)")


if __name__ == "__main__":
    main()
