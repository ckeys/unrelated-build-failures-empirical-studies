"""
GitHub Projects Feature Collection for Unrelated Build Failures Study.

This module extracts features from GitHub PR data for predicting unrelated 
build failures. This is part of the rebuttal experiment to address reviewer 
concerns about generalizability (V2.R3.C5).

Supports multiple projects via command line arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rebuttal_data" / "github"

# Default project configuration (can be overridden via command line)
DEFAULT_PROJECT_NAME = "keycloak"

# These will be set based on command line arguments
PROJECT_NAME = DEFAULT_PROJECT_NAME
PR_DATA_FILE = DATA_DIR / f"{DEFAULT_PROJECT_NAME}_pr_data.json"
BUILD_LOGS_DIR = DATA_DIR / f"{DEFAULT_PROJECT_NAME}_logs"
COMMIT_FILES_DATA_FILE = DATA_DIR / f"{DEFAULT_PROJECT_NAME}_commit_files.json"
OUTPUT_FEATURES_FILE = DATA_DIR / f"{DEFAULT_PROJECT_NAME}_features.csv"

# Random state for reproducibility
RANDOM_STATE: int = 42


def load_pr_data() -> list[dict]:
    """Load the PR data from JSON file.
    
    Returns:
        List of PR dictionaries containing build and comment information.
    """
    with open(PR_DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_commit_files_data() -> dict[str, dict]:
    """Load the commit files data from JSON file.
    
    Returns:
        Dictionary mapping commit_sha to commit file metrics.
    """
    if not COMMIT_FILES_DATA_FILE.exists():
        print(f"Warning: Commit files data not found at {COMMIT_FILES_DATA_FILE}")
        return {}
    
    with open(COMMIT_FILES_DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert list to dict keyed by commit_sha
    return {item['commit_sha']: item for item in data}


def parse_datetime(datetime_str: str) -> datetime:
    """Parse ISO 8601 datetime string to datetime object.
    
    Args:
        datetime_str: ISO 8601 formatted datetime string (e.g., "2025-12-19T10:11:45Z").
        
    Returns:
        Parsed datetime object.
    """
    return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))


def get_date_from_datetime_str(datetime_str: str) -> str:
    """Extract date string (YYYY-MM-DD) from ISO 8601 datetime string.
    
    Args:
        datetime_str: ISO 8601 formatted datetime string.
        
    Returns:
        Date string in YYYY-MM-DD format.
    """
    return datetime_str[:10]


def build_pr_creation_date_index(pr_data: list[dict]) -> dict[str, int]:
    """Build an index of PR creation dates to count PRs created on each date.
    
    Args:
        pr_data: List of PR dictionaries.
        
    Returns:
        Dictionary mapping date string (YYYY-MM-DD) to count of PRs created on that date.
    """
    date_counts: dict[str, int] = {}
    
    for pr in pr_data:
        pr_created_at = pr.get('pr_created_at', '')
        if pr_created_at:
            date_str = get_date_from_datetime_str(pr_created_at)
            date_counts[date_str] = date_counts.get(date_str, 0) + 1
    
    return date_counts


def extract_num_parallel_issues(
    build: dict,
    pr_date_index: dict[str, int]
) -> int:
    """Extract the Number of Parallel Issues feature for a build.
    
    Number of Parallel Issues quantifies the count of issues (PRs) that were
    opened on the same date as the build occurred. This follows the formula:
    
        Sum(Count(T_n = T_i)) for n=1 to m
    
    Where T_n is the date of the n-th issue opened, and T_i is the date of
    the current build.
    
    Args:
        build: Build dictionary containing build information.
        pr_date_index: Dictionary mapping dates to PR counts.
        
    Returns:
        Count of PRs created on the same date as the build.
    """
    build_completed_at = build.get('started_at', '')

    if not build_completed_at:
        return 0
    
    build_date = get_date_from_datetime_str(build_completed_at)
    return pr_date_index.get(build_date, 0)


def extract_num_prior_comments(
    build: dict,
    all_comments: list[dict]
) -> int:
    """Extract the Number of Prior Comments feature for a build.
    
    Number of Prior Comments represents the number of comments that were
    posted before the build started. This follows the formula:
    
        Sum(count(C_i)) for i < curr
    
    Where C_i is the comment at index i, and curr is the index of the
    current build failure.
    
    Args:
        build: Build dictionary containing build information.
        all_comments: List of all comments in the PR.
        
    Returns:
        Count of comments posted before the build started.
    """
    build_started_at = build.get('started_at', '')
    
    if not build_started_at:
        return 0
    
    count = 0
    for comment in all_comments:
        comment_created_at = comment.get('created_at', '')
        # ISO 8601 format allows direct string comparison
        if comment_created_at and comment_created_at < build_started_at:
            count += 1
    
    return count


def extract_ci_latency(
    build: dict,
    pr_created_at: str
) -> float:
    """Extract the CI Latency feature for a build.
    
    CI Latency refers to the duration between when a code patch is uploaded/pushed
    and when the build is triggered for that code patch.
    
    In this implementation, we use:
        CI Latency = build.started_at - pr_created_at
    
    This is an approximation since we don't have the exact commit push time.
    For the first build of a PR, this is accurate. For subsequent builds
    (after new commits), this may overestimate the latency.
    
    Args:
        build: Build dictionary containing build information.
        pr_created_at: ISO 8601 datetime string of when the PR was created.
        
    Returns:
        CI Latency in days (float). Returns 0.0 if calculation is not possible.
    """
    build_started_at = build.get('started_at', '')
    
    if not build_started_at or not pr_created_at:
        return 0.0
    
    try:
        build_start_dt = parse_datetime(build_started_at)
        pr_created_dt = parse_datetime(pr_created_at)
        
        # Calculate the difference in days
        delta = build_start_dt - pr_created_dt
        latency_days = delta.total_seconds() / (24 * 60 * 60)
        
        # Latency should be non-negative
        return max(0.0, latency_days)
    except (ValueError, TypeError):
        return 0.0


def extract_has_code_patch(all_comments: list[dict]) -> bool:
    """Extract the Has Code Patch feature.
    
    Has Code Patch is a boolean flag that indicates whether the PR comments
    include code snippets, code file references, or patch-related content.
    
    This is an adaptation of the original JIRA feature which checked if an
    issue included a code patch file attachment.
    
    Args:
        all_comments: List of all comments in the PR.
        
    Returns:
        True if any comment contains code-related content, False otherwise.
    """
    # Indicators of code patches or code snippets in comments
    code_indicators = [
        '```',           # Markdown code block
        '.java',         # Java file reference
        '.ts',           # TypeScript file reference
        '.tsx',          # TypeScript React file reference
        '.js',           # JavaScript file reference
        '.jsx',          # JavaScript React file reference
        '.py',           # Python file reference
        '.html',         # HTML file reference
        '.css',          # CSS file reference
        '.scss',         # SCSS file reference
        '.diff',         # Diff file
        '.patch',        # Patch file
        'diff --git',    # Git diff format
        '@@',            # Diff hunk header
    ]
    
    for comment in all_comments:
        body = comment.get('body', '')
        if body and any(indicator in body for indicator in code_indicators):
            return True
    
    return False


def load_build_log(build_id: int) -> Optional[dict]:
    """Load build log from the logs directory.
    
    Args:
        build_id: The build/job ID.
        
    Returns:
        Build log dictionary, or None if not found.
    """
    log_file = BUILD_LOGS_DIR / f"{build_id}.json"
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def extract_error_classes_from_log(log_content: str) -> Set[str]:
    """Extract failed class names and file names from build log content.
    
    This function identifies error-related classes and files from CI logs.
    It extracts:
    - Java test classes (e.g., teammates.e2e.cases.xxxTest)
    - TypeScript/JavaScript test files (e.g., src/web/xxx.spec.ts)
    
    Args:
        log_content: The raw log content string.
        
    Returns:
        Set of unique class/file names that appear in error context.
    """
    error_identifiers: Set[str] = set()
    
    if not log_content:
        return error_identifiers
    
    lines = log_content.split('\n')
    
    for line in lines:
        # Pattern 1: Java test class (teammates.xxx.xxxTest)
        # These appear in lines like:
        # "teammates.sqlui.webapi.GetFeedbackQuestionsActionTest > testExecute..."
        java_matches = re.findall(r'(teammates\.[a-zA-Z0-9_.]+(?:Test|E2ETest))', line)
        for match in java_matches:
            error_identifiers.add(match)
        
        # Pattern 2: TypeScript/JavaScript test file (FAIL src/web/xxx.spec.ts)
        # These appear in lines like:
        # "FAIL src/web/services/auth.service.spec.ts"
        if 'FAIL ' in line:
            ts_matches = re.findall(r'(src/web/[a-zA-Z0-9_/.-]+\.spec\.ts)', line)
            for match in ts_matches:
                error_identifiers.add(match)
        
        # Pattern 3: General test file failures
        # "FAIL src/web/app/components/xxx/xxx.component.spec.ts"
        ts_matches = re.findall(r'FAIL\s+(src/[a-zA-Z0-9_/.-]+)', line)
        for match in ts_matches:
            error_identifiers.add(match)
    
    return error_identifiers


def build_error_classes_index(pr_data: list[dict]) -> dict[int, Set[str]]:
    """Build an index of error classes for each build.
    
    Args:
        pr_data: List of PR dictionaries.
        
    Returns:
        Dictionary mapping build_id to set of error class/file names.
    """
    error_index: dict[int, Set[str]] = {}
    
    for pr in pr_data:
        build_associations = pr.get('build_comment_associations', [])
        
        for assoc in build_associations:
            build = assoc.get('build', {})
            build_id = build.get('id')
            conclusion = build.get('conclusion')
            
            if build_id and conclusion == 'failure':
                # Try to load the log for this build
                log_data = load_build_log(build_id)
                if log_data:
                    log_content = log_data.get('log_content', '')
                    error_classes = extract_error_classes_from_log(log_content)
                    error_index[build_id] = error_classes
                else:
                    error_index[build_id] = set()
    
    return error_index


def extract_is_shared_same_emsg(
    build_id: int,
    build_started_at: str,
    error_classes_index: dict[int, Set[str]],
    all_builds_info: list[dict]
) -> bool:
    """Extract the Is Shared Same Emsg feature for a build.
    
    Is Shared Same Emsg is a boolean value that indicates whether the current
    build failure shares the same error message (class/file) with any previous
    build failures.
    
    The comparison is based on:
    - Extracting error class names or file names from the build log
    - Checking if there's any overlap with error classes from previous builds
    
    Args:
        build_id: The current build ID.
        build_started_at: When the current build started (for filtering previous builds).
        error_classes_index: Index mapping build_id to error classes.
        all_builds_info: List of all builds with their info (for temporal ordering).
        
    Returns:
        True if current build shares error classes with any previous build, False otherwise.
    """
    # Get error classes for current build
    current_error_classes = error_classes_index.get(build_id, set())
    
    if not current_error_classes:
        return False
    
    # Check against previous builds (builds that started before current build)
    for prev_build in all_builds_info:
        prev_build_id = prev_build.get('build_id')
        prev_started_at = prev_build.get('started_at', '')
        
        # Skip current build and builds that started after current build
        if prev_build_id == build_id:
            continue
        if not prev_started_at or prev_started_at >= build_started_at:
            continue
        
        # Get error classes for previous build
        prev_error_classes = error_classes_index.get(prev_build_id, set())
        
        # Check for overlap
        if current_error_classes & prev_error_classes:
            return True
    
    return False


def extract_num_similar_failures(
    build_id: int,
    build_started_at: str,
    error_classes_index: dict[int, Set[str]],
    all_builds_info: list[dict]
) -> int:
    """Extract the Number of Similar Failures feature for a build.
    
    Number of Similar Failures represents the count of build failures that have
    an intersection with the set of failed test/exception classes extracted from
    the current build's failure messages.
    
    Formula: Sum(|F_curr ∩ F_pre_i|) for i = 1 to n
    
    Where:
    - F_pre_i is the set of failed test/exception class cases from the i-th previous build
    - F_curr is the set of failed test/exception class cases from the current build
    - n is the number of previous builds
    
    Args:
        build_id: The current build ID.
        build_started_at: When the current build started (for filtering previous builds).
        error_classes_index: Index mapping build_id to error classes.
        all_builds_info: List of all builds with their info (for temporal ordering).
        
    Returns:
        Count of previous builds that share error classes with current build.
    """
    # Get error classes for current build
    current_error_classes = error_classes_index.get(build_id, set())
    
    if not current_error_classes:
        return 0
    
    count = 0
    
    # Count previous builds that have overlap with current build's error classes
    for prev_build in all_builds_info:
        prev_build_id = prev_build.get('build_id')
        prev_started_at = prev_build.get('started_at', '')
        
        # Skip current build and builds that started after current build
        if prev_build_id == build_id:
            continue
        if not prev_started_at or prev_started_at >= build_started_at:
            continue
        
        # Get error classes for previous build
        prev_error_classes = error_classes_index.get(prev_build_id, set())
        
        # Check for overlap (intersection)
        if current_error_classes & prev_error_classes:
            count += 1
    
    return count


def extract_features_for_failed_builds(pr_data: list[dict]) -> pd.DataFrame:
    """Extract features for all failed builds in the dataset.
    
    Args:
        pr_data: List of PR dictionaries.
        
    Returns:
        DataFrame with one row per failed build and extracted features.
    """
    # Build the PR creation date index for num_parallel_issues calculation
    pr_date_index = build_pr_creation_date_index(pr_data)
    
    # Build the error classes index for is_shared_same_emsg calculation
    print("Building error classes index from build logs...")
    error_classes_index = build_error_classes_index(pr_data)
    print(f"  Loaded error classes for {len(error_classes_index)} builds")
    
    # Load commit files data for commit-level features
    print("Loading commit files data...")
    commit_files_index = load_commit_files_data()
    print(f"  Loaded file metrics for {len(commit_files_index)} commits")
    
    # First pass: collect all builds info for temporal ordering
    all_builds_info: list[dict] = []
    for pr in pr_data:
        for assoc in pr.get('build_comment_associations', []):
            build = assoc.get('build', {})
            if build.get('conclusion') == 'failure':
                all_builds_info.append({
                    'build_id': build.get('id'),
                    'started_at': build.get('started_at', ''),
                })
    
    records = []
    
    for pr in pr_data:
        pr_number = pr.get('pr_number')
        pr_url = pr.get('pr_url', '')
        pr_created_at = pr.get('pr_created_at', '')
        pr_author = pr.get('pr_author', '')
        all_comments = pr.get('all_comments', [])
        
        # Process each failed build association
        failed_build_associations = pr.get('failed_build_associations', [])
        
        for association in failed_build_associations:
            build = association.get('build', {})
            
            # Extract build info
            build_id = build.get('id')
            build_name = build.get('name', '')
            build_conclusion = build.get('conclusion', '')
            build_completed_at = build.get('completed_at', '')
            build_started_at = build.get('started_at', '')
            commit_sha = build.get('commit_sha', '')
            build_url = build.get('html_url', '')
            
            # Extract label info
            has_unrelated_discussion = association.get('has_unrelated_discussion', False)
            num_associated_comments = association.get('num_associated_comments', 0)
            
            # Extract features
            num_parallel_issues = extract_num_parallel_issues(build, pr_date_index)
            num_prior_comments = extract_num_prior_comments(build, all_comments)
            ci_latency_days = extract_ci_latency(build, pr_created_at)
            has_code_patch = extract_has_code_patch(all_comments)
            
            # Extract error-related features
            is_shared_same_emsg = extract_is_shared_same_emsg(
                build_id, 
                build_started_at,
                error_classes_index,
                all_builds_info
            )
            num_similar_failures = extract_num_similar_failures(
                build_id,
                build_started_at,
                error_classes_index,
                all_builds_info
            )
            
            # Extract commit-level features (file changes)
            commit_data = commit_files_index.get(commit_sha, {})
            has_config_files = commit_data.get('has_config_files', False)
            config_files_count = commit_data.get('config_files_count', 0)
            config_lines_added = commit_data.get('config_lines_added', 0)
            config_lines_deleted = commit_data.get('config_lines_deleted', 0)
            config_lines_modified = commit_data.get('config_lines_modified', 0)
            has_source_code = commit_data.get('has_source_code', False)
            source_code_files_count = commit_data.get('source_code_files_count', 0)
            source_code_lines_added = commit_data.get('source_code_lines_added', 0)
            source_code_lines_deleted = commit_data.get('source_code_lines_deleted', 0)
            source_code_lines_modified = commit_data.get('source_code_lines_modified', 0)
            total_files_changed = commit_data.get('total_files_changed', 0)
            total_additions = commit_data.get('total_additions', 0)
            total_deletions = commit_data.get('total_deletions', 0)
            
            record = {
                # Identifiers
                'pr_number': pr_number,
                'pr_url': pr_url,
                'build_id': build_id,
                'build_name': build_name,
                'build_url': build_url,
                'commit_sha': commit_sha,
                'build_started_at': build_started_at,
                'build_completed_at': build_completed_at,
                
                # PR info
                'pr_created_at': pr_created_at,
                'pr_author': pr_author,
                
                # Labels
                'has_unrelated_discussion': has_unrelated_discussion,
                'num_associated_comments': num_associated_comments,
                
                # PR/Build-level features
                'num_parallel_issues': num_parallel_issues,
                'num_prior_comments': num_prior_comments,
                'ci_latency_days': ci_latency_days,
                'has_code_patch': has_code_patch,
                'is_shared_same_emsg': is_shared_same_emsg,
                'num_similar_failures': num_similar_failures,
                
                # Commit-level features (file changes)
                'has_config_files': has_config_files,
                'config_files_count': config_files_count,
                'config_lines_added': config_lines_added,
                'config_lines_deleted': config_lines_deleted,
                'config_lines_modified': config_lines_modified,
                'has_source_code': has_source_code,
                'source_code_files_count': source_code_files_count,
                'source_code_lines_added': source_code_lines_added,
                'source_code_lines_deleted': source_code_lines_deleted,
                'source_code_lines_modified': source_code_lines_modified,
                'total_files_changed': total_files_changed,
                'total_additions': total_additions,
                'total_deletions': total_deletions,
            }
            
            records.append(record)
    
    return pd.DataFrame(records)


def main() -> None:
    """Main entry point for feature collection."""
    global PROJECT_NAME, PR_DATA_FILE, BUILD_LOGS_DIR, COMMIT_FILES_DATA_FILE, OUTPUT_FEATURES_FILE
    
    parser = argparse.ArgumentParser(
        description='Extract features from GitHub PR data for unrelated build failure prediction'
    )
    parser.add_argument(
        '--project',
        type=str,
        default=DEFAULT_PROJECT_NAME,
        help=f'Project name (default: {DEFAULT_PROJECT_NAME})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: {project}_features.csv)'
    )
    
    args = parser.parse_args()
    
    # Update global variables based on command line arguments
    PROJECT_NAME = args.project
    PR_DATA_FILE = DATA_DIR / f"{PROJECT_NAME}_pr_data.json"
    BUILD_LOGS_DIR = DATA_DIR / f"{PROJECT_NAME}_logs"
    COMMIT_FILES_DATA_FILE = DATA_DIR / f"{PROJECT_NAME}_commit_files.json"
    OUTPUT_FEATURES_FILE = Path(args.output) if args.output else DATA_DIR / f"{PROJECT_NAME}_features.csv"
    
    print("=" * 70)
    print(f"Feature Collection for {PROJECT_NAME}")
    print("=" * 70)
    print(f"  PR data file: {PR_DATA_FILE}")
    print(f"  Build logs dir: {BUILD_LOGS_DIR}")
    print(f"  Commit files: {COMMIT_FILES_DATA_FILE}")
    print(f"  Output file: {OUTPUT_FEATURES_FILE}")
    
    # Check if required files exist
    if not PR_DATA_FILE.exists():
        print(f"\n❌ Error: PR data file not found: {PR_DATA_FILE}")
        print("Please run github_pr_data_collector.py first.")
        return
    
    print(f"\nLoading {PROJECT_NAME} PR data...")
    pr_data = load_pr_data()
    print(f"Loaded {len(pr_data)} PRs")
    
    print("\nExtracting features for failed builds...")
    df = extract_features_for_failed_builds(pr_data)
    print(f"Extracted features for {len(df)} failed builds")
    
    if len(df) == 0:
        print("\n⚠️ No failed builds found. Exiting.")
        return
    
    # Save features to CSV
    df.to_csv(OUTPUT_FEATURES_FILE, index=False)
    print(f"\n✅ Features saved to: {OUTPUT_FEATURES_FILE}")
    
    # Display summary statistics
    print("\n" + "=" * 60)
    print("Feature Summary Statistics")
    print("=" * 60)
    
    print("\n--- num_parallel_issues ---")
    print(df['num_parallel_issues'].describe())
    
    print("\n--- num_prior_comments ---")
    print(df['num_prior_comments'].describe())
    
    print("\n--- ci_latency_days ---")
    print(df['ci_latency_days'].describe())
    
    print("\n--- has_code_patch ---")
    print(df['has_code_patch'].value_counts())
    print(f"Percentage with code patch: {df['has_code_patch'].mean() * 100:.1f}%")
    
    print("\n--- is_shared_same_emsg ---")
    print(df['is_shared_same_emsg'].value_counts())
    print(f"Percentage with shared error: {df['is_shared_same_emsg'].mean() * 100:.1f}%")
    
    print("\n--- num_similar_failures ---")
    print(df['num_similar_failures'].describe())
    
    print("\n--- Commit-level Features (File Changes) ---")
    print(f"has_config_files: {df['has_config_files'].sum()} ({df['has_config_files'].mean()*100:.1f}%)")
    print(f"has_source_code: {df['has_source_code'].sum()} ({df['has_source_code'].mean()*100:.1f}%)")
    print(f"config_lines_modified: mean={df['config_lines_modified'].mean():.1f}, max={df['config_lines_modified'].max()}")
    print(f"source_code_lines_modified: mean={df['source_code_lines_modified'].mean():.1f}, max={df['source_code_lines_modified'].max()}")
    print(f"source_code_files_count: mean={df['source_code_files_count'].mean():.1f}, max={df['source_code_files_count'].max()}")
    print(f"total_files_changed: mean={df['total_files_changed'].mean():.1f}, max={df['total_files_changed'].max()}")
    
    print("\n" + "=" * 60)
    print("Sample data (first 10 rows)")
    print("=" * 60)
    cols_to_show = ['pr_number', 'build_name', 
                    'num_parallel_issues', 'num_prior_comments', 
                    'ci_latency_days', 'has_code_patch', 
                    'is_shared_same_emsg', 'num_similar_failures']
    if 'has_unrelated_discussion' in df.columns:
        cols_to_show.append('has_unrelated_discussion')
    print(df[cols_to_show].head(10).to_string())


if __name__ == "__main__":
    main()

