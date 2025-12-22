#!/usr/bin/env python3
"""
GitHub Project Selection Script for Replication Study (V2)

This script systematically identifies candidate projects for replication study.

Selection Process:
1. Search for popular open-source projects using GitHub Actions
2. Filter by: PR count, failed builds count
3. Rank by: unrelated failure keyword ratio in PR comments

Usage:
    python github_project_selection_v2.py

Requires:
    - GITHUB_TOKEN environment variable or default token
"""

import os
import time
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict, field

# GitHub API configuration
GITHUB_TOKEN = os.getenv(
    "GITHUB_TOKEN",
    ""
)
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}
BASE_URL = "https://api.github.com"

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "rebuttal_data" / "github" / "project_selection"

# Keywords for unrelated build failures
UNRELATED_KEYWORDS = [
    "unrelated failure",
    "unrelated to this",
    "not related to my change",
    "not related to this PR",
    "flaky test",
    "flaky build", 
    "infrastructure failure",
    "CI flake",
    "pre-existing failure",
    "known flaky",
    "intermittent failure"
]


@dataclass
class ProjectCandidate:
    """Data class for project candidate information."""
    # Required fields (no defaults) - must come first
    repo_full_name: str
    repo_url: str
    description: str
    stars: int
    forks: int
    language: str
    last_updated: str
    # Repository metrics
    commit_count: int
    # PR metrics
    total_prs: int
    closed_prs: int
    # CI metrics
    uses_github_actions: bool
    total_workflow_runs: int
    failed_workflow_runs: int
    failure_rate: float
    # Keyword metrics
    keyword_match_count: int
    total_pr_comments_sampled: int
    keyword_ratio: float
    # Optional fields (with defaults) - must come after required fields
    has_ci_workflows: bool = False  # Whether project has CI/testing workflows
    ci_workflow_names: List[str] = field(default_factory=list)  # Names of CI workflows
    sample_keyword_comments: List[Dict] = field(default_factory=list)
    # Eligibility
    is_eligible: bool = False
    eligibility_reason: str = ""
    # Ranking score
    rank_score: float = 0.0


def rate_limit_wait():
    """Check rate limit and wait if necessary."""
    try:
        response = requests.get(f"{BASE_URL}/rate_limit", headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            core_remaining = data["resources"]["core"]["remaining"]
            search_remaining = data["resources"]["search"]["remaining"]
            
            if core_remaining < 10:
                reset_time = data["resources"]["core"]["reset"]
                wait_time = reset_time - time.time() + 10
                if wait_time > 0:
                    print(f"  ⏳ Core rate limit low ({core_remaining}). Waiting {wait_time:.0f}s...")
                    time.sleep(min(wait_time, 60))
            
            if search_remaining < 5:
                reset_time = data["resources"]["search"]["reset"]
                wait_time = reset_time - time.time() + 10
                if wait_time > 0:
                    print(f"  ⏳ Search rate limit low ({search_remaining}). Waiting {wait_time:.0f}s...")
                    time.sleep(min(wait_time, 60))
    except Exception as e:
        print(f"  Warning: Could not check rate limit: {e}")
        time.sleep(2)


def search_popular_projects(
    min_stars: int = 1500,
    min_forks: int = 3000,
    max_results: int = 100,
    language: str = "Java"
) -> List[Dict]:
    """
    Search for popular open-source projects filtered by language, stars, and forks.
    
    Args:
        min_stars: Minimum star count (default: 1500)
        min_forks: Minimum fork count (default: 3500)
        max_results: Maximum results to return
        language: Programming language filter (default: Java)
    
    Returns:
        List of repository data
    """
    all_repos = []
    
    rate_limit_wait()
    
    # Search for popular repos with language, stars, and forks filter
    # fork:false ensures we only get original projects (not forks)
    query = f"stars:>{min_stars} forks:>{min_forks} archived:false fork:false language:{language}"
    url = f"{BASE_URL}/search/repositories"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": min(100, max_results)
    }
    
    print(f"  Searching {language} projects with >{min_stars} stars and >{min_forks} forks...")
    
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            repos = data.get("items", [])
            print(f"    Found {len(repos)} repositories")
            all_repos.extend(repos)
        else:
            print(f"    Error: {response.status_code} - {response.text[:100]}")
    except Exception as e:
        print(f"    Exception: {e}")
    
    # If we need more results, paginate
    if len(all_repos) < max_results and "Link" in response.headers:
        page = 2
        while len(all_repos) < max_results and page <= 5:  # Limit to 5 pages
            rate_limit_wait()
            params["page"] = page
            try:
                response = requests.get(url, headers=HEADERS, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    repos = data.get("items", [])
                    if not repos:
                        break
                    print(f"    Page {page}: Found {len(repos)} more repositories")
                    all_repos.extend(repos)
                else:
                    break
            except Exception as e:
                print(f"    Exception on page {page}: {e}")
                break
            page += 1
            time.sleep(2)
    
    # Remove duplicates
    seen = set()
    unique_repos = []
    for repo in all_repos:
        if repo["full_name"] not in seen:
            seen.add(repo["full_name"])
            unique_repos.append(repo)
    
    return unique_repos[:max_results]


def get_pr_count(repo_full_name: str) -> Tuple[int, int]:
    """
    Get PR counts for a repository.
    
    Returns:
        Tuple of (total_prs, closed_prs)
    """
    rate_limit_wait()
    
    try:
        # Get total PRs (all states)
        url = f"{BASE_URL}/repos/{repo_full_name}/pulls"
        params = {"state": "all", "per_page": 1}
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        
        if response.status_code != 200:
            return 0, 0
        
        # Parse Link header to get total count
        total_prs = 0
        if "Link" in response.headers:
            links = response.headers["Link"]
            # Extract last page number
            import re
            match = re.search(r'page=(\d+)>; rel="last"', links)
            if match:
                total_prs = int(match.group(1))
        else:
            total_prs = len(response.json())
        
        # Get closed PRs
        params = {"state": "closed", "per_page": 1}
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        
        closed_prs = 0
        if response.status_code == 200:
            if "Link" in response.headers:
                links = response.headers["Link"]
                match = re.search(r'page=(\d+)>; rel="last"', links)
                if match:
                    closed_prs = int(match.group(1))
            else:
                closed_prs = len(response.json())
        
        return total_prs, closed_prs
        
    except Exception as e:
        print(f"    Error getting PR count: {e}")
        return 0, 0


def get_commit_count(repo_full_name: str) -> int:
    """
    Get total commit count for a repository.
    
    Args:
        repo_full_name: Full repository name (owner/repo)
    
    Returns:
        Total number of commits
    """
    rate_limit_wait()
    
    try:
        # Use the contributors API to estimate commits, or commits API
        url = f"{BASE_URL}/repos/{repo_full_name}/commits"
        params = {"per_page": 1}
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        
        if response.status_code != 200:
            return 0
        
        # Parse Link header to get total count
        if "Link" in response.headers:
            links = response.headers["Link"]
            import re
            match = re.search(r'page=(\d+)>; rel="last"', links)
            if match:
                return int(match.group(1))
        
        return len(response.json())
        
    except Exception as e:
        print(f"    Error getting commit count: {e}")
        return 0


def check_github_actions(repo_full_name: str) -> Tuple[bool, int, int, float, bool, List[str]]:
    """
    Check GitHub Actions usage and get workflow run statistics.
    
    Returns:
        Tuple of (uses_actions, total_runs, failed_runs, failure_rate, has_ci_workflows, ci_workflow_names)
    """
    rate_limit_wait()
    
    # Keywords that indicate CI/testing workflows (case-insensitive)
    # Must contain at least one of these to be considered a CI workflow
    CI_WORKFLOW_KEYWORDS = [
        'test', 'testing', 'tests',
        'unit test', 'unit-test', 'unittest',
        'integration test', 'integration-test',
        'e2e', 'end-to-end',
        ' ci', 'ci ', 'ci-', '-ci',  # "CI" with boundaries to avoid false matches
        'continuous integration',
        'build and test', 'test and build',
    ]
    
    try:
        # First, get the list of workflows to check their names
        workflows_url = f"{BASE_URL}/repos/{repo_full_name}/actions/workflows"
        response = requests.get(workflows_url, headers=HEADERS, timeout=10)
        
        ci_workflow_names = []
        has_ci_workflows = False
        
        if response.status_code == 200:
            workflows_data = response.json()
            workflows = workflows_data.get("workflows", [])
            
            for wf in workflows:
                wf_name = wf.get("name", "").lower()
                wf_path = wf.get("path", "").lower()
                combined = f"{wf_name} {wf_path}"
                
                # Check for CI indicators (must have test/ci keywords)
                is_ci_workflow = any(kw in combined for kw in CI_WORKFLOW_KEYWORDS)
                
                if is_ci_workflow:
                    ci_workflow_names.append(wf.get("name", "Unknown"))
                    has_ci_workflows = True
        
        # Now get workflow runs statistics
        rate_limit_wait()
        url = f"{BASE_URL}/repos/{repo_full_name}/actions/runs"
        params = {"per_page": 100}
        
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        
        if response.status_code == 404:
            return False, 0, 0, 0.0, False, []
        
        if response.status_code != 200:
            return False, 0, 0, 0.0, False, []
        
        data = response.json()
        total_count = data.get("total_count", 0)
        
        if total_count == 0:
            return False, 0, 0, 0.0, False, []
        
        runs = data.get("workflow_runs", [])
        failed_count = sum(1 for run in runs if run.get("conclusion") == "failure")
        
        # Estimate total failed runs
        if len(runs) > 0:
            failure_rate = failed_count / len(runs)
            estimated_failed = int(total_count * failure_rate)
        else:
            failure_rate = 0.0
            estimated_failed = 0
        
        return True, total_count, estimated_failed, failure_rate, has_ci_workflows, ci_workflow_names
        
    except Exception as e:
        print(f"    Error checking GitHub Actions: {e}")
        return False, 0, 0, 0.0, False, []


def count_keyword_comments(
    repo_full_name: str, 
    keywords: List[str],
    sample_size: int = 100
) -> Tuple[int, int, float, List[Dict]]:
    """
    Count PR comments containing unrelated failure keywords.
    
    Returns:
        Tuple of (keyword_matches, total_sampled, ratio, sample_comments)
    """
    keyword_matches = 0
    sample_comments = []
    
    for keyword in keywords[:5]:  # Check first 5 keywords to save API calls
        rate_limit_wait()
        
        try:
            query = f'repo:{repo_full_name} "{keyword}" in:comments is:pr'
            url = f"{BASE_URL}/search/issues"
            params = {"q": query, "per_page": 10}
            
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                count = data.get("total_count", 0)
                keyword_matches += count
                
                # Get sample comments
                for item in data.get("items", [])[:2]:
                    sample_comments.append({
                        "keyword": keyword,
                        "pr_title": item.get("title", "")[:100],
                        "pr_url": item.get("html_url", "")
                    })
            
            time.sleep(1)
            
        except Exception as e:
            print(f"    Error searching for '{keyword}': {e}")
    
    # Estimate ratio based on total PRs (rough approximation)
    # We use keyword_matches / sample_size as a proxy
    ratio = keyword_matches / sample_size if sample_size > 0 else 0.0
    
    return keyword_matches, sample_size, ratio, sample_comments


def evaluate_project(repo: Dict) -> Optional[ProjectCandidate]:
    """
    Evaluate a single project against selection criteria.
    
    Selection Criteria:
    - Stars >= 1500
    - Forks >= 3500
    - Commits >= 15000
    - Uses GitHub Actions as primary CI (has test/CI workflows)
    - Workflow runs >= 500
    - Failed builds >= 100
    - PRs >= 500
    - Not a fork
    
    Args:
        repo: Repository data from GitHub API
    
    Returns:
        ProjectCandidate object or None if evaluation failed
    """
    # Minimum thresholds
    MIN_COMMITS = 15000
    MIN_WORKFLOW_RUNS = 500
    MIN_FAILED_RUNS = 100
    MIN_PRS = 500
    
    repo_full_name = repo["full_name"]
    stars = repo.get("stargazers_count", 0)
    forks = repo.get("forks_count", 0)
    
    print(f"\n  📊 Evaluating: {repo_full_name}")
    print(f"    Stars: {stars:,} | Forks: {forks:,}")
    
    # Get commit count first (to filter early)
    commit_count = get_commit_count(repo_full_name)
    print(f"    Commits: {commit_count:,}")
    
    # Early exit: Check commit count
    if commit_count < MIN_COMMITS:
        print(f"    Status: ❌ Insufficient commits ({commit_count:,} < {MIN_COMMITS:,}) - SKIPPED")
        return ProjectCandidate(
            repo_full_name=repo_full_name,
            repo_url=repo.get("html_url", ""),
            description=repo.get("description", "")[:200] if repo.get("description") else "",
            stars=stars, forks=forks,
            language=repo.get("language", "Unknown"),
            last_updated=repo.get("updated_at", ""),
            commit_count=commit_count,
            total_prs=0, closed_prs=0,
            uses_github_actions=False,
            total_workflow_runs=0, failed_workflow_runs=0, failure_rate=0.0,
            keyword_match_count=0, total_pr_comments_sampled=0, keyword_ratio=0.0,
            is_eligible=False,
            eligibility_reason=f"Insufficient commits ({commit_count:,} < {MIN_COMMITS:,})"
        )
    
    # Get PR count
    total_prs, closed_prs = get_pr_count(repo_full_name)
    print(f"    PRs: {total_prs:,} total, {closed_prs:,} closed")
    
    # Check GitHub Actions (now returns CI workflow info)
    uses_actions, total_runs, failed_runs, failure_rate, has_ci_workflows, ci_workflow_names = check_github_actions(repo_full_name)
    print(f"    CI: {total_runs:,} runs, {failed_runs:,} failed ({failure_rate*100:.1f}%)")
    print(f"    Has CI/Test workflows: {has_ci_workflows}")
    if ci_workflow_names:
        print(f"    CI Workflows: {', '.join(ci_workflow_names[:5])}")
    
    # Early exit: Skip projects without GitHub Actions
    if not uses_actions:
        print(f"    Status: ❌ Does not use GitHub Actions - SKIPPED")
        return ProjectCandidate(
            repo_full_name=repo_full_name,
            repo_url=repo.get("html_url", ""),
            description=repo.get("description", "")[:200] if repo.get("description") else "",
            stars=stars, forks=forks,
            language=repo.get("language", "Unknown"),
            last_updated=repo.get("updated_at", ""),
            commit_count=commit_count,
            total_prs=total_prs, closed_prs=closed_prs,
            uses_github_actions=False,
            total_workflow_runs=0, failed_workflow_runs=0, failure_rate=0.0,
            keyword_match_count=0, total_pr_comments_sampled=0, keyword_ratio=0.0,
            is_eligible=False,
            eligibility_reason="Does not use GitHub Actions"
        )
    
    # Early exit: Skip projects without CI workflows
    if not has_ci_workflows:
        print(f"    Status: ❌ No CI/testing workflows - SKIPPED")
        return ProjectCandidate(
            repo_full_name=repo_full_name,
            repo_url=repo.get("html_url", ""),
            description=repo.get("description", "")[:200] if repo.get("description") else "",
            stars=stars, forks=forks,
            language=repo.get("language", "Unknown"),
            last_updated=repo.get("updated_at", ""),
            commit_count=commit_count,
            total_prs=total_prs, closed_prs=closed_prs,
            uses_github_actions=uses_actions,
            total_workflow_runs=total_runs, failed_workflow_runs=failed_runs, failure_rate=failure_rate,
            keyword_match_count=0, total_pr_comments_sampled=0, keyword_ratio=0.0,
            is_eligible=False,
            eligibility_reason="No CI/testing workflows (GitHub Actions not primary CI)"
        )
    
    # Count keyword comments (only for projects with CI workflows)
    keyword_matches, total_sampled, keyword_ratio, sample_comments = count_keyword_comments(
        repo_full_name, UNRELATED_KEYWORDS
    )
    print(f"    Keywords: {keyword_matches} matches (ratio: {keyword_ratio:.4f})")
    
    # Determine eligibility (CI workflow check already passed)
    is_eligible = True
    eligibility_reason = "Meets all criteria"
    
    if total_runs < MIN_WORKFLOW_RUNS:
        is_eligible = False
        eligibility_reason = f"Insufficient workflow runs ({total_runs:,} < {MIN_WORKFLOW_RUNS:,})"
    elif failed_runs < MIN_FAILED_RUNS:
        is_eligible = False
        eligibility_reason = f"Insufficient failed runs ({failed_runs:,} < {MIN_FAILED_RUNS:,})"
    elif total_prs < MIN_PRS:
        is_eligible = False
        eligibility_reason = f"Insufficient PRs ({total_prs:,} < {MIN_PRS:,})"
    
    # Calculate ranking score (higher is better)
    # Weighted combination: PRs (30%), Failed builds (40%), Keyword ratio (30%)
    if is_eligible:
        pr_score = min(total_prs / 10000, 1.0)  # Normalize to 0-1
        failed_score = min(failed_runs / 5000, 1.0)
        keyword_score = min(keyword_ratio * 10, 1.0)
        rank_score = 0.3 * pr_score + 0.4 * failed_score + 0.3 * keyword_score
    else:
        rank_score = 0.0
    
    candidate = ProjectCandidate(
        repo_full_name=repo_full_name,
        repo_url=repo.get("html_url", ""),
        description=repo.get("description", "")[:200] if repo.get("description") else "",
        stars=stars,
        forks=forks,
        language=repo.get("language", "Unknown"),
        last_updated=repo.get("updated_at", ""),
        commit_count=commit_count,
        total_prs=total_prs,
        closed_prs=closed_prs,
        uses_github_actions=uses_actions,
        total_workflow_runs=total_runs,
        failed_workflow_runs=failed_runs,
        failure_rate=failure_rate,
        has_ci_workflows=has_ci_workflows,
        ci_workflow_names=ci_workflow_names,
        keyword_match_count=keyword_matches,
        total_pr_comments_sampled=total_sampled,
        keyword_ratio=keyword_ratio,
        sample_keyword_comments=sample_comments,
        is_eligible=is_eligible,
        eligibility_reason=eligibility_reason,
        rank_score=rank_score
    )
    
    status = "✅ ELIGIBLE" if is_eligible else f"❌ {eligibility_reason}"
    print(f"    Status: {status}")
    
    return candidate


def generate_report(candidates: List[ProjectCandidate], output_path: Path) -> None:
    """Generate a markdown report of the selection process."""
    
    eligible = [c for c in candidates if c.is_eligible]
    ineligible = [c for c in candidates if not c.is_eligible]
    
    # Sort eligible by rank score
    eligible_sorted = sorted(eligible, key=lambda x: x.rank_score, reverse=True)
    
    report = []
    report.append("# GitHub Project Selection Report for Replication Study (Java Projects)\n\n")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append(f"**Language Filter**: Java (to match Apache projects in main study)\n\n")
    report.append(f"**Total Candidates Evaluated**: {len(candidates)}\n\n")
    report.append(f"**Eligible Projects**: {len(eligible)}\n\n")
    report.append(f"**Ineligible Projects**: {len(ineligible)}\n\n")
    
    report.append("## Selection Criteria\n\n")
    report.append("| Criterion | Requirement | Rationale |\n")
    report.append("|-----------|-------------|----------|\n")
    report.append("| C1. Language | Java | Consistent with Apache projects in main study |\n")
    report.append("| C2. Stars | ≥1,500 | Indicates project popularity and community adoption |\n")
    report.append("| C3. Forks | ≥3,000 | Indicates active community engagement |\n")
    report.append("| C4. Commits | ≥15,000 | Ensures substantial development history |\n")
    report.append("| C5. Original Project | Not a fork | Avoid duplicates |\n")
    report.append("| C6. CI Platform | Uses GitHub Actions | Different from Jenkins (Apache projects) |\n")
    report.append("| C7. Primary CI | Has CI/testing workflows | Ensures GitHub Actions is the main CI (not just docs/release) |\n")
    report.append("| C8. Workflow Runs | ≥500 total runs | Sufficient CI history |\n")
    report.append("| C9. Failed Builds | ≥100 failed builds | Sufficient failure data |\n")
    report.append("| C10. PR Activity | ≥500 PRs | Active development |\n\n")
    
    report.append("## Ranking Methodology\n\n")
    report.append("Projects are ranked by a weighted score:\n")
    report.append("- **30%**: PR count (normalized)\n")
    report.append("- **40%**: Failed build count (normalized)\n")
    report.append("- **30%**: Unrelated failure keyword ratio in PR comments\n\n")
    
    report.append("## Eligible Projects (Ranked)\n\n")
    if eligible_sorted:
        report.append("| Rank | Repository | Stars | Forks | Commits | PRs | Failed Builds | Score |\n")
        report.append("|------|------------|-------|-------|---------|-----|---------------|-------|\n")
        
        for i, c in enumerate(eligible_sorted, 1):
            report.append(
                f"| {i} | [{c.repo_full_name}]({c.repo_url}) | "
                f"{c.stars:,} | {c.forks:,} | {c.commit_count:,} | {c.total_prs:,} | "
                f"{c.failed_workflow_runs:,} | {c.rank_score:.3f} |\n"
            )
    else:
        report.append("*No eligible projects found.*\n")
    
    report.append("\n## Top Candidate Details\n\n")
    for i, c in enumerate(eligible_sorted[:5], 1):
        report.append(f"### {i}. {c.repo_full_name}\n\n")
        report.append(f"- **URL**: {c.repo_url}\n")
        report.append(f"- **Description**: {c.description}\n")
        report.append(f"- **Language**: {c.language}\n")
        report.append(f"- **Stars**: {c.stars:,}\n")
        report.append(f"- **Forks**: {c.forks:,}\n")
        report.append(f"- **Commits**: {c.commit_count:,}\n")
        report.append(f"- **Total PRs**: {c.total_prs:,}\n")
        report.append(f"- **Workflow Runs**: {c.total_workflow_runs:,}\n")
        report.append(f"- **Failed Builds**: {c.failed_workflow_runs:,} ({c.failure_rate*100:.1f}%)\n")
        report.append(f"- **Has CI/Test Workflows**: {c.has_ci_workflows}\n")
        if c.ci_workflow_names:
            report.append(f"- **CI Workflows**: {', '.join(c.ci_workflow_names[:5])}\n")
        report.append(f"- **Keyword Matches**: {c.keyword_match_count}\n")
        if c.sample_keyword_comments:
            report.append(f"- **Sample Comments**:\n")
            for comment in c.sample_keyword_comments[:3]:
                report.append(f"  - [{comment['keyword']}] {comment['pr_title'][:50]}...\n")
        report.append("\n")
    
    report.append("\n## Ineligible Projects (Sample)\n\n")
    report.append("| Repository | Language | Stars | Reason |\n")
    report.append("|------------|----------|-------|--------|\n")
    for c in ineligible[:15]:
        report.append(f"| {c.repo_full_name} | {c.language} | {c.stars:,} | {c.eligibility_reason} |\n")
    
    if len(ineligible) > 15:
        report.append(f"\n*... and {len(ineligible) - 15} more ineligible projects*\n")
    
    with open(output_path, "w") as f:
        f.write("".join(report))
    
    print(f"\n📄 Report saved to: {output_path}")


def main():
    """Main function to run the project selection process."""
    
    # Configuration - Java language only for consistency with Apache projects
    TARGET_LANGUAGE = "Java"
    MIN_STARS = 1500
    MIN_FORKS = 3000
    MIN_COMMITS = 15000
    MAX_PROJECTS_TO_EVALUATE = 50
    
    print("=" * 70)
    print("GitHub Project Selection for Replication Study (V3 - Java Only)")
    print("=" * 70)
    print(f"\nSelection Criteria:")
    print(f"  - Language: {TARGET_LANGUAGE}")
    print(f"  - Minimum Stars: {MIN_STARS:,}")
    print(f"  - Minimum Forks: {MIN_FORKS:,}")
    print(f"  - Minimum Commits: {MIN_COMMITS:,}")
    print(f"  - Must NOT be a fork (original project only)")
    print(f"  - Must use GitHub Actions as PRIMARY CI (with test/CI workflows)")
    print(f"  - Minimum 500 workflow runs")
    print(f"  - Minimum 100 failed builds")
    print(f"  - Minimum 500 PRs")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Search for popular Java projects (sorted by stars)
    print(f"\n📌 Step 1: Searching for popular {TARGET_LANGUAGE} projects (by stars)...")
    repos = search_popular_projects(
        min_stars=MIN_STARS,
        min_forks=MIN_FORKS,
        max_results=200,  # Get more to find eligible ones
        language=TARGET_LANGUAGE
    )
    print(f"\nFound {len(repos)} unique {TARGET_LANGUAGE} repositories")
    
    # Step 2: Evaluate each project (in star order - most popular first)
    print(f"\n📌 Step 2: Evaluating top {MAX_PROJECTS_TO_EVALUATE} projects (by stars)...")
    candidates = []
    
    for i, repo in enumerate(repos[:MAX_PROJECTS_TO_EVALUATE], 1):
        print(f"\n[{i}/{min(len(repos), MAX_PROJECTS_TO_EVALUATE)}]", end="")
        candidate = evaluate_project(repo)
        if candidate:
            candidates.append(candidate)
        time.sleep(1)
    
    # Step 3: Save results
    print("\n📌 Step 3: Saving results...")
    
    # Save raw data
    raw_data_path = OUTPUT_DIR / "candidates_java_v3.json"
    with open(raw_data_path, "w") as f:
        json.dump([asdict(c) for c in candidates], f, indent=2, default=str)
    print(f"Raw data saved to: {raw_data_path}")
    
    # Generate report
    report_path = OUTPUT_DIR / "selection_report_java_v3.md"
    generate_report(candidates, report_path)
    
    # Summary
    eligible = [c for c in candidates if c.is_eligible]
    eligible_sorted = sorted(eligible, key=lambda x: x.rank_score, reverse=True)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Language filter: {TARGET_LANGUAGE}")
    print(f"Total candidates evaluated: {len(candidates)}")
    print(f"Eligible projects: {len(eligible)}")
    
    if eligible_sorted:
        print("\n🏆 Top 5 eligible Java projects (by ranking score):")
        for i, c in enumerate(eligible_sorted[:5], 1):
            print(f"  {i}. {c.repo_full_name}")
            print(f"     Stars: {c.stars:,} | Forks: {c.forks:,} | Commits: {c.commit_count:,}")
            print(f"     PRs: {c.total_prs:,} | Failed: {c.failed_workflow_runs:,} | Score: {c.rank_score:.3f}")
            if c.ci_workflow_names:
                print(f"     CI Workflows: {', '.join(c.ci_workflow_names[:3])}")
        
        # Recommend the top project
        top = eligible_sorted[0]
        print(f"\n✅ RECOMMENDED PROJECT: {top.repo_full_name}")
        print(f"   This is the highest-ranked Java project that uses GitHub Actions as primary CI.")
    else:
        print("\n⚠️ No eligible Java projects found with the current criteria.")


def test_teammates_in_search():
    """
    Test function to check if TEAMMATES would appear in the Java project search results.
    This helps verify whether TEAMMATES meets the initial search criteria.
    """
    print("=" * 70)
    print("Testing: Is TEAMMATES in the Java Project Search Results?")
    print("=" * 70)
    
    # First, get TEAMMATES repo info directly
    print("\n📌 Step 1: Fetching TEAMMATES repository info...")
    rate_limit_wait()
    
    teammates_url = f"{BASE_URL}/repos/TEAMMATES/teammates"
    response = requests.get(teammates_url, headers=HEADERS, timeout=10)
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch TEAMMATES info: {response.status_code}")
        return
    
    teammates = response.json()
    print(f"\n📊 TEAMMATES Repository Info:")
    print(f"  - Full Name: {teammates['full_name']}")
    print(f"  - Language: {teammates.get('language', 'Unknown')}")
    print(f"  - Stars: {teammates.get('stargazers_count', 0):,}")
    print(f"  - Forks: {teammates.get('forks_count', 0):,}")
    print(f"  - Is Fork: {teammates.get('fork', False)}")
    print(f"  - Archived: {teammates.get('archived', False)}")
    
    # Get commit count
    commit_count = get_commit_count("TEAMMATES/teammates")
    print(f"  - Commits: {commit_count:,}")
    
    # Check against our criteria
    MIN_STARS = 1500
    MIN_FORKS = 3000
    MIN_COMMITS = 15000
    
    print(f"\n📋 Checking against selection criteria:")
    stars = teammates.get('stargazers_count', 0)
    forks = teammates.get('forks_count', 0)
    is_java = teammates.get('language', '') == 'Java'
    is_fork = teammates.get('fork', False)
    is_archived = teammates.get('archived', False)
    
    criteria_results = [
        ("Language == Java", is_java, f"{teammates.get('language', 'Unknown')}"),
        (f"Stars >= {MIN_STARS:,}", stars >= MIN_STARS, f"{stars:,}"),
        (f"Forks >= {MIN_FORKS:,}", forks >= MIN_FORKS, f"{forks:,}"),
        (f"Commits >= {MIN_COMMITS:,}", commit_count >= MIN_COMMITS, f"{commit_count:,}"),
        ("Not a fork", not is_fork, str(is_fork)),
        ("Not archived", not is_archived, str(is_archived)),
    ]
    
    all_pass = True
    for criterion, passed, value in criteria_results:
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}: {value}")
        if not passed:
            all_pass = False
    
    # Now search for Java projects and see if TEAMMATES appears
    print(f"\n📌 Step 2: Searching for Java projects with our criteria...")
    repos = search_popular_projects(
        min_stars=MIN_STARS,
        min_forks=MIN_FORKS,
        max_results=200,
        language="Java"
    )
    
    repo_names = [r['full_name'] for r in repos]
    teammates_in_results = "TEAMMATES/teammates" in repo_names
    
    print(f"\n📊 Search Results:")
    print(f"  - Total Java repos found: {len(repos)}")
    print(f"  - TEAMMATES in results: {'✅ YES' if teammates_in_results else '❌ NO'}")
    
    if teammates_in_results:
        idx = repo_names.index("TEAMMATES/teammates")
        print(f"  - TEAMMATES position: #{idx + 1} (by stars)")
    
    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if all_pass and teammates_in_results:
        print("✅ TEAMMATES meets all initial search criteria and appears in search results.")
    elif not all_pass:
        print("❌ TEAMMATES does NOT meet all initial search criteria:")
        for criterion, passed, value in criteria_results:
            if not passed:
                print(f"   - Failed: {criterion} (actual: {value})")
    elif not teammates_in_results:
        print("❌ TEAMMATES meets criteria but doesn't appear in search results (API limitation?)")
    
    return all_pass, teammates_in_results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-teammates":
        test_teammates_in_search()
    else:
        main()

