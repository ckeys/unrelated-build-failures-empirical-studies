import argparse
from jira import JIRA

class JiraProjectAnalyzer:
    def __init__(self, jira_url, username, password):
        self.jira = JIRA(server=jira_url, basic_auth=(username, password))
        self.project_info = {}
    def get_top_projects(self, top_n=100):
        projects = self.jira.projects()
        for project in projects:
            project_key = project.key
            print(project.name)
            issue_count = self.jira.search_issues(f'project={project_key}', maxResults=0).total
            self.project_info[project_key] = {
                'name': project.name,
                'issue_count': issue_count
            }
        sorted_projects = sorted(self.project_info.items(), key=lambda x: x[1]['issue_count'], reverse=True)
        return sorted_projects[:top_n]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JIRA Project Analyzer")
    jira_url = 'https://your-jira-instance.com'
    parser.add_argument("-username", help="Username for authentication")
    parser.add_argument("-password", help="Password for authentication")
    args = parser.parse_args()

    analyzer = JiraProjectAnalyzer(jira_url, args.username, args.password)
    top_projects = analyzer.get_top_projects()

    for project_key, project_data in top_projects:
        print(f"Project: {project_data['name']} | Issue Count: {project_data['issue_count']}")