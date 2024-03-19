'''
@FileName       : jira_crawler.py
@Author         : andie.huang
@Date           :2021/12/28


JIRA Crawler

This script is designed to crawl issue reports from the Apache JIRA instance for various Apache projects.
It retrieves XML data of the issues and saves them locally.

Usage:
    python jira_crawler.py [project] [base_path] [issue_idx] [sleeptime]

    - project: The name of the Apache project to crawl (e.g., HIVE, CAMEL, CXF).
    - base_path: The base directory where crawled data will be saved.
    - issue_idx: The starting index of the issue to crawl.
    - sleeptime: Optional. Time in seconds to sleep between each crawl request. Default is 5 seconds.

Example:
    python jira_crawler.py HIVE /Users/user/Documents/jira_data 1 5


'''

import requests
import urllib.request as urllib2
import os
import sys
import time
import random
import argparse
from retrying import retry

'''
We defined the amount of each project by the date of 01/02/2022 to retrieve
'''
project_list = {'HIVE': 23972, 'CAMEL': 18199, 'CXF': 8713, 'DERBY': 7141, 'FELIX': 6540, 'HBASE': 271270,
                'HADOOPC': 18295,
                'LUCENE': 10621, 'OPENEJB': 2153, 'OPENJPA': 2901, 'QPID': 8591, 'WICKET': 6989, 'ATLAS': 4688,
                'YARN': 11347, 'HDDS': 7322, 'AMBARI': 25877, 'HBASE': 27426, 'HDFS': 16802, 'NIFI': 6318,
                'HADOOP': 18495, 'IMPALA': 6268, 'PHOENIX': 6811, 'RANGER': 3945, 'SQOOP': 3474,'RATIS':1732}

class JiraCrawler(object):

    def __init__(self, project_name='HIVE', issue_amount=10000, base_path='/Users/yonghui.huang/Otago/jira',
                 sleeptime=5):
        self.useragents = [
            'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50',
            'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0',
            'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)',
            'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1',
            'Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11',
            'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)',
            'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)',
            'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)',
            'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)',
            'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)']
        self.amount_map = project_list
        store_path = f'{base_path}/{project_name}/'
        if not os.path.exists(store_path):
            os.mkdir(store_path)
        self.project_name = project_name
        self.issue_amount = self.amount_map.get(project_name)
        self.base_path = f'{base_path}/{project_name}/'
        self.sleeptime = sleeptime

    def construct_url(self, project_name, issue_idx):
        return f'https://issues.apache.org/jira/si/jira.issueviews:issue-xml/{project_name}-{issue_idx}/{project_name}-{issue_idx}.xml'

    @retry
    def crawl_issue_report(self, issue_idx, end=None):
        start = issue_idx
        end = self.issue_amount if end is None else end
        for idx in range(start, end + 1):
            print("[Data crawling] Currently crawling issue report {}-{}".format(self.project_name, idx))
            base_url = self.construct_url(self.project_name, idx)
            print(f'On Url {base_url}')
            html = self.getHtml(base_url)
            print("[Data crawling] Getting the html file content is DONE, and then save the html file.")
            self.saveHtml(f'{self.project_name}_{idx}', html)
            print("[Data crawling] Done with issue report {}-{}!".format(self.project_name, idx))
            time.sleep(random.randint(0, self.sleeptime))

    def getHtml(self, url):
        headers = {'User-Agent': random.choice(self.useragents)}
        request = urllib2.Request(url, headers=headers)
        response = urllib2.urlopen(request)
        comments = response.read()
        return comments

    def saveHtml(self, file_name, file_content):
        store_path = self.base_path + file_name + ".xml"
        print(f'Filename:{store_path}')
        import os
        if os.path.exists(store_path):
            return

        with open(store_path, "wb") as f:
            f.write(file_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apache JIRA Crawler")
    parser.add_argument("project", help="Name of the Apache project to crawl (e.g., HIVE, CAMEL, CXF)")
    parser.add_argument("base_path", help="Base directory where crawled data will be saved")
    parser.add_argument("issue_idx", type=int, help="Starting index of the issue to crawl")
    parser.add_argument("--sleeptime", type=int, default=5,
                        help="Time in seconds to sleep between each crawl request. Default is 5 seconds.")

    args = parser.parse_args()

    print("===============================================>")
    print(f'''Currently crawling data for project {args.project} from idx {args.issue_idx} into dir {args.base_path}''')
    JiraCrawler(project_name=args.project, base_path=args.base_path, sleeptime=args.sleeptime).crawl_issue_report(
        issue_idx=args.issue_idx)