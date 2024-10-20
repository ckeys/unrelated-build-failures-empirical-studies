import os
import pandas as pd
import argparse
from xml.dom.minidom import parse

project_list = {'HIVE': 23972, 'CAMEL': 18199, 'CXF': 8713, 'DERBY': 7141, 'FELIX': 6540, 'HBASE': 271270,
                'HADOOPC': 18295, 'LUCENE': 10621, 'OPENEJB': 2153, 'OPENJPA': 2901, 'QPID': 8591, 'WICKET': 6989,
                'ATLAS': 4688, 'YARN': 11347, 'HDDS': 7322, 'AMBARI': 25877, 'HBASE': 27426, 'HDFS': 16802,
                'NIFI': 6318, 'HADOOP': 18495, 'IMPALA': 6268, 'PHOENIX': 6811, 'RANGER': 3945, 'SQOOP': 3474}


class JiraXMLProcessor:
    def __init__(self, base_path, project_name):
        self.base_path = base_path
        self.project_name = project_name
        self.fatal_xml = []

    @staticmethod
    def get_attribute_data(element, attribute_name):
        return element.getAttribute(attribute_name) if element.hasAttribute(attribute_name) else None

    @staticmethod
    def get_elements_data_by_tag_name(element, tag_name):
        element = element.getElementsByTagName(tag_name)[0]
        element_value = element.childNodes[0].data
        return element_value.replace(',', ';') if isinstance(element_value, str) else element_value

    def get_issue_report_dict(self, item):
        issuedict = {}

        def set_issue_dict_value(tag_name, dict_key, single_value=True, default=None):
            elements = item.getElementsByTagName(tag_name)
            if elements:
                element_value = elements[0].childNodes[0].data.replace("\t", " ").replace(",", ";").replace("\n", " ")
                issuedict[dict_key] = element_value if single_value else [e.childNodes[0].data for e in elements]
            else:
                issuedict[dict_key] = default

        set_issue_dict_value('title', 'title')
        set_issue_dict_value('summary', 'summary')
        set_issue_dict_value('status', 'status')
        set_issue_dict_value('resolution', 'resolution')
        set_issue_dict_value('created', 'created')
        set_issue_dict_value('updated', 'updated')
        set_issue_dict_value('resolved', 'resolved')
        set_issue_dict_value('version', 'version')

        if len(item.getElementsByTagName('project')) > 0:
            project = item.getElementsByTagName('project')[0]
            issuedict["project_data"] = self.get_elements_data_by_tag_name(item, "project")
            issuedict["project_id"] = self.get_attribute_data(project, "id")
            issuedict["project_name"] = self.get_attribute_data(project, "key")

        if len(item.getElementsByTagName('description')) > 0:
            description = item.getElementsByTagName("description")[0]
            if description.hasChildNodes():
                description_data = description.childNodes[0].data
                description_data = description_data.replace("\t", " ")
                description_data = description_data.replace(",", ";")
                issuedict["description"] = description_data.replace("\n", " ")

        if len(item.getElementsByTagName('key')) > 0:
            key = item.getElementsByTagName('key')[0]
            key_name = key.childNodes[0].data
            key_name = key_name.split('-')
            key_id = self.get_attribute_data(key, "id")
            issuedict["issue_id"] = key_name[1]
            issuedict["key_id"] = key_id
        if len(item.getElementsByTagName('type')) > 0:
            type_elem = item.getElementsByTagName('type')[0]
            issuedict["type_data"] = self.get_elements_data_by_tag_name(item, "type")
            issuedict["type_id"] = self.get_attribute_data(type_elem, "id")
            issuedict["type_iconUrl"] = self.get_attribute_data(type_elem, "iconUrl")

        if len(item.getElementsByTagName('priority')) > 0:
            priority = item.getElementsByTagName('priority')[0]
            issuedict["priority_data"] = self.get_elements_data_by_tag_name(item, "priority")
            issuedict["priority_id"] = self.get_attribute_data(priority, "id")
            issuedict["priority_iconUrl"] = self.get_attribute_data(priority, "iconUrl")

        if len(item.getElementsByTagName('assignee')) > 0:
            assignee = item.getElementsByTagName('assignee')[0]
            issuedict["assignee_data"] = self.get_elements_data_by_tag_name(item, "assignee")
            issuedict["assignee_username"] = self.get_attribute_data(assignee, "username")

        if len(item.getElementsByTagName('reporter')) > 0:
            reporter = item.getElementsByTagName('reporter')[0]
            issuedict["reporter_data"] = self.get_elements_data_by_tag_name(item, "reporter")
            issuedict["reporter_username"] = self.get_attribute_data(reporter, "username")

        if len(item.getElementsByTagName('issuelinks')) > 0:
            issuelinks = item.getElementsByTagName("issuelinks")[0]
            issuelinktype_dict = {"issue-link": "1"}
            issuelinktypegroup = issuelinks.getElementsByTagName("issuelinktype")
            for issuelinktype in issuelinktypegroup:
                issuelinktype_name = issuelinktype.getElementsByTagName("name")[0].childNodes[0].data
                issuelinktype_dict[f"is_{issuelinktype_name}"] = "0"
                issuelinktype_dict[f"{issuelinktype_name}_issue"] = ""
                if len(issuelinktype.getElementsByTagName('inwardlinks')) > 0:
                    issuelinktype_dict[f"is_{issuelinktype_name}"] = "1"
                    inwardlinks = issuelinktype.getElementsByTagName("inwardlinks")[0]
                    issuelink_group = inwardlinks.getElementsByTagName("issuelink")
                    for issuelink in issuelink_group:
                        issuelinktype_dict[f"{issuelinktype_name}_issue"] += self.get_elements_data_by_tag_name(
                            issuelink, "issuekey") + ";"
                if len(issuelinktype.getElementsByTagName('outwardlinks')) > 0:
                    issuelinktype_dict[f"is_{issuelinktype_name}"] = "1"
                    outwardlinks = issuelinktype.getElementsByTagName("outwardlinks")[0]
                    issuelink_group = outwardlinks.getElementsByTagName("issuelink")
                    for issuelink in issuelink_group:
                        issuelinktype_dict[f"{issuelinktype_name}_issue"] += self.get_elements_data_by_tag_name(
                            issuelink, "issuekey") + ";"
            issuedict.update(issuelinktype_dict)

        return issuedict

    def read_xml(self, xml_data):
        try:
            DOMTree = parse(xml_data)
            rss = DOMTree.documentElement
            channel = rss.getElementsByTagName('channel')[0]
            item = channel.getElementsByTagName("item")[0]
            issuedict = self.get_issue_report_dict(item)
            total_list = []

            if len(item.getElementsByTagName('comments')) > 0:
                comments = item.getElementsByTagName("comments")[0]
                commentgroup = comments.getElementsByTagName("comment")
                for one_comment in commentgroup:
                    comment_dict = {
                        "comment_id": self.get_attribute_data(one_comment, "id"),
                        "comment_author": self.get_attribute_data(one_comment, "author"),
                        "comment_created_at": self.get_attribute_data(one_comment, "created"),
                        "comment_content": one_comment.childNodes[0].data.replace("\t", " ").replace(",", ";").replace(
                            "\n", " ")
                    }
                    comment_dict.update(issuedict)
                    total_list.append(comment_dict)
            else:
                total_list.append(issuedict)

            return pd.DataFrame(total_list)

        except Exception as e:
            print(f"Error processing file {xml_data}: {e}")
            self.fatal_xml.append(os.path.basename(xml_data))
            return pd.DataFrame()

    def process_by_project(self):
        result_data = pd.DataFrame()
        path = os.path.join(self.base_path, self.project_name)
        for i in range(1, project_list.get(self.project_name)):
            # if i in [10108]:
                # for i in range(1, 100):
            xml_file = os.path.join(path, f"{self.project_name}_{i}.xml")
            if os.path.exists(xml_file):
                result_one_data = self.read_xml(xml_file)
                result_data = pd.concat([result_data, result_one_data])
                print(f"Currently Processing Issue {self.project_name}-{i}")
        return result_data

    def process(self):
        result_all_data = self.process_by_project()
        result_all_data = result_all_data.apply(lambda x: x.str.replace(',', ''))
        output_path = os.path.join(self.base_path, f"final_res/{self.project_name}_step1.csv")
        result_all_data.to_csv(output_path)
        print(result_all_data)
        print("Failed to process the following XML files:", self.fatal_xml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JIRA XML data.')
    parser.add_argument('--project_name', type=str, default='HIVE')
    parser.add_argument('--base_path', default='/data/jira/', help='Base path for JIRA XML data')

    args = parser.parse_args()

    processor = JiraXMLProcessor(base_path=args.base_path, project_name=args.project_name)
    processor.process()
