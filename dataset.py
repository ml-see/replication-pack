import os
import sqlite3
import wget

from se_issue_project import Project


class Dataset:
    repo_url = "anonymised"
    datasets_dir = "./datasets/"
    projects = None

    def __init__(self, dataset_name, url=None):
        self.name = dataset_name
        self.conn = None
        if url is not None:
            self.repo_url = url

    def print_id(self, source=""):
        print("==============================")
        print("Dataset: %s " % self.name)
        print("Source: %s" % source)
        print("==============================")

    def get_dataset(self):

        # if dataset DB is available locally or download it
        dataset_path = self.datasets_dir + self.name
        if not os.path.exists(dataset_path):
            if not os.path.exists(self.datasets_dir):
                os.makedirs(self.datasets_dir)
            print('Downloading %s from: %s' % (self.name, self.repo_url + self.name))
            wget.download(self.repo_url + self.name, dataset_path)

        # connect to the dataset DB
        try:
            conn = sqlite3.connect(dataset_path)
        except sqlite3.Error as e:
            print('Can not connect to %s. The Error is: %s' % (self.name, e))
            return None

        # get project key list
        query = "select substr(id,0,instr(id,'-')) proj_key from `case` group by proj_key;"
        self.conn = conn
        project_key_list = self.conn.cursor().execute(query).fetchall()

        self.print_id("Getting Dataset DB")
        print('Number of projects: {:,}\n'.format(len(project_key_list)))

        self.projects = []
        for project_key in project_key_list:
            # if project_key_list.index(project_key) > project_key_list.index(('ACCUMULO',)):continue
            proj = Project(self, project_key[0])
            proj.load_data()
            self.projects.append(proj)

        return self.projects

    def label_type(self):
        label_type = "fibonacci"
        if "ppi." in self.name or "jos." in self.name:
            label_type = "time"
        return label_type

    def pro_index(self, key):
        i = 0
        for p in self.projects:
            if p.key == key:
                return i
        i += 1
