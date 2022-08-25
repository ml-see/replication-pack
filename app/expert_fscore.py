import math

import numpy
from sklearn.metrics import f1_score
import sqlite3


def convert_estimate_to_ordinal( numeric_estimate, labels_index=False):
    time_labels = ['One Hour', 'Half a day', 'A day', 'Half a week', 'One week', 'Two weeks', 'More than two weeks']
    fibonacci_labels = ['1', '2', '3', '5', '8', '13', '20', '40', '100']
    return_label = None

    # convert estimate to hours
    duration = round(float(numeric_estimate) / 3600)

    if duration < 2:
        return_label = time_labels[0]
    elif duration < 6:
        return_label = time_labels[1]
    elif duration < 11:
        return_label = time_labels[2]
    elif duration < 31:
        return_label = time_labels[3]
    elif duration < 61:
        return_label = time_labels[4]
    elif duration < 121:
        return_label = time_labels[5]
    elif duration >= 121:
        return_label = time_labels[6]

    if labels_index:
        return time_labels.index(return_label)
    return return_label

def get_expert_actual_cost():
    dataset = "app/datasets/JOSSE.sqlite3"
    try:
        conn = sqlite3.connect(dataset)
    except sqlite3.Error as e:
        print('Can not connect to JOSSE Dataset (%s). The Error is: %s' % (dataset,e))
        return None

    # get project key list
    cur = conn.cursor()
    query = "select substr(id,0,instr(id,'-')) proj_key, expert_estimated_effort ee, actual_effort ae" \
            " from `case` where ee != 0 order by proj_key;"
    data = cur.execute(query).fetchall()
    project_key=''
    project_efforts={}
    list = {"actual":[],"expert":[]}
    for i in data:
        if project_key != i[0]:
            if project_key != '':
                project_efforts.update({project_key: list})
            project_key = i[0]
            list = {"actual":[],"expert":[]}
        list["actual"].append(convert_estimate_to_ordinal(i[2],True))
        list["expert"].append(convert_estimate_to_ordinal(i[1],True))
    project_efforts.update({project_key: list})


    return project_efforts



if __name__ == "__main__":
    project_efforts = get_expert_actual_cost()
    project_fscore = {}

    for project_key, efforts in project_efforts.items():
        y_true = efforts["actual"]
        y_pred = efforts["expert"]
        f1 = f1_score(y_true, y_pred, average='weighted')
        project_fscore.update({project_key:f1})
    f1s = []

    dota_teams = ["Project", "Number of Issues", "F1"]
    format_row = "{:>12}" * (len(dota_teams) + 1)
    data= []

    for project_key, f1 in project_fscore.items():
        num_issue = len(project_efforts[project_key]["actual"])

        if num_issue>=100:
            f1s.append(f1)
            data.append([project_key, num_issue, f1])

    print("{:<15} {:<20} {:<15}".format(*dota_teams))
    for row in data:
        print("{:<15} {:<20} {:<15}".format(*row))

    print("Average F1 of Expert Estimates is:" + str(numpy.average(f1s)))
