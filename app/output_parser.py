import numpy as np
import sys
import regex as re

def parse_output_performance(file):
    f = open(file, "r")
    content = f.read()
    print("Dataset\tProject\tMethod\tFold\tF1\tAUC_ROC")
    f1, auc_roc = {'RF-BERT':[],'BERT-BERT':[],'RF-BoW':[]},{'RF-BERT':[],'BERT-BERT':[],'RF-BoW':[]}
    for match in re.findall(r'.*\n.*\n.*\n.*Matrix', content):
        if "Average Metrics" in match:
            continue
        if "BERT" in match:
            method_s = "RF-BERT"
        elif "BoW" in match:
            method_s = "RF-BoW"
        else:
            method_s = "BERT-BERT"

        fold = re.search(r'fold\:(\d+)\]', match).group(1)


        if int(fold)<= len (f1[method_s]):

            for k, i in f1.items():
                print("%s\t%s\t%s\t%s\t%s\t%s" % (
                dataset, project, k, len(i), round(np.mean(i),4), round(np.mean(auc_roc[k]),4)))

            f1, auc_roc = {'RF-BERT':[],'BERT-BERT':[],'RF-BoW':[]},{'RF-BERT':[],'BERT-BERT':[],'RF-BoW':[]}

        dataset = re.search(r'dataset: (.*?)\.', match).group(1)
        project = re.search(r'Project: (.*?)\n', match).group(1)


        f1[method_s].append(float(re.search(r'F1 Score: (.*?),', match).group(1)))
        auc_roc[method_s].append(float(re.search(r'ROC AUC: (.*?),', match).group(1)))
    for k, i in f1.items():
        print("%s\t%s\t%s\t%s\t%s\t%s" % (
            dataset, project, k, len(i), round(np.mean(i), 4), round(np.mean(auc_roc[k]), 4)))



if __name__ == "__main__":
    file_name = sys.argv[1]
    parse_output_performance(file_name)