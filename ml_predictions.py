import json

from se_issue_bert import SEIssueBert
from dataset import Dataset
from se_issue_rf import SEIssueRF


def print_out_rejected_issues(project):
    """
    """
    sql = "select id, processed_corpus, grammar_error from `case_results` where substr(id,0, instr(id,'-')) =?"
    cur = project.dataset.conn.cursor()
    data = cur.execute(sql, [project.key])
    text = []

    fetched_data = data.fetchall()
    for issue in fetched_data:
        if len(json.loads(issue[2])) > round(len(issue[1].split()) / 2, 2):
            sql = "update `case_results` set 'rejection_flag' = ? where id = ?"
            cur.execute(sql, ['true', issue[0]])
            text.append("Issue (%s) is eliminated since its has %d errors while it is %d word/s. "
                        "Below is the issue corpus:\n%s" % (
                            issue[0], len(json.loads(issue[2])), len(issue[1].split()), issue[1]))
    project.dataset.conn.commit()

    if len(text) > 0:
        project.print_id("printing %d (%f) rejected issues. Project has %d issues." %
                         (len(text), round(len(text) / len(fetched_data), 2), len(fetched_data)))
        for t in text:
            print(t)


def print_average_metrics(metrics, title=""):
    import statistics
    f1 = []
    roc_auc = []
    cm = []
    for metric in metrics:
        f1.append(metric[3]["f1"])
        roc_auc.append(metric[3]["roc_auc"])
        cm.append(metric[3]["conf_mtrx"])
    r_f1 = round(statistics.mean(f1), 2)
    r_roc_auc = round(statistics.mean(roc_auc), 2)

    print("[%s Average Metrics ] F1: %f, ROC AUC: %f." % (title, r_f1, r_roc_auc,))


def examine_errors(db_name):
    import re
    import language_check
    import statistics
    import scipy
    tool = language_check.LanguageTool('en-US')
    db_obj = Dataset(db_name)
    db_obj.get_dataset()
    ts_err_ratio = []
    txt_err_ratio = []
    tsa = []
    issue_cnt = 0
    for project in db_obj.projects:
        # if proj_obj.key != "NETBEANS": continue
        for idx, issue in project.issues_df.iterrows():

            issue_cnt += 1
            corpus = issue[1]
            if len(corpus) > 200000:
                continue

            ts = re.findall('\n{noformat}.*?{noformat}', corpus, flags=re.DOTALL)
            ts += re.findall('\n{code.*?{code}', corpus, flags=re.DOTALL)
            ts = " ".join(ts)

            if len(ts) > 0:
                tsa.append(ts)
                errors = len(tool.check(ts))
                words = len(corpus.split())
                ts_err_ratio.append({"key": issue[0], "r": errors / words})
            else:
                errors = len(tool.check(corpus))
                words = len(corpus.split())
                txt_err_ratio.append({"key": issue[0], "r": errors / words})
        print("Project Key is:%s" % project.key)
        print("Number of issues with trace stack is: %d out of %d.\n" % (len(tsa), issue_cnt))
        print("Mean(%.2f%%), Median(%.2f%%), STD(%.2f%%), IQR(%.2f%%) "
              "of grammar error ratio to number of words in Trace Stack (code).\n"
              "Mean(%.2f%%), Median(%.2f%%), STD(%.2f%%), IQR(%.2f%%) "
              "of grammar error ratio to number of words in description text.\n"
              % (statistics.mean([i["r"] for i in ts_err_ratio]),
                 statistics.median([i["r"] for i in ts_err_ratio]),
                 statistics.stdev([i["r"] for i in ts_err_ratio]),
                 scipy.stats.iqr([i["r"] for i in ts_err_ratio]),

                 statistics.mean([i["r"] for i in txt_err_ratio]),
                 statistics.median([i["r"] for i in txt_err_ratio]),
                 statistics.stdev([i["r"] for i in txt_err_ratio]),
                 scipy.stats.iqr([i["r"] for i in txt_err_ratio])

                 ))


if __name__ == "__main__":

    from sklearn.model_selection import StratifiedKFold

    #
    datasets = [  "porru.sqlite3","jos.sqlite3", "deep-se.sqlite3"]


    se_issue_bert_models = []
    pro_objs = []
    for name in datasets:
        dataset_obj = Dataset(name)
        dataset_obj.get_dataset()
        results = {"BERT": [], "BERT_RF": [], "BoW_RF": []}
        for proj_obj in dataset_obj.projects:

            label_set = [i for i in set(proj_obj.issues_df["actual_effort_category"])]
            if len(label_set) < 2:
                proj_obj.print_id("Dropping the project because it has only one class: %d" % label_set[0])
                continue
            model = SEIssueBert(proj_obj)

            max_fold = 5
            labels = [i for i in proj_obj.issues_df["actual_effort_category"]]
            class_ = -1
            for label in label_set:
                label_cnt = labels.count(label)
                if max_fold > label_cnt:
                    max_fold = label_cnt
                    class_ = label
            if max_fold < 2:
                proj_obj.print_id("Dropping the project because a class (%d) has only one case." % class_)
                continue
            proj_obj.print_id("Number of folds are: %d" % max_fold)
            kf = StratifiedKFold(n_splits=max_fold, shuffle=True, random_state=model.seed_val)
            i = 1

            for train_index, test_index in kf.split(proj_obj.issues_df, labels):

                fold_id = "[fold:%d] dataset: %s, Project: %s" % (i, proj_obj.dataset.name, proj_obj.key)
                print("====================================== %s" % fold_id)

                train, test = proj_obj.issues_df.iloc[train_index], proj_obj.issues_df.iloc[test_index]
                model.fine_tune(train)
                result = model.evaluate(test, fold_id)
                results["BERT"].append(result)

                model.save_model()
                model.load_saved_model()
                model.embedding()

                rf_model = SEIssueRF(proj_obj)
                result = rf_model.bert_fit_predict(fold_id)
                results["BERT_RF"].append(result)
                result = rf_model.bow_fit_predict(fold_id)
                results["BoW_RF"].append(result)
                i += 1


                if i > 5:
                    break
                proj_obj.data_reset()

        print_average_metrics(results["BERT"], "BERT")
        print_average_metrics(results["BERT_RF"], "BERT_RF")
        print_average_metrics(results["BoW_RF"], "BoW_RF")

