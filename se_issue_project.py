import json
import re
import sqlite3
import string
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd


class Project:
    pre_processing_flag = False
    split_flag = False
    data_retrieval_flag = False
    versions = []

    def __init__(self, dataset, project_key):
        self.dataset = dataset
        self.key = project_key
        self.issues_df = pd.DataFrame()
        self.test_set = None
        self.train_set = None

    def print_id(self, source=""):
        print("==============================")
        print("Dataset: %s, Project Key: %s" % (self.dataset.name, self.key))
        print("Source: %s" % source)
        print("==============================")

    def load_data(self):

        # connect to db
        dataset_path = self.dataset.datasets_dir + self.dataset.name
        conn = sqlite3.connect(dataset_path)
        query = "select id, corpus, actual_effort from `case` where substr(id,0,instr(id,'-')) = '" + self.key + \
                "' order by id; "
        self.issues_df = pd.read_sql_query(query, conn)

        # Report the number of issues.
        self.print_id("Loading Project Data ")
        print('Number of issues: {:,}\n'.format(self.issues_df.shape[0]))
        self.issues_df.sample(10)
        self.data_retrieval_flag = True
        self.pre_process()
        self.to_db()
        self.drop_unqualified_issues()

    def data_reset(self):
        cur = self.dataset.conn.cursor()
        sql = "update `case_results` set set_type=?, bert_embedding=?, bert_prediction=?, RF_bert_prediction=?, " \
              "RF_bow_prediction=? "
        cur.execute(sql, ["train", None, None, None, None])
        self.dataset.conn.commit()

    def set_as_test(self, id_idxs):
        ids = self.issues_df.iloc[id_idxs]
        cur = self.dataset.conn.cursor()
        sql = "update `case_results` set set_type=? where id in (?)"

        cur.execute(sql, ["test", [i[0] for i in ids]])
        self.dataset.conn.commit()

    def to_db(self):
        cur = self.dataset.conn.cursor()
        for idx, row in self.issues_df.iterrows():
            sql = "insert or replace into `case_results` (id, processed_corpus, set_type, grammar_error," \
                  "rejection_flag, bert_embedding, true_label, bert_prediction, RF_bert_prediction, " \
                  "RF_bow_prediction) values ( (select id from `case` where id = ?), ?,?,?,?,?,?,?,?,?);"
            cur.execute(sql,
                        [row[0], row[1], "train", json.dumps([str(r) for r in row[4]]), "false", None, row[3], None,
                         None, None])

        self.dataset.conn.commit()

    def convert_estimate_to_ordinal(self, numeric_estimate, cat_type, labels_index=False):

        time_labels = ['One Hour', 'Half a day', 'A day', 'Half a week', 'One week', 'Two weeks', 'More than two weeks']
        fibonacci_labels = ['1', '2', '3', '5', '8', '13', '20', '40', '100']
        return_label = None

        if cat_type == "time":
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
        elif cat_type == "fibonacci":
            duration = float(numeric_estimate)
            if duration < 2:
                return_label = fibonacci_labels[0]
            elif duration < 3:
                return_label = fibonacci_labels[1]
            elif duration < 4:
                return_label = fibonacci_labels[2]
            elif duration < 7:
                return_label = fibonacci_labels[3]
            elif duration < 11:
                return_label = fibonacci_labels[4]
            elif duration < 16:
                return_label = fibonacci_labels[5]
            elif duration < 30:
                return_label = fibonacci_labels[6]
            elif duration < 70:
                return_label = fibonacci_labels[7]
            elif duration >= 70:
                return_label = fibonacci_labels[8]

            if labels_index:
                return fibonacci_labels.index(return_label)
            return return_label

    def pre_process(self):
        import language_tool_python
        tool = language_tool_python.LanguageTool('en-US')

        if self.pre_processing_flag:
            return self.issues_df

        category_efforts = []
        grammar_error = []
        for idx in range(0, len(self.issues_df)):
            text = self.issues_df.iat[idx, 1]

            # Remove whitespaces
            processed_text = text.strip()

            # remove non-ascii chars
            printable = set(string.printable)
            processed_text = "".join([x for x in filter(lambda x: x in printable, processed_text)])

            # remove any code snippet e.g. trace stack, logs
            processed_text = re.sub('\n{noformat}.*?{noformat}', ' ', processed_text, flags=re.DOTALL)
            processed_text = re.sub('\n{code.*?{code}', ' ', processed_text, flags=re.DOTALL)

            # Remove numbers and some punctuation
            processed_text = re.sub(r'\d+', '', processed_text)
            processed_text = processed_text.translate(
                str.maketrans(string.punctuation, '!  $% \'()  , . :;   ?           '))

            # truncate to 400 words leaving 112 additional space for BERT tokenizing (512 is bert maximum)
            words = processed_text.split()
            word_num = min(400, len(words))
            processed_text = " ".join(words[:word_num])

            # break text string into list for any newlines to later replace it with a stop char
            # make the text in one line with "." as a delimiter
            # if a line not finish by !?;. finish it with "."
            text_lines = re.findall('[^\r\n]+', processed_text)
            for idx2 in range(0, len(text_lines)):
                last_char = text_lines[idx2][-1]
                if last_char not in (".", ";", "?", "!"):
                    text_lines[idx2] += "."
            # concatenate the lines back as without new lines breaks
            processed_text = " ".join(text_lines).strip()

            # Check if the description grammar score is above using
            # Mention the language keyword
            #tool = language_tool_python.LanguageTool('en-US')
            errors = tool.check(processed_text)
            grammar_error.append(errors)

            processed_text = processed_text.lower()

            self.issues_df.iat[idx, 1] = processed_text

            # Convert person-hour to ordinal timings categories
            category_efforts.append(
                self.convert_estimate_to_ordinal(self.issues_df.iat[idx, 2], self.dataset.label_type(), True))

        if len(category_efforts) == len(self.issues_df):
            self.issues_df['actual_effort_category'] = category_efforts
        if len(grammar_error) == len(self.issues_df):
            self.issues_df['grammar_error'] = grammar_error

        self.pre_processing_flag = True
        return self.issues_df

    def drop_unqualified_issues(self):

        # Calculate natural balance
        import statistics as st
        import scipy
        number_errors = [len(row[4]) for idx, row in self.issues_df.iterrows()]
        mean = st.mean(number_errors)
        sigma = st.stdev(number_errors)

        errors_arr = []
        for idx, row in self.issues_df.iterrows():

            # drop any issue that has grammar error more than half number of the words
            threshold = 0.25
            # if len(row[4]) > round(mean+(sigma*3)): # natural tolerance
            if len(row[4]) > round(len(row[1].split()) * threshold):
                errors_arr.append(len(row[4]))

                self.issues_df.drop(idx, inplace=True)
        total = len(errors_arr) + len(self.issues_df)
        print("%d issues have been dropped out of %d (%.2f%%) since the number of grammar errors is more then %d%% "
              "words count." % (len(errors_arr), total, len(errors_arr) / total, threshold * 100))
        if len(errors_arr) > 2:
            me = st.mean(errors_arr)
            md = st.median(errors_arr)
            sd = st.stdev(errors_arr),
            iqr = scipy.stats.iqr(errors_arr)
            print("[Error stats] Mean: %.2f, Median: %.2f, STD: %.2f, IQR: %.2f" % (me, md, sd[0], iqr))

    def plot(self, labels, title):
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()

        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (10, 5)
        labels_ord = ['One Hour', 'Half a day', 'A day', 'Half a week', 'One week', 'Two weeks', 'More than two weeks']
        ay = sns.countplot(y=labels, order=labels_ord)

        plt.title('Class Distribution of project: %s from %s Dataset. %s' % (self.key, self.dataset.name, title))
        plt.xlabel('# of Data points')
        plt.ylabel('Category')

        import matplotlib as mpl
        ay.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

        plt.show()

    def data_split(self, random_state=0, re_split=False):
        if not re_split and self.split_flag:
            return None
        if not self.pre_processing_flag:
            self.pre_process()

        issues_copy = self.issues_df.copy()
        self.train_set = issues_copy.sample(frac=0.75, random_state=random_state)
        self.test_set = issues_copy.drop(self.train_set.index)

        ds_cat_type = self.dataset.label_type()
        self.plot([self.convert_estimate_to_ordinal(act_eff, ds_cat_type) for act_eff in self.train_set.actual_effort],
                  "Training Set")
        self.plot([self.convert_estimate_to_ordinal(act_eff, ds_cat_type) for act_eff in self.test_set.actual_effort],
                  "Testing Set")

        self.split_flag = True

    def get_train_set(self):
        self.data_split()
        return self.train_set

    def get_test_set(self):
        self.data_split()
        return self.test_set

    def confusion_matrix(self, true, pred):
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(true, pred)

    def roc_auc(self, true, pred_prob):
        true_mtrx = []
        class_num = len(pred_prob[0])

        for t in true:
            t_vctr = [0] * class_num
            t_vctr[int(t)] = 1
            true_mtrx.append(t_vctr)

        from sklearn.metrics import roc_auc_score
        weighted_roc_auc_ovr = roc_auc_score(true_mtrx, pred_prob, multi_class="ovr",
                                             average="weighted")
        return weighted_roc_auc_ovr

    def remove_dtop_words(self,text):
        from nltk.corpus import stopwords
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        result = []
        for t in text:
            t= [i for i in t]
            t[1] = " ".join([w for w in t[1].split() if w not in stop_words])
            result.append(t)
        return result

    def get_info_db(self, attribute, set_type, stop_words=True):
        cur = self.dataset.conn.cursor()

        sql = "select cr.id, cr.%s, cr.true_label from `case_results` cr where cr.set_type = ? and " \
              "cr.bert_embedding is not NULL and cr.id like '%s-%%';" % (attribute, self.key)
        cur.execute(sql, [set_type])
        data = cur.fetchall()

        if not stop_words:
            return self.remove_dtop_words(data)

        return data
