import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


class SEIssueRF:
    def __init__(self, data):
        self.data = data

    def get_label_indices(self, labels):
        label_set = [i for i in set(labels)]
        label_set.sort()
        label_idx = []
        for i in labels:
            label_idx.append(label_set.index(i))
        return label_idx

    def predict(self, test_embeddings, test_labels, cls, round_id):

        label_set = [i for i in set(test_labels)]
        label_set.sort()
        test_labels_idx = []
        for i in test_labels:
            test_labels_idx.append(label_set.index(i))

        predictions = cls.predict(test_embeddings)
        predictions_proba = cls.predict_proba(test_embeddings)

        f1 = f1_score(predictions, test_labels_idx, average='weighted')
        conf_mtrx = self.data.confusion_matrix(test_labels_idx, predictions)
        roc_auc = self.data.roc_auc(test_labels_idx, predictions_proba)

        metrics = {"round_id": round_id, "f1": f1, "roc_auc": roc_auc, "conf_mtrx": conf_mtrx}
        print(round_id)
        print('F1 Score: {:.3f}, ROC AUC: {:.3f}, Confusion Matrix: '.format(f1, roc_auc))
        print('%s' % str(conf_mtrx))

        return predictions, metrics

    def bert_fit_predict(self, round_id):
        self.data.print_id("Random Forest BERT embedding Fitting")
        bert_train_data = self.data.get_info_db("bert_embedding", "train")
        train_embeddings = [json.loads(emb[1]) for emb in bert_train_data]
        train_labels = [emb[2] for emb in bert_train_data]
        train_labels_idx = self.get_label_indices(train_labels)

        cls = RandomForestClassifier(n_estimators=10)
        cls.fit(train_embeddings, train_labels_idx)

        # self.data.print_id("Random Forest Prediction")
        bert_test_data = self.data.get_info_db("bert_embedding", "test")
        test_embeddings = [json.loads(emb[1]) for emb in bert_test_data]
        test_labels = [emb[2] for emb in bert_test_data]
        test_id = [emb[0] for emb in bert_test_data]
        predictions, metrics = self.predict(test_embeddings, test_labels, cls, round_id)

        self.to_db("RF_bert_prediction", test_id, predictions)
        return test_id, predictions, test_labels, metrics


    def bow_fit_predict(self, round_id):
        column_trans = ColumnTransformer([('corpus_bow', TfidfVectorizer(), 'corpus')], remainder='passthrough')
        train_data = self.data.get_info_db("processed_corpus", "train", False)
        test_data = self.data.get_info_db("processed_corpus", "test", False)
        fitting_data = [emb[1] for emb in train_data]
        fitting_data.extend([emb[1] for emb in test_data])
        x = pd.DataFrame({"corpus": fitting_data})
        column_trans.fit(x)

        self.data.print_id("Random Forest BoW embedding Fitting")
        train_embeddings = column_trans.transform(pd.DataFrame({"corpus": [emb[1] for emb in train_data]}))
        train_labels = [emb[2] for emb in train_data]
        train_labels_idx = self.get_label_indices(train_labels)

        cls = RandomForestClassifier(n_estimators=10)
        cls.fit(train_embeddings, train_labels_idx)

        # self.data.print_id("Random Forest Prediction")

        test_embeddings = column_trans.transform(pd.DataFrame({"corpus": [emb[1] for emb in test_data]}))
        test_labels = [emb[2] for emb in test_data]
        test_id = [emb[0] for emb in test_data]
        predictions, metrics = self.predict(test_embeddings, test_labels, cls, round_id)

        self.to_db("RF_bow_prediction", test_id, predictions)
        return test_id, predictions, test_labels, metrics

    def to_db(self, field, ids, predictions):
        cur = self.data.dataset.conn.cursor()
        for idx in range(0, len(ids)):
            case_id = ids[idx]
            sql = "update `case_results` set %s = ? where id = ?;" % field
            cur.execute(sql, [int(predictions[idx]), case_id])
        self.data.dataset.conn.commit()
