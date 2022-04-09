import datetime
import json
import os
import time
import random
import numpy as np
import torch
from transformers import BertTokenizer
from sklearn.metrics import f1_score
from scipy.special import softmax


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class SEIssueBert:
    seed_val = None
    tokenizer = None
    data = None
    model = None  # BERT Model

    def __init__(self, data, seed_val=321):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # Set the seed value all over the place to make this reproducible.
        self.seed_val = seed_val
        self.data = data
        self.device = torch.device("cpu")

    # Helper functions
    def get_max_len(self, samples):
        max_len = 0
        for issue in samples:
            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = self.tokenizer.encode(issue, add_special_tokens=True)

            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids))

        # print('Max sentence length: ', max_len)
        return max_len

    def make_smart_batches(self, text_samples, labels, batch_size):

        print(
            'Creating Smart Batches from {:,} examples with batch size {:,}...'.format(len(text_samples), batch_size))

        full_input_ids = []

        max_len = self.get_max_len(text_samples) + 50
        max_lenn = min(max_len, 400)
        print("returned Max Length is: %d and selected one is: %d" % (max_len, max_lenn))
        max_len = max_lenn

        print('Tokenizing {:,} samples...'.format(len(labels)))

        for text in text_samples:
            input_ids = self.tokenizer.encode(text=text,
                                              add_special_tokens=True,
                                              max_length=max_len,
                                              truncation=True,
                                              padding=False)
            full_input_ids.append(input_ids)

        samples = sorted(zip(full_input_ids, labels), key=lambda x: len(x[0]))

        batch_ordered_sentences = []
        batch_ordered_labels = []

        print('Creating batches of size {:}...'.format(batch_size))

        while len(samples) > 0:
            to_take = min(batch_size, len(samples))
            select = random.randint(0, len(samples) - to_take)
            batch = samples[select:(select + to_take)]
            batch_ordered_sentences.append([s[0] for s in batch])
            batch_ordered_labels.append([s[1] for s in batch])

            del samples[select:select + to_take]

        print('DONE - Selected {:,} batches.'.format(len(batch_ordered_sentences)))
        print('Padding out sequences within each batch...')

        py_inputs = []
        py_attn_masks = []
        py_labels = []

        for (batch_inputs, batch_labels) in zip(batch_ordered_sentences, batch_ordered_labels):

            batch_padded_inputs = []
            batch_attn_masks = []

            max_size = max([len(sen) for sen in batch_inputs])

            for sen in batch_inputs:
                num_pads = max_size - len(sen)
                padded_input = sen + [self.tokenizer.pad_token_id] * num_pads
                attn_mask = [1] * len(sen) + [0] * num_pads

                batch_padded_inputs.append(padded_input)
                batch_attn_masks.append(attn_mask)

            py_inputs.append(torch.tensor(batch_padded_inputs))
            py_attn_masks.append(torch.tensor(batch_attn_masks))
            py_labels.append(torch.tensor(batch_labels))

        return py_inputs, py_attn_masks, py_labels

    def connect_gpu(self):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))

    def load_bert_model(self, model_name_dir='bert-base-uncased', label_num=7):
        from transformers import AutoConfig
        from transformers import AutoModelForSequenceClassification
        # labels are: one-hour, half-day, day, half-week, week, two-week, more-than-two-week
        self.data.print_id("Loading BERT model")
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name_dir,
                                            num_labels=label_num)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name_dir,
            config=config)

        if torch.cuda.is_available():
            self.model.cuda()

    def get_label_indices(self, labels):
        label_set = [i for i in set(labels)]
        label_set.sort()
        label_idx = []
        for i in labels:
            label_idx.append(label_set.index(i))
        return label_idx

    ###############################################
    # Train BERT model
    def fine_tune(self, train=None):
        self.connect_gpu()
        from transformers import AdamW
        from transformers import get_linear_schedule_with_warmup

        if train is None:
            train_df = self.data.get_train_set()
        else:
            train_df = train
        train_text = train_df.pop('corpus')
        train_label = train_df.pop('actual_effort_category')
        train_label_idx = self.get_label_indices(train_label)
        #train_label_idx = [i+1000 for i in train_label_idx]
        batch_size = 16

        (py_inputs, py_attn_masks, py_labels) = self.make_smart_batches(train_text, train_label_idx, batch_size)

        self.load_bert_model(label_num=len(set(train_label_idx)))
        optimizer = AdamW(self.model.parameters(), lr=5e-5, eps=1e-8)
        epochs = 3
        total_steps = len(py_inputs) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)

        training_stats = []
        total_t0 = time.time()

        for epoch_i in range(0, epochs):

            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

            if epoch_i > 0:
                (py_inputs, py_attn_masks, py_labels) = self.make_smart_batches(train_text, train_label_idx, batch_size)

            t0 = time.time()
            total_train_loss = 0
            self.model.train()

            for step in range(0, len(py_inputs)):
                b_input_ids = py_inputs[step].to(self.device)
                b_input_mask = py_attn_masks[step].to(self.device)
                b_labels = py_labels[step].to(self.device)

                self.model.zero_grad()
                loss, logits = self.model(b_input_ids,
                                          token_type_ids=None,
                                          attention_mask=b_input_mask,
                                          labels=b_labels)

                total_train_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(py_inputs)
            training_time = format_time(float(time.time() - t0))
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Training Time': training_time,
                }
            )
        print("Total training took {:} (h:mm:ss)".format(format_time(float(time.time() - total_t0))))

    def evaluate(self, test=None, round_id=""):

        if test is None:
            test_df = self.data.get_test_set()
        else:
            test_df = test
        test_text = test_df.pop('corpus')
        test_labels = test_df.pop('actual_effort_category')
        test_ids = test_df.pop('id')
        test_labels_idx = self.get_label_indices(test_labels)
        batch_size = 16

        (py_inputs, py_attn_masks, py_labels) = self.make_smart_batches(test_text, test_labels_idx, batch_size)
        print('Predicting labels for {:,} test sentences...'.format(len(test_labels)))

        self.model.eval()
        predictions, true_labels = [], []

        for step in range(0, len(py_inputs)):
            b_input_ids = py_inputs[step].to(self.device)
            b_input_mask = py_attn_masks[step].to(self.device)
            b_labels = py_labels[step].to(self.device)

            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.append(logits)
            true_labels.append(label_ids)

        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
        preds = np.argmax(predictions, axis=1).flatten()

        f1 = f1_score(preds, true_labels, average='weighted')
        conf_mtrx = self.data.confusion_matrix(true_labels, preds)
        roc_auc = self.data.roc_auc(true_labels, [softmax(p) for p in predictions])
        metrics = {"round_id": round_id, "f1": f1, "roc_auc": roc_auc, "conf_mtrx": conf_mtrx}

        print(round_id)
        print('F1 Score: {:.3f}, ROC AUC: {:.3f}, Confusion Matrix: '.format(f1, roc_auc))
        print('%s' % str(conf_mtrx))

        self.to_db_pred(preds, [i for i in test_ids])
        return predictions, true_labels, test_ids, metrics

    ###############################################
    # Embedding Extraction
    def embedding(self):
        def write_to_file(embeddings_arr):
            output_dir = './embeddings/'
            file_name = '%s_%s.json' % (self.data.dataset.name, self.data.key)

            # self.data.print_id("Dump Embeddings")
            print("Number of embeddings is: %d. Dumped to: %s%s" % (len(embeddings), output_dir, file_name))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_dir + file_name, 'w', encoding='utf-8') as f:
                f.write(json.dumps(embeddings_arr, ensure_ascii=False))

        embeddings = []
        for idx, row in self.data.issues_df.iterrows():
            if row[1] is None:
                continue
            marked_text = "[CLS] " + row[1] + " [SEP]"
            tokenized_text = self.tokenizer.tokenize(marked_text)

            if len(tokenized_text) > 400:
                tokenized_text = tokenized_text[:400]
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)

            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            with torch.no_grad():
                outputs = self.model(tokens_tensor, segments_tensors)
                hidden_states = outputs[2]

            token_vecs = hidden_states[-2][0]

            sentence_embedding = torch.mean(token_vecs, dim=0)
            embeddings.append({row[0]: sentence_embedding.tolist()})
        self.to_db_emb(embeddings)
        write_to_file(embeddings)

    ###############################################
    # Model Saving and Loading
    def to_db_pred(self, pred, ids):
        cur = self.data.dataset.conn.cursor()
        for idx in range(0, len(pred)):
            case_id = ids[idx]
            sql = "update `case_results` set set_type=?, bert_prediction=? where id = ?;"
            cur.execute(sql, ["test", int(pred[idx]), case_id])

        self.data.dataset.conn.commit()

    def to_db_emb(self, embeddings=None):
        if embeddings is None:
            embeddings = []
        cur = self.data.dataset.conn.cursor()
        for idx in range(0, len(embeddings)):
            case_id = [i for i in embeddings[idx].keys()][0]
            sql = "update `case_results` set bert_embedding = ? where id = ?;"
            cur.execute(sql, [json.dumps(embeddings[idx][case_id]), case_id])

        self.data.dataset.conn.commit()

    def save_model(self, output_dir='./model_save/'):

        output_dir = output_dir + self.data.dataset.name + "_" + self.data.key + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_saved_model(self, save_dir='./model_save/'):
        from transformers import BertModel

        save_dir = save_dir + self.data.dataset.name + "_" + self.data.key + "/"
        self.model = BertModel.from_pretrained(save_dir, output_hidden_states=True)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(save_dir)
        self.connect_gpu()
        self.model.to(self.device)
