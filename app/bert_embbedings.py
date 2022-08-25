# The code is adapted from: http://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#3-extracting-embeddings

import re
import string

import torch
from transformers import BertTokenizer, BertModel


def get_data_from_dataset(file, prop="*"):
    import sqlite3
    conn = None
    try:
        conn = sqlite3.connect(file)
    except sqlite3.Error as e:
        print(e)

    cur = conn.cursor()

    cur.execute("select " + prop + " from `case` order by id")
    data = [i[0] for i in cur.fetchall()]
    return data


def text_preprocessing(input_text):
    # Remove whitespaces
    processed_text = input_text.strip()

    # Remove numbers
    processed_text = re.sub(r'\d+', '', processed_text)

    # Remove punctuation
    processed_text = processed_text.translate(str.maketrans(string.punctuation, '!  $% \'()  , . :;   ?           '))

    # Convert text to lowercase
    processed_text = processed_text.lower()

    return processed_text.strip()


def text_bert_marking(processed_text):
    # break text string into list for any newlines to later replace it with [SEP] Mark
    text_lines = re.findall('[^\r\n]+', processed_text)

    # add [SEP] after each of these punctuations
    marked_lines = []
    for line in text_lines:
        # if a line not finish by !?;. finish it with "."
        if line[-1] not in (".", ";", "?", "!"):
            line += "."

        # add [SEP]
        line = " [SEP] ".join(re.findall('[^.?!;]*[.?!;]+', line)) + " [SEP]"
        marked_lines.append(line)

    # add [CLS] mark and join the lines
    marked_text = "[CLS] " + "".join(marked_lines)

    return marked_text


def create_segment_ids(tokenized_text):
    segment_ids = []
    flag = 0

    for token in tokenized_text:
        segment_ids.append(flag)
        if token == "[SEP]":
            flag = 1 - flag

    return segment_ids


def input_formatting(input_text):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    processed_text = text_preprocessing(input_text)
    marked_text = text_bert_marking(processed_text)
    tokenized_text = tokenizer.tokenize(marked_text)
    segment_ids = create_segment_ids(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    return {"tokens": indexed_tokens, "segments": segment_ids}


def main(dataset_file):
    # Load issues corpus from dataset
    data = get_data_from_dataset(dataset_file, "corpus")

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,
                                      )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    for text in data:
        token_indexes_segment_ids = input_formatting(text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([token_indexes_segment_ids["tokens"]])
        segments_tensors = torch.tensor([token_indexes_segment_ids["segments"]])

        # Run the text through BERT, and collect all of the hidden states produced
        # from all 12 layers.
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)

            # Evaluating the model will return a different number of objects based on
            # how it's  configured in the `from_pretrained` call earlier. In this case,
            # because we set `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]
    return hidden_states


if __name__ == "__main__":
    import sys

    dataset_file_name = sys.argv[1]
    main(dataset_file_name)
