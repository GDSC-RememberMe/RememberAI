import os
import pandas as pd
import re
import numpy as np
from transformers import BertTokenizer

DATA_IN_PATH = "./data/in/KOR"
TRAIN_STS_DF = os.path.join(DATA_IN_PATH, "KorSTS", "sts-train.tsv")
DEV_STS_DF = os.path.join(DATA_IN_PATH, "KorSTS", "sts-dev.tsv")


train_data = pd.read_csv(TRAIN_STS_DF, header=0, delimiter='\t', quoting=3)
dev_data = pd.read_csv(DEV_STS_DF, header=0, delimiter='\t', quoting=3)

tokneizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)

def bert_tokenizer(sent1, sent2, MAX_LEN=102):
    encoded_dict = tokneizer.encode_plus(text = sent1, 
                                        text_pair = sent2,
                                        add_special_tokens = True,
                                        max_length = MAX_LEN,
                                        return_attention_mask = True,
                                        truncation = True
                                        )

    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']

    return input_id, attention_mask, token_type_id

def clean_text(sent):
    sent_clean = re.sub("[^a-zA-Z0-9ㄱ-|가-힣\\s]", " ", sent)
    return sent_clean

def prep():
    input_ids = []
    attention_masks = []
    token_type_ids = []
    data_labels = []

    for sent1, sent2, score in train_data[['sentence1', 'sentence2', 'score']].values:
        try:
            input_id, attention_mask, token_type_id = bert_tokenizer(clean_text(sent1), clean_text(sent2), bert_tokenizer.MAX_LEN)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)
            data_labels.append(score)
        except Exception as e:
            print(e)
            print(sent1, sent2)
            pass

    train_input_ids = np.array(input_ids, dtype=int)
    train_attention_masks = np.array(attention_masks, dtype=int)
    train_type_ids = np.array(token_type_ids, dtype=int)

    train_inputs = (train_input_ids, train_attention_masks, train_type_ids)
    train_data_labels = np.array(data_labels)

    return train_inputs, train_data_labels