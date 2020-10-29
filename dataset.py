import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer,BertModel,BertConfig, AdamW
from HyperParams import HyperParams

params = HyperParams()



class BertDataset(Dataset):

    def __init__(self, data):
        super(BertDataset, self).__init__()
        self.title, self.context, self.label = data
        config = BertConfig.from_pretrained(params.bert_config_path)
        self.tokenizer = BertTokenizer.from_pretrained(params.bert_path, config=config)
        self.max_sequence_length = params.max_sequence_length

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        _text = self.tokenizer.encode_plus(self.title[idx],
                                           self.context[idx],
                                           add_special_tokens=True,
                                           max_length=self.max_sequence_length,
                                           padding='max_length',
                                           truncation=True,
                                           return_attention_mask=True,  # Construct attn. masks.
                                           return_tensors='pt')  # Return pytorch tensors.
        # {'input_ids': Tensor, "attention_mask": Tensor, "type_ids":Tensor}
        _label = self.label[idx]
        return _text, _label


def load_dataset(path):
    data = pd.read_csv(path).fillna('[UNK]')
    data = pd.read_csv(path).fillna('[UNK]')
    titles = data["title"].values
    contents = data["content"].values
    labels = data['label'].values
    return titles, contents, labels


def random_sample(lens, test_size):
    random_index = np.random.permutation(lens)
    split_point = int((1 - test_size) * lens)
    train_sample = random_index[:split_point]
    test_sample = random_index[split_point:]
    return train_sample, test_sample


def train_test_splite(titles, contents, labels, test_size=0.2):
    titles, contents, labels = np.asarray(titles), np.asarray(contents), np.asarray(labels)
    train_sample, test_sample = random_sample(len(titles), test_size)

    train_titles, train_contents, train_labels = titles[train_sample], contents[train_sample], labels[train_sample]
    test_titles, test_contents, test_labels = titles[test_sample], contents[test_sample], labels[test_sample]

    return (train_titles, train_contents, train_labels), (test_titles, test_contents, test_labels)


def weighted_accuracy(y_true, y_pred):
    return np.sum((np.array(y_pred).ravel() == np.array(y_true).ravel())) * 1.0 / len(y_true)


def unweighted_accuracy(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    classes = np.unique(y_true)
    classes_accuracies = np.zeros(classes.shape[0])
    for num, cls in enumerate(classes):
        classes_accuracies[num] = weighted_accuracy(y_true[y_true == cls], y_pred[y_true == cls])
    return np.mean(classes_accuracies)

def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

logger = get_logger(params.log_file)
