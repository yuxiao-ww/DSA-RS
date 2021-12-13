import torch
import numpy as np
import logging
import datetime
import torch.nn as nn
from torch.utils.data import TensorDataset
from model import sentenceLevelBert
from torch.autograd import Variable
from transformers import DataProcessor


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2idx = {'<unk>': 0}
        self.word2count = {}
        self.idx2word = {}
        self.word_count = 1

    def addWord(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.word_count
            self.idx2word[self.word_count] = word
            self.word_count += 1
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)


def remove_chars(text, target):
    text = str(text).replace(target, '')
    return text


class Datapreprocessor(DataProcessor):
    def __init__(self, FLAGS, tokenizer, config, filepath):
        self.FLAGS = FLAGS
        self.max_length = 100
        self.tokenizer = tokenizer
        self.filepath = filepath
        self.bert = sentenceLevelBert(FLAGS, config=config)
        self.linear = nn.Linear(768, 128)
        self.train_set = self.build_dataset(self.get_train_examples(self.filepath))
        self.dev_set = self.build_dataset(self.get_dev_examples(self.filepath))
        self.test_set = self.build_dataset(self.get_test_examples(self.filepath))

    def padding(self, seq):
        print(seq)
        seq = seq.squeeze().numpy().tolist()
        print(seq)
        while len(seq) < self.max_length:
            seq.append(0)
        print(seq)
        seq = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
        return seq

    @staticmethod
    def read_files(filepath):
        lines = []
        with open(filepath, 'r', encoding='UTF-8') as f:
            print('FILE: ', f.name)
            for line in f.readlines():
                lines.append(line)
        return lines

    def get_train_examples(self, data_dir):
        return self._create_examples(self.read_files(data_dir + '/train0.txt'))

    def get_test_examples(self, data_dir):
        return self._create_examples(self.read_files(data_dir + '/test0.txt'))

    def get_dev_examples(self, data_dir):
        return self._create_examples(self.read_files(data_dir + '/dev0.txt'))

    def get_labels(self):
        return self.FLAGS.classes

    @staticmethod
    def _create_examples(lines):
        up_vocab = Lang('user_prod')
        target = '<sssss>'
        user_ids = []
        prod_ids = []
        doc_examples = []
        doc_labels = []

        for i, line in enumerate(lines):
            usr, prod, label, doc = line.split('\t\t')
            doc_labels.append(int(label)-1)
            doc = str(doc).split(target)

            # up_vocab.addWord(usr)
            # up_vocab.addWord(prod)

            # user_ids.append(up_vocab.word2idx[usr])
            # prod_ids.append(up_vocab.word2idx[prod])

            doc_examples.append(doc)

        # all_user_ids = torch.tensor(user_ids, dtype=torch.long)
        # all_prod_ids = torch.tensor(prod_ids, dtype=torch.long)

        return doc_examples, doc_labels

    def encode2bert(self, doc_list):
        docs = []
        for doc in doc_list:
            doc_input_ids = []
            for sentence in doc:
                tokenizer = self.tokenizer(
                    sentence,
                    padding=True,
                    truncation=True,
                    max_length=510,
                    return_tensors='pt'  # 返回的类型为 pytorch tensor
                )
                sen_input_ids = tokenizer['input_ids']

                # encode sentence
                encoded_sentence = self.bert(sen_input_ids).detach().numpy()

                # 整合sentence representations成1个vector
                if len(doc_input_ids) == 0:
                    doc_input_ids = encoded_sentence
                else:
                    doc_input_ids = np.concatenate((doc_input_ids, encoded_sentence), axis=1)

            # 调整维度 <= 510
            doc_input_ids = torch.tensor(doc_input_ids)
            if doc_input_ids.shape[1] > 510:
                doc_input_ids = nn.Linear(doc_input_ids.shape[1], 510)(doc_input_ids).detach().numpy()
            docs.append(str(doc_input_ids))

        tokenizer = self.tokenizer(
            docs,
            padding=True,
            truncation=True,
            max_length=510,
            return_tensors='pt'  # 返回的类型为 pytorch tensor
        )

        all_input_ids = tokenizer['input_ids'].float()
        all_token_type_ids = tokenizer['token_type_ids'].float()
        all_attention_mask = tokenizer['attention_mask'].float()

        # print(all_input_ids)
        # print(all_token_type_ids)
        # print(all_attention_mask)

        return all_input_ids, all_token_type_ids, all_attention_mask

    def build_dataset(self, doc_list):
        labels = getVariable(torch.tensor(doc_list[1], dtype=torch.float))
        print(labels)
        docs = doc_list[0]
        input_ids, token_type_ids, attention_mask = self.encode2bert(docs)

        input_ids = getVariable(input_ids)
        token_type_ids = getVariable(token_type_ids)
        attention_mask = getVariable(attention_mask)

        data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)
        return data


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


class Logger(object):
    def __init__(self, log_file_name):
        super(Logger, self).__init__()
        self.__logger = logging.getLogger('myLogger')
        self.__logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_file_name)

        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)

    def get_log(self):
        return self.__logger


def getVariable(tensor):
    return Variable(tensor.float(), requires_grad=True)
