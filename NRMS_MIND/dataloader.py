import os,torch,re,timeit,time,pickle,pandas as pd,numpy as np,random
from random import sample
from torch.utils.data import Dataset
from tqdm import tqdm

def word_tokenize(sent):
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def newsample(news, ratio):
    """Sample ratio samples from news list.
    If length of news is less than ratio, pad zeros.
    Args:
        news (list): input news list
        ratio (int): sample number
    Returns:
        list: output of sample list.
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)

class BaseDataset(Dataset):
    def __init__(self, news_file, behaviors_file, wordDict, userDict, embedding, hparams):
        news_title = [""]
        newsDict = {}
        self.histories_unfold = []
        self.impr_idxs_unfold = []
        self.uidxs_unfold = []
        self.pos_unfold = []
        self.neg_unfold = []
        self.line_num = 0
        col_spliter = '\t'
        with open(news_file, "r", encoding='utf-8') as rd:
            for line in rd:
                nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(col_spliter)
                if nid in newsDict:
                    continue
                newsDict[nid] = len(newsDict) + 1 # start from 1
                title = word_tokenize(title)
                news_title.append(title)
            # TASK : self.news_title_index = [title_seq, word_seq]
            self.news_title_index = np.zeros((len(news_title), hparams['title_size']), dtype="int32")
            for news_index in range(len(news_title)):
                title = news_title[news_index]
                for word_index in range(min(hparams['title_size'], len(title))):
                    if title[word_index] in wordDict:
                        self.news_title_index[news_index, word_index] = wordDict[title[word_index].lower()]

        self.histories = []
        self.imprs = []
        self.labels = []
        self.impr_indexes = []
        self.uindexes = []
        self.history_embeddings = []
        self.impr_embeddings = []
        his_size = 30
        impr_index = 0
        num_lines = sum(1 for line in open(behaviors_file))  # 22034
        with open(behaviors_file, "r", encoding='utf-8') as rd:
            for line in tqdm(rd, desc="behavior file ) behavior log -> history and impr title index list", total=num_lines):
                self.line_num += 1
                uid, time, history, impr = line.strip("\n").split(col_spliter)[-4:]
                history = [newsDict[i] for i in history.split()]
                history = [0] * (his_size - len(history)) + history[:his_size]
                impr_news = [newsDict[i.split("-")[0]] for i in impr.split()]
                label = [int(i.split("-")[1]) for i in impr.split()]
                uindex = userDict[uid] if uid in userDict else 0
                self.histories.append(history)
                self.imprs.append(impr_news)
                self.labels.append(label)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                impr_index += 1

    def __getitem__(self, idx):
        pass
    def __len__(self):
        pass

class DatasetTrn(BaseDataset):
    def __init__(self, news_file, behaviors_file, wordDict, userDict, embedding, hparams):
        super().__init__(news_file, behaviors_file, wordDict, userDict, embedding, hparams)
        self.histories_unfold = []
        self.impr_idxs_unfold = []
        self.uidxs_unfold = []
        self.pos_unfold = []
        self.behaviors_file = behaviors_file
        self.neg_unfold = []
        self.hparams = hparams
        self.npratio = hparams['npratio']

        # TODO : history, positive, negative로 나눠서 저장
        for line in range(len(self.uindexes)):  # 22034
            neg_idxs = [i for i, x in enumerate(self.labels[line]) if x == 0]
            pos_idxs = [i for i, x in enumerate(self.labels[line]) if x == 1]
            if len(pos_idxs) < 1:
                continue
            for pos_idx in pos_idxs:
                self.pos_unfold.append([self.imprs[line][pos_idx]])
                negs = [self.imprs[line][i] for i in neg_idxs]
                self.neg_unfold.append(negs)
                self.histories_unfold.append(self.histories[line])
                self.impr_idxs_unfold.append(self.impr_indexes[line])
                self.uidxs_unfold.append(self.uindexes[line])

    def __getitem__(self, idx):

        his = self.histories_unfold[idx]
        lst = []
        for i in his:
            lst.append(self.news_title_index[i])
        lst = np.array(lst)
        his = torch.tensor(lst)
        neg = torch.tensor(newsample(self.neg_unfold[idx], self.npratio))  # [4, 30, 300] class 'list'
        lst2 = []
        for j in neg:
            lst2.append(self.news_title_index[j])
        lst2 = np.array(lst2)
        neg = torch.tensor(lst2)
        pos = torch.tensor(self.pos_unfold[idx])
        lst3 = []
        for k in pos:
            lst3.append(self.news_title_index[k])
        lst3 = np.array(lst3)
        pos = torch.tensor(lst3)
        return his, pos, neg

    def __len__(self):
        return self.line_num

class DatasetTest(BaseDataset):
    labels = None
    def __init__(self, news_file, behaviors_file, wordDict, userDict, embedding, hparams, label_known=True):

        self.label_known = label_known
        super().__init__(news_file, behaviors_file, wordDict, userDict, embedding, hparams)

        self.histories_unfold = []
        self.imprs_unfold = []

        for i in range(len(self.histories)):
            self.histories_unfold.append(self.histories[i])
            self.imprs_unfold.append(self.imprs[i])

    def __getitem__(self, idx):
        # impr_idx
        # impr_idx = idx

        # his
        his = self.histories_unfold[idx]
        lst1 = []
        for i in his:
            lst1.append(self.news_title_index[i])
        lst1 = np.array(lst1)
        his = torch.tensor(lst1)

        # impr
        lst2 = []
        impr = self.imprs_unfold[idx]
        for i in impr:
            a = self.news_title_index[i]
            lst2.append(a)
        lst2 = np.array(lst2)
        impr = torch.tensor(lst2)

        # label
        label = self.labels[idx]

        # return impr_idx, his, impr, label
        return his, impr, label

    def __len__(self):
        return len(self.uindexes)