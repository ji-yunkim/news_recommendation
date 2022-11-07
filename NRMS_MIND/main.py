import sys, torch, yaml, numpy as np, pickle, torch.nn as nn, os, time
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader import DatasetTrn
from model import NRMS
from evaluation import ndcg_score, mrr_score
from torch import optim
from sklearn.metrics import roc_auc_score

print("System version: {}".format(sys.version))
print("Torch version: {}".format(torch.__version__))
MIND_type = 'demo'

def load_dict(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
def load_lines(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        return f.readlines()
def hparams_setter(yaml_file, **kwargs):
    with open(yaml_file, "r") as f:
        hparams = yaml.load(f, yaml.SafeLoader)
    hparams = flat_hparams(hparams)  # yaml to dict
    hparams.update(kwargs)
    return hparams
def flat_hparams(hparams): # yaml file to a flat dict
    f_hparams = {}
    category = hparams.keys()
    for cate in category:
        for key, val in hparams[cate].items():
            f_hparams[key] = val
    return f_hparams

def main():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    start_time = time.time()
    data_path = "C:\\Users\Jiyun\Desktop\\NRS\datasets\MIND\\" + MIND_type
    train_news_path = os.path.join(data_path, 'train', r'news.tsv')
    train_behaviors_path = os.path.join(data_path, 'train', r'behaviors.tsv')
    valid_news_path = os.path.join(data_path, 'valid', r'news.tsv')
    valid_behaviors_path = os.path.join(data_path, 'valid', r'behaviors.tsv')
    wordEmb_path = os.path.join(data_path, "utils", "embedding.npy")
    userDict_path = os.path.join(data_path, "utils", "uid2index.pkl")
    wordDict_path = os.path.join(data_path, "utils", "word_dict.pkl")
    hparams_path = os.path.join(data_path, "utils", r'nrms.yaml')

    embedding = np.load(wordEmb_path)
    userDict = load_dict(userDict_path)
    wordDict = load_dict(wordDict_path)
    hparams = hparams_setter(hparams_path, wordEmb_file=wordEmb_path, wordDict_file=wordDict_path, userDict_file=userDict_path)
    training_set = DatasetTrn(train_news_path, train_behaviors_path, wordDict=wordDict, userDict=userDict, embedding=embedding, hparams=hparams)
    train_loader = DataLoader(training_set, batch_size=hparams['batch_size'], drop_last=True)
    # valid_set = DatasetTest(valid_news_path, valid_behaviors_path, wordDict=wordDict, userDict=userDict, embedding=embedding, hparams=hparams, label_known=True)
    # valid_loader = DataLoader(valid_set, batch_size = 1, num_workers = 1, shuffle = False)
    wordemb = np.load(wordEmb_path)

    model = NRMS(hparams, wordemb).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(hparams['learning_rate']),
                           weight_decay=float(hparams['weight_decay']))
    criterion = nn.CrossEntropyLoss()
    epochs = hparams['epochs']
    metrics = {metric: 0. for metric in hparams['metrics']}
    eval_every = 3
    print(f'[{time.time() - start_time:5.2f} Sec] Ready for training...')

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        batch_loss = 0.
        for i, (trn_his, trn_pos, trn_neg) in tqdm(enumerate(train_loader), desc='Training', total=len(train_loader)):
            trn_his, trn_pos, trn_neg = trn_his.to(device), trn_pos.to(device), trn_neg.to(device)
            # print(trn_his.shape)
            # print(trn_pos.shape)
            # print(trn_neg.shape)
            model.train()
            optimizer.zero_grad()
            trn_cand = torch.cat((trn_pos, trn_neg), dim=1)
            lst = []
            for i in range(hparams['batch_size']):
                lst.append([1.,0.,0.,0.,0.])
            trn_labels = torch.tensor(lst).to(device)
            # print(trn_cand.shape)
            trn_cand_out = model(trn_cand, source='candidate')
            trn_user_out = model(trn_his, source='history')
            prob = torch.matmul(trn_cand_out, trn_user_out.unsqueeze(2)).squeeze()
            # print("prob",prob.shape)
            # print("trn_labels",trn_labels.shape)
            loss = criterion(prob, trn_labels)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
        inter_time = time.time()
        epoch_loss = batch_loss / (i + 1)
        if epoch % eval_every != 0:
            result = f'Epoch {epoch:3d} [{inter_time - start_time:5.2f}Sec]'f', TrnLoss:{epoch_loss:.4f}'
            print(result)
            continue

        '''
        evaluation
        '''
        # model.eval()
        # with open(os.path.join(f'prediction-{epoch}.txt'), 'w') as f:
        #     # for j in tqdm(range(len(vld_impr)), desc='Evaluation', total=len(vld_impr)):
        #     for j, (impr_idx_j, vld_his_j, vld_cand_j, vld_label_j, vld_pop_j, vld_fresh_j) \
        #         in tqdm(enumerate(valid_loader), desc='Evaluation', total=len(valid_loader)):
        #
        #         # Get model output
        #         vld_global_j = {}
        #         for key in vld_his_j.keys():
        #             vld_his_j[key], vld_pop_j[key], vld_fresh_j[key], vld_cand_j[key] = \
        #             vld_his_j[key].to(device), vld_pop_j[key].to(device), \
        #             vld_fresh_j[key].to(device), vld_cand_j[key].to(device)
        #
        #         vld_user_out_j = model(vld_his_j, source='history')
        #         vld_cand_out_j = model(vld_cand_j, source='candidate')
        #
        #         # Get model output end
        #
        #         scores_j = torch.matmul(vld_cand_out_j, vld_user_out_j.unsqueeze(2)).squeeze()
        #         scores_j = scores_j.detach().cpu().numpy()
        #         argmax_idx = (-scores_j).argsort()
        #         ranks = np.empty_like(argmax_idx)
        #         ranks[argmax_idx] = np.arange(1, scores_j.shape[0]+1)
        #         ranks_str = ','.join([str(r) for r in list(ranks)])
        #         f.write(f'{impr_idx_j.item()} [{ranks_str}]\n')
        #         vld_gt_j = np.array(vld_label_j)
        #         # vld_gt_j = np.array(vld_label[j])
        #
        #         for metric, _ in metrics.items():
        #             if metric == 'auc':
        #                 score = roc_auc_score(vld_gt_j, scores_j)
        #                 metrics[metric] += score
        #             elif metric == 'mrr':
        #                 score = mrr_score(vld_gt_j, scores_j)
        #                 metrics[metric] += score
        #             elif metric.startswith('ndcg'):  # format like: ndcg@5;10
        #                 k = int(metric.split('@')[1])
        #                 score = ndcg_score(vld_gt_j, scores_j, k=k)
        #                 metrics[metric] += score
        #
        # for metric, _ in metrics.items():
        #     metrics[metric] /= len(valid_loader) # len(vld_impr)
        #
        # end_time = time.time()
        #
        # result = f'Epoch {epoch:3d} [{inter_time - start_time:5.2f} / {end_time - inter_time:5.2f} Sec]'f', TrnLoss:{epoch_loss:.4f}, '
        # for enum, (metric, _) in enumerate(metrics.items(), start=1):
        #     result += f'{metric}:{metrics[metric]:.4f}'
        #     if enum < len(metrics):
        #         result += ', '
        # print(result)

if __name__ == "__main__":
    main()