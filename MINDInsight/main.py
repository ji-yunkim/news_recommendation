'''
This code is generated to grasp how users read news.

Author : Jiyun Kim
'''

'''
This code is generated to grasp how users read news.
'''

import os
from tqdm import tqdm

def main():
    MIND_type = 'demo'
    data_path = "C:\\Users\Jiyun\Desktop\\NRS\datasets\MIND\\" + MIND_type
    news_path = os.path.join(data_path, 'train', r'news.tsv')
    behaviors_path = os.path.join(data_path, 'train', r'behaviors.tsv')
    newsDict = {}
    col_spliter = '\t'
    titleDict = {}
    news_title=[""]
    line_num = 0
    with open(news_path, "r", encoding='utf-8') as rd:
        for line in rd:
            nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(col_spliter)
            if nid in titleDict:
                continue
            titleDict[nid] = title
            news_title.append(title)
    histories = []
    pos = []
    neg = []
    num_lines = sum(1 for line in open(behaviors_path))  # 22034
    with open(behaviors_path, "r", encoding='utf-8') as rd:
        for line in tqdm(rd, desc="behavior file ) behavior log -> history and impr title index list", total=num_lines):
            line_num += 1
            uid, time, history, impr = line.strip("\n").split(col_spliter)[-4:]
            lst = []
            for i in history.split():
               lst.append(titleDict[i])
            histories.append(lst)
            lst_pos = []
            lst_neg = []
            for ii in impr.split():
                if ii.split("-")[1]=="1":
                    a = ii.split("-")[0]
                    lst_pos.append(titleDict[a])
                elif ii.split("-")[1]=="0":
                    a = ii.split("-")[0]
                    lst_neg.append(titleDict[a])
            pos.append(lst)
            neg.append(lst)

            print("-----------history--------------")
            for i in lst:
                print(i)
            print("-----------positive-------------")
            for j in lst_pos:
                print(j)
            print("-----------negative-------------")
            for k in lst_neg:
                print(k)



if __name__ == "__main__":
    main()