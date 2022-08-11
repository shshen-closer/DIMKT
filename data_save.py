import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import json
import sys
import time, datetime
from sklearn.model_selection import train_test_split, KFold

pd.set_option('display.float_format',lambda x : '%.2f' % x)
np.set_printoptions(suppress=True)
kfold = KFold(n_splits=5, shuffle=True,random_state=5)

length = int(sys.argv[1])
    
all_data =  pd.read_csv('data/2012-2013-data-with-predictions-4-final.csv', encoding = "ISO-8859-1", low_memory=False)

print(all_data.head())
all_data['timestamp'] =  all_data['end_time'].apply(lambda x:time.mktime(time.strptime(x[:19],'%Y-%m-%d %H:%M:%S')))
order = ['user_id','problem_id','correct','skill_id', 'timestamp']
all_data = all_data[order]
all_data['skill_id'].fillna('nan',inplace=True)
all_data = all_data[all_data['skill_id'] != 'nan'].reset_index(drop=True)

with open('data/difficult2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        difficult2id = eval(line)

with open('data/sdifficult2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        sdifficult2id = eval(line)

with open('data/user2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        user2id = eval(line)
with open('data/problem2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        problem2id = eval(line)
with open('data/skill2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        skill2id = eval(line)

user_id = np.array(all_data['user_id'])
user = sorted(list(set(user_id)))
np.random.seed(100)
np.random.shuffle(user)
train_all_id, test_id = train_test_split(user,test_size=0.2,random_state=5)
train_all_id = np.array(train_all_id)
nones = np.load('data/nones.npy', allow_pickle = True)
nonesk = np.load('data/nonesk.npy', allow_pickle = True)

count = 0
for (train_index, valid_index) in kfold.split(train_all_id):
    
    train_id = train_all_id[train_index]
    valid_id = train_all_id[valid_index]
    np.random.shuffle(train_id)
    q_a_train = []

    for item in tqdm(train_id):

        idx = all_data[(all_data.user_id==item)].index.tolist() 
        temp1 = all_data.iloc[idx]
        temp1 = temp1.sort_values(by=['timestamp']) 
        temp = np.array(temp1)
        if len(temp) < 2:   # 
            continue
        quiz = temp

            
        train_q = []
        train_d = []
        train_a = []
        train_skill = []
        train_sd = []

        for one in range(0,len(quiz)):
            if quiz[one][1] not in nones and quiz[one][3] not in nonesk :            # filtering questions without difficulty value
                train_q.append(problem2id[quiz[one][1]])
                train_d.append(difficult2id[quiz[one][1]])
                train_a.append(int(quiz[one][2]))
                train_skill.append(skill2id[quiz[one][3]])
                train_sd.append(sdifficult2id[quiz[one][3]])
            if len(train_q) >= length :
                q_a_train.append([train_q, train_d, train_a, train_skill, len(train_q), train_sd])
                train_q = []
                train_d = []
                train_a = []
                train_skill = []
                train_sd = []
        if len(train_q)>=2 and len(train_q) < length:
            q_a_train.append([train_q, train_d, train_a, train_skill, len(train_q), train_sd])
    q_a_valid = []
    for item in tqdm(valid_id):

        idx = all_data[(all_data.user_id==item)].index.tolist() 
        temp1 = all_data.iloc[idx]
        temp1 = temp1.sort_values(by=['timestamp']) 
        temp = np.array(temp1)
        if len(temp) < 2:
            continue
        quiz = temp

        test_q = []
        test_d = []
        test_a = []
        test_skill = []
        test_sd = []

        for one in range(0,len(quiz)):
            if quiz[one][1] not in nones and quiz[one][3] not in nonesk :
                test_q.append(problem2id[quiz[one][1]])
                test_d.append(difficult2id[quiz[one][1]])
                test_a.append(int(quiz[one][2]))
                test_skill.append(skill2id[quiz[one][3]])
                test_sd.append(sdifficult2id[quiz[one][3]])
            if len(test_q) >=length :
                q_a_valid.append([test_q,test_d, test_a, test_skill, len(test_q), test_sd])
                test_q = []
                test_d = []
                test_a = []
                test_skill = []
                test_sd = []
        if len(test_q)>=2 and len(test_q) < length:
            q_a_valid.append([test_q,test_d, test_a, test_skill, len(test_q), test_sd])
    np.random.seed(2)
    np.random.shuffle(q_a_train)
    np.random.seed(2)
    np.random.shuffle(q_a_valid)
    np.save("data/train" + str(count) + ".npy",np.array(q_a_train))

    np.save("data/valid" + str(count) + ".npy",np.array(q_a_valid))

    count += 1
    break
q_a_test = []
for item in tqdm(test_id):

    idx = all_data[(all_data.user_id==item)].index.tolist() 
    temp1 = all_data.iloc[idx]
    temp1 = temp1.sort_values(by=['timestamp']) 
    temp = np.array(temp1)
    if len(temp) < 2:
        continue
    quiz = temp

    test_q = []
    test_d = []
    test_a = []
    test_skill = []
    test_sd = []

    for one in range(0,len(quiz)):
        if quiz[one][1] not in nones and quiz[one][3] not in nonesk :
            test_q.append(problem2id[quiz[one][1]])
            test_d.append(difficult2id[quiz[one][1]])
            test_a.append(int(quiz[one][2]))
            test_skill.append(skill2id[quiz[one][3]])
            test_sd.append(sdifficult2id[quiz[one][3]])
        if len(test_q) >=length :
            q_a_test.append([test_q,test_d, test_a, test_skill, len(test_q), test_sd])
            test_q = []
            test_d = []
            test_a = []
            test_skill = []
            test_sd = []
    if len(test_q)>=2 and len(test_q) < length:
        q_a_test.append([test_q,test_d, test_a, test_skill, len(test_q), test_sd])

np.save("data/test.npy",np.array(q_a_test))

print('complete')
            



