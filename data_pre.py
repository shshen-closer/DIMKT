import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import json
import time, datetime
from sklearn.model_selection import train_test_split

pd.set_option('display.float_format',lambda x : '%.2f' % x)
np.set_printoptions(suppress=True)

    
all_data =  pd.read_csv('data/2012-2013-data-with-predictions-4-final.csv', encoding = "ISO-8859-1", low_memory=False)

print(all_data.head())
order = ['user_id','problem_id','correct','skill_id']
all_data2 = all_data[order]
all_data2['skill_id'].fillna('nan',inplace=True)
all_data = all_data2[all_data2['skill_id'] != 'nan'].reset_index(drop=True)

skill_id = np.array(all_data['skill_id'])

skills = set(skill_id)
print('# of skills:',  len(skills))

user_id = np.array(all_data['user_id'])
problem_id = np.array(all_data['problem_id'])
user = set(user_id)
problem = set(problem_id)

print('# of users:', len(user))
print('# of questions:',  len(problem))


user2id ={}
problem2id = {}
skill2id = {}

count = 1
for i in user:
    user2id[i] = count 
    count += 1
count = 1
for i in problem:
    problem2id[i] = count 
    count += 1
count = 0
for i in skills:
    skill2id[i] = count 
    count += 1

with open('data/user2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(user2id))
with open('data/problem2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(problem2id))
with open('data/skill2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(skill2id))






# KC difficulty
sdifficult2id = {}
count = []
nonesk = []   #  dropped with less than 30 answer records
for i in tqdm(skills):
    tttt = []
    idx = all_data[(all_data.skill_id==i)].index.tolist() 
    temp1 = all_data.iloc[idx]
    if len(idx) < 30:
        sdifficult2id[i] = 1.02
        nonesk.append(i)
        continue
    for xxx in np.array(temp1):
        tttt.append(xxx[2])
    if tttt == []:

        sdifficult2id[i] = 1.02
        nonesk.append(i)
        continue
    avg = int(np.mean(tttt)*100)+1
    count.append(avg)
    sdifficult2id[i] = avg 

# Question difficulty
difficult2id = {}
count = []
nones = []
for i in tqdm(problem):
    tttt = []
    idx = all_data[(all_data.problem_id==i)].index.tolist() 
    temp1 = all_data.iloc[idx]
    if len(idx) < 30:
        difficult2id[i] = 1.02
        nones.append(i)
        continue
    for xxx in np.array(temp1):
        tttt.append(xxx[2])
    if tttt == []:
        difficult2id[i] = 1.02
        nones.append(i)
        continue
    avg = int(np.mean(tttt)*100)+1
    count.append(avg)

    difficult2id[i] = avg 


with open('data/difficult2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(difficult2id))
with open('data/sdifficult2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(sdifficult2id))
np.save('data/nones.npy', np.array(nones))
np.save('data/nonesk.npy', np.array(nonesk))

print('complete')
  



 