import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat
from joblib import Parallel, delayed
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gc
from tqdm import tqdm
import os

DIN_SESS_MAX_LEN = 50
ID_OFFSET = 10

def generate_df(data_f):

    df_grouped = data_f.groupby('user_id')

    df_list = applyParallel(df_grouped, gen_level_label_list, n_jobs=10, backend='loky')
    tmp_df = pd.concat(df_list)

    return tmp_df

def gen_level_label_list(uid, t):
    label_list = t["label"].tolist()
    for col in ["cat_id", 'brand_id']:
        t["label_" + col] = trans_label_func(t[col].tolist(), label_list)
    
    return uid, t

def trans_label_func(level_list, label_list):
    t = sorted(zip(level_list, label_list), key=lambda x:x[1], reverse=False)
    d = dict(t)

    return [d.get(j, 0) for j in level_list]

def gen_sess_feature_din(row):
    sess_max_len = DIN_SESS_MAX_LEN
    sess_input_dict = {'cat_id': [0], 'brand_id': [0]}
    sess_input_length = 0
    user, time_stamp = row[1]['user_id'], row[1]['time_stamp']
    if user not in user_hist_session or len(user_hist_session[user]) == 0:

        sess_input_dict['cat_id'] = [0]
        sess_input_dict['brand_id'] = [0]
        sess_input_length = 0
    else:
        cur_sess = user_hist_session[user][0]
        for i in reversed(range(len(cur_sess))):
            if cur_sess[i][2] < time_stamp:
                sess_input_dict['cat_id'] = [e[0]
                                              for e in cur_sess[max(0, i + 1 - sess_max_len):i + 1]]
                sess_input_dict['brand_id'] = [e[1]
                                            for e in cur_sess[max(0, i + 1 - sess_max_len):i + 1]]
                sess_input_length = len(sess_input_dict['brand_id'])
                break
    return sess_input_dict['cat_id'], sess_input_dict['brand_id'], sess_input_length

def gen_session_list_din(uid, t):
    t.sort_values('time_stamp', inplace=True, ascending=True)
    session_list = []
    session = []
    c_b_set=[]
    for row in t.iterrows():

        time_stamp = row[1]['time_stamp']
        # pd_time = pd.to_datetime(timestamp_datetime())
        # delta = pd_time  - last_time
        cat_id = row[1]['cat_id']
        brand = row[1]['brand_id']
        if (cat_id, brand) in c_b_set:
            continue
        c_b_set.append((cat_id, brand))
        session.append((cat_id, brand, time_stamp))

    if len(session) > 2:
        session_list.append(session[:])
    return uid, session_list


def applyParallel(df_grouped, func, n_jobs, backend='multiprocessing'):
    """Use Parallel and delayed """  # backend='threading'
    results = Parallel(n_jobs=n_jobs, verbose=4, backend=backend)(
        delayed(func)(name, group) for name, group in df_grouped)

    return {k: v for k, v in results}

if os.path.exists('data_ijcai/feature_list.pkl') and os.path.exists('data_ijcai/data.pkl') and os.path.exists('data_ijcai/label.pkl'):
  print('IJCAI Data Ready!')
  os._exit(0)

a = pd.read_csv('data_ijcai/data_format1/user_log_format1.csv')
b = pd.read_csv('data_ijcai/data_format1/user_info_format1.csv')

data = pd.merge(a, b, on='user_id')

data = data.loc[data['action_type'] != 1]
data = data.loc[data['action_type'] != 3]
data.to_csv('data_ijcai/data.csv')
del data

a = pd.read_csv('data_ijcai/data.csv')

a['age_range'] = a['age_range'].fillna(-1)
a['item_id'] = a['item_id'].fillna(-1)
a['gender'] = a['gender'].fillna(-1)
a['user_id'] = a['user_id'].fillna(-1)
a['cat_id'] = a['cat_id'].fillna(-1)
a['brand_id'] = a['brand_id'].fillna(-1)
a['seller_id'] = a['seller_id'].fillna(-1)

lbe = LabelEncoder()
unique_cat_id = a['cat_id'].unique()
lbe.fit(unique_cat_id)
a['cat_id'] = lbe.transform(a['cat_id']) + 1

lbe = LabelEncoder()
unique_brand_id = a['brand_id'].unique()
lbe.fit(unique_brand_id)
a['brand_id'] = lbe.transform(a['brand_id']) + 1

lbe = LabelEncoder()
unique_age_range_id = a['age_range'].unique()
lbe.fit(unique_age_range_id)
a['age_range'] = lbe.transform(a['age_range']) + 1

lbe = LabelEncoder()
unique_cat_id = a['gender'].unique()
lbe.fit(unique_cat_id)
a['gender'] = lbe.transform(a['gender']) + 1

lbe = LabelEncoder()
unique_cat_id = a['item_id'].unique()
lbe.fit(unique_cat_id)
a['item_id'] = lbe.transform(a['item_id']) + 1

lbe = LabelEncoder()
unique_cat_id = a['user_id'].unique()
lbe.fit(unique_cat_id)
a['user_id'] = lbe.transform(a['user_id']) + 1

lbe = LabelEncoder()
unique_cat_id = a['seller_id'].unique()
lbe.fit(unique_cat_id)
a['seller_id'] = lbe.transform(a['seller_id']) + 1

a['label'] = a['action_type']//2

df_grouped = a.groupby('user_id')
user_hist_session = applyParallel(
                df_grouped, gen_session_list_din, n_jobs=10, backend='loky')

print('done')

sess_input_dict = {'cat_id': [], 'brand_id': []}
sess_input_length = []
for row in tqdm(a[['user_id', 'time_stamp']].iterrows()):
        aa, bb, cc = gen_sess_feature_din(row)
        sess_input_dict['cat_id'].append(aa)
        sess_input_dict['brand_id'].append(bb)
        sess_input_length.append(cc)
# a['hist_cat_id'] = sess_input_dict['cat_id']
# a['hist_brand_id'] = sess_input_dict['brand_id']
#a.drop(columns=['Unnamed: 0'], inplace=True)

train_idx = np.random.choice(len(a), int(len(a)*0.9), replace = False)
flag = np.ones(len(a))
flag[train_idx] = 0
test_idx = np.squeeze(np.argwhere(flag == 1))
pd.to_pickle(train_idx, 'data_ijcai/train_idx.pkl')
pd.to_pickle(test_idx, 'data_ijcai/test_idx.pkl')
train_data = a.iloc[train_idx]
test_data = a.iloc[test_idx]

train_data = generate_df(train_data)
test_data['label_cat_id'] = 0
test_data['label_brand_id'] = 0

print(train_data['label'].value_counts())
print(test_data['label'].value_counts())
a = pd.concat([train_data, test_data], axis = 0)

print('done')

sparse_features = ['user_id', 'item_id', 'seller_id', 'age_range', 'gender']
sparse_feature_list = [SparseFeat(feat, vocabulary_size=a[feat].max(
    ) + ID_OFFSET, embedding_dim=16) for feat in sparse_features + ['cat_id', 'brand_id']]


sess_feature = ['cat_id', 'brand_id']
feature_dict = {}
for feat in sparse_feature_list:
    feature_dict[feat.name] = a[feat.name].values
    
for feat in sess_feature:
    feature_dict['hist_'+feat] = pad_sequences(sess_input_dict[feat], maxlen=DIN_SESS_MAX_LEN, padding='post')
feature_dict['seq_length'] = np.array(sess_input_length, dtype=np.int64)

sparse_feature_list += [
      VarLenSparseFeat(SparseFeat('hist_cat_id', vocabulary_size=a['cat_id'].max(
      ) + ID_OFFSET, embedding_name='cat_id', embedding_dim=16), maxlen=DIN_SESS_MAX_LEN, length_name='seq_length'),
      VarLenSparseFeat(SparseFeat('hist_brand_id', vocabulary_size=a['brand_id'].max(
      ) + ID_OFFSET, embedding_name='brand_id', embedding_dim=16), maxlen=DIN_SESS_MAX_LEN, length_name='seq_length')]

pd.to_pickle(sparse_feature_list, 'data_ijcai/feature_list.pkl')
pd.to_pickle(feature_dict, 'data_ijcai/data.pkl')
pd.to_pickle(a['label'].values, 'data_ijcai/label.pkl')
pd.to_pickle(a['label_cat_id'].values, 'data_ijcai/label_cate.pkl')
pd.to_pickle(a['label_brand_id'].values, 'data_ijcai/label_brand.pkl')

print('done')