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

    df_grouped = data_f.groupby('userid')

    df_list = applyParallel(df_grouped, gen_level_label_list, n_jobs=10, backend='loky')
    tmp_df = pd.concat(df_list)

    return tmp_df

def gen_level_label_list(uid, t):
    label_list = t["label"].tolist()
    for col in ["cate_id"]:
        t["label_" + col] = trans_label_func(t[col].tolist(), label_list)
    
    return uid, t

def trans_label_func(level_list, label_list):
    t = sorted(zip(level_list, label_list), key=lambda x:x[1], reverse=False)
    d = dict(t)

    return [d.get(j, 0) for j in level_list]

def gen_sess_feature_din(row):
    sess_max_len = DIN_SESS_MAX_LEN
    sess_input_dict = {'cate_id': [0], 'item_id': [0]}
    sess_input_length = 0
    user, time_stamp = row[1]['userid'], row[1]['timestamp']
    if user not in user_hist_session or len(user_hist_session[user]) == 0:

        sess_input_dict['cate_id'] = [0]
        sess_input_dict['item_id'] = [0]
        sess_input_length = 0
    else:
        cur_sess = user_hist_session[user][0]
        for i in reversed(range(len(cur_sess))):
            if cur_sess[i][1] < time_stamp:
                sess_input_dict['cate_id'] = [e[0]
                                              for e in cur_sess[max(0, i + 1 - sess_max_len):i + 1]]
                sess_input_dict['item_id'] = [e[1]
                                            for e in cur_sess[max(0, i + 1 - sess_max_len):i + 1]]
                sess_input_length = len(sess_input_dict['cate_id'])
                break
    return sess_input_dict['cate_id'], sess_input_dict['item_id'], sess_input_length

def gen_session_list_din(uid, t):
    t.sort_values('timestamp', inplace=True, ascending=True)
    session_list = []
    session = []
    c_b_set=[]
    for row in t.iterrows():

        time_stamp = row[1]['timestamp']
        # pd_time = pd.to_datetime(timestamp_datetime())
        # delta = pd_time  - last_time
        cat_id = row[1]['cate_id']
        item_id = row[1]['item_id']
        if (cat_id, item_id) in c_b_set:
            continue
        c_b_set.append((cat_id, item_id))
        session.append((cat_id, item_id, time_stamp))

    if len(session) > 2:
        session_list.append(session[:])
    return uid, session_list


def applyParallel(df_grouped, func, n_jobs, backend='multiprocessing'):
    """Use Parallel and delayed """  # backend='threading'
    results = Parallel(n_jobs=n_jobs, verbose=4, backend=backend)(
        delayed(func)(name, group) for name, group in df_grouped)

    return {k: v for k, v in results}

if os.path.exists('data_user/feature_list.pkl') and os.path.exists('data_user/data.pkl') and os.path.exists('data_user/label.pkl'):
  print('UserBehavior Data Ready!')
  os._exit(0)

a = pd.read_csv('data_user/UserBehavior.csv', names = ['userid', 'item_id', 'cate_id', 'action', 'timestamp'])
a.columns = ['userid', 'item_id', 'cate_id', 'action', 'timestamp']

a = a.loc[a['action'] != 'fav']
a = a.loc[a['action'] != 'cart']
a['label'] = a['action'].map({'pv':0, 'buy':1})

features = ['userid' , 'item_id',  'cate_id']
for item in features:
  lbe = LabelEncoder()
  unique_cat_id = a[item].unique()
  lbe.fit(unique_cat_id)
  a[item] = lbe.transform(a[item]) + 1

df_grouped = a.groupby('userid')
user_hist_session = applyParallel(
                df_grouped, gen_session_list_din, n_jobs=10, backend='loky')


sess_input_dict = {'cate_id': [], 'item_id':[]}
sess_input_length = []
for row in tqdm(a[['userid', 'timestamp']].iterrows()):
        aa, bb, cc = gen_sess_feature_din(row)
        sess_input_dict['cate_id'].append(aa)
        sess_input_dict['item_id'].append(bb)
        sess_input_length.append(cc)
#a.drop(columns=['Unnamed: 0'], inplace=True)

train_idx = np.random.choice(len(a), int(len(a)*0.8), replace = False)
flag = np.ones(len(a))
flag[train_idx] = 0
test_idx = np.squeeze(np.argwhere(flag == 1))

pd.to_pickle(train_idx, 'data_user/train_idx.pkl')
pd.to_pickle(test_idx, 'data_user/test_idx.pkl')
train_data = a.iloc[train_idx]
test_data = a.iloc[test_idx]

train_data = generate_df(train_data)
test_data['label_cate_id'] = 0


print(train_data['label'].value_counts())
print(test_data['label'].value_counts())
a = pd.concat([train_data, test_data], axis = 0)

print(a)
a.to_csv('data_user/data.csv')
print('done')

sparse_features = ['userid']
sparse_feature_list = [SparseFeat(feat, vocabulary_size=a[feat].max(
    ) + ID_OFFSET, embedding_dim=16) for feat in sparse_features + ['cate_id','item_id']]


sess_feature = ['cate_id','item_id']
feature_dict = {}
for feat in sparse_feature_list:
    feature_dict[feat.name] = a[feat.name].values
    
for feat in sess_feature:
    feature_dict['hist_'+feat] = pad_sequences(sess_input_dict[feat], maxlen=DIN_SESS_MAX_LEN, padding='post')
feature_dict['seq_length'] = np.array(sess_input_length, dtype=np.int64)

sparse_feature_list += [
      VarLenSparseFeat(SparseFeat('hist_cate_id', vocabulary_size=a['cate_id'].max(
      ) + ID_OFFSET, embedding_name='cate_id', embedding_dim=16), maxlen=DIN_SESS_MAX_LEN, length_name='seq_length'),
      VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=a['item_id'].max(
      ) + ID_OFFSET, embedding_name='item_id', embedding_dim=16), maxlen=DIN_SESS_MAX_LEN, length_name='seq_length')]

pd.to_pickle(sparse_feature_list, 'data_user/feature_list.pkl')
pd.to_pickle(feature_dict, 'data_user/data.pkl')
pd.to_pickle(a['label'].values, 'data_user/label.pkl')
pd.to_pickle(a['label_cate_id'].values, 'data_user/label_cate.pkl')

print('done')
# a.drop(columns=['Unnamed: 0'], inplace=True)
# a.to_csv('UserBehavior.csv')


