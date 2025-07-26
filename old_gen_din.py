# coding: utf-8

import gc
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat
from tqdm import tqdm

FRAC = 1.0
DIN_SESS_MAX_LEN = 50
DSIN_SESS_COUNT = 5
DSIN_SESS_MAX_LEN = 10
ID_OFFSET = 1000

def gen_sess_feature_din(row):
    sess_max_len = DIN_SESS_MAX_LEN
    sess_input_dict = {'cate_id': [0], 'brand': [0]}
    sess_input_length = 0
    user, time_stamp = row[1]['userid'], row[1]['time_stamp']
    if user not in user_hist_session or len(user_hist_session[user]) == 0:

        sess_input_dict['cate_id'] = [0]
        sess_input_dict['brand'] = [0]
        sess_input_length = 0
    else:
        cur_sess = user_hist_session[user][0]
        for i in reversed(range(len(cur_sess))):
            if cur_sess[i][2] < time_stamp:
                sess_input_dict['cate_id'] = [e[0]
                                              for e in cur_sess[max(0, i + 1 - sess_max_len):i + 1]]
                sess_input_dict['brand'] = [e[1]
                                            for e in cur_sess[max(0, i + 1 - sess_max_len):i + 1]]
                sess_input_length = len(sess_input_dict['brand'])
                break
    return sess_input_dict['cate_id'], sess_input_dict['brand'], sess_input_length

def applyParallel(df_grouped, func, n_jobs, backend='multiprocessing'):
    """Use Parallel and delayed """  # backend='threading'
    results = Parallel(n_jobs=n_jobs, verbose=4, backend=backend)(
        delayed(func)(name, group) for name, group in df_grouped)
    # print(results)

    return [ v for _, v in results ]

def trans_label_func(level_list, label_list):
    t = sorted(zip(level_list, label_list), key=lambda x:x[1], reverse=False)
    d = dict(t)

    return [d.get(j, 0) for j in level_list]

def gen_level_label_list(uid, t):
    label_list = t["clk"].tolist()
    for col in ["cate_id", "brand"]:
        t["label_" + col] = trans_label_func(t[col].tolist(), label_list)
    
    return uid, t

def generate_df(data_f):

    df_grouped = data_f.groupby('userid')

    df_list = applyParallel(df_grouped, gen_level_label_list, n_jobs=10, backend='loky')
    tmp_df = pd.concat(df_list)

    return tmp_df

if __name__ == "__main__":

    if os.path.exists('data_taobao/model_input_old/din_input_' +str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl') and os.path.exists('data_taobao/model_input_old/din_label_' + str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl'):
        print('Taobao Ready!')
        os._exit(0)

    user_hist_session = {}
    if not os.path.exists('data_taobao/model_input_old/'):
        os.mkdir('data_taobao/model_input_old/')
    FILE_NUM = len(
        list(
            filter(lambda x: x.startswith('user_hist_session_' + str(FRAC) + '_din_'), os.listdir('data_taobao/sampled_data_old/'))))

    print('total', FILE_NUM, 'files')
    for i in range(FILE_NUM):
        user_hist_session_ = pd.read_pickle(
            'data_taobao/sampled_data_old/user_hist_session_' + str(FRAC) + '_din_' + str(i) + '.pkl')
        user_hist_session.update(user_hist_session_)
        del user_hist_session_

    sample_sub = pd.read_pickle(
        'data_taobao/sampled_data_old/raw_sample_' + str(FRAC) + '.pkl')
    sample_sub.rename(columns={'user': 'userid'}, inplace=True)

    user = pd.read_pickle('data_taobao/sampled_data_old/user_profile_' + str(FRAC) + '.pkl')
    ad = pd.read_pickle('data_taobao/sampled_data_old/ad_feature_enc_' + str(FRAC) + '.pkl')
    user = user.fillna(-1)
    user.rename(
        columns={'new_user_class_level ': 'new_user_class_level'}, inplace=True)

    data = pd.merge(sample_sub, user, how='left', on='userid', )
    data = pd.merge(data, ad, how='left', on='adgroup_id')
    data.dropna(inplace=True, subset=['cate_id', 'brand'])
    print(data.isnull().sum(axis=0))

    data = data.loc[data.time_stamp >= 1494028800] # >= 0506
    data = data.loc[data.time_stamp < 1494720000] # < 0514

    sess_input_dict = {'cate_id': [], 'brand': []}
    sess_input_length = []
    for row in tqdm(data[['userid', 'time_stamp']].iterrows()):
        a, b, c = gen_sess_feature_din(row)
        sess_input_dict['cate_id'].append(a)
        sess_input_dict['brand'].append(b)
        sess_input_length.append(c)

    print('done')

    sparse_features = ['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
                       'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'campaign_id',
                       'customer']
    dense_features = ['price']

    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()  # or Hash
        data[feat] = lbe.fit_transform(data[feat])
    mms = StandardScaler()
    data[dense_features] = mms.fit_transform(data[dense_features])

    sparse_feature_list = [SparseFeat(feat, vocabulary_size=int(data[feat].max(
    ) + ID_OFFSET), embedding_dim=32) for feat in sparse_features + ['cate_id', 'brand']]

    dense_feature_list = [DenseFeat(feat, dimension=1) for feat in dense_features]

    sess_feature = ['cate_id', 'brand']

    train_data = data.loc[data.time_stamp < 1494633600]
    test_data = data.loc[data.time_stamp >= 1494633600]

    train_data = generate_df(train_data) # labeling
    test_data['label_cate_id'] = 0
    test_data['label_brand'] = 0
    
    train_data.to_csv('data_taobao/model_input_old/din_1.0_train.csv')
    test_data.to_csv('data_taobao/model_input_old/din_1.0_test.csv')
    
    data = pd.concat([train_data, test_data], axis=0)
    del train_data, test_data

    feature_dict = {}
    for feat in sparse_feature_list + dense_feature_list:
        feature_dict[feat.name] = data[feat.name].values
    for feat in sess_feature:
        feature_dict['hist_' + feat] = pad_sequences(
            sess_input_dict[feat], maxlen=DIN_SESS_MAX_LEN, padding='post')
    feature_dict['seq_length'] = np.array(sess_input_length)
    sparse_feature_list += [
        VarLenSparseFeat(SparseFeat('hist_cate_id', vocabulary_size=int(data['cate_id'].max(
        ) + ID_OFFSET), embedding_name='cate_id', embedding_dim=32), maxlen=DIN_SESS_MAX_LEN, length_name='seq_length'),
        VarLenSparseFeat(SparseFeat('hist_brand', vocabulary_size=int(data['brand'].max(
        ) + ID_OFFSET), embedding_name='brand', embedding_dim=32), maxlen=DIN_SESS_MAX_LEN, length_name='seq_length')]
    feature_columns = sparse_feature_list + dense_feature_list
    model_input = feature_dict
    if not os.path.exists('data_taobao/model_input_old/'):
        os.mkdir('data_taobao/model_input_old/')

    pd.to_pickle(model_input, 'data_taobao/model_input_old/din_input_' +
                 str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')
    pd.to_pickle([np.array(sess_input_length)], 'data_taobao/model_input_old/din_input_len_' +
                 str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')

    pd.to_pickle(data['clk'].values, 'data_taobao/model_input_old/din_label_' +
                 str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')
    pd.to_pickle(data['label_cate_id'].values, 'data_taobao/model_input_old/din_label_cate_' +
                 str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')
    pd.to_pickle(data['label_brand'].values, 'data_taobao/model_input_old/din_label_brand_' +
                 str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')
    
    pd.to_pickle(feature_columns,
                 'data_taobao/model_input_old/din_fd_' + str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')

    pd.to_pickle(data['time_stamp'].values, 'data_taobao/model_input_old/din_times_' +
                 str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')           

    print("gen din input done")