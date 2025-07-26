# coding: utf-8
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

FRAC = 1.0
DIN_SESS_MAX_LEN = 50
DSIN_SESS_COUNT = 5
DSIN_SESS_MAX_LEN = 10
ID_OFFSET = 1000

if __name__ == "__main__":
    if os.path.exists('data_taobao/model_input_old/din_input_' +str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl') and os.path.exists('data_taobao/model_input_old/din_label_' + str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl'):
        print('ready')
        os._exit(0)

    user = pd.read_csv('data_taobao/raw_data/user_profile.csv')
    sample = pd.read_csv('data_taobao/raw_data/raw_sample.csv')

    if not os.path.exists('data_taobao/sampled_data_old/'):
        os.mkdir('data_taobao/sampled_data_old/')

    if os.path.exists('data_taobao/sampled_data_old/user_profile_' + str(FRAC) + '_.pkl') and os.path.exists(
            'data_taobao/sampled_data_old/raw_sample_' + str(FRAC) + '_.pkl'):
        user_sub = pd.read_pickle(
            'data_taobao/sampled_data_old/user_profile_' + str(FRAC) + '_.pkl')
        sample_sub = pd.read_pickle(
            'data_taobao/sampled_data_old/raw_sample_' + str(FRAC) + '_.pkl')
    else:

        if FRAC < 1.0:
            user_sub = user.sample(frac=FRAC, random_state=1024)
        else:
            user_sub = user
        sample_sub = sample.loc[sample.user.isin(user_sub.userid.unique())]
        pd.to_pickle(user_sub, 'data_taobao/sampled_data_old/user_profile_' +
                     str(FRAC) + '.pkl')
        pd.to_pickle(sample_sub, 'data_taobao/sampled_data_old/raw_sample_' +
                     str(FRAC) + '.pkl')

    if os.path.exists('data_taobao/raw_data/behavior_log_pv.pkl'):
        log = pd.read_pickle('data_taobao/raw_data/behavior_log_pv.pkl')
    else:
        log = pd.read_csv('data_taobao/raw_data/behavior_log.csv')
        log = log.loc[log['btag'] == 'pv']
        pd.to_pickle(log, 'data_taobao/raw_data/behavior_log_pv.pkl')

    userset = user_sub.userid.unique()
    log = log.loc[log.user.isin(userset)]
    # pd.to_pickle(log, 'data_taobao/sampled_data_old/behavior_log_pv_user_filter_' + str(FRAC) + '_.pkl')

    ad = pd.read_csv('data_taobao/raw_data/ad_feature.csv')
    ad.dropna(inplace=True, subset=['brand'])
    log.dropna(inplace=True, subset=['brand'])
    ad.dropna(inplace=True, subset=['cate_id'])
    log.dropna(inplace=True, subset=['cate'])
    print(ad.isnull().sum(axis=0))
    print(log.isnull().sum(axis=0))
    log = log.loc[log.cate.isin(ad.cate_id.unique())]
    ad['brand'] = ad['brand'].fillna(-1)

    lbe = LabelEncoder()
    # unique_cate_id = ad['cate_id'].unique()
    # log = log.loc[log.cate.isin(unique_cate_id)]

    unique_cate_id = np.concatenate(
        (ad['cate_id'].unique(), log['cate'].unique()))

    lbe.fit(unique_cate_id)
    ad['cate_id'] = lbe.transform(ad['cate_id']) + 1
    log['cate'] = lbe.transform(log['cate']) + 1

    lbe = LabelEncoder()
    # unique_brand = np.ad['brand'].unique()
    # log = log.loc[log.brand.isin(unique_brand)]

    unique_brand = np.concatenate(
        (ad['brand'].unique(), log['brand'].unique()))

    lbe.fit(unique_brand)
    ad['brand'] = lbe.transform(ad['brand']) + 1
    log['brand'] = lbe.transform(log['brand']) + 1

    log = log.loc[log.user.isin(sample_sub.user.unique())]
    log.drop(columns=['btag'], inplace=True)
    log = log.loc[log['time_stamp'] > 0]

    pd.to_pickle(ad, 'data_taobao/sampled_data_old/ad_feature_enc_' + str(FRAC) + '.pkl')
    pd.to_pickle(
        log, 'data_taobao/sampled_data_old/behavior_log_pv_user_filter_enc_' + str(FRAC) + '.pkl')

    print("0_gen_sampled_data done")