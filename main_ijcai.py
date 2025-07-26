import os
import sys

import numpy as np
import pandas as pd
import torch
from models.din import DIN
from models.dcn import DCN
from models.deepfm import DeepFM
from models.autoint import AutoInt
from models.xdeepfm import xDeepFM
from models.finalmlp import FinalMLP
from models.fignn import FiGNN
from sklearn.metrics import log_loss, roc_auc_score

FRAC = 1.0
DIN_SESS_MAX_LEN = 50
DSIN_SESS_COUNT = 5
DSIN_SESS_MAX_LEN = 10
ID_OFFSET = 1000

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == "__main__":
    #f = open('log.txt', 'a+', encoding='utf-8')
    FRAC = FRAC
    SESS_MAX_LEN = DIN_SESS_MAX_LEN


    feature_columns = pd.read_pickle('data_ijcai/feature_list.pkl')

    dnn_feature_columns = []
    linear_feature_columns = []

    scenario_feature_list = ['gender', 'age_range', 'seller_id']

    for item in feature_columns:
      # if 'hist' not in item.name:
      dnn_feature_columns.append(item)
      # linear_feature_columns.append(item)
      if item.name in scenario_feature_list:
        linear_feature_columns.append(item)
    # print(dnn_feature_columns)
    # exit(0)
    # print(type(dnn_feature_columns))
    # exit(0)

    model_input = pd.read_pickle('data_ijcai/data.pkl')
    label = pd.read_pickle('data_ijcai/label.pkl')
    label_cate = pd.read_pickle('data_ijcai/label_cate.pkl')
    label_brand = pd.read_pickle('data_ijcai/label_brand.pkl')

      
    train_idx = pd.read_pickle('data_ijcai/train_idx.pkl')
    test_idx = pd.read_pickle('data_ijcai/test_idx.pkl')


    # train_idx = train_idx[0:50000]
    # test_idx = test_idx[0:50000]

    train_input = {k: v[train_idx] for k, v in model_input.items()}
    #val_input = {k: v[val_idx] for k, v in model_input.items()}
    test_input = {k: v[test_idx] for k, v in model_input.items()}
    train_label = label[train_idx]
    train_label_brand = label_brand[train_idx]
    train_label_cate = label_cate[train_idx]
    #val_label = label[val_idx]
    test_label = label[test_idx]

    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    sess_len_max = SESS_MAX_LEN
    BATCH_SIZE = 8192

    sess_feature_list = ['cat_id', 'brand_id']
    # scenario_feature_list = ['occupation']
    TEST_BATCH_SIZE = 2 ** 14
    aux_weight = 0.005
    models_list = ['DIN','FM','AutoInt','xDeepFM','DCN','FinalMLP','FiGNN']

    times = 1
    all_final_auc = []
    all_main_auc = []
    all_final_loss = []
    all_main_loss = []
    for model_now in models_list:
      print('Now Running: Baseline Model - '+ model_now)
      for t in range(times):
        if model_now == 'DIN':
          model = DIN(feature_columns, sess_feature_list, device=device, att_weight_normalization=True, dnn_hidden_units=(512, 256), dnn_activation='relu',
                      att_hidden_size=(512, 256), seed=t, l2_reg_embedding=1e-6).cuda()
        elif model_now == 'FM':
          model = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(512, 256, 128), dnn_activation='relu', l2_reg_linear=1e-7, l2_reg_embedding=1e-6, l2_reg_dnn=1e-7, seed=t)
        elif model_now == 'AutoInt':
          model = AutoInt(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128), l2_reg_embedding=1e-6, l2_reg_dnn=1e-7, seed=t)
        elif model_now == 'xDeepFM':
          model = xDeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(512, 256, 128), dnn_activation='relu', l2_reg_linear=1e-7, l2_reg_embedding=1e-6, l2_reg_dnn=1e-7, l2_reg_cin=1e-7, seed=t)
        elif model_now == 'DCN':
          model = DCN(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128), dnn_activation='relu', l2_reg_linear=1e-7, l2_reg_embedding=1e-6, l2_reg_dnn=1e-7, l2_reg_cross=1e-7, seed=t)
        elif model_now == 'FinalMLP':
          model = FinalMLP(dnn_feature_columns, mlp1_hidden_units = (400, 400, 400), mlp2_hidden_units=[800], num_heads=10, l2_reg_embedding=1e-6, l2_reg_dnn=1e-7, seed=t)
        elif model_now == 'FiGNN':
          model = FiGNN(dnn_feature_columns, gnn_layers=5, use_residual=True, use_gru=True, reuse_graph_layer=True, l2_reg_embedding=1e-6, l2_reg_dnn=1e-7, seed=t)

        model.compile('adam', 'binary_crossentropy',
                      metrics=['binary_crossentropy', 'auc'])
        history = model.fit(train_input, train_label,
                          batch_size=BATCH_SIZE, epochs=1, initial_epoch=0, verbose=2, validation_data=(test_input, test_label))
        pred_ans = model.predict(test_input, TEST_BATCH_SIZE)

        final_loss = log_loss(test_label, pred_ans)
        final_auc = roc_auc_score(test_label, pred_ans)
        # print("test LogLoss", round(final_loss, 6), "test AUC",
        #     round(final_auc, 6))


        all_final_auc.append(final_auc)
        all_final_loss.append(final_loss)

      
      print('Baseline Method - '+ model_now+' :')
      print('AUC:'+str(np.mean(all_final_auc))+'')
      print('Logloss:'+str(np.mean(all_final_loss))+'')
      print('\n')