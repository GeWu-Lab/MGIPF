# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""
import torch
import torch.nn as nn

from .basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import FM
from .layers import DNN, WeightLayer
from deepctr_torch.layers import PredictionLayer
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat


class DeepFM(BaseModel):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cuda:0', gpus=None):

        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred

class DeepFM_task(BaseModel):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cuda:0', gpus=None, aux_num = 2, aux_units = (64, 32), detach = False):

        super(DeepFM_task, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus, aux_num = aux_num)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.aux1_dnn = DNN(self.compute_input_dim(dnn_feature_columns), aux_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
        self.aux1_dnn_linear = nn.Linear(
            aux_units[-1], 1, bias=False).to(device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.aux1_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.aux1_dnn_linear.weight, l2=l2_reg_dnn)

        if self.aux_num == 2:
            self.aux2_dnn = DNN(self.compute_input_dim(dnn_feature_columns), aux_units,
                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                            init_std=init_std, device=device)
            self.aux2_dnn_linear = nn.Linear(
                aux_units[-1], 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.aux2_dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.aux2_dnn_linear.weight, l2=l2_reg_dnn)

            self.final_linear = nn.Linear(3, 1, bias=False).to(device)
        else:
            self.final_linear = nn.Linear(2, 1, bias=False).to(device)
        self.add_regularization_weight(self.final_linear.weight, l2=l2_reg_dnn)
        self.out_main = PredictionLayer(task,)
        self.out_aux1 = PredictionLayer(task,)
        if self.aux_num == 2:
            self.out_aux2 = PredictionLayer(task,)

        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        public_logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            public_logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            main_logit = dnn_logit + public_logit
        
            aux1_dnn_output = self.aux1_dnn(dnn_input)
            aux1_dnn_logit = self.aux1_dnn_linear(aux1_dnn_output)
            aux1_logit = aux1_dnn_logit + public_logit

            if self.aux_num == 2:
                aux2_dnn_output = self.aux2_dnn(dnn_input)
                aux2_dnn_logit = self.aux2_dnn_linear(aux2_dnn_output)
                aux2_logit = aux2_dnn_logit + public_logit

                final_logit = self.final_linear(torch.cat([main_logit, aux1_logit, aux2_logit], dim=-1))
            else:
                final_logit = self.final_linear(torch.cat([main_logit, aux1_logit], dim=-1))

        y_main = self.out_main(main_logit)
        y_aux1 = self.out_aux1(aux1_logit)
        if self.aux_num == 2:
            y_aux2 = self.out_aux2(aux2_logit)
        y_pred = self.out(final_logit)

        if self.aux_num == 2:
            return y_pred, y_aux1, y_aux2, y_main
        else:
            return y_pred, y_aux1, y_main

class DeepFM_task_noitem(BaseModel):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cuda:0', gpus=None, aux_num = 2, aux_units = (64, 32), detach = False):

        super(DeepFM_task_noitem, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus, aux_num = aux_num)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()
        
        self.item_index = 0
        sparse_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), self.dnn_feature_columns)) if len(self.dnn_feature_columns) else []
        embedding_size = self.embedding_size
        for i in range(len(sparse_columns)):
            if 'item' in sparse_columns[i].name:
                self.item_index = i

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.aux1_dnn = DNN(self.compute_input_dim(dnn_feature_columns) - embedding_size, aux_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
        self.aux1_dnn_linear = nn.Linear(
            aux_units[-1], 1, bias=False).to(device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.aux1_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.aux1_dnn_linear.weight, l2=l2_reg_dnn)

        if self.aux_num == 2:
            self.aux2_dnn = DNN(self.compute_input_dim(dnn_feature_columns) - embedding_size, aux_units,
                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                            init_std=init_std, device=device)
            self.aux2_dnn_linear = nn.Linear(
                aux_units[-1], 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.aux2_dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.aux2_dnn_linear.weight, l2=l2_reg_dnn)

            self.final_linear = nn.Linear(3, 1, bias=False).to(device)
        else:
            self.final_linear = nn.Linear(2, 1, bias=False).to(device)
        self.add_regularization_weight(self.final_linear.weight, l2=l2_reg_dnn)
        self.out_main = PredictionLayer(task,)
        self.out_aux1 = PredictionLayer(task,)
        if self.aux_num == 2:
            self.out_aux2 = PredictionLayer(task,)

        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        # print(len(sparse_embedding_list))
        # print(sparse_embedding_list[0].shape)
        # print(self.item_index)
        # print(self.dnn_feature_columns[self.item_index])
        # exit(0)
        sparse_embedding_list_noitem = sparse_embedding_list[0:self.item_index] + sparse_embedding_list[self.item_index+1 :] 

        linear_logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            public_logit = self.fm(fm_input) + linear_logit

            aux_fm_input = torch.cat(sparse_embedding_list_noitem, dim=1)
            public_logit_aux = self.fm(aux_fm_input) + linear_logit

        if self.use_dnn:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            main_logit = dnn_logit + public_logit

            aux_dnn_input = combined_dnn_input(
                sparse_embedding_list_noitem, dense_value_list)
            aux1_dnn_output = self.aux1_dnn(aux_dnn_input)
            aux1_dnn_logit = self.aux1_dnn_linear(aux1_dnn_output)
            aux1_logit = aux1_dnn_logit + public_logit_aux

            if self.aux_num == 2:
                aux2_dnn_output = self.aux2_dnn(aux_dnn_input)
                aux2_dnn_logit = self.aux2_dnn_linear(aux2_dnn_output)
                aux2_logit = aux2_dnn_logit + public_logit_aux

                final_logit = self.final_linear(torch.cat([main_logit, aux1_logit, aux2_logit], dim=-1))
            else:
                final_logit = self.final_linear(torch.cat([main_logit, aux1_logit], dim=-1))

        y_main = self.out_main(main_logit)
        y_aux1 = self.out_aux1(aux1_logit)
        if self.aux_num == 2:
            y_aux2 = self.out_aux2(aux2_logit)
        y_pred = self.out(final_logit)

        if self.aux_num == 2:
            return y_pred, y_aux1, y_aux2, y_main
        else:
            return y_pred, y_aux1, y_main
