# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.(https://arxiv.org/abs/1810.11921)
"""
import torch
import torch.nn as nn

from .basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import concat_fun, InteractingLayer, FM
from .layers import DNN
from deepctr_torch.layers import PredictionLayer


class AutoInt(BaseModel):
    """Instantiates the AutoInt Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param att_layer_num: int.The InteractingLayer number to be used.
    :param att_head_num: int.The head number in multi-head  self-attention network.
    :param att_res: bool.Whether or not use standard residual connections before output.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_use_bn:  bool. Whether use BatchNormalization before activation or not in DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, att_layer_num=3,
                 att_head_num=2, att_res=True, dnn_hidden_units=(256, 128), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cuda:0', gpus=None, varlen_num=2):

        super(AutoInt, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=0,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)
        if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
            raise ValueError("Either hidden_layer or att_layer_num must > 0")
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        field_num = len(self.embedding_dict)

        embedding_size = self.embedding_size

        if len(dnn_hidden_units) and att_layer_num > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1] + field_num * embedding_size + embedding_size * varlen_num
        elif len(dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif att_layer_num > 0:
            dnn_linear_in_feature = field_num * embedding_size
        else:
            raise NotImplementedError

        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.dnn_hidden_units = dnn_hidden_units
        self.att_layer_num = att_layer_num
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.int_layers = nn.ModuleList(
            [InteractingLayer(embedding_size, att_head_num, att_res, device=device) for _ in range(att_layer_num)])

        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        logit = self.linear_model(X)

        att_input = concat_fun(sparse_embedding_list, axis=1)

        for layer in self.int_layers:
            att_input = layer(att_input)

        att_output = torch.flatten(att_input, start_dim=1)

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        if len(self.dnn_hidden_units) > 0 and self.att_layer_num > 0:  # Deep & Interacting Layer
            deep_out = self.dnn(dnn_input)
            stack_out = concat_fun([att_output, deep_out])
            logit += self.dnn_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        elif self.att_layer_num > 0:  # Only Interacting Layer
            logit += self.dnn_linear(att_output)
        else:  # Error
            pass

        y_pred = self.out(logit)

        return y_pred

class AutoInt_task(BaseModel):
    """Instantiates the AutoInt Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param att_layer_num: int.The InteractingLayer number to be used.
    :param att_head_num: int.The head number in multi-head  self-attention network.
    :param att_res: bool.Whether or not use standard residual connections before output.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_use_bn:  bool. Whether use BatchNormalization before activation or not in DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, att_layer_num=3,
                 att_head_num=2, att_res=True, dnn_hidden_units=(256, 128), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cuda:0', gpus=None, aux_num = 2, aux_units = (64, 32), detach = False, varlen_num=2):

        super(AutoInt_task, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=0,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus, aux_num = aux_num)
        if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
            raise ValueError("Either hidden_layer or att_layer_num must > 0")
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        field_num = len(self.embedding_dict)

        embedding_size = self.embedding_size

        if len(dnn_hidden_units) and att_layer_num > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1] + field_num * embedding_size + embedding_size * varlen_num
        elif len(dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif att_layer_num > 0:
            dnn_linear_in_feature = field_num * embedding_size
        else:
            raise NotImplementedError

        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.dnn_hidden_units = dnn_hidden_units
        self.att_layer_num = att_layer_num
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.aux1_dnn = DNN(self.compute_input_dim(dnn_feature_columns), aux_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
        self.aux1_dnn_linear = nn.Linear(
            aux_units[-1] + field_num * embedding_size + embedding_size * varlen_num, 1, bias=False).to(device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.aux1_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.aux1_dnn_linear.weight, l2=l2_reg_dnn)

        if self.aux_num == 2:
            self.aux2_dnn = DNN(self.compute_input_dim(dnn_feature_columns), aux_units,
                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                            init_std=init_std, device=device)
            self.aux2_dnn_linear = nn.Linear(
                aux_units[-1] + field_num * embedding_size + embedding_size * varlen_num, 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.aux2_dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.aux2_dnn_linear.weight, l2=l2_reg_dnn)

        self.int_layers = nn.ModuleList(
            [InteractingLayer(embedding_size, att_head_num, att_res, device=device) for _ in range(att_layer_num)])
        
        self.final_linear = nn.Linear(self.aux_num+1, 1, bias=False).to(device)
        self.add_regularization_weight(self.final_linear.weight, l2=l2_reg_dnn)
        self.out_main = PredictionLayer(task,)
        self.out_aux1 = PredictionLayer(task,)
        self.out_aux2 = PredictionLayer(task,)

        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        public_logit = self.linear_model(X)

        att_input = concat_fun(sparse_embedding_list, axis=1)

        for layer in self.int_layers:
            att_input = layer(att_input)

        att_output = torch.flatten(att_input, start_dim=1)

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        if len(self.dnn_hidden_units) > 0 and self.att_layer_num > 0:  # Deep & Interacting Layer
            deep_out = self.dnn(dnn_input)
            stack_out = concat_fun([att_output, deep_out])
            main_logit = self.dnn_linear(stack_out) + public_logit
        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        elif self.att_layer_num > 0:  # Only Interacting Layer
            logit += self.dnn_linear(att_output)
        else:  # Error
            pass
        
        aux1_dnn_output = self.aux1_dnn(dnn_input)
        aux1_dnn_output = concat_fun([att_output, aux1_dnn_output])
        aux1_dnn_logit = self.aux1_dnn_linear(aux1_dnn_output)
        aux1_logit = aux1_dnn_logit + public_logit

        if self.aux_num == 2:
            aux2_dnn_output = self.aux2_dnn(dnn_input)
            aux2_dnn_output = concat_fun([att_output, aux2_dnn_output])
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

class AutoInt_task_FM(BaseModel):
    """Instantiates the AutoInt Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param att_layer_num: int.The InteractingLayer number to be used.
    :param att_head_num: int.The head number in multi-head  self-attention network.
    :param att_res: bool.Whether or not use standard residual connections before output.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_use_bn:  bool. Whether use BatchNormalization before activation or not in DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, att_layer_num=3,
                 att_head_num=2, att_res=True, dnn_hidden_units=(256, 128), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cuda:0', gpus=None, aux_num = 2, aux_units = (64, 32), detach = False):

        super(AutoInt_task_FM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=0,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus, aux_num = aux_num)
        if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
            raise ValueError("Either hidden_layer or att_layer_num must > 0")
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        field_num = len(self.embedding_dict)
        
        embedding_size = self.embedding_size

        if len(dnn_hidden_units) and att_layer_num > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1] + field_num * embedding_size
        elif len(dnn_hidden_units) > 0:
            dnn_linear_in_feature = dnn_hidden_units[-1]
        elif att_layer_num > 0:
            dnn_linear_in_feature = field_num * embedding_size
        else:
            raise NotImplementedError

        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
        self.dnn_hidden_units = dnn_hidden_units
        self.att_layer_num = att_layer_num
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        
        self.fm = FM()
        self.aux1_dnn = DNN(self.compute_input_dim(dnn_feature_columns), aux_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
        self.aux1_dnn_linear = nn.Linear(
            aux_units[-1] + field_num * embedding_size, 1, bias=False).to(device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.aux1_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.aux1_dnn_linear.weight, l2=l2_reg_dnn)

        self.aux2_dnn = DNN(self.compute_input_dim(dnn_feature_columns), aux_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
        self.aux2_dnn_linear = nn.Linear(
            aux_units[-1] + field_num * embedding_size, 1, bias=False).to(device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.aux2_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.aux2_dnn_linear.weight, l2=l2_reg_dnn)

        self.int_layers = nn.ModuleList(
            [InteractingLayer(embedding_size, att_head_num, att_res, device=device) for _ in range(att_layer_num)])
        
        self.final_linear = nn.Linear(3, 1, bias=False).to(device)
        self.add_regularization_weight(self.final_linear.weight, l2=l2_reg_dnn)
        self.out_main = PredictionLayer(task,)
        self.out_aux1 = PredictionLayer(task,)
        self.out_aux2 = PredictionLayer(task,)

        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        public_logit = self.linear_model(X)

        att_input = concat_fun(sparse_embedding_list, axis=1)

        for layer in self.int_layers:
            att_input = layer(att_input)

        att_output = torch.flatten(att_input, start_dim=1)

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        if len(self.dnn_hidden_units) > 0 and self.att_layer_num > 0:  # Deep & Interacting Layer
            deep_out = self.dnn(dnn_input)
            stack_out = concat_fun([att_output, deep_out])
            main_logit = self.dnn_linear(stack_out) + public_logit
        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        elif self.att_layer_num > 0:  # Only Interacting Layer
            logit += self.dnn_linear(att_output)
        else:  # Error
            pass
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        aux1_dnn_output = self.aux1_dnn(dnn_input)
        aux1_dnn_output = concat_fun([att_output, aux1_dnn_output])
        aux1_dnn_logit = self.aux1_dnn_linear(aux1_dnn_output)
        aux1_logit = aux1_dnn_logit + public_logit + self.fm(fm_input)

        aux2_dnn_output = self.aux2_dnn(dnn_input)
        aux2_dnn_output = concat_fun([att_output, aux2_dnn_output])
        aux2_dnn_logit = self.aux2_dnn_linear(aux2_dnn_output)
        aux2_logit = aux2_dnn_logit + public_logit + self.fm(fm_input)

        final_logit = self.final_linear(torch.cat([main_logit, aux1_logit, aux2_logit], dim=-1))
        y_main = self.out_main(main_logit)
        y_aux1 = self.out_aux1(aux1_logit)
        y_aux2 = self.out_aux2(aux2_logit)
        y_pred = self.out(final_logit)

        return y_pred, y_aux1, y_aux2, y_main