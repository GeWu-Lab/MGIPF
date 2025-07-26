# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.(https://arxiv.org/abs/1810.11921)
"""
import torch
import torch.nn as nn
from itertools import product
from .basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import concat_fun, InteractingLayer, FM
from .layers import DNN
from deepctr_torch.layers import PredictionLayer
import torch.nn.functional as F
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat

class FiGNN_Layer(nn.Module):
    def __init__(self, 
                 num_fields, 
                 embedding_dim,
                 gnn_layers=3,
                 reuse_graph_layer=False,
                 use_gru=True,
                 use_residual=True):
        super(FiGNN_Layer, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.gnn_layers = gnn_layers
        self.use_residual = use_residual
        self.reuse_graph_layer = reuse_graph_layer
        if reuse_graph_layer:
            self.gnn = GraphLayer(num_fields, embedding_dim)
        else:
            self.gnn = nn.ModuleList([GraphLayer(num_fields, embedding_dim)
                                      for _ in range(gnn_layers)])
        self.gru = nn.GRUCell(embedding_dim, embedding_dim) if use_gru else None
        self.src_nodes, self.dst_nodes = zip(*list(product(range(num_fields), repeat=2)))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.W_attn = nn.Linear(embedding_dim * 2, 1, bias=False)

    def build_graph_with_attention(self, feature_emb):
        src_emb = feature_emb[:, self.src_nodes, :]
        dst_emb = feature_emb[:, self.dst_nodes, :]
        concat_emb = torch.cat([src_emb, dst_emb], dim=-1)
        alpha = self.leaky_relu(self.W_attn(concat_emb))
        alpha = alpha.view(-1, self.num_fields, self.num_fields)
        mask = torch.eye(self.num_fields).to(feature_emb.device)
        alpha = alpha.masked_fill(mask.bool(), float('-inf'))
        graph = F.softmax(alpha, dim=-1) # batch x field x field without self-loops
        return graph

    def forward(self, feature_emb):
        g = self.build_graph_with_attention(feature_emb)
        h = feature_emb
        for i in range(self.gnn_layers):
            if self.reuse_graph_layer:
                a = self.gnn(g, h)
            else:
                a = self.gnn[i](g, h)
            if self.gru is not None:
                a = a.view(-1, self.embedding_dim)
                h = h.view(-1, self.embedding_dim)
                h = self.gru(a, h)
                h = h.view(-1, self.num_fields, self.embedding_dim)
            else:
                h = a + h
            if self.use_residual:
                h += feature_emb
        return h

class GraphLayer(nn.Module):
    def __init__(self, num_fields, embedding_dim):
        super(GraphLayer, self).__init__()
        self.W_in = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        self.W_out = torch.nn.Parameter(torch.Tensor(num_fields, embedding_dim, embedding_dim))
        nn.init.xavier_normal_(self.W_in)
        nn.init.xavier_normal_(self.W_out)
        self.bias_p = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, g, h):
        h_out = torch.matmul(self.W_out, h.unsqueeze(-1)).squeeze(-1) # broadcast multiply
        aggr = torch.bmm(g, h_out)
        a = torch.matmul(self.W_in, aggr.unsqueeze(-1)).squeeze(-1) + self.bias_p
        return a


class AttentionalPrediction(nn.Module):
    def __init__(self, num_fields, embedding_dim):
        super(AttentionalPrediction, self).__init__()
        self.mlp1 = nn.Linear(embedding_dim, 1, bias=False)
        self.mlp2 = nn.Sequential(nn.Linear(num_fields * embedding_dim, num_fields, bias=False),
                                  nn.Sigmoid())

    def forward(self, h):
        score = self.mlp1(h).squeeze(-1) # b x f
        weight = self.mlp2(h.flatten(start_dim=1)) # b x f
        logit = (weight * score).sum(dim=1, keepdim=True)
        return logit

class FiGNN(BaseModel):
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

    def __init__(self, dnn_feature_columns, gnn_layers=3,
                 use_residual=True, use_gru=True, reuse_graph_layer=False, dnn_hidden_units=(256, 128), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cuda:0', gpus=None, varlen_num=2):

        super(FiGNN, self).__init__([], dnn_feature_columns, l2_reg_linear=0,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)

        field_num = len(self.embedding_dict) + varlen_num

        embedding_size = self.embedding_size
        self.have_dense = False
        for item in dnn_feature_columns:
            if isinstance(item, DenseFeat):
                field_num += 1
                self.have_dense=True
        
        if self.have_dense:
          self.dense_linear = nn.Linear(1, embedding_size, bias=False)
          self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dense_linear.named_parameters()), l2=l2_reg_embedding)

        self.fignn = FiGNN_Layer(field_num, 
                                 embedding_size,
                                 gnn_layers=gnn_layers,
                                 reuse_graph_layer=reuse_graph_layer,
                                 use_gru=use_gru,
                                 use_residual=use_residual)
        self.fc = AttentionalPrediction(field_num, embedding_size)

        
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.fignn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.fc.named_parameters()), l2=l2_reg_dnn)      

        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        gnn_input = concat_fun(sparse_embedding_list, axis=1)
        if self.have_dense:
          dense_input = concat_fun(dense_value_list, axis=1)
          dense_input = self.dense_linear(dense_input)
          gnn_input = torch.cat([gnn_input, dense_input.unsqueeze(1)], axis = 1)
        
        h_out = self.fignn(gnn_input)
        main_logit = self.fc(h_out)
    
        y_pred = self.out(main_logit)

        return y_pred

class FiGNN_task(BaseModel):
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

    def __init__(self, dnn_feature_columns, gnn_layers=3,
                 use_residual=True, use_gru=True, reuse_graph_layer=False, dnn_hidden_units=(256, 128), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cuda:0', gpus=None, aux_num = 2, aux_units = (64, 32), detach = False, dynamic=False, aux_use_dnn = False, varlen_num=2):

        super(FiGNN_task, self).__init__([], dnn_feature_columns, l2_reg_linear=0,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus, aux_num = aux_num, dynamic=dynamic)

        field_num = len(self.embedding_dict) + varlen_num
        self.aux_use_dnn = aux_use_dnn

        embedding_size = self.embedding_size
        self.have_dense = False
        for item in dnn_feature_columns:
            if isinstance(item, DenseFeat):
                field_num += 1
                self.have_dense=True

        if self.have_dense:
          self.dense_linear = nn.Linear(1, embedding_size, bias=False)
          self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dense_linear.named_parameters()), l2=l2_reg_embedding)

        self.fignn = FiGNN_Layer(field_num, 
                                 embedding_size,
                                 gnn_layers=gnn_layers,
                                 reuse_graph_layer=reuse_graph_layer,
                                 use_gru=use_gru,
                                 use_residual=use_residual)
        self.fc = AttentionalPrediction(field_num, embedding_size)

        
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.fignn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.fc.named_parameters()), l2=l2_reg_dnn)  
        

        if aux_use_dnn:
            self.aux1_dnn = DNN(embedding_size * field_num, aux_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.aux1_dnn_linear = nn.Linear(
                aux_units[-1], 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.aux1_dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.aux1_dnn_linear.weight, l2=l2_reg_dnn)

            if self.aux_num == 2:
                self.aux2_dnn = DNN(embedding_size * field_num, aux_units,
                                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                init_std=init_std, device=device)
                self.aux2_dnn_linear = nn.Linear(
                    aux_units[-1], 1, bias=False).to(device)
                self.add_regularization_weight(
                    filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.aux2_dnn.named_parameters()), l2=l2_reg_dnn)
                self.add_regularization_weight(self.aux2_dnn_linear.weight, l2=l2_reg_dnn)

        else:
            self.fc_aux1 = AttentionalPrediction(field_num, embedding_size)
            self.add_regularization_weight(
                    filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.fc_aux1.named_parameters()), l2=l2_reg_dnn)  
            if self.aux_num == 2:
                self.fc_aux2 = AttentionalPrediction(field_num, embedding_size)
                
                self.add_regularization_weight(
                    filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.fc_aux2.named_parameters()), l2=l2_reg_dnn)    
        self.out_aux1 = PredictionLayer(task,)
        self.out_aux2 = PredictionLayer(task,)
        self.out_main = PredictionLayer(task,)

        self.final_linear = nn.Linear(self.aux_num + 1, 1, bias=False).to(device)
        self.add_regularization_weight(self.final_linear.weight, l2=l2_reg_dnn) 

        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        gnn_input = concat_fun(sparse_embedding_list, axis=1)
        if self.have_dense:
          dense_input = concat_fun(dense_value_list, axis=1)
          dense_input = self.dense_linear(dense_input)
          gnn_input = torch.cat([gnn_input, dense_input.unsqueeze(1)], axis = 1)
        
        h_out = self.fignn(gnn_input)
        main_logit = self.fc(h_out)

        if self.aux_use_dnn:
            dnn_input = h_out.flatten(start_dim=1)
            aux1_dnn_output = self.aux1_dnn(dnn_input)
            aux1_logit = self.aux1_dnn_linear(aux1_dnn_output)
            if self.aux_num==2:
                aux2_dnn_output = self.aux2_dnn(dnn_input)
                aux2_logit = self.aux2_dnn_linear(aux2_dnn_output)
        else:
            aux1_logit = self.fc_aux1(h_out)
            if self.aux_num == 2:
                aux2_logit = self.fc_aux2(h_out)
        
        
        y_main = self.out_main(main_logit)
        y_aux1 = self.out_aux1(aux1_logit)
        if self.aux_num == 2:
            y_aux2 = self.out_aux2(aux2_logit)
            final_logit = self.final_linear(torch.cat([main_logit, aux1_logit, aux2_logit], dim=-1))
        else:
            final_logit = self.final_linear(torch.cat([main_logit, aux1_logit], dim=-1))
        y_pred = self.out(final_logit)

        if self.aux_num == 2:
            return y_pred, y_aux1, y_aux2, y_main
        else:
            return y_pred, y_aux1, y_main

