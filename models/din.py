# -*- coding:utf-8 -*-
"""
Author:
    Yuef Zhang
Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""

from .basemodel import BaseModel
from .basemodel_nofinal import BaseModel_nofinal
from deepctr_torch.inputs import *
from deepctr_torch.layers import *
from deepctr_torch.layers.sequence import AttentionSequencePoolingLayer
from .layers import DNN, WeightLayer

class DIN(BaseModel):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return:  A PyTorch model instance.

    """

    def __init__(self, dnn_feature_columns, history_feature_list, dnn_use_bn=False,
                 dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=False, l2_reg_dnn=0.0,
                 l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001,
                 seed=1024, task='binary', device='cuda:0', gpus=None):
        super(DIN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.history_feature_list = history_feature_list

        self.history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))

        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        att_emb_dim = self._compute_interest_dim()

        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=att_emb_dim,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)

        self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.to(device)


    def forward(self, X):
        _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        # sequence pooling part
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          return_feat_list=self.history_feature_list, to_list=True)

        keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
                                         return_feat_list=self.history_fc_names, to_list=True)
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                              to_list=True)

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.sparse_varlen_feature_columns)

        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                      self.sparse_varlen_feature_columns, self.device)

        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)                     # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)                       # [B, T, E]

        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)  # [B, 1]

        hist = self.attention(query_emb, keys_emb, keys_length)           # [B, 1, E]

        # deep part
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)

        return y_pred

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim

class DIN_task(BaseModel):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return:  A PyTorch model instance.

    """

    def __init__(self, dnn_feature_columns, history_feature_list, dnn_use_bn=False,
                 dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=False, l2_reg_dnn=0.0,
                 l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001,
                 seed=1024, task='binary', device='cuda:0', gpus=None, aux_num = 2, aux_units = (64, 32), detach = False, dynamic=False):
        super(DIN_task, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device, gpus=gpus, aux_num = aux_num, dynamic=dynamic)

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.history_feature_list = history_feature_list
        self.detach = detach

        self.history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))

        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        att_emb_dim = self._compute_interest_dim()

        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=att_emb_dim,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)

        self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.aux1_dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=aux_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.aux1_dnn_linear = nn.Linear(aux_units[-1], 1, bias=False).to(device)
        if self.aux_num == 2:
            self.aux2_dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=aux_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
            self.aux2_dnn_linear = nn.Linear(aux_units[-1], 1, bias=False).to(device)
            self.final_linear = nn.Linear(3, 1, bias=False).to(device)
            # self.final_linear = DNN(inputs_dim=3,
            #            hidden_units=[4, 4, 1],
            #            activation=dnn_activation,
            #            dropout_rate=dnn_dropout,
            #            l2_reg=l2_reg_dnn,
            #            use_bn=dnn_use_bn,
            #            bias=False)
            self.out_aux2 = PredictionLayer(task,)
        else:
            self.final_linear = nn.Linear(2, 1, bias=False).to(device)
        self.out_main = PredictionLayer(task,)
        self.out_aux1 = PredictionLayer(task,)
        if dynamic:
            # self.weight_net1 = DNN(inputs_dim = 1, hidden_units = (4,4,1), 
            # activation = dnn_activation, output_activation = 'sigmoid',
            # dropout_rate=dnn_dropout,l2_reg=l2_reg_dnn,use_bn=dnn_use_bn)
            # if self.aux_num == 2:
                # self.weight_net2 = DNN(inputs_dim = 1, hidden_units = (4,4,1), 
                # activation = dnn_activation, output_activation = 'sigmoid',
                # dropout_rate=dnn_dropout,l2_reg=l2_reg_dnn,use_bn=dnn_use_bn)
            self.weight_net = WeightLayer()
        self.shared_list = []
        for name, parm in self.named_parameters():
            if 'embedding_dict' in name or 'attention' in name:
                self.shared_list.append(name)
        self.to(device)


    def forward(self, X):
        _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        # sequence pooling part
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          return_feat_list=self.history_feature_list, to_list=True)
        keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
                                         return_feat_list=self.history_fc_names, to_list=True)
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                              to_list=True)

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.sparse_varlen_feature_columns)

        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                      self.sparse_varlen_feature_columns, self.device)

        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)                     # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)                       # [B, T, E]

        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)  # [B, 1]

        hist = self.attention(query_emb, keys_emb, keys_length)           # [B, 1, E]

        # deep part
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        if self.detach:
            aux1_dnn_output = self.aux1_dnn(dnn_input.detach())
            if self.aux_num == 2:
                aux2_dnn_output = self.aux2_dnn(dnn_input.detach())
        else:
            aux1_dnn_output = self.aux1_dnn(dnn_input)
            if self.aux_num == 2:
                aux2_dnn_output = self.aux2_dnn(dnn_input)

        main_dnn_logit = self.dnn_linear(dnn_output)
        aux1_dnn_logit = self.aux1_dnn_linear(aux1_dnn_output)
        if self.aux_num == 2:
            aux2_dnn_logit = self.aux2_dnn_linear(aux2_dnn_output)

        y_main = self.out_main(main_dnn_logit)
        y_aux1 = self.out_aux1(aux1_dnn_logit)
        if self.aux_num == 2:
            y_aux2 = self.out_aux2(aux2_dnn_logit)

        if self.aux_num == 2:
            y_final_logit = self.final_linear(torch.cat([main_dnn_logit, aux1_dnn_logit, aux2_dnn_logit], dim=-1))
            # if not self.training:
            #     print(main_dnn_logit,aux1_dnn_logit, aux2_dnn_logit)
        else:
            y_final_logit = self.final_linear(torch.cat([main_dnn_logit, aux1_dnn_logit], dim=-1))
        y_pred = self.out(y_final_logit)

        aux1_weight = None
        aux2_weight = None
        if self.dynamic:
            aux1_weight = self.weight_net(y_main.detach())
            if self.aux_num == 2:
                aux2_weight = self.weight_net(y_main.detach())

            # print('aux1: ',aux1_weight)
            # print('aux2: ',aux2_weight)
            # print('main: ',y_main)
            # print()
        if self.dynamic and self.training:
            if self.aux_num == 2:
                return y_pred, y_aux1, y_aux2, y_main, aux1_weight, aux2_weight
            else:
                return y_pred, y_aux1, y_main, aux1_weight, aux2_weight
        else:
            if self.aux_num == 2:
                return y_pred, y_aux1, y_aux2, y_main
            else:
                return y_pred, y_aux1, y_main


    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim

class DIN_task_distill(BaseModel):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return:  A PyTorch model instance.

    """

    def __init__(self, dnn_feature_columns, history_feature_list, dnn_use_bn=False,
                 dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=False, l2_reg_dnn=0.0,
                 l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001,
                 seed=1024, task='binary', device='cuda:0', gpus=None, aux_num = 2, aux_units = (64, 32), use_gate=False, gate_feature = None, l2_reg_embedding_gate = 0.001, gate_units = [64, 32]):
        super(DIN_task_distill, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device, gpus=gpus, aux_num = aux_num)

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.history_feature_list = history_feature_list

        self.history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
        self.use_gate = use_gate

        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        att_emb_dim = self._compute_interest_dim()

        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=att_emb_dim,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)

        self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.aux1_dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=aux_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.aux1_dnn_teacher = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=aux_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.aux1_dnn_linear = nn.Linear(aux_units[-1], 1, bias=False).to(device)
        self.aux1_dnn_linear_teacher = nn.Linear(aux_units[-1], 1, bias=False).to(device)
        if self.aux_num == 2:
            self.aux2_dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=aux_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
            self.aux2_dnn_linear = nn.Linear(aux_units[-1], 1, bias=False).to(device)
            self.aux2_dnn_teacher = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=aux_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
            self.aux2_dnn_linear_teacher = nn.Linear(aux_units[-1], 1, bias=False).to(device)
            self.final_linear = nn.Linear(3, 1, bias=False).to(device)
            self.out_aux2 = PredictionLayer(task,)
            self.out_aux2_teacher = PredictionLayer(task,)
        else:
            self.final_linear = nn.Linear(2, 1, bias=False).to(device)
        self.out_main = PredictionLayer(task,)
        self.out_aux1 = PredictionLayer(task,)
        self.out_aux1_teacher = PredictionLayer(task,)

        self.weight_net = WeightLayer()
        self.shared_list = []
        for name, parm in self.named_parameters():
            if 'embedding_dict' in name or 'attention' in name:
                self.shared_list.append(name)
        if use_gate:
            self.gate_embedding_dict = create_embedding_matrix(gate_feature, init_std, sparse=False, device=device)

            self.gate_net = DNN(inputs_dim = self.compute_input_dim(gate_feature),
                                    hidden_units=gate_units + [aux_num + 1],
                                    activation=dnn_activation,
                                    dropout_rate=dnn_dropout,
                                    l2_reg=l2_reg_dnn,
                                    use_bn=dnn_use_bn,
                                    output_activation='sigmoid')
            self.gate_feature = gate_feature
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.gate_net.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.gate_embedding_dict.parameters(), l2=l2_reg_embedding_gate)

        self.to(device)


    def forward(self, X):
        _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        # sequence pooling part
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          return_feat_list=self.history_feature_list, to_list=True)
        keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
                                         return_feat_list=self.history_fc_names, to_list=True)
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                              to_list=True)

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.sparse_varlen_feature_columns)

        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                      self.sparse_varlen_feature_columns, self.device)

        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)                     # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)                       # [B, T, E]

        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)  # [B, 1]

        hist = self.attention(query_emb, keys_emb, keys_length)           # [B, 1, E]

        # deep part
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        dnn_output = self.dnn(dnn_input)

        aux1_dnn_output = self.aux1_dnn(dnn_input)
        aux1_dnn_output_t = self.aux1_dnn_teacher(dnn_input.detach())
        if self.aux_num == 2:
            aux2_dnn_output = self.aux2_dnn(dnn_input)
            aux2_dnn_output_t = self.aux2_dnn_teacher(dnn_input.detach())

        main_dnn_logit = self.dnn_linear(dnn_output)
        aux1_dnn_logit = self.aux1_dnn_linear(aux1_dnn_output)
        aux1_dnn_logit_t = self.aux1_dnn_linear_teacher(aux1_dnn_output_t)
        if self.aux_num == 2:
            aux2_dnn_logit = self.aux2_dnn_linear(aux2_dnn_output)
            aux2_dnn_logit_t = self.aux2_dnn_linear_teacher(aux2_dnn_output_t)

        y_main = self.out_main(main_dnn_logit)
        y_aux1 = self.out_aux1(aux1_dnn_logit)
        y_aux1_t = self.out_aux1_teacher(aux1_dnn_logit_t)
        if self.aux_num == 2:
            y_aux2 = self.out_aux2(aux2_dnn_logit)
            y_aux2_t = self.out_aux2_teacher(aux2_dnn_logit_t)

        if self.use_gate:
            gate_sparse_list, gate_dense_list = self.input_from_feature_columns(X, self.gate_feature, self.gate_embedding_dict)
            gate_input = combined_dnn_input(gate_sparse_list, gate_dense_list)
            gate_output = self.gate_net(gate_input) * 2.0
        else:
            gate_output = 1.0


        if self.aux_num == 2:
            final_logit = torch.cat([main_dnn_logit, aux1_dnn_logit, aux2_dnn_logit], dim=-1) * gate_output
            y_final_logit = self.final_linear(final_logit)
            
        else:
            final_logit = torch.cat([main_dnn_logit, aux1_dnn_logit], dim=-1) * gate_output
            y_final_logit = self.final_linear(final_logit)
        y_pred = self.out(y_final_logit)

        aux1_weight = None
        aux2_weight = None

        aux1_weight = self.weight_net(y_main.detach())
        if self.aux_num == 2:
            aux2_weight = self.weight_net(y_main.detach())
        

        if self.training:
            if self.aux_num == 2:
                return y_pred, y_aux1, y_aux2, y_main, aux1_weight, aux2_weight, y_aux1_t, y_aux2_t
            else:
                return y_pred, y_aux1, y_main, aux1_weight, aux2_weight, y_aux1_t
        else:
            if self.aux_num == 2:
                return y_pred, y_aux1, y_aux2, y_main
            else:
                return y_pred, y_aux1, y_main


    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim

class DIN_task_gate(BaseModel):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return:  A PyTorch model instance.

    """

    def __init__(self, dnn_feature_columns, history_feature_list, dnn_use_bn=False,
                 dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=False, l2_reg_dnn=0.0,
                 l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001,
                 seed=1024, task='binary', device='cuda:0', gpus=None, aux_num = 2, aux_units = (64, 32), detach = False, dynamic=False):
        super(DIN_task_gate, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device, gpus=gpus, aux_num = aux_num, dynamic=dynamic)

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.history_feature_list = history_feature_list
        self.detach = detach

        self.history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))

        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        att_emb_dim = self._compute_interest_dim()

        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=att_emb_dim,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)

        self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.aux1_dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=aux_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.aux1_dnn_linear = nn.Linear(aux_units[-1], 1, bias=False).to(device)
        if self.aux_num == 2:
            self.aux2_dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=aux_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
            self.aux2_dnn_linear = nn.Linear(aux_units[-1], 1, bias=False).to(device)
            # self.final_linear = nn.Linear(3, 1, bias=False).to(device)
            # self.final_linear = DNN(inputs_dim=3,
            #            hidden_units=[16, 16, 1],
            #            activation=dnn_activation,
            #            dropout_rate=dnn_dropout,
            #            l2_reg=l2_reg_dnn,
            #            use_bn=dnn_use_bn)
            self.out_aux2 = PredictionLayer(task,)
        else:
            self.final_linear = nn.Linear(2, 1, bias=False).to(device)
        
        self.gate_net = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=[64, 32, 3],
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn,
                       output_activation='sigmoid')
        self.out_main = PredictionLayer(task,)
        self.out_aux1 = PredictionLayer(task,)
        if dynamic:
            self.weight_net = WeightLayer()
        self.shared_list = []
        for name, parm in self.named_parameters():
            if 'embedding_dict' in name or 'attention' in name:
                self.shared_list.append(name)
        self.to(device)


    def forward(self, X):
        sparse_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        gate_input = combined_dnn_input(sparse_list, dense_value_list)
        gate_output = self.gate_net(gate_input) * 2.0

        # sequence pooling part
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          return_feat_list=self.history_feature_list, to_list=True)
        keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
                                         return_feat_list=self.history_fc_names, to_list=True)
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                              to_list=True)

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.sparse_varlen_feature_columns)

        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                      self.sparse_varlen_feature_columns, self.device)

        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)                     # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)                       # [B, T, E]

        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)  # [B, 1]

        hist = self.attention(query_emb, keys_emb, keys_length)           # [B, 1, E]

        # deep part
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        if self.detach:
            aux1_dnn_output = self.aux1_dnn(dnn_input.detach())
            if self.aux_num == 2:
                aux2_dnn_output = self.aux2_dnn(dnn_input.detach())
        else:
            aux1_dnn_output = self.aux1_dnn(dnn_input)
            if self.aux_num == 2:
                aux2_dnn_output = self.aux2_dnn(dnn_input)

        main_dnn_logit = self.dnn_linear(dnn_output)
        aux1_dnn_logit = self.aux1_dnn_linear(aux1_dnn_output)
        if self.aux_num == 2:
            aux2_dnn_logit = self.aux2_dnn_linear(aux2_dnn_output)

        y_main = self.out_main(main_dnn_logit)
        y_aux1 = self.out_aux1(aux1_dnn_logit)
        if self.aux_num == 2:
            y_aux2 = self.out_aux2(aux2_dnn_logit)

        if self.aux_num == 2:
            y_final_logit = torch.sum(torch.cat([main_dnn_logit, aux1_dnn_logit, aux2_dnn_logit], dim=-1) * gate_output, dim=-1)
            # if not self.training:
            #     print(main_dnn_logit,aux1_dnn_logit, aux2_dnn_logit)
        else:
            y_final_logit = self.final_linear(torch.cat([main_dnn_logit, aux1_dnn_logit], dim=-1))
        y_pred = self.out(y_final_logit)

        aux1_weight = None
        aux2_weight = None
        if self.dynamic:
            aux1_weight = self.weight_net(y_main.detach())
            if self.aux_num == 2:
                aux2_weight = self.weight_net(y_main.detach())

            # print('aux1: ',aux1_weight)
            # print('aux2: ',aux2_weight)
            # print('main: ',y_main)
            # print()
        if self.dynamic and self.training:
            if self.aux_num == 2:
                return y_pred, y_aux1, y_aux2, y_main, aux1_weight, aux2_weight
            else:
                return y_pred, y_aux1, y_main, aux1_weight, aux2_weight
        else:
            if self.aux_num == 2:
                return y_pred, y_aux1, y_aux2, y_main
            else:
                return y_pred, y_aux1, y_main


    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
                interest_dim += feat.embedding_dim
        return interest_dim


if __name__ == '__main__':
    pass
