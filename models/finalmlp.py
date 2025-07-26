import torch
import torch.nn as nn

from .basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import concat_fun, InteractingLayer, FM
from .layers import DNN
from deepctr_torch.layers import PredictionLayer
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat

class FeatureSelection(nn.Module):
    def __init__(self, feature_dim, embedding_dim, fs_hidden_units=[], l2_reg_dnn=1e-7, init_std=0.0001, device='cuda:0'):
        super(FeatureSelection, self).__init__()
        self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, embedding_dim))

        self.fs1_gate = DNN(embedding_dim, fs_hidden_units + [feature_dim],activation='relu', l2_reg=l2_reg_dnn, use_bn=False,
                        init_std=init_std, device=device, output_activation='sigmoid')
        
        self.fs2_gate = DNN(embedding_dim, fs_hidden_units + [feature_dim],activation='relu', l2_reg=l2_reg_dnn, use_bn=False,
                        init_std=init_std, device=device, output_activation='sigmoid')

    def forward(self, flat_emb):
        fs1_input = self.fs1_ctx_bias.repeat(flat_emb.size(0), 1)
        gt1 = self.fs1_gate(fs1_input) * 2
        feature1 = flat_emb * gt1


        fs2_input = self.fs2_ctx_bias.repeat(flat_emb.size(0), 1)
        gt2 = self.fs2_gate(fs2_input) * 2
        feature2 = flat_emb * gt2

        return feature1, feature2

class InteractionAggregation(nn.Module):
    def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
        super(InteractionAggregation, self).__init__()
        assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
            "Input dim must be divisible by num_heads!"
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_x_dim = x_dim // num_heads
        self.head_y_dim = y_dim // num_heads
        self.w_x = nn.Linear(x_dim, output_dim)
        self.w_y = nn.Linear(y_dim, output_dim)
        self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, 
                                              output_dim))
        nn.init.xavier_normal_(self.w_xy)

    def forward(self, x, y):
        output = self.w_x(x) + self.w_y(y)
        head_x = x.view(-1, self.num_heads, self.head_x_dim)
        head_y = y.view(-1, self.num_heads, self.head_y_dim)
        xy = torch.matmul(torch.matmul(head_x.unsqueeze(2), 
                                       self.w_xy.view(self.num_heads, self.head_x_dim, -1)) \
                               .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
                          head_y.unsqueeze(-1)).squeeze(-1)
        output += xy.sum(dim=1)
        return output

class FinalMLP(BaseModel):
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

    def __init__(self, dnn_feature_columns, mlp1_hidden_units=(256, 128), mlp1_activation='relu', mlp1_use_bn=False, mlp1_dropout=0, mlp2_hidden_units=(256, 128), mlp2_activation='relu', mlp2_use_bn=False, mlp2_dropout=0, l2_reg_dnn=0, l2_reg_embedding=1e-5, init_std=0.0001, seed=1024, num_heads=1, use_fs=True, fs_hidden_units=[64],
                 task='binary', device='cuda:0', gpus=None, varlen_num=2):

        super(FinalMLP, self).__init__([], dnn_feature_columns, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)

        field_num = len(self.embedding_dict)

        embedding_size = self.embedding_size

        feature_dim = embedding_size * (field_num + varlen_num)
        for item in dnn_feature_columns:
            if isinstance(item, DenseFeat):
                feature_dim += 1

        self.mlp1 = DNN(self.compute_input_dim(dnn_feature_columns), mlp1_hidden_units,
                        activation=mlp1_activation, l2_reg=l2_reg_dnn, dropout_rate=mlp1_dropout, use_bn=mlp1_use_bn,
                        init_std=init_std, device=device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.mlp1.named_parameters()), l2=l2_reg_dnn)

        self.mlp2 = DNN(self.compute_input_dim(dnn_feature_columns), mlp2_hidden_units,
                        activation=mlp2_activation, l2_reg=l2_reg_dnn, dropout_rate=mlp2_dropout, use_bn=mlp2_use_bn,
                        init_std=init_std, device=device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.mlp2.named_parameters()), l2=l2_reg_dnn)
        
        self.use_fs = use_fs
        if self.use_fs:
            self.fs_module = FeatureSelection(feature_dim, embedding_size, fs_hidden_units, l2_reg_dnn=l2_reg_dnn, init_std=init_std, device=device)
            self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.fs_module.named_parameters()), l2=l2_reg_dnn)

        self.fusion_module = InteractionAggregation(mlp1_hidden_units[-1], 
                                                    mlp2_hidden_units[-1], 
                                                    output_dim=1, 
                                                    num_heads=num_heads)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.fusion_module.named_parameters()), l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        if self.use_fs:
            feat1, feat2 = self.fs_module(dnn_input)
        else:
            feat1, feat2 = dnn_input, dnn_input
        
        main_logit = self.fusion_module(self.mlp1(feat1), self.mlp2(feat2))
        
        y_pred = self.out(main_logit)

        return y_pred

class FinalMLP_task(BaseModel):
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

    def __init__(self, dnn_feature_columns, mlp1_hidden_units=(256, 128), mlp1_activation='relu', mlp1_use_bn=False, mlp1_dropout=0, mlp2_hidden_units=(256, 128), mlp2_activation='relu', mlp2_use_bn=False, mlp2_dropout=0, l2_reg_dnn=0, l2_reg_embedding=1e-5, init_std=0.0001, seed=1024, num_heads=1, use_fs=True, fs_hidden_units=[64],
                 task='binary', device='cuda:0', gpus=None, aux_num = 2, aux_units = (64, 32), detach = False, dynamic=False, varlen_num=2):

        super(FinalMLP_task, self).__init__([], dnn_feature_columns, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus, aux_num = aux_num, dynamic=dynamic)

        field_num = len(self.embedding_dict)

        embedding_size = self.embedding_size

        feature_dim = embedding_size * (field_num + varlen_num)
        for item in dnn_feature_columns:
            if isinstance(item, DenseFeat):
                feature_dim += 1

        self.mlp1 = DNN(self.compute_input_dim(dnn_feature_columns), mlp1_hidden_units,
                        activation=mlp1_activation, l2_reg=l2_reg_dnn, dropout_rate=mlp1_dropout, use_bn=mlp1_use_bn,
                        init_std=init_std, device=device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.mlp1.named_parameters()), l2=l2_reg_dnn)

        self.mlp2 = DNN(self.compute_input_dim(dnn_feature_columns), mlp2_hidden_units,
                        activation=mlp2_activation, l2_reg=l2_reg_dnn, dropout_rate=mlp2_dropout, use_bn=mlp2_use_bn,
                        init_std=init_std, device=device)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.mlp2.named_parameters()), l2=l2_reg_dnn)
        
        self.use_fs = use_fs
        if self.use_fs:
            self.fs_module = FeatureSelection(feature_dim, embedding_size, fs_hidden_units, l2_reg_dnn=l2_reg_dnn, init_std=init_std, device=device)
            self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.fs_module.named_parameters()), l2=l2_reg_dnn)

        self.fusion_module = InteractionAggregation(mlp1_hidden_units[-1], 
                                                    mlp2_hidden_units[-1], 
                                                    output_dim=1, 
                                                    num_heads=num_heads)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.fusion_module.named_parameters()), l2=l2_reg_dnn)
        
        self.fusion_module_aux1 = InteractionAggregation(mlp1_hidden_units[-1], 
                                                    mlp2_hidden_units[-1], 
                                                    output_dim=1, 
                                                    num_heads=num_heads)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.fusion_module_aux1.named_parameters()), l2=l2_reg_dnn)
        
        if self.aux_num == 2:
            self.fusion_module_aux2 = InteractionAggregation(mlp1_hidden_units[-1], 
                                                        mlp2_hidden_units[-1], 
                                                        output_dim=1, 
                                                        num_heads=num_heads)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.fusion_module_aux2.named_parameters()), l2=l2_reg_dnn)
        
        self.out_aux1 = PredictionLayer(task,)
        if self.aux_num == 2:
            self.out_aux2 = PredictionLayer(task,)
        self.out_main = PredictionLayer(task,)
        self.final_linear = nn.Linear(self.aux_num + 1, 1, bias=False).to(device)
        self.add_regularization_weight(self.final_linear.weight, l2=l2_reg_dnn) 
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        if self.use_fs:
            feat1, feat2 = self.fs_module(dnn_input)
        else:
            feat1, feat2 = dnn_input, dnn_input
        
        feat_1 = self.mlp1(feat1)
        feat_2 = self.mlp2(feat2)
        main_logit = self.fusion_module(feat_1, feat_2)
        aux1_logit = self.fusion_module_aux1(feat_1, feat_2)

        if self.aux_num == 2:
            aux2_logit = self.fusion_module_aux2(feat_1, feat_2)


        y_main = self.out(main_logit)
        y_aux1 = self.out_aux1(aux1_logit)
        if self.aux_num == 2:
            y_aux2 = self.out_aux2(aux2_logit)
        
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
