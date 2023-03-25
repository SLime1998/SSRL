import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
import copy
import math

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            # nn.Linear(hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            # nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def updata_beta(self, new):
        self.beta = new

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class ByolCLR(nn.Module):
    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.encoder = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            prev_dim = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(prev_dim, prev_dim),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),
                                            self.encoder.fc)

            self.target_encoder = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.target_encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(prev_dim, prev_dim),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),
                                            self.target_encoder.fc)

            self.online_predictor = MLP(128, 128, 128)
            self.use_momentum = True
            # self.base_beta = 0.99
            self.base_beta = 0.99
            self.target_ema_updater = EMA(self.base_beta)

            self.lamda = 0.8
            self.weight_lambda = 1

    def update_ema(self, curr, total):
        beta = 1 - (1 - self.base_beta) * (numpy.cos(math.pi * curr / total) + 1) / 2
        print("update beta to {} at epoch {}".format(beta, curr))
        self.target_ema_updater.updata_beta(beta)

    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.encoder
        self.encoder = None

    @torch.no_grad()
    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.encoder)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.base_beta + param_q.data * (1. - self.base_beta)

    def forward(self, im1, im2=None, clip=None):
        if not self.pretrain:
            return self.encoder(im1)

        online_proj_one = self.encoder(im1)
        online_proj_two = self.encoder(im2)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        pred_clip = self.online_predictor(self.encoder(clip))

        with torch.no_grad():
            # self.target_encoder = self._get_target_encoder()
            self._momentum_update_key_encoder()
            target_proj_one = self.target_encoder(im1)
            target_proj_two = self.target_encoder(im2)

        loss_one = loss_fn(online_pred_one, target_proj_two)
        loss_two = loss_fn(online_pred_two, target_proj_one)

        loss_clip = loss_fn(pred_clip, target_proj_one) + loss_fn(pred_clip, target_proj_two)

        loss = loss_one + loss_two + loss_clip * self.weight_lambda
#        loss = loss_one + loss_two;
        return loss.mean()
