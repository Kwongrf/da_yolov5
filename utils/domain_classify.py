import torch
import torch.nn as nn
import torch.nn.functional as F
from .grl import GradientScalarLayer

class DC_img(nn.Module):
    """img level domain classifier"""

    def __init__(self,
                 in_channels=512,
                 feat_channels=512,
                 grl_weight=-0.1,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,# TODO 
                     loss_weight=0.1),):
        super(DC_img, self).__init__()
        self.da_conv = nn.Conv2d(in_channels, feat_channels, kernel_size=1, stride=1)
        self.da_cls = nn.Conv2d(feat_channels, 1, kernel_size=1, stride=1)
        self.loss_cls = nn.CrossEntropyLoss()#build_loss(loss_cls)
        self.grl_img = GradientScalarLayer(grl_weight)
        self.sigmoid = nn.Sigmoid()
        self.loss_weight = loss_cls['loss_weight']
        self.init_weights()

    def init_weights(self):
        for l in [self.da_conv, self.da_cls]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        features = self.grl_img(x)
        features = F.relu(self.da_conv(features))
        dc_scores = self.sigmoid(self.da_cls(features))
        return dc_scores

    def loss(self,
             cls_score,
             source,
             reduction_override='sum'):
        # losses = dict()
        N, C, H, W = cls_score.shape
        domain_label = 0 if source else 1
        labels = torch.zeros_like(cls_score, dtype=torch.float32)
        labels[:, :] = domain_label
        cls_score = cls_score.view(N, -1)
        labels = labels.view(N, -1)
        losses = self.loss_weight * self.loss_cls(
                    cls_score,
                    labels,
                    reduction_override=reduction_override)
        return losses