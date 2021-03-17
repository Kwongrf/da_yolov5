import torch
import torch.nn as nn
import torch.nn.functional as F
from .grl import GradientScalarLayer

def flatten(x):
    N = list(x.size())[0]
    #print('dim 0', N, 1024*19*37)
    return x.view(N, -1)
def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)

class netD_img(nn.Module):
    def __init__(self, beta=1, ch_in=512, ch_out=512, W=38, H=75, stride_1=1, padding_1=1, kernel=3):
        super(netD_img, self).__init__()
        self.conv_image = nn.Conv2d(ch_in, ch_out, stride=stride_1, padding=padding_1, kernel_size=kernel)
        self.bn_image = nn.BatchNorm2d(ch_out)
        self.fc_1_image = nn.Linear(8192, 2)
        self.ch_out = ch_out
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.bn_2 = nn.BatchNorm2d(512)
        #self.softmax = nn.Softmax()
        #self.logsoftmax = nn.LogSoftmax()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.conv_image(x)
        x = self.relu(x)
        x = self.bn_image(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.bn_2(x)
        # convert to 1024*W*H x 1.
        x = flatten(x)
        # x = torch.transpose(x, 0, 1)
        # print(x.shape)
        x = self.fc_1_image(x)
        # 1 x n vector
        #y = self.softmax(x)
        #x = self.logsoftmax(x)
        #return x, y
        return x

class netD1(nn.Module):
    def __init__(self, ch_in = 256, context=False):
        super(netD1, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, 256, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
          feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
          x = self.conv3(x)
          return F.sigmoid(x),feat
        else:
          x = self.conv3(x)
          return F.sigmoid(x)

class netD2(nn.Module):
    def __init__(self,ch_in = 512, context=False):
        super(netD2, self).__init__()
        self.conv1 = conv3x3(ch_in, 256, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv3x3(256, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat
        else:
          return x


class netD3(nn.Module):
    def __init__(self, ch_in = 1024, context=False):
        super(netD3, self).__init__()
        self.conv1 = conv3x3(ch_in, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        # self.conv3 = conv3x3(128, 128, stride=2)
        # self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        # x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat
        else:
          return x

class DC_img(nn.Module):
    """img level domain classifier"""

    def __init__(self,
                 in_channels=512,
                 feat_channels=512,
                 grl_weight=-0.1,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,# TODO 
                     loss_weight=1,
                     reduction='mean')):
        super(DC_img, self).__init__()
        self.da_conv = nn.Conv2d(in_channels, feat_channels, kernel_size=1, stride=1)
        self.da_cls = nn.Conv2d(feat_channels, 1, kernel_size=1, stride=1)
        self.loss_cls = nn.BCEWithLogitsLoss(reduction=loss_cls['reduction'])#build_loss(loss_cls)
        self.grl_img = GradientScalarLayer(grl_weight)
        self.loss_weight = loss_cls['loss_weight']
        self.init_weights()

    def init_weights(self):
        for l in [self.da_conv, self.da_cls]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        features = self.grl_img(x)
        features = F.relu(self.da_conv(features))
        dc_scores = self.da_cls(features)
        return dc_scores

    def loss(self,
             cls_score,
             source):
        # losses = dict()
        N, C, H, W = cls_score.shape
        domain_label = 0 if source else 1
        labels = torch.zeros_like(cls_score, dtype=torch.float32)
        labels[:, :] = domain_label
        cls_score = cls_score.view(N, -1)
        labels = labels.view(N, -1)
        losses = self.loss_weight * self.loss_cls(
                    cls_score,
                    labels)
        return losses