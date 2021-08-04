import argparse
import logging
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import sys
sys.path.append("/home/krf/models/faster-rcnn.pytorch/lib")
from model.roi_layers import ROIPool, ROIAlign
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

from utils.domain_classify import DC_img, netD_img, netD1, netD2, netD3, netD_head, netD_inst, flatten
from utils.grl import grad_reverse, gradient_scalar
try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

class Discriminator(nn.Module):
    def __init__(self, in_ch, slope, width=512):
        super(Discriminator, self).__init__()
        # self.l1 = nn.Linear(10647*(n_classes + 5), 512)#TODO \
        self.flat = nn.Flatten()
        self.l1 = nn.Linear(4096*8, width)
        self.l5 = nn.Linear(width, 2)
        self.act = nn.LeakyReLU(slope)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        x = self.act(self.l1(x))
        x = self.l5(x)
        return x

class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.out = None

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # if not self.training:  # inference
            try:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y_ = y.clone()
                y_[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y_[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y_.view(bs, -1, self.no))
                
            except Exception as e:
                print(e)
                pass
        if not self.training:
            return (torch.cat(z, 1), x)
        if len(z) == self.nl:
            return x, z
        return  x, None

    # def forward(self, x):
    #     # x = x.copy()  # for profiling
    #     z = []  # inference output
    #     self.training |= self.export
    #     for i in range(self.nl):
    #         x[i] = self.m[i](x[i])  # conv
    #         bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    #         x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            
    #         if self.grid[i].shape[2:4] != x[i].shape[2:4]:
    #             self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
    #         if self.stride is not None:
    #             y = x[i].sigmoid()
    #             y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
    #             y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #             z.append(y.view(bs, -1, self.no))

    #     return x if self.stride is None else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, da_weight=0.1):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        self.da_weight = da_weight # for domain classify loss weight
        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        # self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.backbone, self.head, self.disc1, self.disc2, self.disc3,\
            self.disc_head1, self.disc_head2, self.disc_head3, \
            self.disc_inst1, self.disc_inst2, self.disc_inst3, self.save  = parse_model(deepcopy(self.yaml), ch=[ch])  # backbone+head=model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])
        self.roi_pools = [ROIPool((7, 7), 0.125),ROIPool((7, 7), 0.0625),ROIPool((7, 7), 0.03125)]
        # Build strides, anchors
        m = self.head[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[0]])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                pred, d_pred = self.forward_once(xi) # forward
                yi = pred[0]
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - 1 - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - 1 - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None, d_pred  # augmented inference, train
        else:
            res = self.forward_once(x, profile) 
            # print(res)
            return res  # single-scale inference, train

    def forward_backbone(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.backbone:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        # print(x.shape,len(y))
        # return None, x, y, dt
        return  x, y, dt 

    def forward_discs(self,x, y):
        # print(y)
        f = self.disc1.f
        # print(f)
        if f != -1:  # if not from previous layer
            x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  # from earlier layers
        # print(x.shape)
        out_d_1 = self.disc1(gradient_scalar(x, -1.0))  # run

        f = self.disc2.f
        # print(f)
        if f != -1:  # if not from previous layer
            x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  # from earlier layers
        # print(x.shape)
        out_d_2 = self.disc2(gradient_scalar(x, -1.0))  # run
        
        f = self.disc3.f
        # print(f)
        if f != -1:  # if not from previous layer
            x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  # from earlier layers
        # print(x)
        # print(x.shape) 
        out_d_3 = self.disc3(gradient_scalar(x, -1.0))  # run

        return out_d_1, out_d_2, out_d_3

    def forword_head(self, x, y, dt,  profile=False):
        def cat(tensors, dim=0):
            """
            Efficient version of torch.cat that avoids a copy if there is only a single element in a list
            """
            assert isinstance(tensors, (list, tuple))
            if len(tensors) == 1:
                return tensors[0]
            return torch.cat(tensors, dim)

        def _convert_to_roi_format(boxes):
            concat_boxes = cat([b for b in boxes], dim=0)[:,0:4]
            device, dtype = concat_boxes.device, concat_boxes.dtype
            ids = cat(
                [
                    torch.full((len(b), 1), i, dtype=dtype, device=device)
                    for i, b in enumerate(boxes)
                ],
                dim=0,
            )
            rois = torch.cat([ids, concat_boxes], dim=1)
            return rois

        for m in self.head:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        
        pred = x[0]
        proposal = x[1]
        # out_d_inst1, out_d_inst2, out_d_inst3 = None, None, None
        if proposal is not None:
            pooled_list = []
            ftr_list = [4,6,9]
            for i in range(len(proposal)):
                # print("proposal.shape", proposal[i].shape)
                boxes = non_max_suppression(proposal[i], conf_thres=0.25, iou_thres=0.45)
                
                rois = _convert_to_roi_format(boxes)
                # print("rois:", rois.shape)
                # print("features:", y[ftr_list[i]].shape)
                # print(rois.device, y[ftr_list[i]].device)
                pooled_list.append(self.roi_pools[i](y[ftr_list[i]].float(), rois.float()))
                # print("roi pooled:",pooled_list[i].shape)
            if pooled_list[0].shape[0] == 0:
                out_d_inst1 = None
            else:
                out_d_inst1 = self.disc_inst1(gradient_scalar(flatten(pooled_list[0]), -1.0))

            if pooled_list[1].shape[0] == 0:
                out_d_inst2 = None
            else:
                out_d_inst2 = self.disc_inst2(gradient_scalar(flatten(pooled_list[1]), -1.0))

            if pooled_list[2].shape[0] == 0:
                out_d_inst3= None
            else:
                out_d_inst3 = self.disc_inst3(gradient_scalar(flatten(pooled_list[2]), -1.0))

    
        out_d_head1 = self.disc_head1(gradient_scalar(y[self.disc_head1.f], -1.0))  # run
        out_d_head2 = self.disc_head2(gradient_scalar(y[self.disc_head2.f], -1.0))  # run
        out_d_head3 = self.disc_head3(gradient_scalar(y[self.disc_head3.f], -1.0))  # run
        if proposal is not None:
            return pred, out_d_head1, out_d_head2, out_d_head3, out_d_inst1, out_d_inst2, out_d_inst3
        else:
            return pred, out_d_head1, out_d_head2, out_d_head3, None, None, None

    def forward_once(self, x, profile=False):
        x, y, dt = self.forward_backbone(x, profile) 
        out_d_1, out_d_2, out_d_3 = self.forward_discs(x, y)
        preds, out_d_head1, out_d_head2, out_d_head3, out_d_inst1, out_d_inst2, out_d_inst3 = self.forword_head(x, y, dt, profile)
        return preds, out_d_head1, out_d_head2, out_d_head3, out_d_inst1, out_d_inst2, out_d_inst3, out_d_1, out_d_2, out_d_3

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.head[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.head[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.backbone.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        for m in self.head.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.head[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.head[-1].i + 1  # index
            self.head.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.head = self.head[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    backbone, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (m_.i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        backbone.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    # disc = None
    
    disc1 = netD1(ch_in = ch[4])
    disc1.f = 4
    disc2 = netD2(ch_in = ch[6])
    disc2.f = 6
    disc3 = netD3(ch_in = ch[9])
    disc3.f = 9
    save.extend([4,6,9])
    # disc = DC_img(ch[-1], grl_weight=1.0) # TODO more config
    # disc = Discriminator(in_ch = ch[-1],slope=0.1)
    head = []   
    for i, (f, n, m, args) in enumerate(d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i + len(d['backbone']), f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (m_.i, f, n, np, t, args))  # print
        save.extend(x % m_.i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        head.append(m_)
        if m_.i == 0:
            ch = []
        ch.append(c2)
    disc_head1 = netD_head(ch_in = ch[17])
    disc_head1.f = 17
    disc_head2 = netD_head(ch_in = ch[20])
    disc_head2.f = 20
    disc_head3 = netD_head(ch_in = ch[23])
    disc_head3.f = 23
    save.extend([17,20,23])

    disc_inst1 = netD_inst(ch_in = 192*7*7)
    disc_inst1.f = 4
    disc_inst2 = netD_inst(ch_in = 384*7*7)
    disc_inst2.f = 6
    disc_inst3 = netD_inst(ch_in = 768*7*7)
    disc_inst3.f = 9
    return nn.Sequential(*backbone), nn.Sequential(*head), disc1, disc2, disc3,\
         disc_head1, disc_head2, disc_head3, disc_inst1, disc_inst2, disc_inst3, sorted(save)

from utils.torch_utils import intersect_dicts
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5m.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    # print(model)
    model.train()

    # for k,v in [*model.backbone.named_parameters(),*model.head.named_parameters()]:
    ckpt = torch.load('yolov5m.pt', map_location=device)  # load checkpoint
    # model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
    # exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else []  # exclude keys
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    # print(ckpt)
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])  # intersect
    model.load_state_dict(state_dict, strict=False)  # load
    logger.info('Transferred %g/%g items from ' % (len(state_dict), len(model.state_dict())))

    # print(ckpt['optimizer'])
    
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k,v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
            print(k)
    # for k, v in model.named_parameters():
    #     print(k)
    # for k, v in ckpt['model'].named_parameters():
    #     print(k)
    # print(pg0, pg1, )
    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
