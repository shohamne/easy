from utils import *
from args import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import rff

class BasicBlockRN12(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlockRN12, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope = 0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope = 0.1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if args.dropout > 0:
            out = F.dropout(out, p=args.dropout, training=self.training, inplace=True)
        return out
    
class ResNet12(nn.Module):
    def __init__(self, feature_maps, input_shape, num_classes, few_shot, rotations):
        super(ResNet12, self).__init__()
        orig_feature_dim = 10 * feature_maps        
        layers = [] 
        layers.append(BasicBlockRN12(input_shape[0], feature_maps))
        layers.append(BasicBlockRN12(feature_maps, int(2.5 * feature_maps)))
        layers.append(BasicBlockRN12(int(2.5 * feature_maps), 5 * feature_maps))
        layers.append(BasicBlockRN12(5 * feature_maps, orig_feature_dim))        
        self.layers = nn.Sequential(*layers)
        feature_dim = orig_feature_dim if args.feature_dim == -1 else args.feature_dim  
        if args.feature_dim != -1:
            if args.encoding == 'rff':
                self.encoding = rff.layers.GaussianEncoding(sigma=1, input_size=feature_dim, encoded_size=int(feature_dim/2))
            else:
                self.T = nn.parameter.Parameter(data=torch.randn([orig_feature_dim, feature_dim])/(orig_feature_dim)**0.5, requires_grad=False)
        self.linear = linear(feature_dim, num_classes)    
        self.rotations = rotations
        self.linear_rot = linear(feature_dim, 4)
        self.mp = nn.MaxPool2d((2,2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, index_mixup = None, lam = -1):
        if lam != -1:
            mixup_layer = random.randint(0, 3)
        else:
            mixup_layer = -1
        out = x
        if mixup_layer == 0:
            out = lam * out + (1 - lam) * out[index_mixup]
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if mixup_layer == i + 1:
                out = lam * out + (1 - lam) * out[index_mixup]
            out = self.mp(F.leaky_relu(out, negative_slope = 0.1))
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        if args.feature_dim != -1:
            if args.encoding == 'rff':
                features = F.normalize(self.encoding(features), dim=1)
            else:
                features = features @ self.T

        if args.distance:
            out = (features**2).sum(dim=1, keepdim=True) \
                - 2*features @ self.linear.weight.t() \
                + (self.linear.weight**2).sum(dim=1, keepdim=True).t()
            out = out**0.5
        else:
            out = self.linear(features)
        if self.rotations:
            out_rot = self.linear_rot(features)
            return (out, out_rot), features

        return out, features
