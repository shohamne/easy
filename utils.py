from args import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st
import numpy as np
import random
from loss import NCEandRCE, StarLoss


### generate random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# function to display timer
def format_time(duration):
    duration = int(duration)
    s = duration % 60
    m = (duration // 60) % 60
    h = (duration // 3600)
    return "{:d}h{:02d}m{:02d}s".format(h,m,s)

def stats(scores, name):
    if len(scores) == 1:
        low, up = 0., 1.
    elif len(scores) < 30:
        low, up = st.t.interval(0.95, df = len(scores) - 1, loc = np.mean(scores), scale = st.sem(scores))
    else:
        low, up = st.norm.interval(0.95, loc = np.mean(scores), scale = st.sem(scores))
    if name == "":
        return np.mean(scores), up - np.mean(scores)
    else:
        print("{:s} {:.2f} (± {:.2f}) (conf: [{:.2f}, {:.2f}]) (worst: {:.2f}, best: {:.2f})".format(name, 100 * np.mean(scores), 100 * np.std(scores), 100 * low, 100 * up, 100 * np.min(scores), 100 * np.max(scores)))

class ncm_output(nn.Module):
    def __init__(self, indim, outdim):
        super(ncm_output, self).__init__()
        self.linear = nn.Linear(indim, outdim)

    def forward(self, x):
        return -1 * torch.norm(x.reshape(x.shape[0], 1, -1) - self.linear.weight.transpose(0,1).reshape(1, -1, x.shape[1]), dim = 2).pow(2) - self.linear.bias

def linear(indim, outdim):
    if args.ncm_loss:
        return ncm_output(indim, outdim)
    else:
        return nn.Linear(indim, outdim)

def criterion(output, target, num_classes):
    special_criterion = None
    if args.label_smoothing > 0:
        special_criterion = LabelSmoothingLoss(num_classes = num_classes, smoothing = args.label_smoothing)
    elif args.one_vs_all_logistic:
        special_criterion = lambda x,y: one_vs_all_logistic(x,y)
    elif args.apl_alpha > 0.0 or args.apl_beta > 0.0:
        special_criterion = NCEandRCE(args.apl_alpha, args.apl_beta, num_classes)
    elif args.star_loss_gamma > 0.0:
        special_criterion = StarLoss(args.star_loss_gamma)
    
    def standard_criterion(x,y):
        return torch.nn.CrossEntropyLoss()(x/args.temperature, y)
    
    if special_criterion is None:
        return standard_criterion(output, target)
    else:
        if not args.no_mix_special_loss:
            return standard_criterion(output, target) + special_criterion(output, target)
        else:
            return special_criterion(output, target)

def criterion_episodic(features, targets, n_shots = args.n_shots[0]):
    targets, sort_idx = targets.sort()
    features = features[sort_idx]
    feat = features.reshape(args.n_ways, -1, features.shape[1])
    feat = preprocess(feat, feat)
    means = torch.mean(feat[:,:n_shots], dim = 1)
    dists = torch.norm(feat[:,n_shots:].unsqueeze(2) - means.unsqueeze(0).unsqueeze(0), dim = 3, p = 2).reshape(-1, args.n_ways)
    if not args.protonet_no_square:
        dists = dists.pow(2)
    test_size = dists.shape[0]//args.n_ways
    tar = torch.arange(0, args.n_ways, 1, device=means.device).repeat_interleave(test_size)
    return criterion(-dists, tar, num_classes=args.n_ways)

def sphering(features):
    return features / torch.norm(features, p = 2, dim = 2, keepdim = True)

def centering(train_features, features):
    return features - train_features.reshape(-1, train_features.shape[2]).mean(dim = 0).unsqueeze(0).unsqueeze(0)

def preprocess(train_features, features, elements_train=None):
    if elements_train != None and "M" in args.preprocessing:
        train_features = torch.cat([train_features[l, torch.arange(elements_train[l]), :] for l in range(len(elements_train))], axis=0).unsqueeze(1)
    
    for i in range(len(args.preprocessing)):
        if args.preprocessing[i] == 'R':
            with torch.no_grad():
                train_features = torch.relu(train_features)
            features = torch.relu(features)
        if args.preprocessing[i] == 'P':
            with torch.no_grad():
                train_features = torch.pow(train_features, 0.5)
            features = torch.pow(features, 0.5)
        if args.preprocessing[i] == 'E':
            with torch.no_grad():
                train_features = sphering(train_features)
            features = sphering(features)
        if args.preprocessing[i] == 'M':
            features = centering(train_features, features)
            with torch.no_grad():
                train_features = centering(train_features, train_features)
    return features


def postprocess(runs):
    # runs shape: [100, 5, 16, 640]
    for i in range(len(args.postprocessing)):
        if args.postprocessing[i] == 'R':
            runs = torch.relu(runs)
        if args.postprocessing[i] == 'P':
            runs = torch.pow(runs, 0.5)
        if args.postprocessing[i] == 'E':
            runs = runs/torch.norm(runs, p=2, dim=3, keepdim=True)
        if args.postprocessing[i] == 'M':
            runs = runs - runs.reshape(runs.shape[0], -1, runs.shape[-1]).mean(dim=1, keepdim=True).unsqueeze(1)
    return runs

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.cls = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        ce = self.ce(pred, target)
        reg = pred.log_softmax(dim=-1).mean()
        return ce + self.smoothing*reg

class SymmetricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target):
        return (-self.ce(pred, target)).exp().mean()

import math
def one_vs_all_logistic(input, target, m=0.0, t=1.0):
    y = F.one_hot(target, num_classes=input.shape[1]).to(device=input.device, dtype=input.dtype) * 2 - 1
    logistic = torch.log(m+torch.exp(-input*y/t))-math.log(m)
    return 2*logistic.mean()

print("utils, ", end='')

class ClassEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, n_classes: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(n_classes).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(n_classes, d_model)
        pe[:,  0::2] = torch.sin(position * div_term)
        pe[:,  1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        shape_x = x.shape
        shape_y = y.shape
        x = x.reshape(-1, x.shape[-1])
        y = y.flatten()
        x = x + self.pe[y]
        x = x.reshape(shape_x)
        y = y.reshape(shape_y)
        return self.dropout(x)

class FewShotTransformer(nn.Module):
    def __init__(self, backbone, backbone_output_dim, max_classes=100, is_transductive=False, dropout=0.1):
        super().__init__()
        self.is_transductive = is_transductive
        self.backbone = backbone
        self.transformer = torch.nn.Transformer(batch_first=True)
        d_model = self.transformer.d_model
        self.class_encoding = ClassEncoding(d_model, dropout=dropout)
        self.fc_transformer_in = nn.Linear(backbone_output_dim, d_model)
        self.fc_transformer_out = nn.Linear(d_model, max_classes)
    def forward(self, x):
        return self.backbone(x)
    def transform(self, support, support_classes, query=None):
        src = self.fc_transformer_in(support)
        tgt = self.fc_transformer_in(query) if query is not None else src
        src = self.class_encoding(src, support_classes)
        t = tgt.shape[1]
        tgt_mask = None
        if not self.is_transductive:
            tgt_mask = -torch.inf*(torch.ones([t,t], device=support.device))
            tgt_mask.fill_diagonal_(0)
        return self.fc_transformer_out(self.transformer(src, tgt, tgt_mask=tgt_mask))

        




