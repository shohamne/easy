### global imports
import math
from cv2 import transform
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import random
import sys
print("Using pytorch version: " + torch.__version__)

### local imports
print("Importing local files: ", end = '')
from args import args
from utils import *
import datasets
import few_shot_eval
import resnet
import wideresnet
import resnet12
import s2m2
import mlp

print("models.")
if args.ema > 0:
    from torch_ema import ExponentialMovingAverage

if args.wandb:
    import wandb


symmetric_loss = SymmetricLoss()
### global variables that are used by the train function
last_update, criterion = 0, torch.nn.CrossEntropyLoss()
### function to either use criterion based on output and target or criterion_episodic based on features and target



def criterion_transformer(features, targets, n_shots = args.n_shots[0]):
    targets, sort_idx = targets.sort()
    features = features[sort_idx]
    feat = features.reshape(args.n_ways, -1, features.shape[1])
    support = feat[:,:n_shots].reshape(-1, features.shape[1])
    query = feat[:,n_shots:].reshape(-1, features.shape[1])
    n_queries = feat.shape[1] - n_shots
    support_labels = torch.repeat_interleave(torch.arange(args.n_ways), n_shots).to(features.device)
    query_labels = torch.repeat_interleave(torch.arange(args.n_ways), n_queries).to(features.device)

    pred = model.transform(support.unsqueeze(0), support_labels.unsqueeze(0), query.unsqueeze(0))
    return criterion(pred.squeeze(0), query_labels)

def crit(output, features, target):
    if args.transformer:
        return criterion_transformer(features, target)
    if args.episodic:
        return criterion_episodic(features, target)
    else:
        return criterion(output, target, num_classes)
 

def train_transformer(model, optimizer, epoch, scheduler):
    elems_train = [elements_per_class] * num_classes if elements_train is None else elements_train
    run_classes, run_indices = few_shot_eval.define_runs(args.n_ways + args.n_unknown, args.n_shots[0], args.n_queries, num_classes, elems_train)  
    features = train_features
    targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
    losses, total = 0., 0
    for batch_idx in range(args.n_runs // args.batch_fs):
        optimizer.zero_grad()
        #print(f"{batch_idx+1} / {(args.n_runs // args.batch_fs)}")
        flat_features = features.reshape([features.shape[0], features.shape[1], -1])
        runs = few_shot_eval.generate_runs(flat_features, run_classes, run_indices, batch_idx)
        runs = runs.to(args.device)
        supports = runs[:,:,:n_shots]
        queries = runs[:,:,n_shots:]
        _, n_ways, n_class_samples, _ = runs.shape
        labels = torch.repeat_interleave(targets, n_class_samples).reshape(n_ways, n_class_samples).unsqueeze(0).repeat(args.batch_fs, 1, 1)
        support_labels = labels[:,:,:n_shots]
        query_labels = labels[:,:,n_shots:]
        support_labels = support_labels.reshape([support_labels.shape[0], -1])
        query_labels = query_labels.reshape([query_labels.shape[0], -1])
        supports = supports.reshape([supports.shape[0], -1, supports.shape[-1]])
        queries = queries.reshape([queries.shape[0], -1, queries.shape[-1]])
        preds = model.transform(supports, support_labels, queries)
        loss = criterion(preds.reshape(-1, preds.shape[-1]), query_labels.flatten())
        loss.backward()
        losses += loss.item() * preds.shape[0] * preds.shape[1]
        total += preds.shape[0] * preds.shape[1]
        # update parameters
        optimizer.step()
        scheduler.step()
        length = args.n_runs//args.batch_fs
        print_train(epoch, scheduler, losses, np.nan, np.nan, total, batch_idx, length)


    return {
        "loss": losses / total,
        "orig_loss": np.nan,
        }




### main train function
def train(model, train_loader, optimizer, epoch, scheduler, F_, m, mixup = False, mm = False):
    model.train()
    losses, neglogdet_losses, orig_losses, total = 0., 0., 0., 0
    for batch_idx, (data, target) in enumerate(train_loader):
            
        data, target = data.to(args.device), target.to(args.device)

        # reset gradients
        optimizer.zero_grad()

        if mm: # as in method S2M2R, to be used in combination with rotations
            # if you do not understand what I just wrote, then just ignore this option, it might be better for now
            new_chunks = []
            sizes = torch.chunk(target, len(args.devices))
            for i in range(len(args.devices)):
                new_chunks.append(torch.randperm(sizes[i].shape[0]))
            index_mixup = torch.cat(new_chunks, dim = 0)
            lam = np.random.beta(2, 2)
            output, features = model(data, index_mixup = index_mixup, lam = lam)
            if args.rotations > 0:
                output, _ = output
            loss_mm = lam * crit(output, features, target) + (1 - lam) * crit(output, features, target[index_mixup])
            loss_mm.backward()

        if True: #args.rotations > 0: # generate self-supervised rotations for improved universality of feature vectors
            bs = data.shape[0] // 4
            target_rot = torch.LongTensor(data.shape[0]).to(args.device)
            target_rot[:bs] = 0
            data[bs:] = data[bs:].transpose(3,2).flip(2)
            target_rot[bs:2*bs] = 1
            data[2*bs:] = data[2*bs:].transpose(3,2).flip(2)
            target_rot[2*bs:3*bs] = 2
            data[3*bs:] = data[3*bs:].transpose(3,2).flip(2)
            target_rot[3*bs:] = 3

        if mixup and args.mm: # mixup or manifold_mixup
            index_mixup = torch.randperm(data.shape[0])
            lam = random.random()            
            if args.mm:
                output, features = model(data, index_mixup = index_mixup, lam = lam)
            else:
                data_mixed = lam * data + (1 - lam) * data[index_mixup]
                output, features = model(data_mixed)
            if args.rotations > 0.0:
                output, output_rot = output
                loss = ((1-args.rotations)*(lam * crit(output, features, target) + (1 - lam) * crit(output, features, target[index_mixup])) \
                    + args.rotations*(lam * crit(output_rot, features, target_rot) + (1 - lam) * crit(output_rot, features, target_rot[index_mixup])))
            else:
                loss = lam * crit(output, features, target) + (1 - lam) * crit(output, features, target[index_mixup])
        else:
            output, features = model(data)
            if args.rotations > 0.0:
                output, output_rot = output
                loss = (1-args.rotations) * crit(output, features, target) + args.rotations * crit(output_rot, features, target_rot)                
            else:
                loss = crit(output, features, target)
        if  args.symmetric_loss > 0.0:
            loss += args.symmetric_loss * symmetric_loss(output, target) 

        orig_losses += loss.item() * data.shape[0]
        if args.logdet_factor is not None and torch.is_tensor(m):
            v = features 
            if args.preprocessing[0] == 'P':
                v = v ** 0.5
            v = F.normalize(v - m, dim=1)
            Fi = v.t() @ v  
            LAM = args.lam/v.shape[1]*torch.eye(v.shape[1], device = v.device) 
            F_ = (1. - args.exp_factor) * F_  +  args.exp_factor * Fi.detach()/v.shape[0]
            Finv = (F_ + LAM).inverse()
            neglogdet_loss = -torch.trace(Finv @ Fi)
            neglogdet_losses += neglogdet_loss.item() * data.shape[0]
            loss += args.logdet_factor*neglogdet_loss
        # backprop loss
        loss.backward()
        losses += loss.item() * data.shape[0]
        total += data.shape[0]
        # update parameters
        optimizer.step()
        scheduler.step()
        if args.ema > 0:
            ema.update()

        if few_shot and args.dataset_size > 0:
            length = args.dataset_size // args.batch_size + (1 if args.dataset_size % args.batch_size != 0 else 0)
        else:
            length = len(train_loader)

        print_train(epoch, scheduler, losses, neglogdet_losses, orig_losses, total, batch_idx, length)

        if few_shot and total >= args.dataset_size and args.dataset_size > 0:
            break
            
    if args.wandb:
        wandb.log({"epoch":epoch, "train_loss": losses / total})

    # return train_loss
    return {
        "loss": losses / total,
        "orig_loss": orig_losses / total,
        }

def print_train(epoch, scheduler, losses, neglogdet_losses, orig_losses, total, batch_idx, length):
    # print advances if at least 100ms have passed since last print
    global last_update
    if (batch_idx + 1 == length) or (time.time() - last_update > 0.1) and not args.quiet:
        if batch_idx + 1 < length:
            print(f"\r{epoch:4d} {1 + batch_idx:4d} / {length:4d} "
                      f"loss: { losses / total:.5f} "
                      f"orig_loss: { orig_losses / total:.5f} "
                      f"neglogdet: { neglogdet_losses / total:.5f} "
                      f"time: {format_time(time.time() - start_time):s} "
                      f"lr: {float(scheduler.get_last_lr()[0]):.5f} ", end = "")
        else:
            print("\r{:4d} loss: {:.5f} ".format(epoch, losses / total), end = '')
        last_update = time.time()


# function to compute accuracy in the case of standard classification
def test(model, test_loader):
    model.eval()
    test_loss, accuracy, accuracy_top_5, total = 0, 0, 0, 0
    
    with torch.no_grad():
        if args.ema > 0:
            ema.store()
            ema.copy_to()
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output, _ = model(data)
            if args.rotations:
                output, _ = output
            test_loss += criterion(output, target).item() * data.shape[0]
            pred = output.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()
            
            # if we want to compute top-5 accuracy
            if top_5:
                preds = output.sort(dim = 1, descending = True)[1][:,:5]
                for i in range(preds.shape[0]):
                    if target[i] in preds[i]:
                        accuracy_top_5 += 1
            # count total number of samples for averaging in the end
            total += target.shape[0]
        if args.ema > 0:
            ema.restore()
    # return results
    model.train()
    
    if args.wandb:
        wandb.log({ "test_loss" : test_loss / total, "test_acc" : accuracy / total, "test_acc_top_5" : accuracy_top_5 / total})

    return { "test_loss" : test_loss / total, "test_acc" : accuracy / total, "test_acc_top_5" : accuracy_top_5 / total}

# function to train a model using args.epochs epochs
# at each args.milestones, learning rate is multiplied by args.gamma
def train_complete(model, loaders, mixup = False):
    global start_time
    start_time = time.time()

    if few_shot:
        train_loader, train_clean, val_loader, novel_loader = loaders
        for i in range(len(few_shot_meta_data["best_val_acc"])):
            few_shot_meta_data["best_val_acc"][i] = 0
    else:
        train_loader, val_loader, test_loader = loaders

    lr = args.lr
    F_ = None
    m = None
    neglogdet = None
    traceF = None
    for epoch in range(args.epochs + args.manifold_mixup):

        if few_shot and args.dataset_size > 0:
            length = args.dataset_size // args.batch_size + (1 if args.dataset_size % args.batch_size != 0 else 0)
        else:
            length = len(train_loader)

        if (args.cosine and epoch % args.milestones[0] == 0) or epoch == 0:
            if lr < 0:
                optimizer = torch.optim.Adam(model.parameters(), lr = -1 * lr)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 5e-4, nesterov = True)
            if args.cosine:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.milestones[0] * length)
                lr = lr * args.gamma
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = list(np.array(args.milestones) * length), gamma = args.gamma)

        if args.transformer and args.test_features != '':
            train_stats = train_transformer(model, optimizer, (epoch + 1), scheduler) 
        else:           
            train_stats = train(model, train_loader, optimizer, (epoch + 1), scheduler, F_, m, mixup = mixup, mm = epoch >= args.epochs)        
        
        if args.save_model != "" and not few_shot:
            if len(args.devices) == 1:
                torch.save(model.state_dict(), args.save_model)
            else:
                torch.save(model.module.state_dict(), args.save_model)
        
        if (epoch + 1) > args.skip_epochs:
            if few_shot:
                if args.ema > 0:
                    ema.store()
                    ema.copy_to()
                with torch.no_grad():
                    res, features = few_shot_eval.update_few_shot_meta_data(model, train_clean, novel_loader, val_loader, few_shot_meta_data, train_features, val_features, test_features, transform=model.transform)
                    if args.preprocessing[0] == 'P':
                        features = features ** 0.5
                    m = features.mean(dim=0, keepdim=True)
                    v = F.normalize(features-m, dim=1)
                    n, d = v.shape[0], v.shape[1]
                    lam = args.lam/d
                    LAM = lam*torch.eye(d, device = v.device) 
                    F_ = (v.t() @ v)/n
                    neglogdet = -torch.linalg.eigvalsh(F_ + LAM).log().sum().item() + d*math.log(1/d+lam)
                    traceF = F_.trace()

                if args.ema > 0:
                    ema.restore()
                for i in range(len(args.n_shots)):
                    print()
                    print(f"val-{args.n_shots[i]:d}: {100 * res[i][0]:.2f}%, "
                          f"nov-{args.n_shots[i]:d}: {100 * res[i][2]:.2f}% ({100 * few_shot_meta_data['best_novel_acc'][i]:.2f}%) "
                          f"val_lin-{args.n_shots[i]:d}: {100 * res[i][4]:.2f}%, "
                          f"nov_lin-{args.n_shots[i]:d}: {100 * res[i][6]:.2f}%  "
                          f"val_nld-{args.n_shots[i]:d}: {100 * res[i][8]:.2f}%, "
                          f"nov_nld-{args.n_shots[i]:d}: {100 * res[i][10]:.2f}%  "
                          f"val_mse-{args.n_shots[i]:d}: {100 * res[i][12]:.2f}%, "
                          f"nov_mse-{args.n_shots[i]:d}: {100 * res[i][14]:.2f}%  "
                          f"val_svm-{args.n_shots[i]:d}: {100 * res[i][16]:.2f}%, "
                          f"nov_svm-{args.n_shots[i]:d}: {100 * res[i][18]:.2f}%  "
                          f"val_mvm-{args.n_shots[i]:d}: {100 * res[i][20]:.2f}%, "
                          f"nov_mvm-{args.n_shots[i]:d}: {100 * res[i][22]:.2f}%  "
                          f"train_loss: {train_stats['loss']} "
                          f"train_orig_loss: {train_stats['orig_loss']} "
                          f"{neglogdet=} {traceF=} ", end = '')
                    if args.wandb:
                        wandb.log({'epoch':epoch, f'val-{args.n_shots[i]}':res[i][0], f'nov-{args.n_shots[i]}':res[i][2], f'best-nov-{args.n_shots[i]}':few_shot_meta_data["best_novel_acc"][i]})

                print()
            else:
                test_stats = test(model, test_loader)
                if top_5:
                    print("top-1: {:.2f}%, top-5: {:.2f}%".format(100 * test_stats["test_acc"], 100 * test_stats["test_acc_top_5"]))
                else:
                    print("test acc: {:.2f}%".format(100 * test_stats["test_acc"]))

    if args.epochs + args.manifold_mixup <= args.skip_epochs:
        if few_shot:
            if args.ema > 0:
                ema.store()
                ema.copy_to()
            res, features = few_shot_eval.update_few_shot_meta_data(model, train_clean, novel_loader, val_loader, few_shot_meta_data, train_features, val_features, test_features, transform=model.transform)
            if args.ema > 0:
                ema.restore()
        else:
            test_stats = test(model, test_loader)

    if few_shot:
        return few_shot_meta_data
    else:
        return test_stats

### process main arguments
loaders, input_shape, num_classes, few_shot, top_5 = datasets.get_dataset(args.dataset)
### initialize few-shot meta data
if few_shot:
    num_classes, val_classes, novel_classes, elements_per_class = num_classes
    if args.dataset.lower() in ["tieredimagenet", "cubfs"]:
        elements_train, elements_val, elements_novel = elements_per_class
    else:
        elements_val, elements_novel = [elements_per_class] * val_classes, [elements_per_class] * novel_classes
        elements_train = None
    print("Dataset contains",num_classes,"base classes,",val_classes,"val classes and",novel_classes,"novel classes.")
    print("Generating runs... ", end='')

    val_runs = list(zip(*[few_shot_eval.define_runs(args.n_ways + args.n_unknown, s, args.n_queries, val_classes, elements_val) for s in args.n_shots]))
    val_run_classes, val_run_indices = val_runs[0], val_runs[1]
    novel_runs = list(zip(*[few_shot_eval.define_runs(args.n_ways + args.n_unknown, s, args.n_queries, novel_classes, elements_novel) for s in args.n_shots]))
    novel_run_classes, novel_run_indices = novel_runs[0], novel_runs[1]
    print("done.")
    few_shot_meta_data = {
        "elements_train":elements_train,
        "val_run_classes" : val_run_classes,
        "val_run_indices" : val_run_indices,
        "novel_run_classes" : novel_run_classes,
        "novel_run_indices" : novel_run_indices,
        "best_val_acc" : [0] * len(args.n_shots),
        "best_val_acc_ever" : [0] * len(args.n_shots),
        "best_novel_acc" : [0] * len(args.n_shots)
    }

# can be used to compute mean and std on training data, to adjust normalizing factors
if False:
    train_loader, _, _ = loaders
    try:
        for c in range(input_shape[0]):
            print("Mean of canal {:d}: {:f} and std: {:f}".format(c, train_loader.data[:,c,:,:].reshape(train_loader.data[:,c,:,:].shape[0], -1).mean(), train_loader.data[:,c,:,:].reshape(train_loader.data[:,c,:,:].shape[0], -1).std()))
    except:
        pass

### prepare stats
run_stats = {}
if args.output != "":
    f = open(args.output, "a")
    f.write(str(args))
    f.close()

### function to create model
def create_model():
    if args.model.lower() == "resnet18":
        return resnet.ResNet18(args.feature_maps, input_shape, num_classes, few_shot, args.rotations).to(args.device)
    if args.model.lower() == "resnet20":
        return resnet.ResNet20(args.feature_maps, input_shape, num_classes, few_shot, args.rotations).to(args.device)
    if args.model.lower() == "wideresnet":
        return wideresnet.WideResNet(args.feature_maps, input_shape, few_shot, args.rotations, num_classes = num_classes).to(args.device)
    if args.model.lower() == "resnet12":
        return resnet12.ResNet12(args.feature_maps, input_shape, num_classes, few_shot, args.rotations).to(args.device)
    if args.model.lower()[:3] == "mlp":
        return mlp.MLP(args.feature_maps, int(args.model[3:]), input_shape, num_classes, args.rotations, few_shot).to(args.device)
    if args.model.lower() == "s2m2r":
        return s2m2.S2M2R(args.feature_maps, input_shape, args.rotations, num_classes = num_classes).to(args.device)

def load_features(num_classes, val_classes):
    try:
        print(f'args.test_features={args.test_features}')
        filenames = eval(args.test_features)
    except:
        filenames = args.test_features
    if isinstance(filenames, str):
        filenames = [filenames]
    print(f'filenames={filenames}, {len(filenames)=}')
    test_features = torch.cat([torch.load(fn, map_location=torch.device(args.device)).to(args.dataset_device) for fn in filenames], dim = 2)
    if len(test_features.shape) == 4:
        test_features = test_features.reshape(test_features.shape[0], test_features.shape[1], args.sample_aug, -1, test_features.shape[3]).mean(dim=3)
    print("Testing features of shape", test_features.shape)
    train_features = test_features[:num_classes]
    val_features = test_features[num_classes:num_classes + val_classes]
    test_features = test_features[num_classes + val_classes:]
    return test_features,train_features,val_features

if args.test_features:
    test_features, train_features, val_features = load_features(num_classes, val_classes)
else:
    test_features = train_features = val_features = None

if args.test_features != "" and not args.transformer:
    if not args.transductive:
        result = []
        for i in range(len(args.n_shots)):
            (val_acc, val_conf, test_acc, test_conf, 
            val_acc_med, val_conf_med, test_acc_med, test_conf_med,
            val_acc_me1, val_conf_me1, test_acc_me1, test_conf_me1, 
            val_acc_me2, val_conf_me2, test_acc_me2, test_conf_me2, 
            val_acc_me3, val_conf_me3, test_acc_me3, test_conf_me3, 
            val_acc_me4, val_conf_me4, test_acc_me4, test_conf_me4, 
            val_acc_me5, val_conf_me5, test_acc_me5, test_conf_me5, 
            val_acc_me6, val_conf_me6, test_acc_me6, test_conf_me6, 
            val_acc_me7, val_conf_me7, test_acc_me7, test_conf_me7, 
            val_acc_me8, val_conf_me8, test_acc_me8, test_conf_me8, 
            val_acc_me9, val_conf_me9, test_acc_me9, test_conf_me9,
            val_acc_maj, val_conf_maj, test_acc_maj, test_conf_maj,  
            val_acc_lin, val_conf_lin, test_acc_lin, test_conf_lin, 
            val_nld, val_conf_nld, test_nld, test_conf_nld,
            val_mse, val_conf_mse, test_mse, test_conf_mse,
            val_svm, val_conf_svm, test_svm, test_conf_svm, 
            val_mvm, val_conf_mvm, test_mvm, test_conf_mvm) = \
                few_shot_eval.evaluate_shot(i, train_features, val_features, test_features, few_shot_meta_data, transform=model.transform)
            print(f"Inductive {args.n_shots[i]:d}-shot: "
                  f"acc: {100 * test_acc:.2f}% (± {100 * test_conf:.2f}%) "
                  f"me1: {100 * test_acc_me1:.2f}% (± {100 * test_conf_med:.2f}%) "
                  f"me2: {100 * test_acc_me2:.2f}% (± {100 * test_conf_med:.2f}%) "
                  f"me3: {100 * test_acc_me3:.2f}% (± {100 * test_conf_med:.2f}%) "
                  f"me4: {100 * test_acc_me4:.2f}% (± {100 * test_conf_med:.2f}%) "
                  f"me5: {100 * test_acc_me5:.2f}% (± {100 * test_conf_med:.2f}%) "
                  f"me6: {100 * test_acc_me6:.2f}% (± {100 * test_conf_med:.2f}%) "
                  f"me7: {100 * test_acc_me7:.2f}% (± {100 * test_conf_med:.2f}%) "
                  f"me8: {100 * test_acc_me8:.2f}% (± {100 * test_conf_med:.2f}%) "
                  f"me9: {100 * test_acc_me9:.2f}% (± {100 * test_conf_med:.2f}%) "
                  f"med: {100 * test_acc_med:.2f}% (± {100 * test_conf_med:.2f}%) "
                  f"maj: {100 * test_acc_maj:.2f}% (± {100 * test_conf_maj:.2f}%) "
                  f"lin: {100 * test_acc_lin:.2f}% (± {100 * test_conf_lin:.2f}%) "
                  f"nld: {100 * test_nld:.2f}% (± {100 * test_conf_nld:.2f}%) "
                  f"mse: {100 * test_mse:.2f}% (± {100 * test_conf_mse:.2f}%) "
                  f"svm: {100 * test_svm:.2f}% (± {100 * test_conf_svm:.2f}%) "
                  f"mvm: {100 * test_mvm:.2f}% (± {100 * test_conf_mvm:.2f}%) ")
            res = dict(shots=args.n_shots[i],
                val_acc    =val_acc    , val_conf    =val_conf    , test_acc    =test_acc    , test_conf    =test_conf    ,
                val_acc_me1=val_acc_me1, val_conf_me1=val_conf_me1, test_acc_me1=test_acc_me1, test_conf_me1=test_conf_me1,
                val_acc_me2=val_acc_me2, val_conf_me2=val_conf_me2, test_acc_me2=test_acc_me2, test_conf_me2=test_conf_me2,
                val_acc_me3=val_acc_me3, val_conf_me3=val_conf_me3, test_acc_me3=test_acc_me3, test_conf_me3=test_conf_me3,
                val_acc_me4=val_acc_me4, val_conf_me4=val_conf_me4, test_acc_me4=test_acc_me4, test_conf_me4=test_conf_me4,
                val_acc_me5=val_acc_me5, val_conf_me5=val_conf_me5, test_acc_me5=test_acc_me5, test_conf_me5=test_conf_me5,
                val_acc_me6=val_acc_me6, val_conf_me6=val_conf_me6, test_acc_me6=test_acc_me6, test_conf_me6=test_conf_me6,
                val_acc_me7=val_acc_me7, val_conf_me7=val_conf_me7, test_acc_me7=test_acc_me7, test_conf_me7=test_conf_me7,
                val_acc_me8=val_acc_me8, val_conf_me8=val_conf_me8, test_acc_me8=test_acc_me8, test_conf_me8=test_conf_me8,
                val_acc_me9=val_acc_me9, val_conf_me9=val_conf_me9, test_acc_me9=test_acc_me9, test_conf_me9=test_conf_me9,
                val_acc_med=val_acc_med, val_conf_med=val_conf_med, test_acc_med=test_acc_med, test_conf_med=test_conf_med, 
                val_acc_maj=val_acc_maj, val_conf_maj=val_conf_maj, test_acc_maj=test_acc_maj, test_conf_maj=test_conf_maj,  
                val_acc_lin=val_acc_lin, val_conf_lin=val_conf_lin, test_acc_lin=test_acc_lin, test_conf_lin=test_conf_lin, 
                val_nld    =val_nld    , val_conf_nld=val_conf_nld, test_nld    =test_nld    , test_conf_nld=test_conf_nld,
                val_mse    =val_mse    , val_conf_mse=val_conf_mse, test_mse    =test_mse    , test_conf_mse=test_conf_mse,
                val_svm    =val_svm    , val_conf_svm=val_conf_svm, test_svm    =test_svm    , test_conf_svm=test_conf_svm, 
                val_mvm    =val_mvm    , val_conf_mvm=val_conf_mvm, test_mvm    =test_mvm    , test_conf_mvm=test_conf_mvm)
            result.append(res)
        import pandas as pd
        result_path = args.test_features + f".lam{args.lam}_nt{args.label_noise_test}"+ '.csv' if args.result_path is None else args.result_path
        pd.DataFrame(result).to_csv(result_path)
        
    else:
        for i in range(len(args.n_shots)):
            val_acc, val_conf, test_acc, test_conf = few_shot_eval.evaluate_shot(i, train_features, val_features, test_features, few_shot_meta_data, transductive = True)
            print("Transductive {:d}-shot: {:.2f}% (± {:.2f}%)".format(args.n_shots[i], 100 * test_acc, 100 * test_conf))
    sys.exit()

for i in range(args.runs):

    if not args.quiet:
        print(args)
    if args.wandb:
        wandb.init(project="few-shot", 
            entity=args.wandb, 
            tags=[f'run_{i}', args.dataset], 
            notes=str(vars(args))
            )
        wandb.log({"run": i})
    
    if args.test_features != "" and args.transformer:
        model = None
    else:
        model = create_model()

    if args.ema > 0:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema)
        

    backbone_output_dim = 640 if args.feature_dim == -1 else args.feature_dim
    if args.test_features == "":
        if args.load_model != "":
            model.load_state_dict(torch.load(args.load_model, map_location=torch.device(args.device)))
            model.to(args.device)

        if len(args.devices) > 1:
            model = torch.nn.DataParallel(model, device_ids = args.devices)
        

        if args.transformer:
            model = FewShotTransformer(model, backbone_output_dim)
            model.backbone.requires_grad_(False)
            model.to(args.device)
    else:
        model = FewShotTransformer(None, backbone_output_dim)
        model.to(args.device)


    if i == 0:
        print("Number of trainable parameters in model is: " + str(np.sum([p.numel() for p in model.parameters()])))

    # training
    test_stats = train_complete(model, loaders, mixup = args.mixup)

    # assemble stats
    for item in test_stats.keys():
        if i == 0:
            run_stats[item] = [test_stats[item].copy() if isinstance(test_stats[item], list) else test_stats[item]]
        else:
            run_stats[item].append(test_stats[item].copy() if isinstance(test_stats[item], list) else test_stats[item])

    # write file output 
    if args.output != "":
        f = open(args.output, "a")
        f.write(", " + str(run_stats))
        f.close()

    # print stats
    print("Run", i + 1, "/", args.runs)
    if few_shot:
        for index in range(len(args.n_shots)):
            stats(np.array(run_stats["best_novel_acc"])[:,index], "{:d}-shot".format(args.n_shots[index]))
            if args.wandb:
                wandb.log({"run": i+1,"test acc {:d}-shot".format(args.n_shots[index]):np.mean(np.array(run_stats["best_novel_acc"])[:,index])})
    else:
        stats(run_stats["test_acc"], "Top-1")
        if top_5:
            stats(run_stats["test_acc_top_5"], "Top-5")

if args.output != "":
    f = open(args.output, "a")
    f.write("\n")
    f.close()
