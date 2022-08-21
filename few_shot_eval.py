import torch
import numpy as np
from args import *
from utils import *
from classification_heads import MetaOptNetHead_SVM_CS as svm
n_runs = args.n_runs
batch_few_shot_runs = args.batch_fs
assert(n_runs % batch_few_shot_runs == 0)

def define_runs(n_ways, n_shots, n_queries, num_classes, elements_per_class):
    shuffle_classes = torch.LongTensor(np.arange(num_classes))
    run_classes = torch.LongTensor(n_runs, n_ways).to(args.device)
    run_indices = torch.LongTensor(n_runs, n_ways, n_shots + n_queries).to(args.device)
    for i in range(n_runs):
        run_classes[i] = torch.randperm(num_classes)[:n_ways]
        for j in range(n_ways):
            run_indices[i,j] = torch.randperm(elements_per_class[run_classes[i, j]])[:n_shots + n_queries]
    return run_classes, run_indices

def generate_runs(data, run_classes, run_indices, batch_idx):
    n_runs, n_ways, n_samples = run_classes.shape[0], run_classes.shape[1], run_indices.shape[2]
    n_shots = n_samples - args.n_queries
    
    classes_to_gather = run_classes[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs]
    indices_to_gather = run_indices[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs]
    classes_to_gather = classes_to_gather.unsqueeze(2).unsqueeze(3).repeat(1,1,data.shape[1], data.shape[2])
    indices_to_gather = indices_to_gather.unsqueeze(3).repeat(1, 1, 1, data.shape[2])

    rho = (args.label_noise_test * n_ways)/(n_ways - 1)
    noisy_labels = torch.rand(classes_to_gather.shape[0], classes_to_gather.shape[1], classes_to_gather.shape[2]) < rho
    new_labels = torch.randint(classes_to_gather.shape[1],[classes_to_gather.shape[0], classes_to_gather.shape[1], classes_to_gather.shape[2]])
    new_labels = torch.gather(run_classes.unsqueeze(2).repeat(1, 1, new_labels.shape[2]), 1, new_labels)
    new_labels = new_labels.unsqueeze(3).repeat([1, 1, 1, classes_to_gather.shape[3]])
    noisy_labels = noisy_labels.unsqueeze(3).repeat([1, 1, 1, classes_to_gather.shape[3]])
    noisy_classes_to_gather = noisy_labels*new_labels + noisy_labels.logical_not()*classes_to_gather 

    indices_to_gather_train = indices_to_gather[:, :, :n_shots, :]
    indices_to_gather_test = indices_to_gather[:, :, n_shots:, :]
    datas = data.unsqueeze(0).repeat(batch_few_shot_runs, *([1]*len(data.shape))) 

    cclasses = torch.gather(datas, 1, classes_to_gather)
    res_test = torch.gather(cclasses, 2, indices_to_gather_test)
    
    noisy_cclasses = torch.gather(datas, 1, noisy_classes_to_gather)
    res_train = torch.gather(noisy_cclasses, 2, indices_to_gather_train)

    
    res = torch.cat([res_train, res_test], 2)

    return res

def ncm(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[-1]
        aug = features.shape[2] if len(features.shape) == 4 else 1
        device = features.device
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features.reshape([train_features.shape[0], -1, dim]), features.reshape([features.shape[0], -1, dim]), elements_train=elements_train).reshape(features.shape)
        features, run_classes, run_indices = features.cpu(), run_classes.cpu(), run_indices.cpu()
        scores = []
        scores_lin = []
        scores_med = []
        scores_me1 = []
        scores_me2 = []
        scores_me3 = []
        scores_me4 = []
        scores_me5 = []
        scores_me6 = []
        scores_me7 = []
        scores_me8 = []
        scores_me9 = []
        neglogdets = []
        mses = []
        mses_svm = []
        scores_svm = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            #print(f"{batch_idx+1} / {(n_runs // batch_few_shot_runs)}")
            flat_features = features.reshape([features.shape[0], features.shape[1], -1])
            runs_aug = generate_runs(flat_features, run_classes, run_indices, batch_idx).to(args.device)
            runs = runs_aug.reshape([runs_aug.shape[0], runs_aug.shape[1], -1, aug, dim]).mean(dim=3)
            runs_aug = runs_aug.reshape([runs.shape[0], runs.shape[1], -1, dim])
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            distances = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim), dim = 4, p = 2)
            winners = torch.min(distances, dim = 2)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
            n_test = runs.shape[2] - n_shots 

            if args.save_features == "":
                Z = runs[:,:,:n_shots]
                medoids_inds = torch.cdist(Z,Z).sum(dim=3).argmin(dim=2)
                medoids = Z.gather(2, medoids_inds.unsqueeze(2).unsqueeze(3).repeat([1,1,1,Z.shape[3]]))
                distances_med = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim) - medoids.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim), dim = 4, p = 2)

                winners = torch.min(distances_med, dim = 2)[1]
                scores_med += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
                
                winners = torch.min(0.1*distances_med + 0.9*distances, dim = 2)[1]
                scores_me1 += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())

                winners = torch.min(0.2*distances_med + 0.8*distances, dim = 2)[1]
                scores_me2 += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())

                winners = torch.min(0.3*distances_med + 0.7*distances, dim = 2)[1]
                scores_me3 += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())

                winners = torch.min(0.4*distances_med + 0.6*distances, dim = 2)[1]
                scores_me4 += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())

                winners = torch.min(0.5*distances_med + 0.5*distances, dim = 2)[1]
                scores_me5 += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())

                winners = torch.min(0.6*distances_med + 0.4*distances, dim = 2)[1]
                scores_me6 += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())

                winners = torch.min(0.7*distances_med + 0.3*distances, dim = 2)[1]
                scores_me7 += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())

                winners = torch.min(0.8*distances_med + 0.2*distances, dim = 2)[1]
                scores_me8 += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())

                winners = torch.min(0.9*distances_med + 0.1*distances, dim = 2)[1]
                scores_me9 += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())

                X = runs_aug[:,:,:n_shots*aug].reshape(runs_aug.shape[0],-1,runs_aug.shape[3])
                if args.bias:
                    X = F.pad(X, (0,1), value=1)
                b, n, d =  X.shape
                device = X.device
                Xt = X.transpose(2,1)
                V = runs[:,:,n_shots:].reshape(runs.shape[0],-1,runs.shape[3])
                if args.bias:
                    V = F.pad(V, (0,1), value=1)
                Y = torch.kron(torch.eye(args.n_ways, device=X.device)*(1+1/(args.n_ways-1))-1/(args.n_ways-1), torch.ones([n_shots*aug, 1], device=device))
                Ytest = torch.kron(torch.eye(args.n_ways, device=X.device)*(1+1/(args.n_ways-1))-1/(args.n_ways-1), torch.ones([n_test, 1], device=device))
                lam = args.lam/d
                LAM = lam*torch.eye(d, device=X.device)
                M = Xt @ X + LAM
                Yhat = V @ M.inverse() @ Xt @ Y 
                E = Yhat - Ytest.unsqueeze(0)
                distances_lin = -Yhat
                winners_lin = torch.min(distances_lin, dim = 2)[1].reshape(winners.shape)
                scores_lin += list((winners_lin == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
                mses += list(E.pow(2).mean(dim = 1).mean(dim = 1).to("cpu").numpy())
                neglogdets += list(-torch.linalg.eigvalsh(M).log().sum(dim=1).cpu().numpy() + d*np.log(n/d + lam))
            else:
                scores_lin.append(np.nan)
                scores_med.append(np.nan)
                mses.append(np.nan)
                neglogdets.append(np.nan)

            if args.svm_c >= 0 and args.save_features == "":
                labels = torch.tile(torch.tile(torch.tensor(range(args.n_ways)).unsqueeze(1), (1, n_shots*aug)).flatten().unsqueeze(0), (b, 1)).to(device)
                with torch.cuda.device(device), torch.no_grad():
                    Ysvm = svm(V, X, labels, args.n_ways, n_shots*aug, args.svm_c)
                distances_svm = -Ysvm
                winners_svm = torch.min(distances_svm, dim = 2)[1].reshape(winners.shape)
                scores_svm += list((winners_svm == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
                mses_svm += list((Ysvm - Ytest.unsqueeze(0)).pow(2).mean(dim = 1).mean(dim = 1).to("cpu").numpy())
            else:
                scores_svm.append(np.nan)
                mses_svm.append(np.nan)

        return \
            stats(scores, ""), stats(scores_med, ""), stats(scores_me1, ""), stats(scores_me2, ""), stats(scores_me3, ""),stats(scores_me4, ""),\
            stats(scores_me5, ""), stats(scores_me6, ""), stats(scores_me7, ""), stats(scores_me8, ""), stats(scores_me9, ""),     \
            stats(scores_lin, ""), stats(neglogdets, ""), stats(mses, ""), stats(scores_svm, ""), stats(mses_svm, "") 

def transductive_ncm(train_features, features, run_classes, run_indices, n_shots, n_iter_trans = args.transductive_n_iter, n_iter_trans_sinkhorn = args.transductive_n_iter_sinkhorn, temp_trans = args.transductive_temperature, alpha_trans = args.transductive_alpha, cosine = args.transductive_cosine, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        if cosine:
            features = features / torch.norm(features, dim = 2, keepdim = True)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            if cosine:
                means = means / torch.norm(means, dim = 2, keepdim = True)
            for _ in range(n_iter_trans):
                if cosine:
                    similarities = torch.einsum("bswd,bswd->bsw", runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim), means.reshape(batch_few_shot_runs, 1, args.n_ways, dim))
                    soft_sims = torch.softmax(temp_trans * similarities, dim = 2)
                else:
                    similarities = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                    soft_sims = torch.exp( -1 * temp_trans * similarities)
                for _ in range(n_iter_trans_sinkhorn):
                    soft_sims = soft_sims / soft_sims.sum(dim = 2, keepdim = True) * args.n_ways
                    soft_sims = soft_sims / soft_sims.sum(dim = 1, keepdim = True) * args.n_queries
                new_means = ((runs[:,:,:n_shots].mean(dim = 2) * n_shots + torch.einsum("rsw,rsd->rwd", soft_sims, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3])))) / runs.shape[2]
                if cosine:
                    new_means = new_means / torch.norm(new_means, dim = 2, keepdim = True)
                means = means * alpha_trans + (1 - alpha_trans) * new_means
                if cosine:
                    means = means / torch.norm(means, dim = 2, keepdim = True)
            if cosine:
                winners = torch.max(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            else:
                winners = torch.min(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def kmeans(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            for i in range(500):
                similarities = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                new_allocation = (similarities == torch.min(similarities, dim = 2, keepdim = True)[0]).float()
                new_allocation = new_allocation / new_allocation.sum(dim = 1, keepdim = True)
                allocation = new_allocation
                means = (runs[:,:,:n_shots].mean(dim = 2) * n_shots + torch.einsum("rsw,rsd->rwd", allocation, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3])) * args.n_queries) / runs.shape[2]
            winners = torch.min(similarities.reshape(runs.shape[0], runs.shape[1], runs.shape[2] - n_shots, -1), dim = 3)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def softkmeans(train_features, features, run_classes, run_indices, n_shots, transductive_temperature_softkmeans=args.transductive_temperature_softkmeans, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            runs = postprocess(runs)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            for i in range(30):
                similarities = torch.norm(runs[:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means.reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                soft_allocations = F.softmax(-similarities.pow(2)*args.transductive_temperature_softkmeans, dim=2)
                means = torch.sum(runs[:,:,:n_shots], dim = 2) + torch.einsum("rsw,rsd->rwd", soft_allocations, runs[:,:,n_shots:].reshape(runs.shape[0], -1, runs.shape[3]))
                means = means/(n_shots+soft_allocations.sum(dim = 1).reshape(batch_few_shot_runs, -1, 1))
            winners = torch.min(similarities, dim = 2)[1]
            winners = winners.reshape(batch_few_shot_runs, args.n_ways, -1)
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def ncm_cosine(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        dim = features.shape[2]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        features = sphering(features)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:n_shots], dim = 2)
            means = sphering(means)
            distances = torch.einsum("bwysd,bwysd->bwys",runs[:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim), means.reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim))
            winners = torch.max(distances, dim = 2)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def get_features(model, loader, n_aug = args.sample_aug):
    model.eval()
    features_total = 0 if args.average else [] 
    for augs in range(n_aug):
        all_features, offset, max_offset = [], 1000000, 0
        for batch_idx, (data, target) in enumerate(loader):        
            with torch.no_grad():
                data, target = data.to(args.device), target.to(args.device)
                _, features = model(data)
                all_features.append(features)
                offset = min(min(target), offset)
                max_offset = max(max(target), max_offset)
        num_classes = max_offset - offset + 1
        print(".", end='')
        all_features = torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
        features_total += all_features if args.average else [all_features.unsqueeze(2).cpu()]  
    features_total = features_total / n_aug if args.average\
         else torch.cat(features_total, dim=2)
    return features_total
def eval_few_shot(train_features, val_features, novel_features, val_run_classes, val_run_indices, novel_run_classes, novel_run_indices, n_shots, transductive = False,elements_train=None):
    if transductive:
        if args.transductive_softkmeans:
            return softkmeans(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), softkmeans(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)
        else:
            return kmeans(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), kmeans(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)
    else:
        return ncm(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), ncm(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)

def update_few_shot_meta_data(model, train_clean, novel_loader, val_loader, few_shot_meta_data):

    if "M" in args.preprocessing or args.save_features != '':
        train_features = get_features(model, train_clean)
    else:
        train_features = torch.Tensor(0,0,0)
    val_features = get_features(model, val_loader)
    novel_features = get_features(model, novel_loader)

    res = []
    for i in range(len(args.n_shots)):
        res.append(evaluate_shot(i, train_features, val_features, novel_features, few_shot_meta_data, model = model))

    return res, train_features.reshape(-1, train_features.shape[2])

def evaluate_shot(index, train_features, val_features, novel_features, few_shot_meta_data, model = None, transductive = False):
    (
    ((val_acc, val_conf), (val_acc_med, val_conf_med),
    (val_acc_me1, val_conf_me1), (val_acc_me2, val_conf_me2), (val_acc_me3, val_conf_me3), (val_acc_me4, val_conf_me4), 
    (val_acc_me5, val_conf_me5), (val_acc_me6, val_conf_me6), (val_acc_me7, val_conf_me7), (val_acc_me8, val_conf_me8), (val_acc_me9, val_conf_me9), 
    (val_acc_lin, val_conf_lin), (val_nld, val_conf_nld), (val_mse, val_conf_mse), (val_svm, val_conf_svm), (val_mvm, val_conf_mvm)), 
    ((novel_acc, novel_conf), (novel_acc_med, novel_conf_med), 
    (novel_acc_me1, novel_conf_me1), (novel_acc_me2, novel_conf_me2), (novel_acc_me3, novel_conf_me3), (novel_acc_me4, novel_conf_me4), 
    (novel_acc_me5, novel_conf_me5), (novel_acc_me6, novel_conf_me6), (novel_acc_me7, novel_conf_me7), (novel_acc_me8, novel_conf_me8), (novel_acc_me9, novel_conf_me9), 
    (novel_acc_lin, novel_conf_lin), (novel_nld, novel_conf_nld), (novel_mse, novel_conf_mse), (novel_svm, novel_conf_svm), (novel_mvm, novel_conf_mvm)) 
    )= eval_few_shot(train_features, val_features, novel_features, few_shot_meta_data["val_run_classes"][index], few_shot_meta_data["val_run_indices"][index], few_shot_meta_data["novel_run_classes"][index], few_shot_meta_data["novel_run_indices"][index], args.n_shots[index], transductive = transductive, elements_train=few_shot_meta_data["elements_train"])
    if val_acc > few_shot_meta_data["best_val_acc"][index]:
        if val_acc > few_shot_meta_data["best_val_acc_ever"][index]:
            few_shot_meta_data["best_val_acc_ever"][index] = val_acc
            if args.save_model != "":
                if len(args.devices) == 1:
                    torch.save(model.state_dict(), args.save_model + str(args.n_shots[index]))
                else:
                    torch.save(model.module.state_dict(), args.save_model + str(args.n_shots[index]))
            if args.save_features != "":
                torch.save(torch.cat([train_features, val_features, novel_features], dim = 0), args.save_features + str(args.n_shots[index]))
        few_shot_meta_data["best_val_acc"][index] = val_acc
        few_shot_meta_data["best_novel_acc"][index] = novel_acc
    return (val_acc,     val_conf,     novel_acc,     novel_conf,     
            val_acc_med, val_conf_med, novel_acc_med, novel_conf_med,
            val_acc_me1, val_conf_me1, novel_acc_me1, novel_conf_me1,
            val_acc_me2, val_conf_me2, novel_acc_me2, novel_conf_me2,
            val_acc_me3, val_conf_me3, novel_acc_me3, novel_conf_me3,
            val_acc_me4, val_conf_me4, novel_acc_me4, novel_conf_me4,
            val_acc_me5, val_conf_me5, novel_acc_me5, novel_conf_me5,
            val_acc_me6, val_conf_me6, novel_acc_me6, novel_conf_me6,
            val_acc_me7, val_conf_me7, novel_acc_me7, novel_conf_me7,
            val_acc_me8, val_conf_me8, novel_acc_me8, novel_conf_me8,
            val_acc_me9, val_conf_me9, novel_acc_me9, novel_conf_me9, 
            val_acc_lin, val_conf_lin, novel_acc_lin, novel_conf_lin, 
            val_nld,     val_conf_nld, novel_nld,     novel_conf_nld, 
            val_mse,     val_conf_mse, novel_mse,     novel_conf_mse,
            val_svm,     val_conf_svm, novel_svm,     novel_conf_svm, 
            val_mvm,     val_conf_mvm, novel_mvm,     novel_conf_mvm)

print("eval_few_shot, ", end='')
