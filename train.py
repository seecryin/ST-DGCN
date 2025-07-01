import argparse
import os
import time
import gc
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn.functional as F
from net import GNNStack
from utils import AverageMeter, accuracy, log_msg, get_default_train_val_test_loader

parser = argparse.ArgumentParser(description='PyTorch UEA Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='chapman2500')
parser.add_argument('--num_layers', type=int, default=3, help='the number of GNN layers')
parser.add_argument('--groups', type=int, default=1, help='the number of time series groups (num_graphs)')
parser.add_argument('--fft', type=int, default=0,help='the number of time series groups (num_graphs)')
parser.add_argument('--stgere', type=int, default=1,help='the number of time series groups (num_graphs)')
parser.add_argument('--printxshape', type=int, default=0,help='the number of time series groups (num_graphs)')
parser.add_argument('--relu2', type=int, default=0,help='the number of time series groups (num_graphs)')
parser.add_argument('--fft_channel', type=int, default=12, help='the number of time series groups (num_graphs)')
parser.add_argument('--mtpool', type=int, default=1, help='the number of time series groups (num_graphs)')
# parser.add_argument('--num_class', type=int, default=5, help='the number of time series groups (num_graphs)')
parser.add_argument('--pool_ratio', type=float, default=0.2, help='the ratio of pooling for nodes')
parser.add_argument('--kern_size', type=str, default="9,5,3", help='list of time conv kernel size for each layer')
parser.add_argument('--in_dim', type=int, default=64, help='input dimensions of GNN stacks')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimensions of GNN stacks')
parser.add_argument('--out_dim', type=int, default=256, help='output dimensions of GNN stacks')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--val-batch-size', default=8, type=int, metavar='V',
                    help='validation batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--use_benchmark', dest='use_benchmark', action='store_true',
                    default=True, help='use benchmark')
parser.add_argument('--tag', default='date', type=str,
                    help='the tag for identifying the log and model files. Just a string.')
parser.add_argument('-t', '--test', dest='test', action='store_true',
                    help='evaluate model on test set')

def main():
    args = parser.parse_args()
    
    args.kern_size = [ int(l) for l in args.kern_size.split(",") ]

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    main_work(args)


def main_work(args):
    # init acc
    best_acc1 = 0
    
    if args.tag == 'date':
        local_date = time.strftime('%m.%d', time.localtime(time.time()))
        args.tag = local_date
    dsid = args.dataset

    data_file = f'../log/{dsid}'
    model_file = f'../exp/{dsid}'
    config_file = '{}_groups{}_fft{}_mtpool{}_stgere{}_layer{}_{}exp.txt'.format(args.tag, args.groups, args.fft, args.mtpool, args.stgere, args.num_layers,args.dataset)
    # log_file = '../log/{}_groups{}_{}_{}_exp.txt'.format(args.tag, args.groups, args.fft, args.dataset)
    log_file = os.path.join(data_file, config_file)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


    # dataset
    train_loader, val_loader, test_loader, num_nodes, seq_length, num_classes = get_default_train_val_test_loader(args)
    
    # training model from net.py
    model = GNNStack(gnn_model_type=args.arch, num_layers=args.num_layers, 
                     groups=args.groups, pool_ratio=args.pool_ratio, kern_size=args.kern_size, 
                     in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim, 
                     seq_len=seq_length,num_nodes=args.fft_channel ,fft_channel=args.fft_channel, stgere=args.stgere, num_classes=num_classes, fft = args.fft,relu2=args.relu2, mtpool=args.mtpool)

    # print & log
    # log_msg('epochs {}, lr {}, weight_decay {}'.format(args.epochs, args.lr, args.weight_decay), log_file)
    log_msg('epochs {}, lr {}, weight_decay {}'.format(args.epochs, args.lr, args.weight_decay), log_file)
    #

    # determine whether GPU or not
    if not torch.cuda.is_available():
        print("Warning! Using CPU!!!")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)

        # collect cache
        gc.collect()
        torch.cuda.empty_cache()

        model = model.cuda(args.gpu)
        if args.use_benchmark:
            cudnn.benchmark = True
        print('Using cudnn.benchmark.')
    else:
        print("Error! We only have one gpu!!!")

    # define loss function(criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                               patience=50, verbose=True)


    # validation
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    # if args.test:
    #     test_results = test(test_loader, model, criterion)
    #     print(f"测试集评估结果：{test_results}")


    # train & valid
    print('****************************************************')
    print(args.dataset)

    dataset_time = AverageMeter('Time', ':6.3f')

    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    loss_test = []
    acc_test = []
    epoches = []
    best_epoch = 0
    best_accval = 0
    best_acctest = 0
    best_total = 0

    end = time.time()
    for epoch in range(args.epochs):
        epoches += [epoch]
        start_time = datetime.datetime.now()

        # train for one epoch
        acc_train_per, loss_train_per = train(train_loader, model, criterion, optimizer, lr_scheduler, args)
        
        acc_train += [acc_train_per]
        loss_train += [loss_train_per]

        msg = f'TRAIN, epoch {epoch}, loss {loss_train_per}, acc {acc_train_per}'
        log_msg(msg, log_file)


        # evaluate on validation set
        acc_val_per, loss_val_per, valaverage_precision, valaverage_recall, valf1, valf1avg, valauc = validate(val_loader, model, criterion, args, epoch, num_classes)

        acc_val += [acc_val_per]
        loss_val += [loss_val_per]

        msg = f'VAL, epoch {epoch},   loss {loss_val_per}, acc {acc_val_per}, F1: {valf1:.4f}, F1avg: {valf1avg:.4f}, Auc: {valauc:.4f}, Precision: {valaverage_precision:.4f}, Recall: {valaverage_recall:.4f}'
        log_msg(msg, log_file)
        score_total = acc_val_per + valaverage_recall + valaverage_precision + valf1

        # test data
        acc_test_per, loss_test_per, average_precision, average_recall, testf1, testf1avg, testauc = test(test_loader, model, criterion, args, epoch, num_classes)

        acc_test += [acc_test_per]
        loss_test += [loss_test_per]

        msg = f'TEST, epoch {epoch},  loss {loss_test_per}, acc {acc_test_per}, F1: {testf1:.4f}, F1avg: {testf1avg:.4f}, Auc: {testauc:.4f}, Precision: {average_precision:.4f}, Recall: {average_recall:.4f}'
        log_msg(msg, log_file)
        # score_total = acc_test_per + average_recall + average_precision + f1
        # msg = f' * bestval: {best_accval}at epoch {best_epochval},*besttest: {best_acctest}at epoch {best_epochtest}'
        # msg = f'  '
        # log_msg(msg, log_file)

        # remember best acc
        if acc_val_per > best_accval:
            best_accval = acc_val_per
            best_epochval = epoch

        if acc_test_per > best_acctest:
            best_acctest = acc_test_per
            best_epochtest = epoch
            if epoch > 20:
                expconfig_file = '{}_{}_groups{}_FFT{}_epoch{}_acc{}.pth'.format(args.tag, args.dataset, args.groups, args.fft_channel, epoch,
                                                                           best_acctest)
                model_path = os.path.join(model_file, expconfig_file)
                torch.save(model, model_path)
                print(f'best model saved at epoch {epoch}.')

        end_time = datetime.datetime.now()
        msg = f' * bestval: {best_accval}at epoch {best_epochval},*besttest: {best_acctest}at epoch {best_epochtest}'
        log_msg(msg, log_file)
        msg = f'  '
        log_msg(msg, log_file)
        print(f' * besttest: {best_accval}at epoch {best_epochval},*besttest: {best_acctest}at epoch {best_epochtest}')

        print(f"Time taken for this epoch: {end_time - start_time}")
        print(f' ')
        # model_path = '../exp/{}model{}_{}_{}_{}.pth'.format(args.tag,args.dataset, args.groups, epoch, acc_val_per)
        # model_file = f'../exp/{dsid}'
        # expconfig_file = '{}_{}_groups{}_epoch{}_acc{}.pth'.format(args.tag,args.dataset, args.groups, epoch, acc_val_per)
        # model_path = os.path.join(model_file, expconfig_file)


        # # torch.save(model, model_path)
        #
        # if epoch > 20:
        #     if score_total > best_total:
        #         best_total = score_total
        #         torch.save(model, model_path)
        #         print(f'best model saved at epoch {epoch}.')
        #
        # if (epoch + 1) % 20 == 0 or epoch == 200 - 1:
        #     # 保存整个模型
        #     torch.save(model, model_path)
        #     print(f'Model saved at epoch {epoch}.')


    # measure elapsed time
    dataset_time.update(time.time() - end)

    # log & print the best_acc
    msg = f'\n\n * BEST_VAL_ACC: {best_accval}at epoch {best_epochval}\n * TIME: {dataset_time}\n'
    log_msg(msg, log_file)
    msg = f'\n\n * BEST_TEST_ACC: {best_acctest}at epoch {best_epochtest}\n * TIME: {dataset_time}\n'
    log_msg(msg, log_file)

    print(f' * val: {best_accval}at epoch {best_epochval}')
    print(f' * test: {best_acctest}at epoch {best_epochtest}')
    print(f' * time: {dataset_time}')
    print('****************************************************')


    # collect cache
    gc.collect()
    torch.cuda.empty_cache()


def train(train_loader, model, criterion, optimizer, lr_scheduler, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')

    # switch to train mode
    model.train()

    for count, (data, label) in enumerate(train_loader):
        # data in cuda
        data = data.cuda(args.gpu).type(torch.float)
        label = label.cuda(args.gpu).type(torch.long)

        # compute output
        output = model(data)

        loss = criterion(output, label)

        # output_prob = F.softmax(output, dim=1)
        #
        # # 将标签和概率转换为 NumPy 数组
        # label_np = label.cpu().numpy()
        # output_prob_np = output_prob.detach().cpu().numpy()
        #
        # # 计算多分类问题的 AUC 值，使用一对多（OvR）策略
        # auc_ovr = roc_auc_score(label_np, output_prob_np, multi_class='ovr')
        # print("AUC (One-vs-Rest):", auc_ovr)
        #
        # # 计算多分类问题的 AUC 值，使用一对一（OvO）策略
        # auc_ovo = roc_auc_score(label_np, output_prob_np, multi_class='ovo')
        # print("AUC (One-vs-One):", auc_ovo)

        acc1, pre, re,fno, auc  = acc(output, label)
        losses.update(loss.item(), data.size(0))
        top1.update(acc1, data.size(0))
        # label = label.cpu()
        # outputs = output_prob.detach().cpu().numpy()
        # auc_ovr = roc_auc_score(label, outputs, multi_class='ovr')
        # print("AUC (One-vs-Rest):", auc_ovr)
        #
        # auc_ovo = roc_auc_score(label, outputs, multi_class='ovo')
        # print("AUC (One-vs-One):", auc_ovo)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lr_scheduler.step(top1.avg)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args, epoch,num_classes):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    average_auc = AverageMeter('Auc', ':6.2f')
    average_precision = AverageMeter('Avg Precision', ':.4f')
    average_recall = AverageMeter('Avg Recall', ':.4f')
    average_f1 = AverageMeter('Avg F1 Score', ':.4f')
    f1 = AverageMeter(' F1 Score', ':.4f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for count, (data, label) in enumerate(val_loader):
            if args.gpu is not None:
                data = data.cuda(args.gpu, non_blocking=True).type(torch.float)
            if torch.cuda.is_available():
                label = label.cuda(args.gpu, non_blocking=True).type(torch.long)

            # compute output
            output = model(data)

            loss = criterion(output, label)

            accval, pre, re ,f1zero,auc = acc(output, label)
            f1_score = ff(output, label,num_classes)  # 确保 f1 函数返回的是单个 F1 分数
            losses.update(loss.item(), data.size(0))
            top1.update(accval, data.size(0))
            average_precision.update(pre, data.size(0))
            average_auc.update(auc, data.size(0))
            average_recall.update(re, data.size(0))
            average_f1.update(f1_score, data.size(0))
            f1.update(f1zero, data.size(0))


            # 打印 epoch 信息和指标
        print(f'val:Epoch: {epoch + 1}, Loss: {losses.avg:.4e}, '
              f'Accuracy: {top1.avg:.2f}, F1: {f1.avg:.4f}, F1avg: {average_f1.avg:.4f}, Auc: {average_auc.avg:.4f} '
              f'Precision: {average_precision.avg:.4f}, Recall: {average_recall.avg:.4f} ')

    return top1.avg, losses.avg, average_precision.avg, average_recall.avg, f1.avg, average_f1.avg, average_auc.avg

def test(test_loader, model, criterion, args, epoch, num_classes):
    losses = AverageMeter('Loss2', ':.4e')
    top1 = AverageMeter('Acc@2', ':6.2f')
    average_precision = AverageMeter('testAvg Precision', ':.4f')
    average_recall = AverageMeter('testAvg Recall', ':.4f')
    average_f1 = AverageMeter('testAvg F1 Score', ':.4f')
    f1 = AverageMeter(' tesF1 Score', ':.4f')
    average_auc = AverageMeter('Auc', ':6.2f')
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for count, (data, label) in enumerate(test_loader):
            if args.gpu is not None:
                data = data.cuda(args.gpu, non_blocking=True).type(torch.float)
            if torch.cuda.is_available():
                label = label.cuda(args.gpu, non_blocking=True).type(torch.long)

            # compute output
            output = model(data)

            loss = criterion(output, label)

            acctest, pre, re, f1zero, auc = acc(output, label)
            f1_score = ff(output, label, num_classes)  # 确保 f1 函数返回的是单个 F1 分数
            losses.update(loss.item(), data.size(0))
            top1.update(acctest, data.size(0))
            average_auc.update(auc, data.size(0))
            average_precision.update(pre, data.size(0))
            average_recall.update(re, data.size(0))
            average_f1.update(f1_score, data.size(0))
            f1.update(f1zero, data.size(0))


            # 打印 epoch 信息和指标
        print(f'val:Epoch: {epoch + 1}, Loss: {losses.avg:.4e}, '
              f'Accuracy: {top1.avg:.2f}, F1: {f1.avg:.4f}, F1avg: {average_f1.avg:.4f}, Auc: {average_auc.avg:.4f} '
              f'Precision: {average_precision.avg:.4f}, Recall: {average_recall.avg:.4f} ')

    return top1.avg, losses.avg, average_precision.avg, average_recall.avg, f1.avg, average_f1.avg, average_auc.avg

def ff(output, target,num_classes):
    # 将输出转换为预测标签
    pred_labels = output.argmax(dim=1)
    numclasses = num_classes

    class_f1_scores = []

    for i in range(0, numclasses):
        # 计算当前正类的 TP、FP、FN 和 TN
        TP = ((pred_labels == i) & (target == i)).sum().item()
        FP = ((pred_labels == i) & (target != i)).sum().item()
        FN = ((pred_labels != i) & (target == i)).sum().item()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_f1_scores.append(f1_score)
    f1 = sum(class_f1_scores) / numclasses

    return f1


def acc(output, target):
    # 将输出转换为预测标签
    pred_labels = output.argmax(dim=1)
    output_prob = F.softmax(output, dim=1).detach().cpu().numpy()
    label=target.cpu().numpy()
    TP = ((pred_labels == target) & (target > 0)).sum().item()  # 预测和目标相等且为正类
    FP = ((pred_labels != target) & (pred_labels > 0)).sum().item()  # 预测为正类但目标不为正类
    FN = ((pred_labels == 0) & (target > 0)).sum().item()  # 预测为0但目标为正类
    TN = ((pred_labels == target) & (target == 0)).sum().item()  # 预测和目标都是0
    # 计算 Accuracy, Precision, Recall, F1 Score
    acc = (TP + TN) / (TP + TN + FP + FN)
    pre = TP / (TP + FP) if (TP + FP) > 0 else 0
    re = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1zero = 2 * (pre * re) / (pre + re) if (pre + re) > 0 else 0
    auc = multiclass_roc_auc_score(label, output_prob, average='macro')

    # output_prob = F.softmax(output, dim=1)
    # label = target.cpu()
    # outputs = output_prob.detach().cpu().numpy()
    # auc_ovr = roc_auc_score(label, outputs, multi_class='ovr')
    # print("AUC (One-vs-Rest):", auc_ovr)
    #
    # auc_ovo = roc_auc_score(label, outputs, multi_class='ovo')
    # print("AUC (One-vs-One):", auc_ovo)
    return acc, pre, re ,f1zero, auc


def multiclass_roc_auc_score(labels, pred_probs, average='macro'):
    # 获取所有唯一的类别
    unique_classes = np.unique(labels)

    roc_auc = []
    for i in unique_classes:
        # 为当前类别创建二元标签
        y_true = (labels == i).astype(int)
        y_score = pred_probs[:, i]

        # 检查当前类别的正样本和负样本是否都存在
        if np.sum(y_true) > 0 and np.sum(1 - y_true) > 0:
            roc_auc_value = roc_auc_score(y_true, y_score)
            roc_auc.append(roc_auc_value)
        else:
            # 如果某个类别的样本全是正样本或负样本，则无法计算 AUC，可以将其 AUC 设置为 0 或其他值
            roc_auc.append(0.5)  # 或者 np.nan

    if average == 'macro':
        return np.mean(roc_auc)
    else:
        return roc_auc




if __name__ == '__main__':
    main()
