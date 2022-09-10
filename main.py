import argparse
import shutil
import os
import time
import warnings
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from models.resnet import resnet19
from models.vggsnn import VGGSNNwoAP
from models.m2spike import spikem2
from models.test import *
from dataloaders.cifar10 import build_cifar10
from dataloaders.cifar100 import build_cifar100
from dataloaders.cifar10_dvs import build_cifar10dvs
from dataloaders.mnist import build_mnist
from dataloaders.imagenet import build_imagenet
from functions.functions import seed_all
from functions.loss import TET_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('--Dname',
                    type = str, 
                    help = 'CIFAR10, CIFAR100, CIFAR10DVS, MNIST, IMAGENET')

parser.add_argument('--DataDownload',
                    default = True,
                    type = bool)

parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b',
                    '--batch-size',
                    default=16,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when ')

parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')

parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')

parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--seed',
                    default=1000,
                    type=int,
                    help='seed for initializing training. ')

parser.add_argument('--T',
                    default=2,
                    type=int,
                    metavar='N',
                    help='snn simulation time (default: 2)')

parser.add_argument('--means',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='make all the potential increment around the means (default: 1.0)')

parser.add_argument('--TET',
                    default=True,
                    type=bool,
                    metavar='N',
                    help='if use Temporal Efficient Training (default: True)')

parser.add_argument('--lamb',
                    default=0.0,
                    type=float,
                    metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')

args = parser.parse_args()

def main():
    if args.seed is not None:
        seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    best_acc1 = .0

    load_names = None
    save_names = None

    # load_names = 'dvs-cifar1048_V1_D05_VGGSNN_distribute_ensemble_gama05.pth'
    # save_names = 'dvs-cifar1048_V1_D05_VGG11SNN_distribute_ensemble_gamad.pth'
    Dname = args.Dname 
    if Dname == "CIFAR10DVS":
        model = ...
    else:
        model = MobileNetV2().to(device) #resnet19().to(device)
    
    model.T = args.T

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    if args.Dname == 'CIFAR10':
        build_data = build_cifar10(args.DataDownload)
    elif args.Dname == 'CIFAR100':
        build_data = build_cifar100(args.DataDownload)
    elif args.Dname == 'CIFAR10DVS':
        build_data = build_cifar10dvs(args.DataDownload)
    elif args.Dname == 'MNIST':
        build_data = build_mnist(args.DataDownload)
    elif args.Dname == 'IMAGENET':
        build_data = build_imagenet(args.DataDownload) 
       
    # Data loading code
    train_dataset, val_dataset = build_data
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             pin_memory=True)

    
    for t in range(args.start_epoch, args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, criterion, optimizer, t)
        validate(val_loader, model, criterion)
    print("Done!")
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model")

#다시 봐야함(이게 없어도 돌아감 이부분 살려서 local_rank 지우고 돌려도 결과는 똑같게 돌아가고 나옴)
    # for epoch in range(args.start_epoch, args.epochs):
    #     t1 = time.time()
 
    #     # train for one epoch
    #     train(train_loader, model, criterion, optimizer, epoch, local_rank,
    #           args)

    #     # evaluate on validation set
    #     acc1 = validate(val_loader, model, criterion, local_rank, args)

    #     scheduler.step()
    #     # remember best acc@1 and save checkpoint
    #     is_best = acc1 > best_acc1
    #     best_acc1 = max(acc1, best_acc1)
    #     t2 = time.time()
    #     print('Time elapsed: ', t2 - t1)
    #     print('Best top-1 Acc: ', best_acc1)
    #     if is_best and save_names != None:
    #         if args.local_rank == 0:
    #             torch.save(model.module.state_dict(), save_names)
    #     # save_checkpoint(
    #     #     {
    #     #         'epoch': epoch + 1,
    #     #         'arch': args.arch,
    #     #         'state_dict': model.module.state_dict(),
    #     #         'best_acc1': best_acc1,
    #     #     }, is_best)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    #size = len(train_loader)
    for batch, (X, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        #X=input
        X, target = X.to(device), target.to(device)

        # Compute prediction error
        output = model(X)
        mean_out = torch.mean(output, dim=1)
        if not args.TET:
            loss = criterion(mean_out, target)
        else:
            loss = TET_loss(output, target, criterion, args.means, args.lamb)
            
        # measure accuracy and record loss
        acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))
        
        losses.update(loss.item(), X.size(0))
        top1.update(acc1[0], X.size(0))
        top5.update(acc5[0], X.size(0))
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
                # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch % args.print_freq == 0:
            progress.display(batch)

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e') 
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')
    
    #size = len(val_loader)
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for batch, (X, target) in enumerate(val_loader):
            #X=input
            X, target = X.to(device), target.to(device)

            # compute output
            output = model(X)
            mean_out = torch.mean(output, dim=1)
            loss = criterion(mean_out, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))
            
            losses.update(loss.item(), X.size(0))
            top1.update(acc1[0], X.size(0))
            top5.update(acc5[0], X.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch % args.print_freq == 0:
                progress.display(batch)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))
    return top1.avg

    
    #test_loss, correct = 0, 0
    # with torch.no_grad():
    #     for X, y in val_loader:
    #         X, y = X.to(device), y.to(device)
    #         pred = model(X)
    #         test_loss += criterion(pred, y).item()
    #     if not args.TET:
    #         loss = criterion(mean_out, target)
    #     else:
    #         loss = TET_loss(output, target, criterion, args.means, args.lamb)
    #         correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # test_loss /= size
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") 





# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))   

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()