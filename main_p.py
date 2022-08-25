from functions.functions import *
from layers import *
import argparse
from models.vggsnn import VGGSNNwoAP

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='TET + PLIF')
parser.add_argument('-j',
                    '--workers',
                    default=10,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')
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
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
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


if __name__ == '__main__':
    seed_all(args.seed)
    # train_dataset, val_dataset = data_loaders.build_dvscifar('cifar-dvs')
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                                            num_workers=args.workers, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
    #                                           shuffle=False, num_workers=args.workers, pin_memory=True)

    model = VGGSNNwoAP()

    parallel_model = torch.nn.DataParallel(model)
    parallel_model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    best_acc = 0
    best_epoch = 0

    logger = get_logger('exp.log')
    logger.info('start training!')

    for epoch in range(args.epochs):

        loss, acc = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, args.epochs, loss, acc))
        scheduler.step()
        facc = test(parallel_model, test_loader, device)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch, args.epochs, facc))

        if best_acc < facc:
            best_acc = facc
            best_epoch = epoch + 1
            # torch.save(parallel_model.module.state_dict(), 'VGGSNN_woAP.pth')
        logger.info('Best Test acc={:.3f}'.format(best_acc))
        print('\n')