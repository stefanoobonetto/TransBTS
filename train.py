# distributed configuration (not available on windows)
# python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train_spup3.py

import os
import time
import json
import torch
import random
import logging
import argparse
import torch.optim
import numpy as np
import setproctitle
from torch import nn
from data.BraTS import BraTS
from models import criterions
import torch.distributed as dist
from types import SimpleNamespace
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from utils.utils import adjust_learning_rate, log_args
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS

os.environ['USE_LIBUV'] = '0'

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return SimpleNamespace(**config_dict)  # allows dot notation access

args = load_config()

device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu' 
print('Using device:', device)

def main_worker():
    if args.local_rank == 0:    # only the main process will execute this (distributed configuration) to avoid I/O race conditions.
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment+args.date)
        mkdir = os.path.exists(log_dir)
        if not mkdir:
            os.makedirs(log_dir)
        log_file = log_dir + '.txt'
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))

    # set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(args.local_rank)

    _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")

    model.cuda(args.local_rank)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                # find_unused_parameters=True)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)


    model = model.to(device)

    criterion = getattr(criterions, args.criterion)

    if args.local_rank == 0:
        checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.experiment+args.date)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    resume = ''

    writer = SummaryWriter()

    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    train_list = os.path.join('train.txt')
    train_root = os.path.join('data\BraTS2020_TrainingData\\train\images')

    train_set = BraTS(train_list, train_root, args.mode)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_sampler = None
    logging.info('Samples for train = {}'.format(len(train_set)))

    num_gpu = (len(args.gpu)+1) // 2

    train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size // num_gpu,
                              drop_last=True, num_workers=args.num_workers, pin_memory=True)

    start_time = time.time()

    torch.set_grad_enabled(True)

    for epoch in range(args.start_epoch, args.end_epoch):
        # train_sampler.set_epoch(epoch)  # shuffle
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch+1, args.end_epoch))
        start_epoch = time.time()

        for i, data in enumerate(train_loader):

            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            x, target = data
            x = x.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)
            
            output = model(x)

            # loss1: Dice loss for label 1
            # loss2: Dice loss for label 2
            # loss3: Dice loss for label 4 (BraTS uses 1, 2, 4 for tumor subregions)
            loss, loss1, loss2, loss3 = criterion(output, target)
            reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
            reduce_loss1 = all_reduce_tensor(loss1, world_size=num_gpu).data.cpu().numpy()
            reduce_loss2 = all_reduce_tensor(loss2, world_size=num_gpu).data.cpu().numpy()
            reduce_loss3 = all_reduce_tensor(loss3, world_size=num_gpu).data.cpu().numpy()

            if args.local_rank == 0:
                logging.info('Epoch: {}_Iter:{}  loss: {:.5f} || 1:{:.4f} | 2:{:.4f} | 3:{:.4f} ||'
                             .format(epoch, i, reduce_loss, reduce_loss1, reduce_loss2, reduce_loss3))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_epoch = time.time()
        if args.local_rank == 0:
            if (epoch + 1) % int(args.save_freq) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 1) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 2) == 0 \
                    or (epoch + 1) % int(args.end_epoch - 3) == 0:
                file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)

            writer.add_scalar('lr:', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('loss:', reduce_loss, epoch)
            writer.add_scalar('loss1:', reduce_loss1, epoch)
            writer.add_scalar('loss2:', reduce_loss2, epoch)
            writer.add_scalar('loss3:', reduce_loss3, epoch)

        if args.local_rank == 0:
            epoch_time_minute = (end_epoch-start_epoch)/60
            remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60
            logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
            logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

    if args.local_rank == 0:
        writer.close()

        final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
        torch.save({
            'epoch': args.end_epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            final_name)
    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('Cuda available: {}'.format(torch.cuda.is_available()))
    print('Cuda version : ', torch.version.cuda)
    print('Cuda curret device: ', torch.cuda.current_device(), ' - name: ',torch.cuda.get_device_name(0))


    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()
