# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, Label_transfer

from reg import Regularization_LabelSpaceFocusing

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        features, outputs1, outputs2 = net(images)
        loss1 = loss_function1(outputs1, labels) 

        regularization = Regularization_LabelSpaceFocusing(features, labels)

        if epoch >= args.high_start:
            up_labels = label_transfer.transfer(labels)
            loss2 = loss_function2(outputs2, up_labels)
            loss = loss1 + args.high * loss2

        else:
            loss = loss1
            
        ent_loss = loss + REG * regularization
        ent_loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    finish = time.time()

    #print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    correct5 = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        features, outputs1, outputs2 = net(images)
        _, psudo_target = torch.max( outputs2, 1 )

        loss = loss_function2(outputs1, labels)

        test_loss += loss.item()
        _, preds = outputs1.max(1)
        correct += preds.eq(labels).sum()

        _, indices5 = outputs1.topk(5, dim = 1, largest=True, sorted=True)
        targets = labels.expand_as((indices5.T))
        total5 = (indices5.T).eq(targets).reshape(-1).float().sum()
        correct5 += total5

    finish = time.time()

    if epoch >= (args.high_start-1):
        label_transfer.Combine_Similar_Label(net, cifar100_training_loader)
        label_map = label_transfer.Get_Map()
        class_num = torch.unique(label_map).shape[0]

    else:
        class_num = 100

    #if args.gpu:
    #    print('GPU INFO.....')
    #    print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Top1 Accuracy: {:.4f}, Top5 Accuracy: {:.4f}, High class num: {}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        correct5.float() / len(cifar100_test_loader.dataset),
        class_num,
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Top1 Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Top5 Accuracy', correct5.float() / len(cifar100_test_loader.dataset), epoch)

        writer.add_scalar('Test/High class numer', class_num, epoch)

    return correct.float() / len(cifar100_test_loader.dataset), correct5.float() / len(cifar100_test_loader.dataset), class_num

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='resnet18_upper', help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-reg', type=float, default=0.0, help='reg parameter')
    parser.add_argument('-high_start', type=int, default=120, help='reg parameter')
    parser.add_argument('-high', type=float, default=0.1, help='reg parameter')
    parser.add_argument('-smth', type=float, default=0.1, help='smth parameter')
    parser.add_argument('-cnum', type=int, default=10, help='class num parameter')

    args = parser.parse_args()


    net = get_network(args)

    if args.gpu:
        net = net.cuda()


    MILESTONES = [60, 120, 160]
    EPOCH = 200
    
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    if os.path.exists('./Test_Acc_'+args.net+'_upper'+str(args.high)+'_smooth'+str(args.smth)+'_reg'+str(args.reg)+'.csv') == False:
        with open('./Test_Acc_'+args.net+'_upper'+str(args.high)+'_smooth'+str(args.smth)+'_reg'+str(args.reg)+'.csv', 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['name','top1 acc', 'top5 acc', 'high when/weight', 'upper class num'])

    REG = args.reg
    loss_function1 = nn.CrossEntropyLoss(label_smoothing = args.smth)
    loss_function2 = nn.CrossEntropyLoss()

    optimizer = optim.RAdam(net.parameters(), lr = args.lr)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, str(args.high_start)+'_'+str(args.high)+'_'+str(args.cnum), recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, str(args.high_start)+'_'+str(args.high)+'_'+str(args.cnum), settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, str(args.high_start)+'_'+str(args.high)+'_'+str(args.cnum), settings.TIME_NOW))


    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    best_acc5 = 0.0
    final_class_num = 0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, str(args.high_start)+'_'+str(args.high)+'_'+str(args.cnum), recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, str(args.high_start)+'_'+str(args.high)+'_'+str(args.cnum), recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc, class_num = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, str(args.high_start)+'_'+str(args.high)+'_'+str(args.cnum), recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, str(args.high_start)+'_'+str(args.high)+'_'+str(args.cnum), recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, str(args.high_start)+'_'+str(args.high)+'_'+str(args.cnum), recent_folder))

    label_transfer = Label_transfer(100, args.cnum)

    for epoch in range(1, EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc, top5acc, class_num = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            best_acc5 = top5acc
            final_class_num = class_num
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()


with open('./Test_Acc_'+args.net+'_upper'+str(args.high)+'_smooth'+str(args.smth)+'_reg'+str(args.reg)+'.csv', 'a') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([args.net, str(best_acc.item()), str(best_acc5.item()), str(args.high_start)+'_'+str(args.high)+'_'+str(args.cnum) , str(final_class_num)])
