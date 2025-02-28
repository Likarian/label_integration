import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

def get_network(args):
    """ return given network
    """

    if args.net == 'vgg11_upper':
        from models.vgg_upper import vgg11_bn_upper
        net = vgg11_bn_upper()
    elif args.net == 'vgg13_upper':
        from models.vgg_upper import vgg13_bn_upper
        net = vgg13_bn_upper()
    elif args.net == 'vgg16_upper':
        from models.vgg_upper import vgg16_bn_upper
        net = vgg16_bn_upper()
    elif args.net == 'vgg19_upper':
        from models.vgg_upper import vgg19_bn_upper
        net = vgg19_bn_upper()
    elif args.net == 'resnet18_upper':
        from models.resnet_upper import resnet18_upper
        net = resnet18_upper()
    elif args.net == 'resnet34_upper':
        from models.resnet_upper import resnet34_upper
        net = resnet34_upper()
    elif args.net == 'resnet50_upper':
        from models.resnet_upper import resnet50_upper
        net = resnet50_upper()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]



class Label_transfer(object):
    def __init__(self, init_num_classes, combine_num):
        self.init_num_classes = init_num_classes
        self.combine_num = combine_num
        self.label_transfer_map = torch.tensor( range(self.init_num_classes) )
    
    def reset_map(self):
        self.label_transfer_map = torch.tensor( range(self.init_num_classes) )

    def Combine_Similar_Label(self, network, training_loader):
        network.eval()

        for batch_index, (images, labels) in enumerate(training_loader):

            labels = labels.cuda()
            images = images.cuda()

            feature, outputs1, outputs2 = network(images)

            if batch_index == 0:
                feature_bank = feature
                label_bank = labels
            else:
                feature_bank = torch.cat( (feature_bank, feature), 0)
                label_bank = torch.cat( (label_bank, labels), 0)
        
        median_vector_bank = self.Get_Median_Vector(feature_bank, label_bank)
        similaritry_map = self.Get_Similarity_Map(median_vector_bank)
        self.Change_Transfer_Map(similaritry_map)

    def Get_Median_Vector(self, features, targets):
        # features (A, feature)
        # targets (A, )
        output = torch.unique(targets)
        median_vector_bank = torch.empty( output.shape[0], features.shape[1] ).cuda()


        for i in range(output.shape[0]):

            # divided the targets per label
            extracted = torch.where( targets==output[i] )[0]

            # find a median vector per label and save it in bank
            extracted_features = features[extracted,:]
            extracted_features_median = torch.median(extracted_features, 0)[0]
            median_vector_bank[output[i],:] = extracted_features_median

        return median_vector_bank

    def Get_Similarity_Map(self, median_vector_bank):
        similaritry_map = torch.zeros( median_vector_bank.shape[0], self.combine_num )

        for i in range(median_vector_bank.shape[0]):

            current_median_vector = median_vector_bank[i:i+1,:]
            # calculate a cosine similarity
            cosine_similarity_between_labels = F.cosine_similarity(current_median_vector,median_vector_bank, dim=1)
            _, indices = torch.topk(cosine_similarity_between_labels, self.combine_num+1)
            similaritry_map[i,:] = indices[1:]

        return similaritry_map

    def Change_Transfer_Map(self, similaritry_map):
        self.reset_map()
        for i in range( similaritry_map.shape[0] ):
            current_indice = similaritry_map[i,:]

            for j in range(current_indice.shape[0]):
                target_label = current_indice[j]
                #print(similaritry_map.shape, current_indice.shape, target_label)
                if int(target_label.item()) > i:
                    target_indice = similaritry_map[int(target_label.item()),:]
                    if torch.sum( target_indice == i ) > 0:
                        self.label_transfer_map[int(target_label.item())] = i

    def Get_Map(self):
        return self.label_transfer_map

    def transfer(self, target):
        new_target = target.clone().detach()
        for i in range(target.shape[0]):
            new_target[i] = self.label_transfer_map[target[i]]
        return new_target