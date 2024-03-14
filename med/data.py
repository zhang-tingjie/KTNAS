import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import medmnist
from medmnist import INFO


def get_dataset(config, dataset_name, is_test=False, is_all=False):
   
    data_path = config.get('data_path')

    if is_test:
        if dataset_name == 'PathMNIST':
            return get_medmnist_test(config, os.path.join(data_path, 'PathMNIST'))
        elif dataset_name == 'OrganAMNIST':
            return get_medmnist_test(config, os.path.join(data_path, 'OrganAMNIST'), dataset_name='OrganAMNIST')
        elif dataset_name == 'OrganCMNIST':
            return get_medmnist_test(config, os.path.join(data_path, 'OrganCMNIST'), dataset_name='OrganCMNIST')
        elif dataset_name == 'OrganSMNIST':
            return get_medmnist_test(config, os.path.join(data_path, 'OrganSMNIST'), dataset_name='OrganSMNIST')

    elif is_all:
        if dataset_name == 'PathMNIST':
            return get_medmnist_all(config, os.path.join(data_path, 'PathMNIST'))
        elif dataset_name == 'OrganAMNIST':
            return get_medmnist_all(config, os.path.join(data_path, 'OrganAMNIST'), dataset_name='OrganAMNIST')
        elif dataset_name == 'OrganCMNIST':
            return get_medmnist_all(config, os.path.join(data_path, 'OrganCMNIST'), dataset_name='OrganCMNIST')
        elif dataset_name == 'OrganSMNIST':
            return get_medmnist_all(config, os.path.join(data_path, 'OrganSMNIST'), dataset_name='OrganSMNIST')
    
    else:
        if dataset_name == 'CIFAR10':
            return get_cifar(config, os.path.join(data_path, 'CIFAR10'))
        elif dataset_name == 'CIFAR100':
            return get_cifar(config, os.path.join(data_path, 'CIFAR100'), dataset_name='CIFAR100')
        
        elif dataset_name == 'PathMNIST':
            return get_medmnist(config, os.path.join(data_path, 'PathMNIST'))
        elif dataset_name == 'OrganAMNIST':
            return get_medmnist(config, os.path.join(data_path, 'OrganAMNIST'), dataset_name='OrganAMNIST')
        elif dataset_name == 'OrganCMNIST':
            return get_medmnist(config, os.path.join(data_path, 'OrganCMNIST'), dataset_name='OrganCMNIST')
        elif dataset_name == 'OrganSMNIST':
            return get_medmnist(config, os.path.join(data_path, 'OrganSMNIST'), dataset_name='OrganSMNIST')
        
        else:
            raise Exception('unkown dataset type')


def get_cifar(config, data_path, dataset_name='CIFAR10'):
    train_transform = transforms.Compose([])
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    if dataset_name == 'CIFAR10':
        n_class = 10
        trainset_train = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                                download=True, transform=train_transform)

        trainset_val = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                                      download=True, transform=train_transform)
    
        index = {}
        i=0
        for value in trainset_train.targets:
            index.setdefault(value, []).append(i)
            i = i+1

        num_1 = 1000 # 验证集上每个类别的样本数
        num_temp = (i//n_class)//num_1
        num_end = (i//n_class)-1
        num = [i for i in range(0,num_end,num_temp)]
        train_index = []
        val_index = []
        for value in index.values():
            train_index.extend([value[j] for j in list(set(i for i in range(len(value))) - set(num))])
            val_index.extend([value[i] for i in num])

        train_index.sort()
        val_index.sort()
     
        for i in reversed(val_index):
            del trainset_train.targets[i]
        
        trainset_train.data = np.delete(trainset_train.data, val_index, axis=0)

       
        for i in reversed(train_index):
            del trainset_val.targets[i]
        
        trainset_val.data = np.delete(trainset_val.data, train_index, axis=0)


        train_train_loader = torch.utils.data.DataLoader(trainset_train, batch_size=config.get('batch_size'), shuffle=True, num_workers=8)

        train_val_loader = torch.utils.data.DataLoader(trainset_val, batch_size=config.get('batch_size_val'), shuffle=True, num_workers=8)


    elif dataset_name == 'CIFAR100':
        n_class = 100

        trainset_train = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                 download=True, transform=train_transform)
        trainset_val = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                       download=True, transform=train_transform)
        index = {}
        i = 0
        for value in trainset_train.targets:
            index.setdefault(value, []).append(i)
            i = i + 1

        num_1 = 100
        num_temp = (i // n_class) // num_1
        num_end = (i // n_class) - 1
        num = [i for i in range(0, num_end, num_temp)]
        train_index = []
        val_index = []
        for value in index.values():
            train_index.extend([value[j] for j in list(set(i for i in range(len(value))) - set(num))])
            val_index.extend([value[i] for i in num])

        train_index.sort()
        val_index.sort()

        for i in reversed(val_index):
            del trainset_train.targets[i]

        trainset_train.data = np.delete(trainset_train.data, val_index, axis=0)


        for i in reversed(train_index):
            del trainset_val.targets[i]

        trainset_val.data = np.delete(trainset_val.data, train_index, axis=0)

        train_train_loader = torch.utils.data.DataLoader(trainset_train, batch_size=config.get('batch_size'), shuffle=True,
                                                         num_workers=8)
        train_val_loader = torch.utils.data.DataLoader(trainset_val, batch_size=config.get('batch_size_val'), shuffle=True,
                                                       num_workers=8)


    else:
        raise Exception('unkown dataset' + dataset_name)

 
    return train_train_loader, train_val_loader, n_class


def get_medmnist(config, data_path, dataset_name='PathMNIST'):
    info = INFO[dataset_name.lower()]
    DataClass = getattr(medmnist, info['python_class'])

    train_transform = transforms.Compose([])
    normalize = transforms.Normalize(mean=[.5], std=[.5])
    train_transform.transforms.append(transforms.RandomCrop(28, padding=2))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    if dataset_name == 'PathMNIST':
        n_class = len(info['label'])
        trainset_train = DataClass(root=data_path,split='train',
                                                download=True, transform=train_transform)

        trainset_val = DataClass(root=data_path,split='val',
                                                download=True, transform=train_transform)

        train_train_loader = torch.utils.data.DataLoader(trainset_train, batch_size=config.get('batch_size'), shuffle=True, num_workers=8)

        train_val_loader = torch.utils.data.DataLoader(trainset_val, batch_size=config.get('batch_size_val'), shuffle=True, num_workers=8)

    elif dataset_name in ['OrganAMNIST','OrganCMNIST','OrganSMNIST']:
        n_class = len(info['label'])
        trainset_train = DataClass(root=data_path,split='train',
                                                download=True, transform=train_transform, as_rgb=True)

        trainset_val = DataClass(root=data_path,split='val',
                                                download=True, transform=train_transform, as_rgb=True)

        train_train_loader = torch.utils.data.DataLoader(trainset_train, batch_size=config.get('batch_size'), shuffle=True, num_workers=8)

        train_val_loader = torch.utils.data.DataLoader(trainset_val, batch_size=config.get('batch_size_val'), shuffle=True, num_workers=8)

    else:
        raise Exception('unkown dataset' + dataset_name)

 
    return train_train_loader, train_val_loader, n_class


def get_medmnist_test(config, data_path, dataset_name='PathMNIST'):
    info = INFO[dataset_name.lower()]
    DataClass = getattr(medmnist, info['python_class'])

    train_transform = transforms.Compose([])
    normalize = transforms.Normalize(mean=[.5], std=[.5])
    train_transform.transforms.append(transforms.RandomCrop(28, padding=2))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    if dataset_name == 'PathMNIST':
        n_class = len(info['label'])
        test_data = DataClass(root=data_path,split='test',
                                                download=True, transform=train_transform)

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.get('batch_size_val'), shuffle=True, num_workers=8)

    elif dataset_name in ['OrganAMNIST','OrganCMNIST','OrganSMNIST']:
        n_class = len(info['label'])
        test_data = DataClass(root=data_path,split='test',
                                                download=True, transform=train_transform, as_rgb=True)

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.get('batch_size_val'), shuffle=True, num_workers=8)

    else:
        raise Exception('unkown dataset' + dataset_name)

 
    return test_loader, n_class


def get_medmnist_all(config, data_path, dataset_name='PathMNIST'):
    info = INFO[dataset_name.lower()]
    DataClass = getattr(medmnist, info['python_class'])

    train_transform = transforms.Compose([])
    normalize = transforms.Normalize(mean=[.5], std=[.5])
    train_transform.transforms.append(transforms.RandomCrop(28, padding=2))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    if dataset_name == 'PathMNIST':
        n_class = len(info['label'])
        trainset = DataClass(root=data_path,split='train',
                                                download=True, transform=train_transform)

        valset = DataClass(root=data_path,split='val',
                                                download=True, transform=train_transform)
        
        testset = DataClass(root=data_path,split='test',
                                                download=True, transform=train_transform)

        train_valset = torch.utils.data.ConcatDataset([trainset,valset])

        train_loader = torch.utils.data.DataLoader(train_valset, batch_size=config.get('batch_size'), shuffle=True, num_workers=8)

        val_loader = torch.utils.data.DataLoader(testset, batch_size=config.get('batch_size_test'), shuffle=True, num_workers=8)

    elif dataset_name in ['OrganAMNIST','OrganCMNIST','OrganSMNIST']:
        n_class = len(info['label'])
        trainset = DataClass(root=data_path,split='train',
                                                download=True, transform=train_transform, as_rgb=True)

        valset = DataClass(root=data_path,split='val',
                                                download=True, transform=train_transform, as_rgb=True)
        
        testset = DataClass(root=data_path,split='test',
                                                download=True, transform=train_transform, as_rgb=True)

        train_valset = torch.utils.data.ConcatDataset([trainset,valset])

        train_loader = torch.utils.data.DataLoader(train_valset, batch_size=config.get('batch_size'), shuffle=True, num_workers=8)

        val_loader = torch.utils.data.DataLoader(testset, batch_size=config.get('batch_size_test'), shuffle=True, num_workers=8)

    else:
        raise Exception('unkown dataset' + dataset_name)

 
    return train_loader, val_loader, n_class