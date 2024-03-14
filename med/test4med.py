import time
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.optim as optim
import os
import pickle
import argparse

import gnas
from models import model_cnn
from cnn_utils import evaluate_single, evaluate_single_best, evaluate_individual_list,evaluate_parents_individual_list, evaluate_transferred_individual_list, uptate_parents_individual_list, uptate_children_individual_list, uptate_transferred_individual_list
from data import get_dataset
from common import make_log_dir
from config import get_config, load_config, save_config
from modules.drop_module import DropModuleControl
from modules.cosine_annealing import CosineAnnealingLR

def main():

    parser = argparse.ArgumentParser(description='med')
    parser.add_argument('--Task_1', type=str, choices=['PathMNIST','OrganAMNIST','OrganCMNIST','OrganSMNIST'], help='the working data',
                        default='PathMNIST')
    parser.add_argument('--Task_2', type=str, choices=['PathMNIST','OrganAMNIST','OrganCMNIST','OrganSMNIST'], help='the working data',
                        default='OrganAMNIST')
    parser.add_argument('--Task_3', type=str, choices=['PathMNIST','OrganAMNIST','OrganCMNIST','OrganSMNIST'], help='the working data',
                        default='OrganCMNIST')
    parser.add_argument('--Task_4', type=str, choices=['PathMNIST','OrganAMNIST','OrganCMNIST','OrganSMNIST'], help='the working data',
                        default='OrganSMNIST')
    parser.add_argument('--config_file', type=str, help='location of the config file')
    parser.add_argument('--data_path', type=str, default='../dataset', help='location of the dataset')
    parser.add_argument('--log_dir', type=str, default='./logs/search-20240226-132315', help='log dir') # model path
    args = parser.parse_args()
    #######################################
    # Search Working Device
    #######################################
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    working_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(working_device)
    #######################################
    # Parameters
    #######################################
    config = get_config()
    if args.config_file is not None:
        print("Loading config file:" + args.config_file)
        config.update(load_config(args.config_file))
    config.update({'data_path': args.data_path, 'Task_1': args.Task_1, 'Task_2': args.Task_2, 'Task_3': args.Task_3, 'Task_4': args.Task_4, 'working_device': str(working_device)})
    print(config)
    ######################################
    # dataset
    ######################################
    testloader_1, n_param_1 = get_dataset(config, args.Task_1, is_test=True)
    testloader_2, n_param_2 = get_dataset(config, args.Task_2, is_test=True)
    testloader_3, n_param_3 = get_dataset(config, args.Task_3, is_test=True)
    testloader_4, n_param_4 = get_dataset(config, args.Task_4, is_test=True)
    ######################################
    # Config model and search space
    ######################################
    n_cell_type = gnas.SearchSpaceType(config.get('n_block_type') - 1)
    dp_control = DropModuleControl(config.get('drop_path_keep_prob'))
    ss = gnas.get_gnas_cnn_search_space(config.get('n_nodes'), dp_control, n_cell_type)

    net_1 = model_cnn.Net(config.get('n_blocks'), config.get('n_channels_1'), n_param_1, config.get('dropout'), ss).to(working_device)
    net_2 = model_cnn.Net(config.get('n_blocks'), config.get('n_channels_2'), n_param_2, config.get('dropout'), ss).to(working_device)
    net_3 = model_cnn.Net(config.get('n_blocks'), config.get('n_channels_3'), n_param_3, config.get('dropout'), ss).to(working_device)
    net_4 = model_cnn.Net(config.get('n_blocks'), config.get('n_channels_4'), n_param_4, config.get('dropout'), ss).to(working_device)
    ##################################################
    # Load Best Individuals
    ##################################################
    with open(os.path.join(args.log_dir, 'best_individual_1.pickle'), 'rb') as f:
        best_individual_1 = pickle.load(f)
    with open(os.path.join(args.log_dir, 'best_individual_2.pickle'), 'rb') as f:
        best_individual_2 = pickle.load(f)
    with open(os.path.join(args.log_dir, 'best_individual_3.pickle'), 'rb') as f:
        best_individual_3 = pickle.load(f)
    with open(os.path.join(args.log_dir, 'best_individual_4.pickle'), 'rb') as f:
        best_individual_4 = pickle.load(f)
    ##################################################
    # Test Accuracy
    ##################################################
    mean_1, best_1 = evaluate_single_best(best_individual_1, net_1, testloader_1, working_device, config.get('batch_size_val'), model_path=os.path.join(args.log_dir, 'best_model_1.pt'))
    mean_2, best_2 = evaluate_single_best(best_individual_2, net_2, testloader_2, working_device, config.get('batch_size_val'), model_path=os.path.join(args.log_dir, 'best_model_2.pt'))
    mean_3, best_3 = evaluate_single_best(best_individual_3, net_3, testloader_3, working_device, config.get('batch_size_val'), model_path=os.path.join(args.log_dir, 'best_model_3.pt'))
    mean_4, best_4 = evaluate_single_best(best_individual_4, net_4, testloader_4, working_device, config.get('batch_size_val'), model_path=os.path.join(args.log_dir, 'best_model_4.pt'))

    print('{}  Mean Acc:{:2.3f}%, Best Acc:{:2.3f}%'.format(args.Task_1, mean_1, best_1))
    print('{}  Mean Acc:{:2.3f}%, Best Acc:{:2.3f}%'.format(args.Task_2, mean_2, best_2))
    print('{}  Mean Acc:{:2.3f}%, Best Acc:{:2.3f}%'.format(args.Task_3, mean_3, best_3))
    print('{}  Mean Acc:{:2.3f}%, Best Acc:{:2.3f}%'.format(args.Task_4, mean_4, best_4))
    ##################################################
    # Search Cost
    ##################################################
    from gnas.common.result import ResultAppender
    ra=ResultAppender.load_result(args.log_dir)
    res_dict=ra.result_dict
    print('Search Cost:{:.5f} GPU Day'.format(res_dict['Time Sum'][-1] / 60 / 24))
        
if __name__ == '__main__':
    main()
