import torch
import torch.cuda
from gnas.genetic_algorithm.population_dict import PopulationDict


def evaluate_single(input_individual, input_model, data_loader, device):
    correct = 0
    total = 0
    input_model = input_model.eval()
    input_model.set_individual(input_individual)
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = input_model(images)
            labels = labels.squeeze().long()
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def evaluate_single_best(input_individual, input_model, data_loader, device, batch_size, model_path):
    correct = 0
    total = 0
    best = 0
    input_model = input_model.eval()
    input_model.set_individual(input_individual)
    input_model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda'))) #load weights
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = input_model(images)
            labels = labels.squeeze().long()
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            fmax = (predicted == labels).sum().item()
            if fmax > best:
                best = fmax
            correct += fmax
    return 100 * correct / total, 100 * best / batch_size

def evaluate_individual_list(input_individual_list, ga, input_model, data_loader, device, num):
    correct = 0
    total = 0
    input_model = input_model.eval()
    i = 0
    with torch.no_grad():
        while len(input_individual_list) > i:
            for data in data_loader:
                if len(input_individual_list) <= i:
                    pass
                else:
                    ind = input_individual_list[i]
                    input_model.set_individual(ind)
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = input_model(images)
                    labels = labels.squeeze().long()
                    _, predicted = torch.max(outputs[0].data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    acc = 100 * correct / total
                    ga.update_current_individual_fitness(ind, acc, num)
                    # print(i,' completed')
                    i += 1

def evaluate_parents_individual_list(input_individual_list, ga, input_model, data_loader, device, num):
    correct = 0
    total = 0
    input_model = input_model.eval()
    i = 0
    with torch.no_grad():
        while len(input_individual_list) > i:
            for data in data_loader:
                if len(input_individual_list) <= i:
                    pass
                else:
                    ind = input_individual_list[i]
                    input_model.set_individual(ind)
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = input_model(images)
                    labels = labels.squeeze().long()
                    _, predicted = torch.max(outputs[0].data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    acc = 100 * correct / total
                    ga.update_max_individual_fitness(ind, acc, num)
                    i += 1

def evaluate_transferred_individual_list(input_individual_list, ga, input_model, data_loader, device, num):
    correct = 0
    total = 0
    input_model = input_model.eval()
    i = 0
    with torch.no_grad():
        while len(input_individual_list) > i:
            for data in data_loader:
                if len(input_individual_list) <= i:
                    pass
                else:
                    ind = input_individual_list[i]
                    input_model.set_individual(ind)
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = input_model(images)
                    labels = labels.squeeze().long()
                    _, predicted = torch.max(outputs[0].data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    acc = 100 * correct / total
                    ga.update_tranferred_individual_fitness(ind, acc, num)
                    i += 1

def uptate_parents_individual_list(input_individual_list, ga, num): # Set 0
    i = 0
    while len(input_individual_list) > i:
        ind = input_individual_list[i]
        if num == 1:
            ga.max_dict_1.update_2({ind: 0})
        elif num == 2:
            ga.max_dict_2.update_2({ind: 0})
        i += 1

def uptate_children_individual_list(input_individual_list, ga, num): # Set 0
    i = 0
    while len(input_individual_list) > i:
        ind = input_individual_list[i]
        if num == 1:
            ga.current_dict_1.update({ind: 0})
        elif num == 2:
            ga.current_dict_2.update({ind: 0})
        i += 1

def uptate_transferred_individual_list(input_individual_list, ga, num): # Set 0
    if num == 1:
        ga.transferred_dict_1 = PopulationDict()
    elif num == 2:
        ga.transferred_dict_2 = PopulationDict()

    i = 0
    while len(input_individual_list) > i:
        ind = input_individual_list[i]
        if num == 1:
            ga.transferred_dict_1.update({ind: 0})
        elif num == 2:
            ga.transferred_dict_2.update({ind: 0})
        i += 1
