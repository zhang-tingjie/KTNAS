import sys
sys.path.append('/home/ztj/codes/TREMT-NAS4Med/')


import numpy as np
from random import choices
from gnas.search_space.search_space import SearchSpace
from gnas.search_space.cross_over import individual_uniform_crossover, individual_block_crossover
from gnas.search_space.mutation import individual_flip_mutation
from gnas.genetic_algorithm.ga_results import GenetricResult
from gnas.genetic_algorithm.population_dict import PopulationDict
import random
import math
from collections import OrderedDict
from transfer_rank import select_transferred_individuals
import operator

# modules for test
from gnas.search_space.factory import get_gnas_cnn_search_space, SearchSpaceType
from modules.drop_module import DropModuleControl
from config import get_config
from cnn_utils import uptate_parents_individual_list
import warnings
warnings.filterwarnings("ignore")


def genetic_algorithm_searcher(search_space: SearchSpace, generation_size=20, population_size=20, keep_size=0,
                               min_objective=False, mutation_p=None, p_cross_over=None, n_epochs=300, 
                               n_transferred_individual=2, n_generation_saved_negative=3, ratio_elite_individual=0.1):

    def population_initializer(p_size):
        return search_space.generate_population(p_size)

    def mutation_function(x):
        return individual_flip_mutation(x, mutation_p)

    def cross_over_function(x0, x1):
        return individual_block_crossover(x0, x1, p_cross_over)

    def selection_function(p, n_transfer):\
        # binary tournament selection
        new_population_idx = []
        transfer_idx = list(range(-n_transfer,0))
        for _ in range(len(p)):
            candidates_idx = np.random.choice(list(range(len(p))),2,replace=False)
            is_transfer_1, is_transfer_2 = False, False
            for idx in transfer_idx:
                if candidates_idx[0]==idx:
                    is_transfer_1 = True
                if candidates_idx[1]==idx:
                    is_transfer_2 = True
            
            if not is_transfer_1 and not is_transfer_2:
                candidates_fitness = p[candidates_idx]
                winner_idx = np.argmax(candidates_fitness)
            elif is_transfer_1 and not is_transfer_2:
                winner_idx = candidates_idx[0]
            elif not is_transfer_1 and is_transfer_2:
                winner_idx = candidates_idx[1]
            elif is_transfer_1 and is_transfer_2:
                winner_idx = np.random.choice(candidates_idx)
            
            new_population_idx.append(winner_idx)
        return np.reshape(np.asarray(new_population_idx),[-1,2])

    return GeneticAlgorithms(population_initializer, mutation_function, cross_over_function, selection_function,
                             min_objective=min_objective, generation_size=generation_size,
                             population_size=population_size, keep_size=keep_size,n_epochs=n_epochs, 
                             n_transferred_individual=n_transferred_individual, n_generation_saved_negative=n_generation_saved_negative, ratio_elite_individual=ratio_elite_individual)


class GeneticAlgorithms(object):
    def __init__(self, population_initializer, mutation_function, cross_over_function, selection_function,
                 population_size=20, generation_size=20, keep_size=20, min_objective=False, n_epochs=300, 
                 n_transferred_individual=2, n_generation_saved_negative=3, ratio_elite_individual=0.1):
        ####################################################################
        # Functions
        ####################################################################
        self.population_initializer = population_initializer
        self.mutation_function = mutation_function
        self.cross_over_function = cross_over_function
        self.selection_function = selection_function
        self.n_epochs = n_epochs
        ####################################################################
        # parameters
        ####################################################################
        self.population_size = population_size
        self.generation_size = generation_size
        self.keep_size = keep_size
        self.min_objective = min_objective
        # self.RMP = RMP
        ####################################################################
        # status
        ####################################################################
        self.max_dict_1 = PopulationDict()
        self.ga_result_1 = GenetricResult()
        self.current_dict_1 = dict()
        self.new_current_dict_1 = dict()
        self.generation_1 = self._create_random_generation() # 父代和子代个体都暂存在这里，然后添加到max_dict或current_dict中
        self.i = 0
        self.best_individual_1 = None
        self.avg_individual_1_fitness = None

        self.max_dict_2 = PopulationDict()
        self.ga_result_2 = GenetricResult()
        self.current_dict_2 = dict()
        self.new_current_dict_2 = dict()
        self.generation_2 = self._create_random_generation()
        self.best_individual_2 = None
        self.avg_individual_2_fitness = None
        
        self.max_dict_3 = PopulationDict()
        self.ga_result_3 = GenetricResult()
        self.current_dict_3 = dict()
        self.new_current_dict_3 = dict()
        self.generation_3 = self._create_random_generation()
        self.best_individual_3 = None
        self.avg_individual_3_fitness = None
        
        self.max_dict_4 = PopulationDict()
        self.ga_result_4 = GenetricResult()
        self.current_dict_4 = dict()
        self.new_current_dict_4 = dict()
        self.generation_4 = self._create_random_generation()
        self.best_individual_4 = None
        self.avg_individual_4_fitness = None

        self.old_current_dict_1 = dict() # 存储历史出现过的父代个体和适应值
        self.old_current_dict_2 = dict()
        self.old_current_dict_3 = dict()
        self.old_current_dict_4 = dict()
        self.mapping_1 = dict() # 存储子代：父代的映射
        self.mapping_2 = dict()
        self.mapping_3 = dict()
        self.mapping_4 = dict()
        ####################################################################
        # transferred population
        ####################################################################
        self.n_transferred_individual = n_transferred_individual
        self.n_generation_saved_negative = n_generation_saved_negative

        self.transferred_dict_1 = PopulationDict() # 生成HTS用到
        self.transferred_generation_1 = self._create_random_transferred_generation(1) # 迁移个体暂存在这里，然后添加到transferred_dict中
        self.historical_transferred_set_1 = dict()
        self.positive_class_set_1 = dict()
        self.negtive_class_set_1 = dict()
        self.historcial_negtive_class_set_1 = OrderedDict()
        self.negtive_class_set_size_1 = [] # 记录每代负迁移的个体数

        self.transferred_dict_2 = PopulationDict()
        self.transferred_generation_2 = self._create_random_transferred_generation(2)
        self.historical_transferred_set_2 = dict()
        self.positive_class_set_2 = dict()
        self.negtive_class_set_2 = dict()
        self.historcial_negtive_class_set_2 = OrderedDict()
        self.negtive_class_set_size_2 = []

        self.transferred_dict_3 = PopulationDict()
        self.transferred_generation_3 = self._create_random_transferred_generation(3)
        self.historical_transferred_set_3 = dict()
        self.positive_class_set_3 = dict()
        self.negtive_class_set_3 = dict()
        self.historcial_negtive_class_set_3 = OrderedDict()
        self.negtive_class_set_size_3 = []
        
        self.transferred_dict_4 = PopulationDict()
        self.transferred_generation_4 = self._create_random_transferred_generation(4)
        self.historical_transferred_set_4 = dict()
        self.positive_class_set_4 = dict()
        self.negtive_class_set_4 = dict()
        self.historcial_negtive_class_set_4 = OrderedDict()
        self.negtive_class_set_size_4 = []
        
        self.ratio_elite_individual = ratio_elite_individual # 父代种群中精英个体的占比

    def _create_random_generation(self):
        return self.population_initializer(self.generation_size)

    def _create_new_generation_old(self):
        # discard
        self.mapping_1 = dict()
        self.mapping_2 = dict()
        population_fitness_1 = np.asarray(
            list(self.max_dict_1.values())).flatten()
        population_1 = np.asarray(list(self.max_dict_1.keys())).flatten()
        population_fitness_2 = np.asarray(
            list(self.max_dict_2.values())).flatten()
        population_2 = np.asarray(list(self.max_dict_2.keys())).flatten()
        # print('type of individual:', type(population_1[0]))
        # print('individual_1 of population_1:', str(population_1[0]))
        # print('individual_1 of population_2:', population_2[0])
        # print('shape of population_1:', population_1.shape,
        #       ',shape of population_2:', population_2.shape)

        p_1 = population_fitness_1 / np.nansum(population_fitness_1)
        p_2 = population_fitness_2 / np.nansum(population_fitness_2)
        # print('shape of p_1:', p_1.shape, ',shape of p_2:', p_2.shape)
        p = np.hstack((p_1, p_2))

        population = np.hstack((population_1, population_2))

        if self.min_objective:
            p = 1 - p
        # print('shape of p:', p.shape)
        couples = self.selection_function(p)  # select by fitness of individuals
        # print('shape of couples:', couples.shape)

        child_1 = []
        child_2 = []

        for c in couples: # c[0],c[1]分别代表task1,2的个体的基因型索引。前10为task1，后10为task2。
            if (len(child_1) >= self.population_size) & (len(child_2) >= self.population_size):
                break
            # same task? Yes.
            elif (c[0] < self.population_size) & (c[1] < self.population_size) & (len(child_1)+2 <= self.population_size):
                child_p_1, child_p_2 = self.cross_over_function(
                    population[c[0]], population[c[1]])  # cross-over
                new_child_p_1 = self.mutation_function(child_p_1)
                new_child_p_2 = self.mutation_function(child_p_2)

                child_1 = np.hstack((child_1, new_child_p_1))
                child_1 = np.hstack((child_1, new_child_p_2))
                self.mapping_1.update({new_child_p_1: population[c[0]]})
                self.mapping_1.update({new_child_p_2: population[c[1]]})
            elif (c[0] >= self.population_size) & (c[1] >= self.population_size) & (len(child_2)+2 <= self.population_size):
                child_p_1, child_p_2 = self.cross_over_function(
                    population[c[0]], population[c[1]])  # cross-over
                new_child_p_1 = self.mutation_function(child_p_1)
                new_child_p_2 = self.mutation_function(child_p_2)

                child_2 = np.hstack((child_2, new_child_p_1))
                child_2 = np.hstack((child_2, new_child_p_2))
                self.mapping_2.update({new_child_p_1: population[c[0]]})
                self.mapping_2.update({new_child_p_2: population[c[1]]})
            # same task? No.
            elif ((c[0] < self.population_size) & (c[1] >= self.population_size)) | ((c[0] >= self.population_size) & (c[1] < self.population_size)):
                # arch transfer? Yes. cross_over and mutation on two tasks arch.
                if random.random() < self.RMP:
                    child_p_1, child_p_2 = self.cross_over_function(
                        population[c[0]], population[c[1]])  # cross-over
                    new_child_p_1 = self.mutation_function(child_p_1)
                    new_child_p_2 = self.mutation_function(child_p_2)
                else:  # arch transfer? No. only mutation on two tasks arch, respectively.
                    new_child_p_1 = self.mutation_function(population[c[0]])
                    new_child_p_2 = self.mutation_function(population[c[1]])

                if random.random() < 0.5:
                    if len(child_1) + 1 <= self.population_size:
                        child_1 = np.hstack((child_1, new_child_p_1))
                        if c[0] < self.population_size:
                            self.mapping_1.update(
                                {new_child_p_1: population[c[0]]})
                        if c[1] < self.population_size:
                            self.mapping_1.update(
                                {new_child_p_1: population[c[1]]})

                    if len(child_2) + 1 <= self.population_size:
                        child_2 = np.hstack((child_2, new_child_p_2))
                        if c[0] >= self.population_size:
                            self.mapping_2.update(
                                {new_child_p_2: population[c[0]]})
                        if c[1] >= self.population_size:
                            self.mapping_2.update(
                                {new_child_p_2: population[c[1]]})
                else:
                    if len(child_1) + 1 <= self.population_size:
                        child_1 = np.hstack((child_1, new_child_p_2))
                        if c[0] < self.population_size:
                            self.mapping_1.update(
                                {new_child_p_2: population[c[0]]})
                        if c[1] < self.population_size:
                            self.mapping_1.update(
                                {new_child_p_2: population[c[1]]})

                    if len(child_2) + 1 <= self.population_size:
                        child_2 = np.hstack((child_2, new_child_p_1))
                        if c[0] >= self.population_size:
                            self.mapping_2.update(
                                {new_child_p_1: population[c[0]]})
                        if c[1] >= self.population_size:
                            self.mapping_2.update(
                                {new_child_p_1: population[c[1]]})

        self.generation_1 = np.asarray(child_1)
        self.generation_2 = np.asarray(child_2)

    def _create_new_generation(self):
        self.mapping_1 = dict()
        population_fitness_1 = np.asarray(
            list(self.max_dict_1.values())).flatten()
        population_1 = np.asarray(list(self.max_dict_1.keys())).flatten()
        p_1 = population_fitness_1 / np.nansum(population_fitness_1)
        if self.min_objective:
            p_1 = 1 - p_1
        couples_1 = self.selection_function(p_1, self.n_transferred_individual)  # tournament selection
        child_1 = []
        for c in couples_1: # c[0],c[1]分别代表父代个体1,2的基因型索引
            if len(child_1) >= self.population_size:
                break
            else:
                child_p_1, child_p_2 = self.cross_over_function(population_1[c[0]], population_1[c[1]])  # cross-over
                new_child_p_1 = self.mutation_function(child_p_1)
                new_child_p_2 = self.mutation_function(child_p_2)

                child_1 = np.hstack((child_1, new_child_p_1))
                child_1 = np.hstack((child_1, new_child_p_2))
                self.mapping_1.update({new_child_p_1: population_1[c[0]]})
                self.mapping_1.update({new_child_p_2: population_1[c[1]]})


        self.mapping_2 = dict()
        population_fitness_2 = np.asarray(
            list(self.max_dict_2.values())).flatten()
        population_2 = np.asarray(list(self.max_dict_2.keys())).flatten()
        p_2 = population_fitness_2 / np.nansum(population_fitness_2)
        if self.min_objective:
            p_2 = 1 - p_2
        couples_2 = self.selection_function(p_2, self.n_transferred_individual)
        child_2 = []
        for c in couples_2:
            if len(child_2) >= self.population_size:
                break
            else:
                child_p_1, child_p_2 = self.cross_over_function(population_2[c[0]], population_2[c[1]])  # cross-over
                new_child_p_1 = self.mutation_function(child_p_1)
                new_child_p_2 = self.mutation_function(child_p_2)

                child_2 = np.hstack((child_2, new_child_p_1))
                child_2 = np.hstack((child_2, new_child_p_2))
                self.mapping_2.update({new_child_p_1: population_2[c[0]]})
                self.mapping_2.update({new_child_p_2: population_2[c[1]]})

        
        self.mapping_3 = dict()
        population_fitness_3 = np.asarray(
            list(self.max_dict_3.values())).flatten()
        population_3 = np.asarray(list(self.max_dict_3.keys())).flatten()
        p_3 = population_fitness_3 / np.nansum(population_fitness_3)
        if self.min_objective:
            p_3 = 1 - p_3
        couples_3 = self.selection_function(p_3, self.n_transferred_individual)
        child_3 = []
        for c in couples_3:
            if len(child_3) >= self.population_size:
                break
            else:
                child_p_1, child_p_2 = self.cross_over_function(population_3[c[0]], population_3[c[1]])  # cross-over
                new_child_p_1 = self.mutation_function(child_p_1)
                new_child_p_2 = self.mutation_function(child_p_2)

                child_3 = np.hstack((child_3, new_child_p_1))
                child_3 = np.hstack((child_3, new_child_p_2))
                self.mapping_3.update({new_child_p_1: population_3[c[0]]})
                self.mapping_3.update({new_child_p_2: population_3[c[1]]})
        
        
        self.mapping_4 = dict()
        population_fitness_4 = np.asarray(
            list(self.max_dict_4.values())).flatten()
        population_4 = np.asarray(list(self.max_dict_4.keys())).flatten()
        p_4 = population_fitness_4 / np.nansum(population_fitness_4)
        if self.min_objective:
            p_4 = 1 - p_4
        couples_4 = self.selection_function(p_4, self.n_transferred_individual)
        child_4 = []
        for c in couples_4:
            if len(child_4) >= self.population_size:
                break
            else:
                child_p_1, child_p_2 = self.cross_over_function(population_4[c[0]], population_4[c[1]])  # cross-over
                new_child_p_1 = self.mutation_function(child_p_1)
                new_child_p_2 = self.mutation_function(child_p_2)

                child_4 = np.hstack((child_4, new_child_p_1))
                child_4 = np.hstack((child_4, new_child_p_2))
                self.mapping_4.update({new_child_p_1: population_4[c[0]]})
                self.mapping_4.update({new_child_p_2: population_4[c[1]]})
        
        self.generation_1 = np.asarray(child_1)
        self.generation_2 = np.asarray(child_2)
        self.generation_3 = np.asarray(child_3)
        self.generation_4 = np.asarray(child_4)

    def first_population(self):
        self.i += 1
        # Task 1
        n_diff_1 = self.population_size

        self.current_dict_1 = dict()
        population_fitness_1 = np.asarray(list(self.max_dict_1.values())).flatten()
        population_1 = np.asarray(list(self.max_dict_1.keys())).flatten()

        self.best_individual_1 = population_1[np.argmax(population_fitness_1)]

        fp_mean_1 = np.mean(population_fitness_1)
        fp_var_1 = np.var(population_fitness_1)
        fp_max_1 = np.max(population_fitness_1)
        fp_min_1 = np.min(population_fitness_1)
        self.ga_result_1.add_population_result(population_fitness_1, population_1)
        print(
           "population results 1 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |".format(
               fp_mean_1, fp_var_1, fp_max_1, fp_min_1))

        # Task 2
        n_diff_2 = self.population_size

        population_fitness_2 = np.asarray(list(self.max_dict_2.values())).flatten()
        population_2 = np.asarray(list(self.max_dict_2.keys())).flatten()
        self.best_individual_2 = population_2[np.argmax(population_fitness_2)]

        fp_mean_2 = np.mean(population_fitness_2)
        fp_var_2 = np.var(population_fitness_2)
        fp_max_2 = np.max(population_fitness_2)
        fp_min_2 = np.min(population_fitness_2)
        self.ga_result_2.add_population_result(population_fitness_2, population_2)

        print(
            "population results 2 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |".format(
                fp_mean_2, fp_var_2, fp_max_2, fp_min_2))
        
        # Task 3
        n_diff_3 = self.population_size

        population_fitness_3 = np.asarray(list(self.max_dict_3.values())).flatten()
        population_3 = np.asarray(list(self.max_dict_3.keys())).flatten()
        self.best_individual_3 = population_3[np.argmax(population_fitness_3)]

        fp_mean_3 = np.mean(population_fitness_3)
        fp_var_3 = np.var(population_fitness_3)
        fp_max_3 = np.max(population_fitness_3)
        fp_min_3 = np.min(population_fitness_3)
        self.ga_result_3.add_population_result(population_fitness_3, population_3)

        print(
            "population results 3 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |".format(
                fp_mean_3, fp_var_3, fp_max_3, fp_min_3))
        
        # Task 4
        n_diff_4 = self.population_size

        population_fitness_4 = np.asarray(list(self.max_dict_4.values())).flatten()
        population_4 = np.asarray(list(self.max_dict_4.keys())).flatten()
        self.best_individual_4 = population_4[np.argmax(population_fitness_4)]

        fp_mean_4 = np.mean(population_fitness_4)
        fp_var_4 = np.var(population_fitness_4)
        fp_max_4 = np.max(population_fitness_4)
        fp_min_4 = np.min(population_fitness_4)
        self.ga_result_4.add_population_result(population_fitness_4, population_4)

        print(
            "population results 4 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |".format(
                fp_mean_4, fp_var_4, fp_max_4, fp_min_4))

        return fp_mean_1, fp_var_1, fp_max_1, fp_min_1, n_diff_1, fp_mean_2, fp_var_2, fp_max_2, fp_min_2, n_diff_2, fp_mean_3, fp_var_3, fp_max_3, fp_min_3, n_diff_3, fp_mean_4, fp_var_4, fp_max_4, fp_min_4, n_diff_4

    def second_population(self):
        self.i += 1
        # Task 1
        generation_fitness_1 = np.asarray(list(self.current_dict_1.values()))
        generation_1 = list(self.current_dict_1.keys())
        self.ga_result_1.add_generation_result(generation_fitness_1, generation_1)

        f_mean_1 = np.mean(generation_fitness_1)
        f_var_1 = np.var(generation_fitness_1)
        f_max_1 = np.max(generation_fitness_1)
        f_min_1 = np.min(generation_fitness_1)
        
        total_dict_1 = self.max_dict_1.copy()
        total_dict_1.update(self.current_dict_1) # 合并P、O、TP
        best_max_dict_1 = total_dict_1.filter_top_n(self.population_size, min_max=not self.min_objective)
        n_diff_1 = self.max_dict_1.get_n_diff(best_max_dict_1)
        self.max_dict_1 = best_max_dict_1

        self.current_dict_1 = dict()
        population_fitness_1 = np.asarray(list(self.max_dict_1.values())).flatten()
        population_1 = np.asarray(list(self.max_dict_1.keys())).flatten()
        self.best_individual_1 = population_1[np.argmax(population_fitness_1)]

        fp_mean_1 = np.mean(population_fitness_1)
        fp_var_1 = np.var(population_fitness_1)
        fp_max_1 = np.max(population_fitness_1)
        fp_min_1 = np.min(population_fitness_1)
        self.ga_result_1.add_population_result(population_fitness_1, population_1)

        print(
            "Update generation 1 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |population size {:d}|".format(
                f_mean_1, f_var_1, f_max_1, f_min_1, len(population_1)))
        print(
            "population results 1 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |".format(
                fp_mean_1, fp_var_1, fp_max_1, fp_min_1))

        # Task 2
        generation_fitness_2 = np.asarray(list(self.current_dict_2.values()))
        generation_2 = list(self.current_dict_2.keys())
        self.ga_result_2.add_generation_result(generation_fitness_2, generation_2)

        f_mean_2 = np.mean(generation_fitness_2)
        f_var_2 = np.var(generation_fitness_2)
        f_max_2 = np.max(generation_fitness_2)
        f_min_2 = np.min(generation_fitness_2)
        
        total_dict_2 = self.max_dict_2.copy()
        total_dict_2.update(self.current_dict_2)
        best_max_dict_2 = total_dict_2.filter_top_n(self.population_size, min_max=not self.min_objective)
        n_diff_2 = self.max_dict_2.get_n_diff(best_max_dict_2)
        self.max_dict_2 = best_max_dict_2

        self.current_dict_2 = dict()
        population_fitness_2 = np.asarray(list(self.max_dict_2.values())).flatten()
        population_2 = np.asarray(list(self.max_dict_2.keys())).flatten()
        self.best_individual_2 = population_2[np.argmax(population_fitness_2)]
        fp_mean_2 = np.mean(population_fitness_2)
        fp_var_2 = np.var(population_fitness_2)
        fp_max_2 = np.max(population_fitness_2)
        fp_min_2 = np.min(population_fitness_2)
        self.ga_result_2.add_population_result(population_fitness_2, population_2)
        
        print(
            "Update generation 2 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |population size {:d}|".format(
                f_mean_2, f_var_2, f_max_2, f_min_2, len(population_2)))
        print(
            "population results 2 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |".format(
                fp_mean_2, fp_var_2, fp_max_2, fp_min_2))
        
        # Task 3
        generation_fitness_3 = np.asarray(list(self.current_dict_3.values()))
        generation_3 = list(self.current_dict_3.keys())
        self.ga_result_3.add_generation_result(generation_fitness_3, generation_3)

        f_mean_3 = np.mean(generation_fitness_3)
        f_var_3 = np.var(generation_fitness_3)
        f_max_3 = np.max(generation_fitness_3)
        f_min_3 = np.min(generation_fitness_3)
        
        total_dict_3 = self.max_dict_3.copy()
        total_dict_3.update(self.current_dict_3)
        best_max_dict_3 = total_dict_3.filter_top_n(self.population_size, min_max=not self.min_objective)
        n_diff_3 = self.max_dict_3.get_n_diff(best_max_dict_3)
        self.max_dict_3 = best_max_dict_3

        self.current_dict_3 = dict()
        population_fitness_3 = np.asarray(list(self.max_dict_3.values())).flatten()
        population_3 = np.asarray(list(self.max_dict_3.keys())).flatten()
        self.best_individual_3 = population_3[np.argmax(population_fitness_3)]
        fp_mean_3 = np.mean(population_fitness_3)
        fp_var_3 = np.var(population_fitness_3)
        fp_max_3 = np.max(population_fitness_3)
        fp_min_3 = np.min(population_fitness_3)
        self.ga_result_3.add_population_result(population_fitness_3, population_3)
        
        print(
            "Update generation 3 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |population size {:d}|".format(
                f_mean_3, f_var_3, f_max_3, f_min_3, len(population_3)))
        print(
            "population results 3 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |".format(
                fp_mean_3, fp_var_3, fp_max_3, fp_min_3))
        
        # Task 4
        generation_fitness_4 = np.asarray(list(self.current_dict_4.values()))
        generation_4 = list(self.current_dict_4.keys())
        self.ga_result_4.add_generation_result(generation_fitness_4, generation_4)

        f_mean_4 = np.mean(generation_fitness_4)
        f_var_4 = np.var(generation_fitness_4)
        f_max_4 = np.max(generation_fitness_4)
        f_min_4 = np.min(generation_fitness_4)
        
        total_dict_4 = self.max_dict_4.copy()
        total_dict_4.update(self.current_dict_4)
        best_max_dict_4 = total_dict_4.filter_top_n(self.population_size, min_max=not self.min_objective)
        n_diff_4 = self.max_dict_4.get_n_diff(best_max_dict_4)
        self.max_dict_4 = best_max_dict_4

        self.current_dict_4 = dict()
        population_fitness_4 = np.asarray(list(self.max_dict_4.values())).flatten()
        population_4 = np.asarray(list(self.max_dict_4.keys())).flatten()
        self.best_individual_4 = population_4[np.argmax(population_fitness_4)]
        fp_mean_4 = np.mean(population_fitness_4)
        fp_var_4 = np.var(population_fitness_4)
        fp_max_4 = np.max(population_fitness_4)
        fp_min_4 = np.min(population_fitness_4)
        self.ga_result_4.add_population_result(population_fitness_4, population_4)

        print(
            "Update generation 4 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |population size {:d}|".format(
                f_mean_4, f_var_4, f_max_4, f_min_4, len(population_4)))
        print(
            "population results 4 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |".format(
                fp_mean_4, fp_var_4, fp_max_4, fp_min_4))

        return f_mean_1, f_var_1, f_max_1, f_min_1, n_diff_1, f_mean_2, f_var_2, f_max_2, f_min_2, n_diff_2, f_mean_4, f_var_4, f_max_4, f_min_3, n_diff_3, f_mean_4, f_var_4, f_max_4, f_min_4, n_diff_4

    def get_current_generation(self, num):
        if num == 1:
            return self.generation_1
        elif num == 2:
            return self.generation_2
        elif num == 3:
            return self.generation_3
        elif num == 4:
            return self.generation_4

    def get_max_generation(self, num):
        if num == 1:
            return np.asarray(list(self.max_dict_1.keys())).flatten()
        elif num == 2:
            return np.asarray(list(self.max_dict_2.keys())).flatten()
        elif num == 3:
            return np.asarray(list(self.max_dict_3.keys())).flatten()
        elif num == 4:
            return np.asarray(list(self.max_dict_4.keys())).flatten()

    def get_transferred_generation(self, num):
        if num == 1:
            return self.transferred_generation_1
        elif num == 2:
            return self.transferred_generation_2
        elif num == 3:
            return self.transferred_generation_3
        elif num == 4:
            return self.transferred_generation_4
    
    def get_transferred_dict_generation(self, num):
        if num == 1:
            return np.asarray(list(self.transferred_dict_1.keys())).flatten()
        elif num == 2:
            return np.asarray(list(self.transferred_dict_2.keys())).flatten()
        elif num == 3:
            return np.asarray(list(self.transferred_dict_3.keys())).flatten()
        elif num == 4:
            return np.asarray(list(self.transferred_dict_4.keys())).flatten()

    def update_current_individual_fitness(self, individual, individual_fitness, num):
        if num == 1:
           self.current_dict_1.update({individual: individual_fitness})
        elif num == 2:
           self.current_dict_2.update({individual: individual_fitness})
        elif num == 3:
           self.current_dict_3.update({individual: individual_fitness})
        elif num == 4:
           self.current_dict_4.update({individual: individual_fitness})

    def update_max_individual_fitness(self, individual, individual_fitness, num):
        if num == 1:
           self.max_dict_1.update_2({individual: individual_fitness})
        elif num == 2:
           self.max_dict_2.update_2({individual: individual_fitness})
        elif num == 3:
           self.max_dict_3.update_2({individual: individual_fitness})
        elif num == 4:
           self.max_dict_4.update_2({individual: individual_fitness})

    def update_tranferred_individual_fitness(self, individual, individual_fitness, num):
        if num == 1:
           self.transferred_dict_1.update_2({individual: individual_fitness})
        elif num == 2:
           self.transferred_dict_2.update_2({individual: individual_fitness})
        elif num == 3:
           self.transferred_dict_3.update_2({individual: individual_fitness})
        elif num == 4:
           self.transferred_dict_4.update_2({individual: individual_fitness})

    def sample_child(self, num, flag):
        # 采集个体用于epoch下的batch训练
        if flag == 0: # 仅采集父代个体
            if num == 1:
                couples = choices(list(self.max_dict_1.keys()), k=2)  # random select two indivuals from parents Task1
            elif num == 2:
                couples = choices(list(self.max_dict_2.keys()), k=2)  # random select two indivuals from parents Task2
            elif num == 3:
                couples = choices(list(self.max_dict_3.keys()), k=2)
            elif num == 4:
                couples = choices(list(self.max_dict_4.keys()), k=2)
            return couples[0]  # select the first then mutation
        else: # 采集父代子代个体
            if random.random() < 0.5:
                if num == 1:
                    couples = choices(list(self.max_dict_1.keys()), k=2)  # random select two indivuals from parents Task1
                elif num == 2:
                    couples = choices(list(self.max_dict_2.keys()), k=2)  # random select two indivuals from parents Task2
                elif num == 3:
                    couples = choices(list(self.max_dict_3.keys()), k=2)
                elif num == 4:
                    couples = choices(list(self.max_dict_4.keys()), k=2)
            else:
                if num == 1:
                    couples = choices(list(self.current_dict_1.keys()), k=2)  # random select two indivuals from offspring Task1
                elif num == 2:
                    couples = choices(list(self.current_dict_2.keys()), k=2)  # random select two indivuals from offspring Task2
                elif num == 3:
                    couples = choices(list(self.current_dict_3.keys()), k=2)
                elif num == 4:
                    couples = choices(list(self.current_dict_4.keys()), k=2)
            return couples[0]

    def get_historical_transferred_set(self, num):
        if num == 1:
            return self.historical_transferred_set_1
        elif num == 2:
            return self.historical_transferred_set_2
        elif num == 3:
            return self.historical_transferred_set_3
        elif num == 4:
            return self.historical_transferred_set_4

    def _create_random_transferred_generation(self, num):
        # 创建对象时，初始种群存在self.generation_1/2，从中随机选择，存到self.transferred_dict_1/2
        if num == 1:
            return random.sample(self.generation_2+self.generation_3+self.generation_4, self.n_transferred_individual)
        elif num == 2:
            return random.sample(self.generation_1+self.generation_3+self.generation_4, self.n_transferred_individual)
        elif num == 3:
            return random.sample(self.generation_1+self.generation_2+self.generation_4, self.n_transferred_individual)
        elif num == 4:
            return random.sample(self.generation_1+self.generation_2+self.generation_3, self.n_transferred_individual)
    
    def _select_topk_max(self):
        total_dict_1 = self.max_dict_1.copy()
        best_max_dict_1 = total_dict_1.filter_top_n(self.population_size, min_max=not self.min_objective)
        self.max_dict_1 = best_max_dict_1

        total_dict_2 = self.max_dict_2.copy()
        best_max_dict_2 = total_dict_2.filter_top_n(self.population_size, min_max=not self.min_objective)
        self.max_dict_2 = best_max_dict_2
        
        total_dict_3 = self.max_dict_3.copy()
        best_max_dict_3 = total_dict_3.filter_top_n(self.population_size, min_max=not self.min_objective)
        self.max_dict_3 = best_max_dict_3
        
        total_dict_4 = self.max_dict_4.copy()
        best_max_dict_4 = total_dict_4.filter_top_n(self.population_size, min_max=not self.min_objective)
        self.max_dict_4 = best_max_dict_4

    def merge_transferred_to_max(self):
        self.max_dict_1 = self.max_dict_1.merge(self.transferred_dict_1)
        self.max_dict_2 = self.max_dict_2.merge(self.transferred_dict_2)
        self.max_dict_3 = self.max_dict_3.merge(self.transferred_dict_3)
        self.max_dict_4 = self.max_dict_4.merge(self.transferred_dict_4)

    def _updata_hts_multiple_generation(self, generation):
        # max_dict/transferred_dict的数据格式:{Individuals对象:fitness,...}
        self.positive_class_set_1.clear()
        self.negtive_class_set_1.clear()
        for c in self.transferred_dict_1.keys():
            # 正迁移：迁移个体产生的子代在新的父代种群中是否排序top K
            reverse_mapping = {v:k for k,v in self.mapping_1.items()}
            if c not in reverse_mapping.keys():
                continue
            o = [c]
            is_positive=False
            for i in range(int(self.population_size*self.ratio_elite_individual)):
                if o == list(self.max_dict_1.keys())[i]:
                    self.positive_class_set_1.update({c:1})
                    is_positive=True
                    break
            if not is_positive:
                self.negtive_class_set_1.update({c:-1})
        
        self.positive_class_set_2.clear()
        self.negtive_class_set_2.clear()
        for c in self.transferred_dict_2.keys():
            reverse_mapping = {v:k for k,v in self.mapping_2.items()}
            if c not in reverse_mapping.keys():
                continue
            o = [c]
            is_positive=False
            for i in range(int(self.population_size*self.ratio_elite_individual)):
                if o == list(self.max_dict_2.keys())[i]:
                    self.positive_class_set_2.update({c:1})
                    is_positive=True
                    break
            if not is_positive:
                self.negtive_class_set_2.update({c:-1})
                
        self.positive_class_set_3.clear()
        self.negtive_class_set_3.clear()
        for c in self.transferred_dict_3.keys():
            reverse_mapping = {v:k for k,v in self.mapping_3.items()}
            if c not in reverse_mapping.keys():
                continue
            o = [c]
            is_positive=False
            for i in range(int(self.population_size*self.ratio_elite_individual)):
                if o == list(self.max_dict_3.keys())[i]:
                    self.positive_class_set_3.update({c:1})
                    is_positive=True
                    break
            if not is_positive:
                self.negtive_class_set_3.update({c:-1})
                
        self.positive_class_set_4.clear()
        self.negtive_class_set_4.clear()
        for c in self.transferred_dict_4.keys():
            reverse_mapping = {v:k for k,v in self.mapping_4.items()}
            if c not in reverse_mapping.keys():
                continue
            o = [c]
            is_positive=False
            for i in range(int(self.population_size*self.ratio_elite_individual)):
                if o == list(self.max_dict_4.keys())[i]:
                    self.positive_class_set_4.update({c:1})
                    is_positive=True
                    break
            if not is_positive:
                self.negtive_class_set_4.update({c:-1})
        
        # 记录正在historcal_negtive_class_set中每代negtive_class_set的个体数
        self.negtive_class_set_size_1.append(len(self.negtive_class_set_1))
        self.negtive_class_set_size_2.append(len(self.negtive_class_set_2))
        self.negtive_class_set_size_3.append(len(self.negtive_class_set_3))
        self.negtive_class_set_size_4.append(len(self.negtive_class_set_4))
        
        self.historical_transferred_set_1.clear()
        self.historical_transferred_set_2.clear()
        self.historical_transferred_set_3.clear()
        self.historical_transferred_set_4.clear()
        if generation < self.n_generation_saved_negative: # generation starts from 0
            self.historcial_negtive_class_set_1.update(self.negtive_class_set_1)
            self.historcial_negtive_class_set_2.update(self.negtive_class_set_2)
            self.historcial_negtive_class_set_3.update(self.negtive_class_set_3)
            self.historcial_negtive_class_set_4.update(self.negtive_class_set_4)
        else:
            self.historcial_negtive_class_set_1.update(self.negtive_class_set_1)
            for _ in range(self.negtive_class_set_size_1[generation - self.n_generation_saved_negative]):
                if len(self.historcial_negtive_class_set_1) > 0:
                    self.historcial_negtive_class_set_1.popitem(last=False) 
            self.historcial_negtive_class_set_2.update(self.negtive_class_set_2)
            for _ in range(self.negtive_class_set_size_2[generation - self.n_generation_saved_negative]):
                if len(self.historcial_negtive_class_set_2) > 0:
                    self.historcial_negtive_class_set_2.popitem(last=False)
            self.historcial_negtive_class_set_3.update(self.negtive_class_set_3)
            for _ in range(self.negtive_class_set_size_3[generation - self.n_generation_saved_negative]):
                if len(self.historcial_negtive_class_set_3) > 0:
                    self.historcial_negtive_class_set_3.popitem(last=False)
            self.historcial_negtive_class_set_4.update(self.negtive_class_set_4)
            for _ in range(self.negtive_class_set_size_4[generation - self.n_generation_saved_negative]):
                if len(self.historcial_negtive_class_set_4) > 0:
                    self.historcial_negtive_class_set_4.popitem(last=False) 
        self.historical_transferred_set_1.update(self.positive_class_set_1)
        self.historical_transferred_set_1.update(self.historcial_negtive_class_set_1)
        self.historical_transferred_set_2.update(self.positive_class_set_2)
        self.historical_transferred_set_2.update(self.historcial_negtive_class_set_2)
        self.historical_transferred_set_3.update(self.positive_class_set_3)
        self.historical_transferred_set_3.update(self.historcial_negtive_class_set_3)
        self.historical_transferred_set_4.update(self.positive_class_set_4)
        self.historical_transferred_set_4.update(self.historcial_negtive_class_set_4)

    def _create_new_transferred_generation(self):
        if len(self.historical_transferred_set_1) > 0:
            self.transferred_generation_1 = select_transferred_individuals(self.historical_transferred_set_1, self.max_dict_2.merge(self.max_dict_3.merge(self.max_dict_4)), self.n_transferred_individual)
        if len(self.historical_transferred_set_2) > 0:
            self.transferred_generation_2 = select_transferred_individuals(self.historical_transferred_set_2, self.max_dict_1.merge(self.max_dict_3.merge(self.max_dict_4)), self.n_transferred_individual)
        if len(self.historical_transferred_set_3) > 0:
            self.transferred_generation_3 = select_transferred_individuals(self.historical_transferred_set_3, self.max_dict_1.merge(self.max_dict_2.merge(self.max_dict_4)), self.n_transferred_individual)
        if len(self.historical_transferred_set_4) > 0:
            self.transferred_generation_4 = select_transferred_individuals(self.historical_transferred_set_4, self.max_dict_1.merge(self.max_dict_2.merge(self.max_dict_3)), self.n_transferred_individual)

    def _create_new_transferred_generation_all(self):
        self.transferred_generation_1 = list((self.max_dict_2.merge(self.max_dict_3.merge(self.max_dict_4))).keys())
        self.transferred_generation_2 = list((self.max_dict_1.merge(self.max_dict_3.merge(self.max_dict_4))).keys())
        self.transferred_generation_3 = list((self.max_dict_1.merge(self.max_dict_2.merge(self.max_dict_4))).keys())
        self.transferred_generation_4 = list((self.max_dict_1.merge(self.max_dict_2.merge(self.max_dict_3))).keys())

    def _select_transferred_individuals(self):
        self.transferred_dict_1 = self.transferred_dict_1.filter_top_n(n=self.n_transferred_individual)
        self.transferred_dict_2 = self.transferred_dict_2.filter_top_n(n=self.n_transferred_individual)
        self.transferred_dict_3 = self.transferred_dict_3.filter_top_n(n=self.n_transferred_individual)
        self.transferred_dict_4 = self.transferred_dict_4.filter_top_n(n=self.n_transferred_individual)

    def _create_new_transferred_generation_random(self):
        self.transferred_generation_1 = random.sample(list((self.max_dict_2.merge(self.max_dict_3.merge(self.max_dict_4))).keys()), self.n_transferred_individual)
        self.transferred_generation_2 = random.sample(list((self.max_dict_1.merge(self.max_dict_3.merge(self.max_dict_4))).keys()), self.n_transferred_individual)
        self.transferred_generation_3 = random.sample(list((self.max_dict_1.merge(self.max_dict_2.merge(self.max_dict_4))).keys()), self.n_transferred_individual)
        self.transferred_generation_4 = random.sample(list((self.max_dict_1.merge(self.max_dict_2.merge(self.max_dict_3))).keys()), self.n_transferred_individual)

    def _create_new_transferred_generation_elite(self):
        self.transferred_generation_1 = list({k:v for k,v in sorted((self.max_dict_2.merge(self.max_dict_3.merge(self.max_dict_4))).items(),key=operator.itemgetter(1),reverse=True)}.keys())[:self.n_transferred_individual]
        self.transferred_generation_2 = list({k:v for k,v in sorted((self.max_dict_1.merge(self.max_dict_3.merge(self.max_dict_4))).items(),key=operator.itemgetter(1),reverse=True)}.keys())[:self.n_transferred_individual]
        self.transferred_generation_3 = list({k:v for k,v in sorted((self.max_dict_1.merge(self.max_dict_2.merge(self.max_dict_4))).items(),key=operator.itemgetter(1),reverse=True)}.keys())[:self.n_transferred_individual]
        self.transferred_generation_4 = list({k:v for k,v in sorted((self.max_dict_1.merge(self.max_dict_2.merge(self.max_dict_3))).items(),key=operator.itemgetter(1),reverse=True)}.keys())[:self.n_transferred_individual]

    def save_old_current_generation(self):
        self.old_current_dict_1 = dict()
        for key, value in self.max_dict_1.items():
            self.old_current_dict_1.update({key: value})

        self.old_current_dict_2 = dict()
        for key, value in self.max_dict_2.items():
            self.old_current_dict_2.update({key: value})
            
        self.old_current_dict_3 = dict()
        for key, value in self.max_dict_3.items():
            self.old_current_dict_3.update({key: value})
            
        self.old_current_dict_4 = dict()
        for key, value in self.max_dict_4.items():
            self.old_current_dict_4.update({key: value})


# def main():
#     config = get_config()
#     ss = get_gnas_cnn_search_space(n_nodes=5, drop_path_control=DropModuleControl(
#         0.90), n_cell_type=SearchSpaceType(1))
#     ga = genetic_algorithm_searcher(ss, generation_size=config.get('generation_size'), population_size=config.get('population_size'), keep_size=config.get('keep_size'),
#                                     mutation_p=config.get('mutation_p'), p_cross_over=config.get('p_cross_over'), n_epochs=config.get('n_generations'), RMP=config.get('RMP'),)
    
#     uptate_parents_individual_list(ga.get_current_generation(1), ga, 1)
#     inds = ga.get_max_generation(1)
#     print('traditional:',len(inds))
#     for i in range(len(inds)):
#         print(inds[i])
#     # transfer
#     uptate_parents_individual_list(ga.get_transferred_population(1), ga, 1)
#     inds_transfer = ga.get_max_generation(1)
#     print('trasfer:',len(inds_transfer))
#     for i in range(len(inds_transfer)):
#         print(inds_transfer[i])
#     # select
#     ga._select_topk_max()
#     inds_select = ga.get_max_generation(1)
#     print('select:',len(inds_select))
#     for i in range(len(inds_select)):
#         print(inds_select[i])
    
#     # uptate_parents_individual_list(ga.get_current_generation(2), ga, 2)
#     # ga._create_new_generation()


# if __name__ == "__main__":
#     main()
