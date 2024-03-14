import sys
sys.path.append('/home/ztj/codes/TREMT-NAS/')
import numpy as np
from gnas.search_space.individual import Individual, MultipleBlockIndividual


class SearchSpace(object):
    def __init__(self, operation_config_list: list, single_block=True):
        self.single_block = single_block
        self.ocl = operation_config_list # CnnNodeConfig对象（节点的配置）列表
        if single_block:
            self.n_elements = sum([len(self.generate_vector(o.max_values_vector(i))) for i, o in enumerate(self.ocl)])
        else:
            self.n_elements = sum(
                [sum([len(self.generate_vector(o.max_values_vector(i))) for i, o in enumerate(block)]) for block in
                 self.ocl])

    def get_operation_configs(self):
        return self.ocl

    def get_n_nodes(self):
        if self.single_block:
            return len(self.ocl)
        else:
            return [len(ocl) for ocl in self.ocl]

    def get_max_values_vector(self, index=0):
        if self.single_block:
            return [o.max_values_vector(i) for i, o in enumerate(self.ocl)]
        else:
            return [o.max_values_vector(i) for i, o in enumerate(self.ocl[index])]

    def get_opeartion_config(self, index=0):
        if self.single_block:
            return self.ocl
        else:
            return self.ocl[index]

    def generate_vector(self, max_values):
        return np.asarray([np.random.randint(0, mv + 1) for mv in max_values])

    def _generate_individual_single(self, ocl, index=0):
        operation_vector = [self.generate_vector(o.max_values_vector(i)) for i, o in enumerate(ocl)]
        max_inputs = [i for i, _ in enumerate(ocl)]
        return Individual(operation_vector, max_inputs, self, index=index)

    def generate_individual(self): # 返回单个MultipleBlockIndividual对象
        if self.single_block:
            return self._generate_individual_single(self.ocl)
        else:
            return MultipleBlockIndividual(
                [self._generate_individual_single(ocl, index=i) for i, ocl in enumerate(self.ocl)])

    def generate_population(self, size): # 返回MultipleBlockIndividual对象列表
        return [self.generate_individual() for _ in range(size)]
    
# if __name__=='__main__':
#     # import modules for test
#     from gnas.search_space.factory import _two_input_cell
#     from gnas.search_space.search_space import SearchSpace
#     from graph.graph_similarity import get_pair_sim

#     node_config_list_a = _two_input_cell(5, 0.9)
#     node_config_list_b = _two_input_cell(5, 0.9)
#     ss = SearchSpace([node_config_list_a, node_config_list_b], single_block=False)
#     inds = ss.generate_population(3) # 3个MultipleBlockIndividual对象列表
#     # for ind in inds:
#     #     print(ind)
#     ind_0 = inds[0]
#     ind_1 = inds[1]
#     print(get_pair_sim(ind_0.get_code(),ind_1.get_code()))
#     print('gene:',ind_0.get_code())
#     print('complete cell:',ind_0,'type:',type(ind_0))
#     print('normal cell:',ind_0.get_individual(0),'type:',type(ind_0.get_individual(0)))
#     print('reduction cell:',ind_0.get_individual(1),'type:',type(ind_0.get_individual(1)))
#     normal_cell = ind_0.get_individual(0)
#     print('individual_vector:',normal_cell.iv,type(normal_cell.iv))
#     print('max_inputs',normal_cell.mi)
#     print('index',normal_cell.index)
#     print('config_list',normal_cell.config_list)
#     print(normal_cell)
    
#     # for node in node_config_list_a:
#     #     mv_vector = node.max_values_vector(0)
#     #     print('max index vector',node.node_id, ':', mv_vector)
