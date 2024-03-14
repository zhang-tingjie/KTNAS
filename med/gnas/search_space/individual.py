import numpy as np


class Individual(object):
    def __init__(self, individual_vector, max_inputs, search_space, index=0):
        self.iv = individual_vector
        self.mi = max_inputs
        self.ss = search_space
        # Generate config when generating individual
        self.index = index
        self.config_list = [oc.parse_config(iv) for iv, oc in zip(self.iv, self.ss.get_opeartion_config(self.index))]
        self.code = np.concatenate(self.iv, axis=0)

    def get_length(self):
        return len(self.code)

    def get_n_op(self):
        return len(self.iv)

    def copy(self):
        return Individual(self.iv, self.mi, self.ss, index=self.index)

    def generate_node_config(self):
        return self.config_list

    def update_individual(self, individual_vector):
        return Individual(individual_vector, self.mi, self.ss, index=self.index)

    def __eq__(self, other):
        return np.array_equal(self.code, other.code)

    def __str__(self):
        return "code:" + str(self.code)

    def __hash__(self):
        return hash(str(self))
    
    def get_code(self):
        return str(self.code)


class MultipleBlockIndividual(object):
    def __init__(self, individual_list):
        self.individual_list = individual_list
        self.code = np.concatenate([i.code for i in self.individual_list])

    def get_individual(self, index):
        return self.individual_list[index]

    def generate_node_config(self, index):
        return self.individual_list[index].generate_node_config()

    def update_individual(self, individual_vector):
        raise NotImplemented

    def __eq__(self, other):
        return np.array_equal(self.code, other.code)

    def __str__(self):
        return "code:" + str(self.code)

    def __hash__(self):
        return hash(str(self))
    
    def get_code(self):
        return str(self.code)

def get_individual_vector(code, index):
    split_code=code.replace('[','').replace(']','').split(' ')
    if index==0:
        code_0=split_code[:20]
    elif index==1:
        code_0=split_code[20:]
    iv=[]
    for i in range(0,len(code_0),4):
            iv_tmp=[]
            iv_tmp.append(code_0[i])
            iv_tmp.append(code_0[i+1])
            iv_tmp.append(code_0[i+2])
            iv_tmp.append(code_0[i+3])
            iv.append(np.asarray(iv_tmp).astype(int))
    return iv


if __name__=='__main__':
    import sys
    sys.path.append('/home/ztj/codes/TREMT-NAS/')
    from gnas.search_space.factory import _two_input_cell
    from gnas.search_space.search_space import SearchSpace

    node_config_list_a = _two_input_cell(5, 0.9)
    node_config_list_b = _two_input_cell(5, 0.9)
    ss = SearchSpace([node_config_list_a, node_config_list_b], single_block=False)
    mi = [0,1,2,3,4]

    code='[1 1 8 1 2 1 1 3 1 2 6 6 4 3 6 6 1 3 5 1 0 1 1 0 0 0 5 6 3 2 3 6 0 1 3 6 2 0 2 3]'
    print('code:',code)
    iv_0=get_individual_vector(code,0)
    print('individual vector 0:',iv_0)
    iv_1=get_individual_vector(code,1)

    ind_0=Individual(iv_0,mi,ss,0)
    ind_1=Individual(iv_1,mi,ss,1)
    print('ind 0:',ind_0.get_code())
    print('ind 1:',ind_1.get_code())

    ind=MultipleBlockIndividual([ind_0,ind_1])
    print('ind:',ind.get_code())

    from models import model_cnn
    from config import get_config
    import torch
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    working_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    net_1 = model_cnn.Net(config.get('n_blocks'), config.get('n_channels_1'), 10,
                          0.10, ss).to(working_device)
    net_1.set_individual(ind)

    # input = torch.randn(16,3,8,8).to(working_device)
    # output = net_1(input)
    # print(output.shape)

