import numpy as np

class CnnNodeConfig(object): # Block：2 source nodes, 2 operation nodes
    def __init__(self, node_id, inputs: list, op_list, drop_path_control):
        self.node_id = node_id
        self.inputs = inputs # 可选source node ids
        self.op_list = op_list
        self.drop_path_control = drop_path_control

    def max_values_vector(self, max_inputs): # 前2个为source node的最大id，后2个为operation nodes的最大索引
        max_inputs = len(self.inputs)
        if max_inputs > 1:
            return np.asarray([max_inputs - 1, max_inputs - 1, len(self.op_list) - 1, len(self.op_list) - 1])
        return np.asarray([len(self.op_list) - 1, len(self.op_list) - 1])

    def get_n_inputs(self): # source nodes数量:2
        return len(self.inputs)

    def parse_config(self, oc):
        if len(self.inputs) == 1:
            op_a = oc[0]
            op_b = oc[1]
            input_index_a = 0
            input_index_b = 0
        else:
            input_index_a = oc[0]
            input_index_b = oc[1]
            op_a = oc[2]
            op_b = oc[3]
        input_a = self.inputs[input_index_a]
        input_b = self.inputs[input_index_b]
        return input_a, input_b, input_index_a, input_index_b, op_a, op_b
