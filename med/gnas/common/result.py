import os
import pickle


class ResultAppender(object):
    def __init__(self):
        self.result_dict = dict()

    def add_epoch_result(self, result_name: str, result_var: float):
        if self.result_dict.get(result_name) is None:
            self.result_dict.update({result_name: [result_var]})
        else:
            self.result_dict.get(result_name).append(result_var)

    def add_result(self, result_name: str, result_array):
        self.result_dict.update({result_name: result_array})

    def save_result(self, input_path):
        pickle.dump(self, open(os.path.join(input_path, 'ga_result.pickle'), "wb"))

    @staticmethod
    def load_result(input_path):
        return pickle.load(open(os.path.join(input_path, 'ga_result.pickle'), "rb"))
        

if __name__=='__main__':
    import sys
    sys.path.append('/home/ztj/codes/TREMT-NAS/')
    from gnas.common.result import ResultAppender

    input_path='/home/ztj/codes/TREMT-NAS4Med/logs/search-20231225-094537-as'
    ra=ResultAppender.load_result(input_path)
    res_dict=ra.result_dict
    # print('keys:',res_dict.keys())
    # print('Generation:',res_dict['Generation'])
    # print('best 1:',res_dict['Best 1'])
    # print('fitness population 1:',res_dict['Fitness-Population 1'])
    # print('time sum:',res_dict['Time Sum'])
    # print('generation:',res_dict['N 1'])
    # print('best 1:',res_dict['Best 1'])
    # print('validation acc 1:',res_dict['Validation Accuracy 1'])
    # print('validation acc 2:',res_dict['Validation Accuracy 2'])
    # print('best 2:',res_dict['Best 2'])
    print('Search Cost:{:.5f} GPU Day'.format(res_dict['Time Sum'][-1] / 60 / 24))
