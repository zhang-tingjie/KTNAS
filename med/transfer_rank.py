from graph.graph_similarity import get_pair_sim
import operator

def calc_transfer_rank(hts, p):
    transfer_rank = dict.fromkeys(p.keys(),0) #存放P中个体的迁移等级
    for hts_ind in hts.keys():
        max_sim = -1.0
        max_ind = None # 记录p_inds中与hts_ind相似度最大的个体
        for p_ind in p.keys():
            current_sim = get_pair_sim(hts_ind.get_code(), p_ind.get_code())
            if current_sim > max_sim:
                max_sim = current_sim
                max_ind = p_ind
        transfer_rank[max_ind] = transfer_rank[max_ind] + hts[hts_ind]
    return {k:v for k,v in sorted(transfer_rank.items(), key=operator.itemgetter(1), reverse=True)}

def select_transferred_individuals(hts, p, n_inds):
    sorted_transfer_rank = calc_transfer_rank(hts, p)
    return list(sorted_transfer_rank.keys())[:n_inds]

# if __name__=='__main__':
#     hts = {'[1 1 8 1 2 1 1 3 1 2 6 6 4 3 6 6 1 3 5 1 0 1 1 0 0 0 5 6 3 2 3 6 0 1 3 6 2 0 2 3]':1, '[1 0 6 6 2 2 0 0 3 0 6 1 4 0 7 7 2 0 0 4 0 1 3 5 1 1 2 3 1 0 8 1 4 1 4 1 5 4 8 7]':-1}
#     p = {'[1 1 8 1 2 1 1 3 1 2 6 6 4 3 6 6 1 3 5 1 0 1 1 0 0 0 5 6 3 2 3 6 1 4 0 7 7 2 0 0]':0.33, '[6 6 4 3 6 6 1 3 3 0 6 1 4 0 7 7 2 0 0 4 0 1 3 5 1 1 2 3 1 0 8 1 4 1 4 1 5 4 8 7]':0.2}
#     sorted_transfer_rank = calc_transfer_rank(hts, p)
#     print('transfer rank:', sorted_transfer_rank)
#     select_inds = select_transferred_individuals(hts, p, 1)
#     print(select_inds)