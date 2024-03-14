from graph.graph_similarity import get_pair_sim
import operator

def calc_transfer_rank(hts, p):
    transfer_rank = dict.fromkeys(p.keys(),0) # save transfer rank of individuals in p
    for hts_ind in hts.keys():
        max_sim = -1.0
        max_ind = None # record the individual in p that the most similar to hts_ind
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
