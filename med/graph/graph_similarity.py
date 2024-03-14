import sys
sys.path.append('/home/ztj/codes/TREMT-NAS/')
import numpy as np
from graph.emb_vector import generate_emb_vecs


def euclidian_distance(vecs1, vecs2):
    """
    计算两个图的embedding vector的欧式距离
    """
    num_of_nodes = len(vecs1)
    sum_of_dist = 0.0
    for i in range(1, num_of_nodes+1):
        v1 = vecs1[str(i)]
        v2 = vecs2[str(i)]
        dist = np.linalg.norm(v1-v2)
        sum_of_dist += dist
    return sum_of_dist / num_of_nodes


def vec_similarity(vecs1, vecs2):
    """
    计算两个图的embedding vector的相似度1到0,对应夹角0度-180度
    """
    num_of_nodes = len(vecs1)
    sum_of_sim = 0.0
    for i in range(0, num_of_nodes):
        v1 = vecs1[str(i)]
        v2 = vecs2[str(i)]
        sim = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
        sum_of_sim += sim
    mean_of_sim = sum_of_sim / num_of_nodes
    angle = np.degrees(np.arccos(mean_of_sim))
    sim = 1 - angle / 180
    return sim


def get_pair_sim(encode1, encode2):
    # 输入为字符串类型的基因型，'[1 1 8 1 ......]'
    N = 10
    sum_of_pair_sim = 0.0
    for _ in range(N):
        vecs1 = generate_emb_vecs(encode1)
        vecs2 = generate_emb_vecs(encode2)
        sum_of_pair_sim += vec_similarity(vecs1, vecs2)
    return sum_of_pair_sim / N


if __name__ == '__main__':
    encode = '[1 1 8 1 2 1 1 3 1 2 6 6 4 3 6 6 1 3 5 1 0 1 1 0 0 0 5 6 3 2 3 6 0 1 3 6 2 0 2 3]'
    encode_2 = '[1 0 6 6 2 2 0 0 3 0 6 1 4 0 7 7 2 0 0 4 0 1 3 5 1 1 2 3 1 0 8 1 4 1 4 1 5 4 8 7]'
    # print(get_pair_sim(encode, encode_2))
    vecs = generate_emb_vecs(encode)
    print('vector:',vecs,type(encode))
    print('vector 0:',vecs['0'],type(vecs['0']))
