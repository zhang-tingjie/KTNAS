
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
    计算两个图的embedding vector的相似度-1到1,对应夹角-180~0度
    """
    num_of_nodes = len(vecs1)
    sum_of_sim = 0.0
    for i in range(0, num_of_nodes):
        v1 = vecs1[str(i)]
        v2 = vecs2[str(i)]
        sim = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
        sum_of_sim += sim
    mean_of_sim = sum_of_sim / num_of_nodes
    return mean_of_sim


def get_pair_sim(encode1, encode2):
    # 输入为字符串类型的基因型，'[1 1 8 1 ......]'
    N = 10
    sum_of_pair_sim = 0.0
    for _ in range(N):
        vecs1 = generate_emb_vecs(encode1)
        vecs2 = generate_emb_vecs(encode2)
        sum_of_pair_sim += vec_similarity(vecs1, vecs2)
    return sum_of_pair_sim / N

