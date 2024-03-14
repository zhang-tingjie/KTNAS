'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import sys
sys.path.append('/home/ztj/codes/TREMT-NAS/graph')
import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec

parser = argparse.ArgumentParser(description="Run node2vec.")

parser.add_argument('--dimensions', type=int, default=128,
					help='Number of dimensions. Default is 128.')

parser.add_argument('--walk-length', type=int, default=80,
					help='Length of walk per source. Default is 80.')

parser.add_argument('--num-walks', type=int, default=10,
					help='Number of walks per source. Default is 10.')

parser.add_argument('--window-size', type=int, default=10,
					help='Context size for optimization. Default is 10.')

parser.add_argument('--iter', default=1, type=int,
					help='Number of epochs in SGD')

parser.add_argument('--workers', type=int, default=8,
					help='Number of parallel workers. Default is 8.')

parser.add_argument('--p', type=float, default=1,
					help='Return hyperparameter. Default is 1.')

parser.add_argument('--q', type=float, default=1,
					help='Inout hyperparameter. Default is 1.')

parser.add_argument('--weighted', dest='weighted', action='store_true',
					help='Boolean specifying (un)weighted. Default is unweighted.')
parser.add_argument('--unweighted', dest='unweighted', action='store_false')
# parser.set_defaults(weighted=False)
parser.set_defaults(weighted=True)

parser.add_argument('--directed', dest='directed', action='store_true',
					help='Graph is (un)directed. Default is undirected.')
parser.add_argument('--undirected', dest='undirected', action='store_false')
# parser.set_defaults(directed=False)
parser.set_defaults(directed=True) # 有向图

args=parser.parse_args()
	

def encode_to_edge_list(encode):
	'''
	Architecture encode converts to edge list
	'''
	edge_ls=[]

	encode_ls = encode.replace('[','').replace(']','').split(' ')
	s1=encode_ls[:20]
	for i in range(0,len(s1),4):
		s_1=s1[i]
		s_2=s1[i+1]
		o_1=s1[i+2]
		o_2=s1[i+3]
		curr=str(i//4+2) #source node编号从2到6
		edge_ls.append(curr+' '+s_1+' '+o_1)
		edge_ls.append(curr+' '+s_2+' '+o_2)
	
	s2=encode_ls[20:]
	for i in range(0,len(s2),4):
		s_1=s2[i]
		s_2=s2[i+1]
		o_1=s2[i+2]
		o_2=s2[i+3]
		curr=str(i//4+2)
		edge_ls.append(curr+' '+s_1+' '+o_1)
		edge_ls.append(curr+' '+s_2+' '+o_2)

	return edge_ls


def read_graph(edge_list):
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.parse_edgelist(edge_list, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.parse_edgelist(edge_list, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
	
	kv=model.wv
	node_idx=kv.key_to_index
	emb_vecs={}
	for node in node_idx.keys():
		emb_vecs[node]=kv[node]
	return emb_vecs

def generate_emb_vecs(encode):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	edge_list=encode_to_edge_list(encode)
	nx_G = read_graph(edge_list)
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	# emb vecs has 7 keys:0,1,...,6
	emb_vecs=learn_embeddings(walks)
	return emb_vecs


# if __name__ == "__main__":
# 	encode='[1 1 8 1 2 1 1 3 1 2 6 6 4 3 6 6 1 3 5 1 0 1 1 0 0 0 5 6 3 2 3 6 0 1 3 6 2 0 2 3]'
# 	encode_2='[1 0 6 6 2 2 0 0 3 0 6 1 4 0 7 7 2 0 0 4 0 1 3 5 1 1 2 3 1 0 8 1 4 1 4 1 5 4 8 7]'
# 	emb_vecs=generate_emb_vecs(encode_2)
# 	print(emb_vecs['6'])
	

	
