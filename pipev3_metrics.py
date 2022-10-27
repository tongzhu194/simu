#import pygraphviz as pgv
from tqdm import tqdm
#from IPython.display import Image

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import tensorflow_gnn as tfgnn
#import tensorflow_datasets as tfds

from tensorflow_gnn import runner
from tensorflow_gnn.models import gat_v2

import networkx as nx
import numpy as np

#import pandas as pd
from pandas import read_parquet as rp

print(f'Using TensorFlow v{tf.__version__} and TensorFlow-GNN v{tfgnn.__version__}')
print(f'GPUs available: {tf.config.list_physical_devices("GPU")}')

import knndatap


##########################

graph_schema = tfgnn.read_schema("/n/home05/tzhu/work/icgen2/algorithm/schema_v3_metrics.txt")
gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)


########################
def makeGT(id):
    print("Generating GraphTensor for this event: ID =",id)
    hit_num = len(simu.loc[id][2]['sensor_pos_x'])+len(simu.loc[id][3]['sensor_pos_x'])
    hits = np.array([np.append(simu.loc[id][2]['t'],simu.loc[id][3]['t']),np.append(simu.loc[id][2]['sensor_pos_x'],simu.loc[id][3]['sensor_pos_x']),np.append(simu.loc[id][2]['sensor_pos_y'],simu.loc[id][3]['sensor_pos_y']),np.append(simu.loc[id][2]['sensor_pos_z'],simu.loc[id][3]['sensor_pos_z'])]).transpose()
    hits = hits[hits[:,0].argsort()] 
    ind = []
    for i in range(hit_num):
        if hits[i,0]<0: ind.append(i)
        else:break
    hits = np.delete(hits,ind,axis=0)
    hit_num = hit_num-len(ind)
    print("There is",hit_num,"hit(s) in this event.")
    #print(hits)
    fenergy = simu.loc[id][1]['injection_energy']
    fzenith = simu.loc[id][1]['injection_zenith']
    fazimuth = simu.loc[id][1]['injection_azimuth']
    
    hits_input = hits.flatten()

    stbinding = knndatap.knndatap(hits_input,hit_num)

    
    if bool(stbinding)==False:
        return 0
    

    #print(tmp)
    hit_sources = stbinding[1::3]
    hit_targets = stbinding[2::3]
    metrics = np.array(stbinding[3::3])
    metrics= metrics[:,np.newaxis]
    #for i in range(5):
        #print(hit_sources[i],hit_targets[i])

    #print("tmp",len(tmp))
    #print('**********')
    hit_adjacency = tfgnn.Adjacency.from_indices(source=("hit",tf.cast(hit_sources,dtype=tf.int32)),target=("hit",tf.cast(hit_targets,dtype=tf.int32)))
    #print("@@@@@@@@@@")
    ###generate GT###
    hit = tfgnn.NodeSet.from_fields(
        sizes = [hit_num],
        features={ 
            "4Dvec":tf.cast(hits,dtype=tf.float32),
        })
    coincidence = tfgnn.EdgeSet.from_fields(
        sizes = tf.shape(hit_sources),
        features={
            "metric": tf.cast(metrics,dtype=tf.float32)
        },
        adjacency=hit_adjacency)

    context = tfgnn.Context.from_fields(
        features={"injection_zenith":[fzenith],"event_energy":[fenergy],"injection_azimuth":[fazimuth]})

    graphtensor = tfgnn.GraphTensor.from_pieces(node_sets={"hit":hit},edge_sets={"coincidence":coincidence},context=context)
    #print("%%%%%%%%%%%")
    return graphtensor



##############################
counter =0
with tf.io.TFRecordWriter('/n/home05/tzhu/work/icgen2/data_graph/SAM_E6nnmetrics_1050_1450') as writer:
    for x in range(10,15):
        simulationpath = '/n/home05/tzhu/work/icgen2/gputest2/data/EMinus_Hadrons_seed_'+str(x)+'50_meta_data.parquet'
        simu = rp(simulationpath)
        print(f'#######begin file {x}#######')
        for i in range(1000):
            graph = makeGT(i)
            if graph ==0:
                counter+=1
                continue
            else:
                example = tfgnn.write_example(graph)
                writer.write(example.SerializeToString())

#################################
print(counter)
