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

#Last edit: 1027 we have id in context now
#Last edit: 1102 we have ortho filter for interstring things now
##########################

graph_schema = tfgnn.read_schema("/n/home05/tzhu/work/icgen2/algorithm/ic_schema_1127.txt")
gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

interfilter = False
index = 2
efffile = f'/n/home05/tzhu/work/icgen2/simu/ortho_eff_{index}.npy'

########################
def makeGT(id,counter):
    print("Generating GraphTensor for this event: ID =",id)
    hit_num = len(simu.loc[id][2]['sensor_pos_x'])+len(simu.loc[id][3]['sensor_pos_x'])
    hits = np.array([np.append(simu.loc[id][2]['t'],simu.loc[id][3]['t']),np.append(simu.loc[id][2]['sensor_pos_x'],simu.loc[id][3]['sensor_pos_x']),np.append(simu.loc[id][2]['sensor_pos_y'],simu.loc[id][3]['sensor_pos_y']),np.append(simu.loc[id][2]['sensor_pos_z'],simu.loc[id][3]['sensor_pos_z']),np.append(simu.loc[id][2]['string_id'],simu.loc[id][3]['string_id'])]).transpose()
 
    hits = hits[hits[:,0].argsort()] 
    ind = []
    for i in range(hit_num):
        if hits[i,0]<0: ind.append(i)
        else:break
    hits = np.delete(hits,ind,axis=0)
    hit_num = hit_num-len(ind)
    #print(f'####:{hit_num}')
    if interfilter == True:
        eff = np.load(efffile)
        map = np.isin(hits[:,4],eff)
        hits = hits[map]
        hits = hits[:,0:4]
        hit_num= len(hits)
    else:
        hits = hits[:,0:4]
    print("There is",hit_num,"hit(s) in this event.")
    #print(hits)
    tmin = hits[0,0]
    tmax = hits[-1,0]
    fmean = np.mean(hits,axis=0)
    fcc = np.concatenate(([tmin,tmax],fmean),axis = 0)
    fenergy = simu.loc[id][1]['injection_energy']
    fzenith = simu.loc[id][1]['injection_zenith']
    fazimuth = simu.loc[id][1]['injection_azimuth']
    feventid = tf.cast(counter,dtype=tf.int64)
    fhitnum = tf.cast(hit_num,dtype = tf.int64) 
    hits_input = hits.flatten()
    if(hit_num>1300000):
        return 0
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
        features={"injection_zenith":[fzenith],"event_energy":[fenergy],"injection_azimuth":[fazimuth],"event_id":[feventid],"hit_num":[fhitnum],"cc":tf.cast(fcc,dtype=tf.float32)})

    graphtensor = tfgnn.GraphTensor.from_pieces(node_sets={"hit":hit},edge_sets={"coincidence":coincidence},context=context)
    #print("%%%%%%%%%%%")
    return graphtensor



##############################
counter =0
with tf.io.TFRecordWriter('/n/holyscratch01/arguelles_delgado_lab/Everyone/tzhu/IC_Mu_test') as writer:
#with tf.io.TFRecordWriter(f'/n/holyscratch01/arguelles_delgado_lab/Everyone/tzhu/SAM_6nnIDmetrics_1_70') as writer:
    for x in range(1,31):
        #simulationpath = '/n/holyscratch01/arguelles_delgado_lab/Everyone/tzhu/simu/hexa/data/MuMinus_Hadrons_seed_'+str(x)+'_meta_data.parquet'
        #simulationpath = '/n/home05/tzhu/work/icgen2/gputest2/data/MuMinus_Hadrons_seed_'+str(x)+'_meta_data.parquet'
        #simulationpath = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/tzhu/simudata/ortho240/data/MuMinus_Hadrons_seed_'+str(x)+'_meta_data.parquet'
        empty=[17,18,19,20,27]
        if(np.isin(x,empty)==True):
            continue

        simulationpath = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/jlazar/ic_ssnet_sim/data/MuMinus_Hadrons_seed_'+str(x)+'_meta_data.parquet'

        simu = rp(simulationpath)
        print(f'#######begin file {x}#######')
        for i in range(5000):
            graph = makeGT(i,counter)
            if graph ==0:
                continue
            else:
                counter+=1
                example = tfgnn.write_example(graph)
                writer.write(example.SerializeToString())
        print(f'Now we have {counter} events in file')

#################################
print(counter)
