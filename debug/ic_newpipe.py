import tensorflow_gnn as tfgnn
#import tensorflow_datasets as tfds
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



#Last edit: 1027 we have id in context now
#Last edit: 1102 we have ortho filter for interstring things now
##########################

graph_schema = tfgnn.read_schema("/n/home05/tzhu/work/icgen2/algorithm/ic_schema_1231.txt")
gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

interfilter = False
index = 2
efffile = f'/n/home05/tzhu/work/icgen2/simu/ortho_eff_{index}.npy'
#########################
class TimesNotOrderedError(Exception):
    """Rasied if hit times are not in ascending order"""
    def __init__(self):
        self.message = "Input times are not in ascending order"
        super().__init__(self.message)

class IncompatibleLengthsError(Exception):

    def __init__(self, id_type, nts, nids):
        self.message = f"Number of {id_type} incompatible with number of times."
        self.message += f"Expected {nts} but got {nids}"
        super().__init__(self.message)

def has_HLC(
    string_ids,
    sensor_ids,
    times,
    hlc_dt = 5000.0
):
    #TONG
    #hlc_dt = 5000.0
    has_hlc = False
    rstring_ids = string_ids[::-1]
    rsensor_ids = sensor_ids[::-1]
    rtimes = times[::-1]
    for idx, (time, sensor_id, string_id) in enumerate(zip(rtimes, rsensor_ids, rstring_ids)):
        slc = slice(idx+1, None, None)
        is_neighbor = np.logical_and(
            0 != np.abs(sensor_ids[slc] - sensor_id),
            np.abs(sensor_ids[slc] - sensor_id) <= 2
        )
        is_samestring = string_ids[slc] == string_id
        can_hlc = np.logical_and(is_neighbor, is_samestring)
        did_hlc = np.abs(times[slc][can_hlc] - time) <= hlc_dt
        hlc_times = times[slc][can_hlc][did_hlc]
        if len(hlc_times) > 0:
            has_hlc = True
            break
    if has_hlc:
        return has_hlc, time
    else:
        return has_hlc, np.max(times)


def SMT(
    string_ids,
    sensor_ids,
    times,
    multiplicity:int = 8,
    hlc_dt:float = 5000.0,
) -> bool:
    """
    Function to determine if event passed SMT-N
    """
    # Make sure lengths are compatible
    ntimes = len(times)
    if len(sensor_ids) != ntimes:
        raise IncompatibleLengthsError("sensor_ids", ntimes, len(sensor_ids))
    if len(string_ids) != ntimes:
        raise IncompatibleLengthsError("string_ids", ntimes, len(string_ids))

    # Make sure times are ordered
    if np.any(np.diff(times) < 0):
        raise TimesNotOrderedError

    times = np.array(times)
    sensor_ids = np.array(sensor_ids)
    string_ids = np.array(string_ids)
    n_hlc = 0
    hlc_times = np.array([])
    did_smt = False
    for idx, (time, sensor_id, string_id) in enumerate(zip(times, sensor_ids, string_ids)):
        slc = slice(idx+1, None, None)
        is_neighbor = np.logical_and(
            0 != np.abs(sensor_ids[slc] - sensor_id),
            np.abs(sensor_ids[slc] - sensor_id) <= 2
        )
        is_samestring = string_ids[slc] == string_id
        can_hlc = np.logical_and(is_neighbor, is_samestring)
        did_hlc = times[slc][can_hlc] - time <= hlc_dt
        n_hlc += np.count_nonzero(did_hlc)
        hlc_times = np.append(hlc_times, times[slc][can_hlc][did_hlc])
        if n_hlc >= multiplicity:
            did_smt = True
            break
    if not did_smt:
        return False, None, None
    # Find when the trigger should stop recording
    has_hlc = True
    min_time = np.max(hlc_times)
    while has_hlc:
        max_time = min_time + hlc_dt
        mask = np.logical_and(
            min_time < times,
            times < max_time
        )
        if len(string_ids[mask])==0:
            has_hlc = False
            min_time = min_time + hlc_dt
        else:
            has_hlc, min_time = has_HLC(
            string_ids[mask],
            sensor_ids[mask],
            times[mask],
            hlc_dt=hlc_dt
        )
    return did_smt, np.min(hlc_times), max(np.min(hlc_times)+hlc_dt, min_time)

########################
def makeGT(id,counter):
    print("Generating GraphTensor for this event: ID =",id)
    hit_num = len(simu.loc[id][2]['sensor_pos_x'])+len(simu.loc[id][3]['sensor_pos_x'])
    hits = np.array([np.append(simu.loc[id][2]['t'],simu.loc[id][3]['t']),np.append(simu.loc[id][2]['sensor_pos_x'],simu.loc[id][3]['sensor_pos_x']),np.append(simu.loc[id][2]['sensor_pos_y'],simu.loc[id][3]['sensor_pos_y']),np.append(simu.loc[id][2]['sensor_pos_z'],simu.loc[id][3]['sensor_pos_z']),np.append(simu.loc[id][2]['string_id'],simu.loc[id][3]['string_id']),np.append(simu.loc[id][2]['sensor_id'],simu.loc[id][3]['sensor_id'])]).transpose()

    hits = hits[hits[:,0].argsort()] 
    ind = []
    for i in range(hit_num):
        if hits[i,0]<0: ind.append(i)
        else:break
    hits = np.delete(hits,ind,axis=0)
    hit_num = hit_num-len(ind)
    #print(f'####:{hit_num}')

    ssid = hits[:,4:6]#string+sensor
    hits = hits[:,0:4]
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
    fbjorkeny = simu.loc[id][1]['injection_bjorkeny']

    ###place cut here###
    ###SMT8###
    smt8 = SMT(ssid[:,0],ssid[:,1],hits[:,0],multiplicity=8,hlc_dt=5000.0)
    smt12 = SMT(ssid[:,0],ssid[:,1],hits[:,0],multiplicity=12,hlc_dt=5000.0)

    
    if smt12==False:
        return 0,0
    
    rank = np.arange(0,len(hits))[:,np.newaxis]
    hits = np.concatenate((hits,rank),axis=1)
    uhits,inver,uindex,ucounts = np.unique(hits[:,1:4],axis=0,return_inverse=True,return_index=True,return_counts=True)
    suhits = hits[inver][np.argsort(hits[inver][:,0])]
    source_t = np.array([])
    target_t = np.array([])
    metric_t = np.array([])
    for i in range(len(suhits)):
        if ucounts[i]==1: continue
        DOM = hits[np.isin(uindex,i)]
        if ucounts[i]==2:
            source = np.array([DOM[0,-1]])
            target = np.array([DOM[1,-1]])

        elif ucounts[i]==3:
            source = np.array([DOM[0,-1],DOM[0,-1],DOM[1,-1]])
            target = np.array([DOM[1,-1],DOM[1,-1],DOM[2,-1]])
        else:
            plus = np.sum((DOM[:,0]-DOM[0,0])>50)
            source =   [DOM[0,-1] for x in range(len(DOM)-1+plus)]
            source = np.concatenate((source,DOM[1:(len(DOM)-1),-1],DOM[1:(len(DOM)-2),-1]))
            target = np.concatenate((DOM[1:,-1],DOM[:,-1][(DOM[:,0]-DOM[0,0])>50],DOM[2:,-1],DOM[3:,-1]))
        
        nmetric = (hits[np.int32(target),0]-hits[np.int32(source),0])*0.3
        source_t = np.append(source_t,source)
        target_t = np.append(target_t,target)
        metric_t = np.append(metric_t,nmetric)
    source_u = np.concatenate((suhits[:len(suhits)-1,-1],suhits[:len(suhits)-2,-1]))
    target_u =np.concatenate(( suhits[1:,-1],suhits[2:,-1]))
    metric_u = (hits[np.int32(target_u)][:,0:4]-hits[np.int32(source_u)][:,0:4])*[0.3,1,1,1]
    metric_u = np.linalg.norm(metric_u,axis=1)
    hit_sources = np.concatenate((source_t,source_u,suhits[:,-1]))
    hit_targets = np.concatenate((source_t,source_u,suhits[:,-1]))
    u_m = np.concatenate(([metric_u**2],[metric_u**-1],[metric_u**-2],[np.zeros(len(metric_u))]),axis = 0).T
    t_m = np.concatenate(([metric_t**2],[metric_t**-1],[metric_t**-2],[np.ones(len(metric_t))]),axis = 0).T
    s_m = np.concatenate(([np.zeros(len(suhits))],[np.ones(len(suhits))],[np.ones(len(suhits))],[np.zeros(len(suhits))]),axis = 0).T
    metrics = np.concatenate((t_m,u_m,s_m))
    
    hits = hits[:,:4]
    #print(tmp)

    hit_adjacency = tfgnn.Adjacency.from_indices(source=("hit",tf.cast(hit_sources,dtype=tf.int32)),target=("hit",tf.cast(hit_targets,dtype=tf.int32)))
    #print("@@@@@@@@@@")
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
        features={"injection_zenith":[fzenith],"event_energy":[fenergy],"injection_azimuth":[fazimuth],"event_id":[feventid],"hit_num":[fhitnum],"cc":tf.cast([fcc],dtype=tf.float32)})

    graphtensor = tfgnn.GraphTensor.from_pieces(node_sets={"hit":hit},edge_sets={"coincidence":coincidence},context=context)
    #print("%%%%%%%%%%%")
    print(f"{len(metrics)} edges,{len(hits)} hits")
    return graphtensor, np.concatenate((fcc,[fenergy,fzenith,fazimuth,hit_num,fbjorkeny,smt12[0],cut60,smt8[0]]),axis = 0)



##############################
counter =0
filename = 'ICnewtrain'

mapinfo = []
#with tf.io.TFRecordWriter('/n/holyscratch01/arguelles_delgado_lab/Everyone/tzhu/ortho300SAM_test') as writer:
#with tf.io.TFRecordWriter(f'/n/holyscratch01/arguelles_delgado_lab/Everyone/tzhu/SAM_6nnIDmetrics_1_70') as writer:
with tf.io.TFRecordWriter(f'/n/holyscratch01/arguelles_delgado_lab/Everyone/tzhu/{filename}') as writer:
        simulationpath = "/n/holyscratch01/arguelles_delgado_lab/Everyone/felixyu/ic_ssnet_sim_smt_round2.parquet"

        simu = rp(simulationpath)
        #print(f'#######begin file {x}#######')
        for i in range(149627):
            graph,info = makeGT(i,counter)
            if graph ==0:
                continue
            else:
                counter+=1
                mapinfo.append(np.concatenate((info,[i]),axis = 0))
                example = tfgnn.write_example(graph)
                writer.write(example.SerializeToString())
        print(f'Now we have {counter} events in file')

#################################
print(counter)
mapinfo = np.array(mapinfo)
np.save(f'/n/home05/tzhu/work/icgen2/simu/{filename}_info',mapinfo)
print("successfully saved mapinfo. Have a nice day!")
