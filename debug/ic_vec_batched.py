import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn import runner
from tensorflow_gnn.models import gcn

import numpy as np
import matplotlib.pyplot as plt
graph_schema = tfgnn.read_schema("/n/home05/tzhu/work/icgen2/algorithm/ic_schema_1127.txt")
gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)



gpus = tf.config.list_physical_devices(device_type = 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
###############################
D=4000.
###############################
def extract_labels(graph_tensor):
    zenith = graph_tensor.context['injection_zenith']
    azimuth = graph_tensor.context['injection_azimuth']
    true_vec = [tf.sin(zenith)*tf.cos(azimuth),tf.sin(zenith)*tf.sin(azimuth),tf.cos(zenith)]
    return graph_tensor,true_vec

################################
def initialize_args():
    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument(
        "-md",
        dest="mdfile",
        type=str,
        required=True
    )
    parser.add_argument(
        "-date",
        dest="date",
        type=int,
        required=True
    )
    parser.add_argument(
        "-sub",
        dest="sub",
        type=int,
        required=True
    )
    args = parser.parse_args()
    return args


    args = parser.parse_args()
    return args

args = initialize_args()
mdfile = args.mdfile
date = args.date
sub = args.sub

##############################
train_filepattern = f"/n/holyscratch01/arguelles_delgado_lab/Everyone/tzhu/ICbigtrain_{sub}"
valid_filepattern = f"/n/holyscratch01/arguelles_delgado_lab/Everyone/tzhu/ICbigvalid_{sub}"


print(train_filepattern)
print(valid_filepattern)
train_ds_provider = runner.TFRecordDatasetProvider(file_pattern=train_filepattern)
valid_ds_provider = runner.TFRecordDatasetProvider(file_pattern=valid_filepattern)

##############################
train_dataset = train_ds_provider.get_dataset(context=tf.distribute.InputContext())
train_dataset = train_dataset.map(lambda serialized: tfgnn.parse_single_example(serialized=serialized,spec=gtspec))
train_dataset = train_dataset.map(lambda graph_tensor: extract_labels(graph_tensor=graph_tensor))


valid_dataset = valid_ds_provider.get_dataset(context=tf.distribute.InputContext())
valid_dataset = valid_dataset.map(lambda serialized: tfgnn.parse_single_example(serialized=serialized,spec=gtspec))
valid_dataset = valid_dataset.map(lambda graph_tensor: extract_labels(graph_tensor=graph_tensor))

train_Number = 0
for i,data in train_dataset.enumerate():
        train_Number +=1
valid_Number =0
for i,data in valid_dataset.enumerate():
        valid_Number+=1

print("train data:",train_Number)
print("valid data:",valid_Number)

batch_size = 16
test_trds = train_ds_provider.get_dataset(context=tf.distribute.InputContext())
test_trds = test_trds.map(lambda serialized: tfgnn.parse_single_example(serialized=serialized,spec=gtspec))
test_trds_batched = test_trds.batch(batch_size=batch_size)

valid_trds = valid_ds_provider.get_dataset(context=tf.distribute.InputContext())
valid_trds = valid_trds.map(lambda serialized: tfgnn.parse_single_example(serialized=serialized,spec=gtspec))
valid_trds_batched = valid_trds.batch(batch_size=batch_size)



def merge(graph_tensor_batch):
    return graph_tensor_batch.merge_batch_to_components()

def spextract_labels(graph_tensor):
    zenith = graph_tensor.context['injection_zenith']
    azimuth = graph_tensor.context['injection_azimuth']
    true_vec = tf.transpose([tf.sin(zenith)*tf.cos(azimuth),tf.sin(zenith)*tf.sin(azimuth),tf.cos(zenith)])
    return graph_tensor,true_vec 


test_trds_scalar = test_trds_batched.map(lambda graph_tensor_batch:merge(graph_tensor_batch=graph_tensor_batch))
test_trds_scalar = test_trds_scalar.map(lambda graph_tensor: spextract_labels(graph_tensor=graph_tensor))
valid_trds_scalar = valid_trds_batched.map(lambda graph_tensor_batch:merge(graph_tensor_batch=graph_tensor_batch))
valid_trds_scalar = valid_trds_scalar.map(lambda graph_tensor: spextract_labels(graph_tensor=graph_tensor))


###########################
class Gather(tf.keras.layers.Layer):
    def __init__(self,hidden_size,name='compute_attention',**kwargs):
        super().__init__(name=name,**kwargs)
        self.hidden_size = hidden_size
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'hops': self.hops
        })
        return config


    def call(self,graph:tfgnn.GraphTensor,edge_set_name='coincidence'):
        
        #return tf.keras.layers.Dense(units=self.hidden_size,activation='relu')(pooled)
        feat = graph.edge_sets['coincidence']['hidden_state']   
        source_bcast = tfgnn.broadcast_node_to_edges(graph,edge_set_name='coincidence',node_tag=tfgnn.SOURCE,feature_name=tfgnn.HIDDEN_STATE)
        target_bcast = tfgnn.broadcast_node_to_edges(graph,edge_set_name='coincidence',node_tag=tfgnn.TARGET,feature_name=tfgnn.HIDDEN_STATE)
        new = tf.concat([feat,source_bcast,target_bcast],axis =1)

        return new
   
###########################
def edge_gatherinfo(name='edgegather'):
    return tfgnn.keras.layers.GraphUpdate(
         edge_sets={
            "coincidence": Gather(hidden_size=16)}#the hid unused
        ,name = name)

#############################
def get_norm_map():

    def node_sets_fn(node_set, node_set_name):
        if node_set_name == 'hit':
            return (node_set['4Dvec']/D)
    def edge_sets_fn(edge_set,edge_set_name):
        if edge_set_name =='coincidence':
            return (edge_set['metric']/D)
    return tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn,edge_sets_fn=edge_sets_fn,name='normalize')

def midenseEdge(hidden_size=12,activation='relu',name='mid'):
    def edge_sets_fn(edge_set,edge_set_name):
        if edge_set_name =='coincidence':
            return tf.keras.layers.Dense(units=hidden_size,activation=activation)(edge_set['hidden_state'])
    return tfgnn.keras.layers.MapFeatures(edge_sets_fn=edge_sets_fn,name=name)

def bi_get_initial_map_features(hidden_size, activation='tanh',name='undefined'):
    """
    Initial pre-processing layer for a GNN (use as a class constructor).
    """
    def dense(units):
        return tf.keras.layers.Dense(units=units,activation='relu')
    def node_sets_fn(node_set, node_set_name):
        if node_set_name == 'hit':
            return tf.keras.layers.Dense(units=hidden_size, activation=activation)(node_set['hidden_state'])
            #return (node_set['4Dvec']/D)
    def edge_sets_fn(edge_set,edge_set_name):
        if edge_set_name =='coincidence':
            return tf.keras.layers.Dense(1,activation='sigmoid')(edge_set['hidden_state'])
    
    return tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn,edge_sets_fn=edge_sets_fn,name=name)


def get_initial_map_features(hidden_size, activation='tanh',name='undefined'):
    """
    Initial pre-processing layer for a GNN (use as a class constructor).
    """
    def node_sets_fn(node_set, node_set_name):
        if node_set_name == 'hit':
            return tf.keras.layers.Dense(units=hidden_size, activation=activation)(node_set['hidden_state'])
            #return (node_set['4Dvec']/D)
    
    return tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn,name=name)


#######pseudo Cell only for visualization################
################
class custom_d_MPNN(tf.keras.layers.Layer):
    def __init__(self,hidden_size,name='test_gcn',**kwargs):
        super().__init__(name=name,**kwargs)
        self.hidden_size = hidden_size
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'hops': self.hops
        })
        return config


    def call(self,graph:tfgnn.GraphTensor,edge_set_name='coincidence'):
        edge_adj = graph.edge_sets[edge_set_name].adjacency
        nnodes = tf.cast(graph.node_sets['hit'].total_size, tf.int64)
        float_type = graph.node_sets['hit']['hidden_state'].dtype
        edge_set = graph.edge_sets[edge_set_name]

        D1= tf.squeeze(tfgnn.pool_edges_to_node(graph,edge_set_name,tfgnn.TARGET,'sum',feature_name=tfgnn.HIDDEN_STATE), -1)
        #D2 = tf.squeeze(tfgnn.pool_edges_to_node(graph,edge_set_name,tfgnn.SOURCE,'sum',feature_value=edge_ones), -1)
        in_degree = D1
        in_degree += 1
        invsqrt_deg = tf.math.rsqrt(in_degree)
  

        # Calculate \hat{D^{-1/2}}X first
        normalized_values = (graph.node_sets['hit']['hidden_state'])

        # Calculate A\hat{D^{-1/2}}X by broadcasting then pooling
        
        source_bcast2 = tfgnn.broadcast_node_to_edges(
            graph,
            edge_set_name,
            tfgnn.SOURCE,
            feature_value=normalized_values
        )

        weights = edge_set['hidden_state']
        
        weighted_source = weights*source_bcast2

        pooled = tfgnn.pool_edges_to_node(graph, edge_set_name, tfgnn.TARGET, 'sum', feature_value=weighted_source)
 
        # left-multiply the result by \hat{D^{-1/2}}
        pooled = invsqrt_deg[:, tf.newaxis] * pooled
        pooled = invsqrt_deg[:, tf.newaxis] * pooled
        pooled += invsqrt_deg[:, tf.newaxis] * (invsqrt_deg[:, tf.newaxis] * normalized_values)

        #return tf.keras.layers.Dense(units=self.hidden_size,activation='relu')(pooled)
        return pooled
   
   

#########################

def message_passing(hidden_size ,name):
    return tfgnn.keras.layers.GraphUpdate(
         node_sets={
            "hit": tfgnn.keras.layers.NodeSetUpdate(
                {"coincidence": custom_d_MPNN(hidden_size=hidden_size)},
                tfgnn.keras.layers.SingleInputNextState())}
        ,name = name)

#########################
##########Task define abandoned later########
class GraphMeanSquaredError(runner.GraphMeanSquaredError):
    def __init__(self, hidden_dim, *args, **kwargs):
        self._hidden_dim = hidden_dim
        super().__init__(*args, **kwargs)     
               
    def adapt(self, model):
        hidden_state = tfgnn.pool_nodes_to_context(model.output,
                                                   node_set_name=self._node_set_name,
                                                   reduce_type=self._reduce_type,
                                                   feature_name=self._state_name)
        hidden_state = tf.keras.layers.Dense(units=self._hidden_dim, activation='relu', name='hidden_layer')(hidden_state)
        logits = tf.keras.layers.Dense(units=self._units, name='logits')(hidden_state)
        print(self._units)
        return tf.keras.Model(inputs=model.inputs, outputs=logits)

task = GraphMeanSquaredError(hidden_dim=16,node_set_name = 'hit')
################################

def complex_mpnn_model(hidden_size_emb,hidden_gcn1,hidden_gcn2,hidden_gcn3,hidden_gcn4,graph_tensor_spec):
    norm_fn = get_norm_map()
    edgegather = edge_gatherinfo(name='gather_pre')
    midE1 = midenseEdge(hidden_size=12,activation='relu',name='midpre1')
    midE2 = midenseEdge(hidden_size=36,activation='relu',name='midpre2')
    midE3 = midenseEdge(hidden_size=9,activation='relu',name='midpre3')
    init_states_fn = bi_get_initial_map_features(hidden_size=hidden_size_emb,name='embedding')

    message_passing_fn_1 = message_passing(hidden_size = hidden_gcn1,name='message_passing_1')
    message_passing_fn_2 = message_passing(hidden_size = hidden_gcn2,name = 'message_passing_2')
    message_passing_fn_3 = message_passing(hidden_size = hidden_gcn3,name = 'message_passing_3')
    message_passing_fn_4 = message_passing(hidden_size = hidden_gcn4,name = 'message_passing_4')
    #message_passing_fn_5 = message_passing(hidden_size = hidden_gcn5,name = 'message_passing_5')
    #message_passing_fn_6 = message_passing(hidden_size = hidden_gcn6,name = 'message_passing_6')
    node_dense_fn_1=get_initial_map_features(hidden_size=hidden_gcn1,name='gcn1')
    node_dense_fn_2=get_initial_map_features(hidden_size=hidden_gcn2,name='gcn2')
    node_dense_fn_3=get_initial_map_features(hidden_size=hidden_gcn3,name='gcn3')
    node_dense_fn_4=get_initial_map_features(hidden_size=hidden_gcn4,name='gcn4')
    #node_dense_fn_5=get_initial_map_features(hidden_size=hidden_gcn5,name='gcn5')
    #node_dense_fn_6=get_initial_map_features(hidden_size=hidden_gcn6,name='gcn6')

    graph_tensor = tf.keras.layers.Input(type_spec = graph_tensor_spec)
    normalized_graph = norm_fn(graph_tensor)
    gathered_graph = edgegather(normalized_graph)
    gathered_graph = midE1(gathered_graph)
    gathered_graph = midE2(gathered_graph)
    gathered_graph = midE3(gathered_graph)
    embedded_graph = init_states_fn(gathered_graph)

    passed_graph_1 = message_passing_fn_1(embedded_graph)
    gcned_graph_1 = node_dense_fn_1(passed_graph_1)

    edgegather = edge_gatherinfo(name='gather_1')
    #midE1 = midenseEdge(hidden_size=9,activation='relu',name='mid1_1')
    #midE2 = midenseEdge(hidden_size=36,activation='relu',name='mid1_2')
    #midE3 = midenseEdge(hidden_size=3,activation='relu',name='mid1_3')
    midE4 = midenseEdge(hidden_size=1,activation='sigmoid',name='mid1_4')
    gcned_graph_1= edgegather(gcned_graph_1)
    #gcned_graph_1=midE1(gcned_graph_1)
    #gcned_graph_1=midE2(gcned_graph_1)
    #gcned_graph_1=midE3(gcned_graph_1)
    gcned_graph_1=midE4(gcned_graph_1)

    graph_pool_1_mean = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'mean',node_set_name='hit',name='pool1_mean')(gcned_graph_1)
    graph_pool_1_max = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'max',node_set_name='hit',name='pool1_max')(gcned_graph_1)
    graph_pool_1_min = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'min',node_set_name='hit',name='pool1_min')(gcned_graph_1)
    graph_pool_1 = tf.concat([graph_pool_1_mean,graph_pool_1_max,graph_pool_1_min],axis = 1)
    graph_pool_1 = tf.keras.layers.Dense(units=12,activation='relu'  )(graph_pool_1)
    graph_pool_1 = tf.keras.layers.Dense(units=8,activation='relu' )(graph_pool_1)
    graph_pool_1 = tf.keras.layers.Dense(units=6,activation='relu',name ='embed_pool_1' )(graph_pool_1)
    


    passed_graph_2 = message_passing_fn_2(gcned_graph_1)
    gcned_graph_2 = node_dense_fn_2(passed_graph_2)

    edgegather = edge_gatherinfo(name='gather_2')
    #midE1 = midenseEdge(hidden_size=9,activation='relu',name='mid2_1')
    #midE2 = midenseEdge(hidden_size=36,activation='relu',name='mid2_2')
    #midE3 = midenseEdge(hidden_size=3,activation='relu',name='mid2_3')
    midE4 = midenseEdge(hidden_size=1,activation='sigmoid',name='mid2_4')
    gcned_graph_2= edgegather(gcned_graph_2)
    #gcned_graph_2=midE1(gcned_graph_2)
    #gcned_graph_2=midE2(gcned_graph_2)
    #gcned_graph_2=midE3(gcned_graph_2)
    gcned_graph_2=midE4(gcned_graph_2)
    graph_pool_2_mean = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'mean',node_set_name='hit',name='pool2_mean')(gcned_graph_2)
    graph_pool_2_max = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'max',node_set_name='hit',name='pool2_max')(gcned_graph_2)
    graph_pool_2_min = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'min',node_set_name='hit',name='pool2_min')(gcned_graph_2)
    graph_pool_2 = tf.concat([graph_pool_2_mean,graph_pool_2_max,graph_pool_2_min],axis = 1)
    graph_pool_2 = tf.keras.layers.Dense(units=12,activation='relu' )(graph_pool_2)
    graph_pool_2 = tf.keras.layers.Dense(units=8,activation='relu' )(graph_pool_2)
    graph_pool_2 = tf.keras.layers.Dense(units=6,activation='relu' ,name ='embed_pool_2')(graph_pool_2)
    
    


    passed_graph_3 = message_passing_fn_3(gcned_graph_2)
    gcned_graph_3 = node_dense_fn_3(passed_graph_3)

    edgegather = edge_gatherinfo(name='gather_3')
    #midE1 = midenseEdge(hidden_size=9,activation='relu',name='mid3_1')
    #midE2 = midenseEdge(hidden_size=36,activation='relu',name='mid3_2')
    #midE3 = midenseEdge(hidden_size=3,activation='relu',name='mid3_3')
    midE4 = midenseEdge(hidden_size=1,activation='sigmoid',name='mid3_4')
    gcned_graph_3= edgegather(gcned_graph_3)
    #gcned_graph_3=midE1(gcned_graph_3)
    #gcned_graph_3=midE2(gcned_graph_3)
    #gcned_graph_3=midE3(gcned_graph_3)
    gcned_graph_3=midE4(gcned_graph_3)
    graph_pool_3_mean = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'mean',node_set_name='hit',name='pool3_mean')(gcned_graph_3)
    graph_pool_3_max = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'max',node_set_name='hit',name='pool3_max')(gcned_graph_3)
    graph_pool_3_min = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'min',node_set_name='hit',name='pool3_min')(gcned_graph_3)
    graph_pool_3 = tf.concat([graph_pool_3_mean,graph_pool_3_max,graph_pool_3_min],axis = 1)
    graph_pool_3 = tf.keras.layers.Dense(units=12,activation='relu')(graph_pool_3)
    graph_pool_3 = tf.keras.layers.Dense(units=8,activation='relu' )(graph_pool_3)
    graph_pool_3 = tf.keras.layers.Dense(units=6,activation='relu',name ='embed_pool_3' )(graph_pool_3)

    passed_graph_4 = message_passing_fn_4(gcned_graph_3)
    gcned_graph_4 = node_dense_fn_4(passed_graph_4)
    edgegather = edge_gatherinfo(name='gather_4')
    #midE1 = midenseEdge(hidden_size=9,activation='relu',name='mid4_1')
    #midE2 = midenseEdge(hidden_size=36,activation='relu',name='mid4_2')
    #midE3 = midenseEdge(hidden_size=3,activation='relu',name='mid4_3')
    midE4 = midenseEdge(hidden_size=1,activation='sigmoid',name='mid4_4')
    gcned_graph_4= edgegather(gcned_graph_4)
    #gcned_graph_4=midE1(gcned_graph_4)
    #gcned_graph_4=midE2(gcned_graph_4)
    #gcned_graph_4=midE3(gcned_graph_4)
    gcned_graph_4=midE4(gcned_graph_4)
    graph_pool_4_mean = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'mean',node_set_name='hit',name='pool4_mean')(gcned_graph_4)
    graph_pool_4_max = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'max',node_set_name='hit',name='pool4_max')(gcned_graph_4)
    graph_pool_4_min = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'min',node_set_name='hit',name='pool4_min')(gcned_graph_4)
    graph_pool_4 = tf.concat([graph_pool_4_mean,graph_pool_4_max,graph_pool_4_min],axis = 1)
    graph_pool_4 = tf.keras.layers.Dense(units=12,activation='relu' )(graph_pool_4)
    graph_pool_4 = tf.keras.layers.Dense(units=8,activation='relu' )(graph_pool_4)
    graph_pool_4 = tf.keras.layers.Dense(units=6,activation='relu',name ='embed_pool_4' )(graph_pool_4)
    
    graph_pool = tf.concat([graph_pool_1,graph_pool_2,graph_pool_3,graph_pool_4],axis = 1)
    graph_pool = tf.keras.layers.Dense(units=8,activation='relu',name='dense_after_pool' )(graph_pool)
    logits = tf.keras.layers.Dense(units=3,name='logits')(graph_pool)
    #gcned_graph = directedgcn_fn(embedded_graph)

    return tf.keras.Model(inputs = graph_tensor,outputs =logits)


################################
simple_ts_model =  complex_mpnn_model(hidden_size_emb=126,hidden_gcn1=42,hidden_gcn2=56,hidden_gcn3=36,hidden_gcn4=14,graph_tensor_spec=gtspec)
simple_ts_model.summary()



#########
cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)


def AngularDifference(y_actual,y_pred):
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    p = -cosine_loss(y_actual,y_pred)
    p = tf.maximum(p,-1+1e-9)
    p = tf.minimum(p,1-1e-9)
    angle = tf.acos(p)

    return angle

def mae_cos(y_actual,y_pred):
    mae = tf.keras.losses.MeanAbsoluteError()
    s = tf.reduce_sum(y_pred**2,axis=1)
    s = tf.maximum(s,1e-9)
    s = tf.sqrt(s)

    s =tf.transpose( [s for x in range(3)])
    pred = y_pred/s

    ss = tf.reduce_sum(y_actual**2,axis=1)
    ss = tf.sqrt(ss)+1e-9
    #ss =tf.transpose( [ss for x in range(3)])
    #actual = y_actual/ss

    map = [0,0,1]
    cos_actual  = tf.reduce_sum(y_actual*map,axis=1)
    cos_pred = tf.reduce_sum(pred*map,axis=1)
    loss = tf.abs(cos_actual-cos_pred)
    return ss

def mae_zenith(y_actual,y_pred):
    mae = tf.keras.losses.MeanAbsoluteError()
    s = tf.reduce_sum(y_pred**2,axis=1)
    s = tf.maximum(s,1e-9)
    s = tf.sqrt(s)


    s =tf.transpose( [s for x in range(3)])
    pred = y_pred/s

    ss = tf.reduce_sum(y_actual**2,axis=1)
    ss = tf.sqrt(ss)+1e-9
    ss =tf.transpose( [ss for x in range(3)])
    actual = y_actual/ss

    map = [0,0,1]
    cos_actual  = tf.reduce_sum(actual*map,axis=1)
    cos_pred = tf.reduce_sum(pred*map,axis=1)
    zenith_actual =tf.acos(cos_actual)
    zenith_pred = tf.acos(cos_pred)
    loss = tf.abs(zenith_actual-zenith_pred)
    return tf.reduce_mean(pred,axis=1)

def radium(y_actual,y_pred):
    s = tf.reduce_sum(y_pred**2,axis=1)
    s = tf.maximum(s,1e-9)
    s = tf.sqrt(s)

    return s

########
metrics = [AngularDifference,mae_cos,mae_zenith,radium]
###################
list_layer=[]
for i in range(len(simple_ts_model.layers)):
    if (len(simple_ts_model.layers[i].get_weights())!=0):
        list_layer.append(i)
################
###SAVE AND LOAD
def save_model_weights(model,list_layer=list_layer,path='./'):
    for i in list_layer:
        w = model.layers[i].get_weights()
        for j in range(len(w)):
            tmp = path+f'_layer_{i}_w{j}.npy'
            np.save(tmp,w[j])
    print(f'weights are saved to {path}')


def load_model_weights(model,list_layer=list_layer,path='./'):
    for i in list_layer:
        num = len(model.layers[i].get_weights())
        w = []
        for j in range(num):
            tmp = path+f'_layer_{i}_w{j}.npy'
            w.append(np.load(tmp))
        model.layers[i].set_weights(w)
    print("weights are loaded sucessfully")



#####################
checkpoint_filepath = './checkpoint/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

termina = tf.keras.callbacks.TerminateOnNaN()

##########################
md_path = f'./md_weights/{date}_ICbig{sub}_{mdfile}/'

md_path_result = md_path+'100ep'
new_model =complex_mpnn_model(hidden_size_emb=126,hidden_gcn1=42,hidden_gcn2=56,hidden_gcn3=36,hidden_gcn4=14,graph_tensor_spec=gtspec)
read_path =  f'./md_weights/1128_ICbig{sub}_{mdfile}/'+'200ep'
load_model_weights(model=simple_ts_model,list_layer=list_layer,path=read_path)

simple_ts_model.compile(tf.keras.optimizers.Adam(learning_rate=0.001),loss=cosine_loss,metrics=metrics)
simplehistory_test = simple_ts_model.fit(test_trds_scalar,epochs = 100,verbose=2,validation_data=valid_trds_scalar, callbacks=[model_checkpoint_callback,termina])

save_model_weights(model=simple_ts_model,list_layer=list_layer,path=md_path_result)

sim2 = new_model
load_model_weights(model=sim2,list_layer=list_layer,path=md_path_result)
sim2.compile(tf.keras.optimizers.Adam(learning_rate=0.001),loss=AngularDifference,metrics=metrics)
simplehistory_test2 = sim2.fit(test_trds_scalar,epochs = 100,verbose=2,validation_data=valid_trds_scalar, callbacks=[model_checkpoint_callback,termina])

save_model_weights(model=sim2,list_layer=list_layer,path=md_path+'200ep')

simplehistory_test3 = sim2.fit(test_trds_scalar,epochs = 100,verbose=2,validation_data=valid_trds_scalar, callbacks=[model_checkpoint_callback,termina])
save_model_weights(model=sim2,list_layer=list_layer,path=md_path+'300ep')




#num =1
#for i in range(2,400):
        #read_path = md_path+f'{num}ep'
        #new_model =complex_mpnn_model(hidden_size_emb=126,hidden_gcn1=42,hidden_gcn2=56,hidden_gcn3=36,hidden_gcn4=14,graph_tensor_spec=gtspec)

        #load_model_weights(model=new_model,list_layer=list_layer,path=read_path)
        #new_model.compile(tf.keras.optimizers.Adam(learning_rate=0.001),loss=AngularDifference,metrics=metrics)
        #history = new_model.fit(test_trds_scalar,epochs = 1,verbose=2,validation_data=valid_trds_scalar, callbacks=[model_checkpoint_callback,termina])

        #num+=1
        #md_path_result = md_path+f'{num}ep'

        #save_model_weights(model=simple_ts_model,list_layer=list_layer,path=md_path_result)




