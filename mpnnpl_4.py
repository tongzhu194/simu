import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn import runner
from tensorflow_gnn.models import gcn

import numpy as np
import matplotlib.pyplot as plt
graph_schema = tfgnn.read_schema("/n/home05/tzhu/work/icgen2/algorithm/schema_v3_metrics.txt")
gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)



gpus = tf.config.list_physical_devices(device_type = 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
###############################
def extract_labels(graph_tensor):
    index = graph_tensor.context['injection_zenith']
    return graph_tensor,index


##############################
train_filepattern = "/n/holyscratch01/arguelles_delgado_lab/Everyone/tzhu/SAM_3nnIDmetrics_1_70"
valid_filepattern = "/n/holyscratch01/arguelles_delgado_lab/Everyone/tzhu/SAM_3nnIDmetrics_100_105"
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
def merge(graph_tensor_batch):
    return graph_tensor_batch.merge_batch_to_components()

test_trds_scalar = test_trds_batched.map(lambda graph_tensor_batch:merge(graph_tensor_batch=graph_tensor_batch))
test_trds_scalar = test_trds_scalar.map(lambda graph_tensor: extract_labels(graph_tensor=graph_tensor))
print("Finished batching, name=test_trds_scalar")
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
            return (node_set['4Dvec']/720.)
    def edge_sets_fn(edge_set,edge_set_name):
        if edge_set_name =='coincidence':
            return (edge_set['metric']/720.)
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
            #return (node_set['4Dvec']/720.)
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
            #return (node_set['4Dvec']/720.)
    
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
    logits = tf.keras.layers.Dense(units=1,name='logits')(graph_pool)
    #gcned_graph = directedgcn_fn(embedded_graph)

    return tf.keras.Model(inputs = graph_tensor,outputs =logits)




################################
simple_ts_model =  complex_mpnn_model(hidden_size_emb=126,hidden_gcn1=42,hidden_gcn2=56,hidden_gcn3=36,hidden_gcn4=14,graph_tensor_spec=gtspec)
simple_ts_model.summary()

#########
def pre_loss(y_actual,y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    loss1 = mse(tf.cos(y_actual),tf.cos(y_pred))
    loss2 = mse(y_actual,y_pred)
    loss = loss2+loss1
    return loss

def custom_loss(y_actual,y_pred):
   
    mae = tf.keras.losses.MeanAbsoluteError()

    loss1 = mae(tf.cos(y_actual),tf.cos(y_pred))
    loss2 = mae(tf.sin(y_actual),tf.sin(y_pred))

    
    loss = loss1*100+loss2*100
    return loss

def mae_loss(y_actual,y_pred):
    mae = tf.keras.losses.MeanAbsoluteError()
    loss1 = mae(tf.cos(y_actual),tf.cos(y_pred))
    loss2 = mae(y_actual,y_pred)
    loss = loss1*100 
    return loss

########
datee = 'knn1012'
def show_layer(layer,name='None'):
    plt.figure(dpi=300)
    hi1 = np.array(layer.get_weights()[0]).flatten()
    plt.hist(hi1,label='weights')

    hi2 = np.array(layer.get_weights()[1]).flatten()
    plt.hist(hi2,label='bias')

    plt.legend()
    plt.title(name)

    plt.savefig(f'./out_pic/{datee}/{name}_{datee}.jpg')

def visualize_model(simple_ts_model2,simplehistory2,model_name,layer_list):
    

    for k, hist in simplehistory2.history.items():
        plt.figure(dpi=300)
        plt.plot(hist)
        plt.title(k)
        plt.show()
        plt.savefig(f'./out_pic/{datee}/{model_name}_{k}.jpg')


    zenithtrue=[]
    zenithreco=[]
    iterator = iter(train_dataset)
    for id in range(train_Number):
        graphy = next(iterator)
        zenithtrue.append(graphy[1])
        zenithreco.append(simple_ts_model2(graphy[0]))   
    plt.figure(dpi=300)
    plt.scatter(zenithtrue,zenithreco,marker='.')
    plt.xlabel('true')
    plt.ylabel('reco')
    a=np.arange(1.4,3.2,0.1)
    plt.plot(a,a,'r--')
    plt.savefig(f'./out_pic/{datee}/{model_name}_train_regression.jpg')

    vzenithtrue=[]
    vzenithreco=[]
    iterator = iter(valid_dataset)
    for id in range(valid_Number):
        graphy = next(iterator)
        vzenithtrue.append(graphy[1])
        vzenithreco.append(simple_ts_model2(graphy[0]))   
    plt.figure(dpi=300)
    plt.scatter(vzenithtrue,vzenithreco,marker='.')
    plt.xlabel('true')
    plt.ylabel('reco')
    a=np.arange(1.4,3.2,0.1)
    plt.plot(a,a,'r--')
    plt.savefig(f'./out_pic/{datee}/{model_name}_valid_regression.jpg')


    for i in layer_list:
        show_layer(simple_ts_model2.layers[i],f'{modle_name}_layer_{i}')

    normalized_graph = simple_ts_model2.layers[1](ps)
    embedded_graph = simple_ts_model2.layers[2](normalized_graph)
    passed_graph_1 = simple_ts_model2.layers[3](embedded_graph)
    gcned_graph_1 = simple_ts_model2.layers[4](passed_graph_1)
    passed_graph_2 = simple_ts_model2.layers[5](gcned_graph_1)
    gcned_graph_2 = simple_ts_model2.layers[6](passed_graph_2)
    pooled_graph = simple_ts_model2.layers[7](gcned_graph_2)
    densed_graph = simple_ts_model2.layers[8](pooled_graph)
    logit = simple_ts_model2.layers[9](densed_graph)




    print(ps.node_sets['hit']['4Dvec'])
    print('----------------')
    print(normalized_graph.node_sets['hit']['hidden_state'])
    print('----------------')
    print(embedded_graph.node_sets['hit']['hidden_state'])
    print('----------------')
    print(passed_graph_1.node_sets['hit']['hidden_state'])
    print('----------------')
    print(gcned_graph_1.node_sets['hit']['hidden_state'])
    print('----------------')
    print(passed_graph_2.node_sets['hit']['hidden_state'])
    print('----------------')
    print(gcned_graph_2.node_sets['hit']['hidden_state'])
    print('----------------')
    print(pooled_graph)
    print('----------------')
    print(densed_graph)
    print('----------------')
    print(logit)
 
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
        

###################
md_path = './md_weights/3nn_mpnnpl_zenith/'
md_path_pre = md_path+'pre'
new_model =complex_mpnn_model(hidden_size_emb=126,hidden_gcn1=42,hidden_gcn2=56,hidden_gcn3=36,hidden_gcn4=14,graph_tensor_spec=gtspec)


simple_ts_model.compile(tf.keras.optimizers.Adam(),loss=pre_loss,metrics=task.metrics())
simplehistory_test = simple_ts_model.fit(test_trds_scalar,epochs = 50,verbose=2,validation_data=valid_dataset)

save_model_weights(model=simple_ts_model,list_layer=list_layer,path=md_path_pre)


simple_ts_model2 = new_model
load_model_weights(model=simple_ts_model2,list_layer=list_layer,path=md_path_pre)
simple_ts_model2.compile(tf.keras.optimizers.Adam(),loss=custom_loss,metrics=task.metrics())
simplehistory2 = simple_ts_model2.fit(test_trds_scalar,epochs = 50,verbose=2,validation_data=valid_dataset)
md_path_result1=md_path+'_50ep'
save_model_weights(model=simple_ts_model2,list_layer=list_layer,path=md_path_result1)
#visualize_model(simple_ts_model2=simple_ts_model2,simplehistory2=simplehistory2,model_name = 'model1012_200',layer_list = list_layer)


simple_ts_model3 = simple_ts_model2
simple_ts_model3.compile(tf.keras.optimizers.Adam(),loss=custom_loss,metrics=task.metrics())
simplehistory3 = simple_ts_model3.fit(test_trds_scalar,epochs = 50,verbose=2,validation_data=valid_dataset)

md_path_result2=md_path+'_100ep'
save_model_weights(model=simple_ts_model3,list_layer=list_layer,path=md_path_result2)
#visualize_model(simple_ts_model2=simple_ts_model3,simplehistory2=simplehistory3,model_name = 'model1012_300ep',layer_list=list_layer)

simple_ts_model4 = simple_ts_model3
simple_ts_model4.compile(tf.keras.optimizers.Adam(),loss=custom_loss,metrics=task.metrics())
simplehistory4 = simple_ts_model4.fit(test_trds_scalar,epochs = 100,verbose=2,validation_data=valid_dataset)

md_path_result3=md_path+'_200ep'
save_model_weights(model=simple_ts_model4,list_layer=list_layer,path=md_path_result3)
#visualize_model(simple_ts_model2=simple_ts_model4,simplehistory2=simplehistory4,model_name = 'model1012_400ep',layer_list=list_layer)


simple_ts_model5 = simple_ts_model4
simple_ts_model5.compile(tf.keras.optimizers.Adam(),loss=custom_loss,metrics=task.metrics())
simplehistory5 = simple_ts_model5.fit(test_trds_scalar,epochs = 200,verbose=2,validation_data=valid_dataset)

md_path_result4=md_path+'_400ep'
save_model_weights(model=simple_ts_model5,list_layer=list_layer,path=md_path_result4)


