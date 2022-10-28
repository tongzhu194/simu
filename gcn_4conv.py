import tensorflow as tf
import tensorflow_gnn as tfgnn

from tensorflow_gnn import runner
from tensorflow_gnn.models import gcn

import numpy as np
import matplotlib.pyplot as plt
graph_schema = tfgnn.read_schema("/n/home05/tzhu/work/icgen2/algorithm/ic2_gnn_v1.txt")
gtspec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)



gpus = tf.config.list_physical_devices(device_type = 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
###############################
def extract_labels(graph_tensor):
    index = graph_tensor.context['injection_zenith']
    return graph_tensor,index


##############################
train_ds_provider = runner.TFRecordDatasetProvider(file_pattern="/n/home05/tzhu/work/icgen2/data_graph/SAM_knn_v4_1_70")
valid_ds_provider = runner.TFRecordDatasetProvider(file_pattern="/n/home05/tzhu/work/icgen2/data_graph/SAM_knn_v4_100_105")

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
def get_norm_map():

    def node_sets_fn(node_set, node_set_name):
        if node_set_name == 'hit':
            return (node_set['4Dvec']/720.)
    
    return tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn,name='normalize')


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
#pseudo
hit_sources = [0,0,0,1]
hit_targets = [1,2,3,2]
hits= [[1.,1.,1.,1.],[2.,1.,1.,1.],[3.,1.,1.,1.],[4.,1.,1.,1.]]
fzenith = 3.14
energy=1000.
fenergy = tf.cast(energy,dtype=tf.float32)

hit_adjacency = tfgnn.Adjacency.from_indices(source=("hit",tf.cast(hit_sources,dtype = tf.int32)),target=("hit",tf.cast(hit_targets,tf.int32)))

hit = tfgnn.NodeSet.from_fields(
    sizes = [len(hits)],
    features={ 
        "4Dvec":tf.cast(hits,dtype=tf.float32),
    })
coincidence = tfgnn.EdgeSet.from_fields(
        sizes = tf.shape(hit_sources),
        adjacency=hit_adjacency)

context = tfgnn.Context.from_fields(
        features={"injection_zenith":[fzenith],"event_energy":[fenergy]})

ps = tfgnn.GraphTensor.from_pieces(node_sets={"hit":hit},edge_sets={"coincidence":coincidence},context=context)
################
class custom_d_GCN(tf.keras.layers.Layer):
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
        edge_ones = tf.ones([edge_set.total_size, 1])

        D1= tf.squeeze(tfgnn.pool_edges_to_node(graph,edge_set_name,tfgnn.TARGET,'sum',feature_value=edge_ones), -1)
        #D2 = tf.squeeze(tfgnn.pool_edges_to_node(graph,edge_set_name,tfgnn.SOURCE,'sum',feature_value=edge_ones), -1)
        in_degree = D1
        in_degree += 1
        invsqrt_deg = tf.math.rsqrt(in_degree)
  

        # Calculate \hat{D^{-1/2}}X first
        normalized_values = (invsqrt_deg[:, tf.newaxis] * graph.node_sets['hit']['hidden_state'])

        # Calculate A\hat{D^{-1/2}}X by broadcasting then pooling
        
        source_bcast2 = tfgnn.broadcast_node_to_edges(
            graph,
            edge_set_name,
            tfgnn.SOURCE,
            feature_value=normalized_values
        )
        pooled = tfgnn.pool_edges_to_node(graph, edge_set_name, tfgnn.TARGET, 'sum', feature_value=source_bcast2)
        # left-multiply the result by \hat{D^{-1/2}}
        pooled = invsqrt_deg[:, tf.newaxis] * pooled

        pooled += invsqrt_deg[:, tf.newaxis] * normalized_values

        #return tf.keras.layers.Dense(units=self.hidden_size,activation='relu')(pooled)
        return pooled
   

#########################
def message_passing(hidden_size,name):
    return tfgnn.keras.layers.GraphUpdate(
         node_sets={
            "hit": tfgnn.keras.layers.NodeSetUpdate(
                {"coincidence": custom_d_GCN(hidden_size=hidden_size)},
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

def complex_gcn_model(hidden_size_emb,hidden_gcn1,hidden_gcn2,hidden_gcn3,hidden_gcn4,graph_tensor_spec):
    norm_fn = get_norm_map()
    init_states_fn = get_initial_map_features(hidden_size=hidden_size_emb,name='embedding')
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
    embedded_graph = init_states_fn(normalized_graph)

    passed_graph_1 = message_passing_fn_1(embedded_graph)
    gcned_graph_1 = node_dense_fn_1(passed_graph_1)
    graph_pool_1_mean = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'mean',node_set_name='hit',name='pool1_mean')(gcned_graph_1)
    graph_pool_1_max = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'max',node_set_name='hit',name='pool1_max')(gcned_graph_1)
    graph_pool_1_min = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'min',node_set_name='hit',name='pool1_min')(gcned_graph_1)
    graph_pool_1 = tf.concat([graph_pool_1_mean,graph_pool_1_max,graph_pool_1_min],axis = 1)
    graph_pool_1 = tf.keras.layers.Dense(units=12,activation='relu'  )(graph_pool_1)
    graph_pool_1 = tf.keras.layers.Dense(units=6,activation='relu' )(graph_pool_1)
    graph_pool_1 = tf.keras.layers.Dense(units=8,activation='relu',name ='embed_pool_1' )(graph_pool_1)
    

    passed_graph_2 = message_passing_fn_2(gcned_graph_1)
    gcned_graph_2 = node_dense_fn_2(passed_graph_2)
    graph_pool_2_mean = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'mean',node_set_name='hit',name='pool2_mean')(gcned_graph_2)
    graph_pool_2_max = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'max',node_set_name='hit',name='pool2_max')(gcned_graph_2)
    graph_pool_2_min = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'min',node_set_name='hit',name='pool2_min')(gcned_graph_2)
    graph_pool_2 = tf.concat([graph_pool_2_mean,graph_pool_2_max,graph_pool_2_min],axis = 1)
    graph_pool_2 = tf.keras.layers.Dense(units=12,activation='relu' )(graph_pool_2)
    graph_pool_2 = tf.keras.layers.Dense(units=6,activation='relu' )(graph_pool_2)
    graph_pool_2 = tf.keras.layers.Dense(units=8,activation='relu' ,name ='embed_pool_2')(graph_pool_2)
    
    


    passed_graph_3 = message_passing_fn_3(gcned_graph_2)
    gcned_graph_3 = node_dense_fn_3(passed_graph_3)
    graph_pool_3_mean = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'mean',node_set_name='hit',name='pool3_mean')(gcned_graph_3)
    graph_pool_3_max = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'max',node_set_name='hit',name='pool3_max')(gcned_graph_3)
    graph_pool_3_min = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'min',node_set_name='hit',name='pool3_min')(gcned_graph_3)
    graph_pool_3 = tf.concat([graph_pool_3_mean,graph_pool_3_max,graph_pool_3_min],axis = 1)
    graph_pool_3 = tf.keras.layers.Dense(units=12,activation='relu')(graph_pool_3)
    graph_pool_3 = tf.keras.layers.Dense(units=6,activation='relu' )(graph_pool_3)
    graph_pool_3 = tf.keras.layers.Dense(units=8,activation='relu',name ='embed_pool_3' )(graph_pool_3)

    passed_graph_4 = message_passing_fn_4(gcned_graph_3)
    gcned_graph_4 = node_dense_fn_4(passed_graph_4)
    graph_pool_4_mean = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'mean',node_set_name='hit',name='pool4_mean')(gcned_graph_4)
    graph_pool_4_max = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'max',node_set_name='hit',name='pool4_max')(gcned_graph_4)
    graph_pool_4_min = tfgnn.keras.layers.Pool(tfgnn.CONTEXT,'min',node_set_name='hit',name='pool4_min')(gcned_graph_4)
    graph_pool_4 = tf.concat([graph_pool_4_mean,graph_pool_4_max,graph_pool_4_min],axis = 1)
    graph_pool_4 = tf.keras.layers.Dense(units=12,activation='relu' )(graph_pool_4)
    graph_pool_4 = tf.keras.layers.Dense(units=6,activation='relu' )(graph_pool_4)
    graph_pool_4 = tf.keras.layers.Dense(units=8,activation='relu',name ='embed_pool_4' )(graph_pool_4)
    
    graph_pool = tf.concat([graph_pool_1,graph_pool_2,graph_pool_3,graph_pool_4],axis = 1)
    graph_pool = tf.keras.layers.Dense(units=8,activation='relu',name='dense_after_pool' )(graph_pool)
    logits = tf.keras.layers.Dense(units=1,name='logits')(graph_pool)
    #gcned_graph = directedgcn_fn(embedded_graph)

    return tf.keras.Model(inputs = graph_tensor,outputs =logits)


################################
simple_ts_model =  complex_gcn_model(hidden_size_emb=54,hidden_gcn1=100,hidden_gcn2=60,hidden_gcn3=12,hidden_gcn4=16,graph_tensor_spec=gtspec)
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
 

################
###SAVE AND LOAD
def save_model_weights(model,list_layer,path):
    for i in list_layer:
        w = model.layers[i].get_weights()
        tmp0 = path+f'_layer_{i}_w.npy'
        tmp1 = path+f'_layer_{i}_b.npy'
        np.save(tmp0,w[0])
        np.save(tmp1,w[1])
    print(f"weights are saved to {path}")


def load_model_weights(model,list_layer,path):
    for i in list_layer:
        tmp0 = path+f'_layer_{i}_w.npy'
        tmp1 = path+f'_layer_{i}_b.npy'
        b0 = np.load(tmp0)
        b1 = np.load(tmp1)
        b = [b0,b1]
        model.layers[i].set_weights(b)
    print("weights are loaded sucessfully")
###################
list_layer=[]
for i in range(len(simple_ts_model.layers)):
    if (len(simple_ts_model.layers[i].get_weights())!=0):
        list_layer.append(i)

md_path = './md_weights/gcncomplex_v2/'
md_path_pre = md_path+'pre'
new_model =complex_gcn_model(hidden_size_emb=54,hidden_gcn1=100,hidden_gcn2=60,hidden_gcn3=12,hidden_gcn4=16,graph_tensor_spec=gtspec)

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

