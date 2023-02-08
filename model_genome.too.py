################### model explain
# argument 1 = input location
# argument 2 = max trials
# argument 3 = GPU number

################### import library ###################

import os
import sys
import random
import numpy as np
import pandas as pd
import random
import pickle
import math
import joblib
import re
import datetime

import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, Activation, Dense, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D, Input, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.models import load_model
from tensorflow import nn
from tensorflow.keras.backend import shape


from sklearn.metrics import precision_score, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.utils import class_weight
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

start_time = datetime.datetime.now()

################### set gpu usage ###################
os.environ["CUDA_VISIBLE_DEVICES"]="%s"%sys.argv[3]


################### define function ###################

def mish(x):
    return x*K.tanh(K.softplus(x))


def create_model(num_dense_layers, num_dense_nodes, learning_rate, activation, drop_rate, weight_decay):
    input1 = keras.Input(shape=(genome_train_data.shape[1],))
    x1 = Dense(num_dense_nodes, name="input_layer")(input1)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activation)(x1)
    x1 = Dropout(drop_rate)(x1)
    for i in range(num_dense_layers):
        name = "layer_dense_{0}".format(i+1)
        x1 = Dense(num_dense_nodes, name=name, kernel_initializer='glorot_uniform')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation(activation)(x1)
        x1 = Dropout(drop_rate)(x1)
    output = Dense(num_classes, activation="softmax", bias_initializer=output_bias, kernel_initializer='glorot_uniform')(x1)
    model = keras.Model(input1, output)
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['CategoricalAccuracy'])
    return model


def get_result(model, x, y):
    pred_prob = model.predict(x)
    pred_value = np.argmax(pred_prob, axis=1) 
    true = y
    accuracy = accuracy_score(true, pred_value)
    weighted_accuracy = balanced_accuracy_score(true, pred_value)
    f1 = f1_score(true, pred_value,average='macro')
    return accuracy, weighted_accuracy, f1



#### hyperparameter setting
dim_num_dense_layers = Integer(low=1, high=10, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')

dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
dim_activation = Categorical(categories=['relu', mish], name='activation')
dim_drop_rate = Real(low=0.3, high=0.7, prior='uniform', name='drop_rate')
dim_weight_decay = Real(low=1e-6, high=1e-2, prior='log-uniform', name='weight_decay')


dimensions = [dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_learning_rate,
              dim_activation,
              dim_drop_rate,
              dim_weight_decay
              ]

default_parameters = [1, 5, 1e-5, 'relu', 0.5, 1e-3]

best_loss = 1.0
num_model = 0

num_model = 0
train_scores = []
vali_scores = []

@use_named_args(dimensions=dimensions)
def fitness(num_dense_layers, num_dense_nodes, learning_rate, activation, drop_rate, weight_decay):
    global num_model
    num_model += 1
    print('====Model_{0}===='.format(num_model))
    print('num_dense_layers: ', num_dense_layers)
    print('num_dense_nodes: ', num_dense_nodes)
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('activation: ', activation)
    print('drop rate: {0:.1e}'.format(drop_rate))
    print('weight decay: {0:.1e}'.format(weight_decay))
    print('\n')
    model = create_model(num_dense_layers = num_dense_layers,
                         num_dense_nodes = num_dense_nodes,
                         learning_rate=learning_rate,
                         activation=activation,
                         drop_rate=drop_rate,
                         weight_decay=weight_decay)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint(path_tmp_best_model, monitor='val_loss', mode='min', save_best_only=True)
    history = model.fit(x=genome_train_data,
                        y=train_labels,
                        epochs=200,
                        batch_size=32,
                        validation_data=(genome_vali_data, vali_labels),
                        class_weight=class_weight_dict,
                        callbacks=[es, mc])
    pred_val = np.round(model.predict(x=genome_vali_data).ravel())
    print(pred_val)
    train_loss = min(history.history['loss'])
    vali_loss = min(history.history['val_loss'])
    print('\n')
    print("loss: {0}".format(vali_loss))
    print("\n")
    global train_scores
    global vali_scores
    global best_loss
    train_scores.append(train_loss)
    vali_scores.append(vali_loss)
    if vali_loss < best_loss:
        model.save(path_best_model)
        best_loss = vali_loss
    del model
    K.clear_session()
    return vali_loss



################### hyperparameter tuning ###################
#### load data
random.seed(92)
np.random.seed(92)
tf.random.set_seed(92)

genome_train_data_file = sys.argv[1]
ncall = int(sys.argv[2])

path_bo_plot= re.sub('.npz','.dnn_bo_loss_plot.png',genome_train_data_file.split('/')[-1])
path_real_bo_plot = re.sub('.npz','.dnn_bo_real_loss_plot.png',genome_train_data_file.split('/')[-1])
hyperparameter_file =  re.sub('.npz','.dnn.loss.best_model.search_result.summary.pkl', genome_train_data_file.split('/')[-1])
path_tmp_best_model = re.sub('.npz','.dnn.loss.tmp_best_model.h5',genome_train_data_file.split('/')[-1])
path_best_model = re.sub('.npz','.dnn.loss.best_model.h5',genome_train_data_file.split('/')[-1])
final_best_model_idx =  re.sub('.npz','.dnn.loss.best_model.trained',genome_train_data_file.split('/')[-1])
#output_file = re.sub('.npz','.performance.txt',genome_train_data_file.split('/')[-1])
output_file = "./performance_genome.too.txt"
#dir_name = re.sub('.npz','',genome_train_data_file.split('/')[-1])


genome_data = np.load(genome_train_data_file, allow_pickle=True)

genome_train_data = genome_data['train_x']
genome_vali_data = genome_data['valid_x']
genome_test_data = genome_data['test_x']
genome_train_labels = genome_data['train_y']
genome_vali_labels = genome_data['valid_y']
genome_test_labels = genome_data['test_y']

train_labels = np.array(genome_train_labels).astype(int)
vali_labels = np.array(genome_vali_labels).astype(int)
test_labels = np.array(genome_test_labels).astype(int)

freq = [(train_labels.tolist().count(x))/len(train_labels) for x in np.unique(train_labels)]

num_classes = len(set(train_labels))
train_labels = tf.one_hot(train_labels, depth=num_classes)
vali_labels = tf.one_hot(vali_labels, depth=num_classes)
test_labels = tf.one_hot(test_labels, depth=num_classes)



## calculate bias and class weight
train_integers = np.argmax(train_labels, axis=1)
class_weight1 = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_integers), y=train_integers)
class_weight_dict = dict(enumerate(class_weight1))

# initial bias
initial_bias = np.log(freq)
output_bias = tf.keras.initializers.Constant(initial_bias)


## hyperparameter tuning
search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI', n_calls=ncall, x0=default_parameters)

## hyperparameter save
with open(hyperparameter_file,'wb') as f:
    pickle.dump(search_result,f)

## Save convergence plot
#fig = plt.figure()
#ax = plot_convergence(search_result)
#ax = fig.add_subplot(ax)
#fig.savefig(path_bo_plot)

## Save each loss
#fig = plt.figure(figsize=(10,8))
#plt.plot(range(1,len(train_scores)+1),train_scores, label='Train loss')
#plt.plot(range(1,len(vali_scores)+1),vali_scores, label='Validation loss')
#plt.xlabel('epochs')
#plt.ylabel('loss')
#plt.xlim(0, len(train_scores)+1) # 일정한 scale
#plt.legend()
#plt.tight_layout()
#plt.show()
#fig.savefig(path_real_bo_plot, bbox_inches = 'tight')


## Report Best Hyper-parameter and accuracy
dimension_name = []
for i in range(len(dimensions)):
    dimension_name.append(dimensions[i].name)
print("Best Hyper-parameter: " ,zip(dimension_name, search_result.x))
print("Best loss: ",-search_result.fun)



################### Model prediction ###################

num_dense_layers = search_result.x[0]
num_dense_nodes = search_result.x[1]
learning_rate = search_result.x[2]
activation = search_result.x[3]
drop_rate = search_result.x[4]
weight_decay = search_result.x[5]

os.system('mkdir model_genome.too')
out = open(output_file, 'w')
for i in range(90,120):
    random.seed(i)
    np.random.seed(i)
    tf.random.set_seed(i)
    final_best_model_path = './model_genome.too/%s.seed_%s.h5'%(final_best_model_idx,i)
    model = create_model(num_dense_layers = num_dense_layers,
                         num_dense_nodes = num_dense_nodes,
                         learning_rate=learning_rate,
                         activation=activation,
                         drop_rate=drop_rate,
                         weight_decay=weight_decay)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    mc = ModelCheckpoint(final_best_model_path, monitor='val_loss', mode='min', save_best_only=True)
    history = model.fit(x=genome_train_data,
                        y=train_labels,
                        epochs=200,
                        batch_size=32,
                        validation_data=(genome_vali_data, vali_labels),
                        class_weight=class_weight_dict,
                        callbacks=[es, mc])
    train_loss = min(history.history['loss'])
    vali_loss = min(history.history['val_loss'])
    train_acc, train_weighted_acc, train_f1 = get_result(model, genome_train_data, genome_train_labels)
    valid_acc, valid_weighted_acc, valid_f1 = get_result(model, genome_vali_data, genome_vali_labels)
    test_acc, test_weighted_acc, test_f1 = get_result(model, genome_test_data, genome_test_labels)
    out.write("%s"%i + "\t" + str(train_loss) + "\t" + str(vali_loss) + "\t" + str(train_acc) + "\t" + str(valid_acc) + "\t" + str(test_acc) + "\t" + str(train_weighted_acc)  + "\t" +  str(valid_weighted_acc)  + "\t" + str(test_weighted_acc)  + "\t" + str(train_f1) + '\t' + str(valid_f1) + '\t'+ str(test_f1) +  "\n")

out.close()

end_time = datetime.datetime.now()
print(end_time - start_time)


df = pd.read_csv("performance_genome.too.txt", sep="\t", header=None, index_col=0)
idx = df.sort_values(by=2).head(1).index.values[0]
os.system("cp ./model_genome.too/input_genome.too.dnn.loss.best_model.trained.seed_100.h5 model_genome.too.best.h5"%(idx))

os.system("rm input_genome.too.dnn.loss.best_model.h5")
os.system("rm input_genome.too.dnn.loss.best_model.search_result.summary.pkl")
os.system("rm input_genome.too.dnn.loss.tmp_best_model.h5")
os.system("rm -r model_genome.too")


