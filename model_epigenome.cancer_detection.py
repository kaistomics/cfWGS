# argument 1 = input location
# argument 2 = max trials
# argument 3 = GPU number

import os
import sys
import datetime
import numpy as np
import pandas as pd
import random
import tensorflow as tf
#import tensorflow_addons as tfa
#from adabelief_tf import AdaBeliefOptimizer
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dense, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import joblib
import keras_tuner as kt
import IPython
from sklearn.metrics import confusion_matrix
from sklearn import metrics
#from kerastuner import Objective

start_time = datetime.datetime.now()

random.seed(2002)
np.random.seed(2002)
tf.random.set_seed(2002)


os.environ["CUDA_VISIBLE_DEVICES"]="%s"%sys.argv[3]

input_file = sys.argv[1]
max_trials = int(sys.argv[2])
epochs = 200
frag_dim = 100
pos_dim = 250



def model_builder(hp):
    hp_unit1 = hp.Int('units1', min_value=10, max_value=100, step=5)
    hp_unit2 = hp.Int('units2', min_value=10, max_value=100, step=5)
    hp_unit3 = hp.Int('units3', min_value=10, max_value=100, step=5)
    hp_kernel_size1 = hp.Int('kernel_size', min_value=1, max_value=50, step=1)
    hp_kernel_size2 = hp.Int('kernel_size2', min_value=1, max_value=50, step=1)
    hp_dropout_rate1 = hp.Float('dropout_rate1', min_value=0, max_value=0.5,  default=0.2, step=0.05)
    hp_dropout_rate2 = hp.Float('dropout_rate2', min_value=0, max_value=0.5,  default=0.2, step=0.05)
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling="LOG", default=1e-2)
    input = keras.Input(shape=(frag_dim,pos_dim,25))
    x = Conv2D(hp_unit1, kernel_size=(hp_kernel_size1, pos_dim))(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(hp_dropout_rate1)(x)
    x = Conv2D(hp_unit2, kernel_size=(hp_kernel_size2, 1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(hp_dropout_rate2)(x)
    x = Flatten()(x)
    x = Dense(hp_unit3, activation="relu")(x)
    output = Dense(1, activation="sigmoid", bias_initializer=output_bias)(x)
    model = keras.Model(input, output)
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate = hp_learning_rate), metrics=[tf.keras.metrics.AUC()])
    return model


def get_result(model, x, y):
    pred = model.predict(x)
    true = y
    pred2 = []
    for i in range(pred.shape[0]):
        if pred[i][0] > 0.5:
            pred2.append(1)
        else:
            pred2.append(0)
    tn,fp,fn,tp = confusion_matrix(true, pred2).ravel()
    conf=[]; conf.append(str(tn)); conf.append(str(fp)); conf.append(str(fn)); conf.append(str(tp))
    confmat=";".join(conf)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc, confmat


def run(num1):
    os.system("mkdir ./model_epigenome.cancer_detection")
    data = np.load("%s"%(num1), allow_pickle=True)
    x_train = data['train_x']
    x_vali = data['vali_x']
    x_test = data['test_x']
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3]))
    x_vali = np.reshape(x_vali, (x_vali.shape[0], x_vali.shape[1]*x_vali.shape[2]*x_vali.shape[3]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3]))
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    joblib.dump(scaler, "minmax_epigenome.cancer_detection.joblib")
    x_train_norm = scaler.transform(x_train)
    x_train = np.reshape(x_train_norm, (x_train.shape[0], 100, 250, 25))
    x_vali_norm = scaler.transform(x_vali)
    x_vali = np.reshape(x_vali_norm, (x_vali.shape[0], 100, 250, 25))
    x_test_norm = scaler.transform(x_test)
    x_test = np.reshape(x_test_norm, (x_test.shape[0], 100, 250, 25))
    y_train = data['train_y']
    y_vali = data['vali_y']
    y_test = data['test_y']
    global output_bias
    global class_weight
    num_classes = 2
    pos = sum(y_train==1)
    neg = sum(y_train==0)
    output_bias = tf.keras.initializers.Constant(np.log(pos/neg))
    class_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y = y_train)
    class_weight = dict(enumerate(class_weight))
    print(class_weight)
    tuner = kt.BayesianOptimization(model_builder, max_trials=max_trials, objective = 'val_loss', directory = "epigenome_hyperparameter", project_name = "cancer_detection")
    es = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
    class ClearTrainingOutput(tf.keras.callbacks.Callback):
        def on_train_end(*args, **kwargs):
            IPython.display.clear_output(wait = True)
    tuner.search(x_train, y_train, epochs = epochs, validation_data = (x_vali, y_vali), callbacks = [ClearTrainingOutput(), es, keras.callbacks.TensorBoard("./epigenome_log/logcancer_detection")], class_weight=class_weight)
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    out = open("./performance_epigenome.cancer_detection.txt", 'w')
    for number in range(30):
        model = tuner.hypermodel.build(best_hps)
        filename = "./model_epigenome.cancer_detection/model_epigenome_%s.h5"%(number)
        checkpoint = ModelCheckpoint(filename, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
        model.fit(x_train, y_train, validation_data=(x_vali, y_vali), epochs=epochs, callbacks=[checkpoint], class_weight=class_weight)
        model = tf.keras.models.load_model("./model_epigenome.cancer_detection/model_epigenome_%s.h5"%(number))
        train_loss, train_auc = model.evaluate(x_train, y_train)
        vali_loss, vali_auc = model.evaluate(x_vali, y_vali)
        test_loss, test_auc = model.evaluate(x_test, y_test)
        train_auc, train_confmat = get_result(model, x_train, y_train)
        vali_auc, vali_confmat = get_result(model, x_vali, y_vali)
        test_auc, test_confmat = get_result(model, x_test, y_test)
        out.write(str(number) + "\t" + str(train_loss) + "\t" + str(vali_loss) + "\t" + str(test_loss) + "\t" + str(train_auc) + "\t" + str(vali_auc) + "\t" + str(test_auc) + "\t" + train_confmat + "\t" + vali_confmat + "\t" + test_confmat + "\n")
    out.close()

run(input_file)


end_time = datetime.datetime.now()

print(end_time - start_time)

df = pd.read_csv("performance_epigenome.cancer_detection.txt", sep="\t", header=None, index_col=0)
idx = df.sort_values(by=2).head(1).index.values[0]
os.system("cp ./model_epigenome.cancer_detection/model_epigenome_%s.h5 model_epigenome.cancer_detection.best.h5"%(idx))

os.system("rm -r model_epigenome.cancer_detection")
os.system("rm performance_epigenome.cancer_detection.txt")
os.system("rm minmax_epigenome.cancer_detection.joblib")
os.system("rm -r epigenome_hyperparameter")
os.system("rm -r epigenome_log")


