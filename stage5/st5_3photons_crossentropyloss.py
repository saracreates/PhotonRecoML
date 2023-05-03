# own library
import helperfile5 as hf
# generalls libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import uproot
from tensorflow.keras.optimizers import Adam
import datetime
import time
import keras.backend as k
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
print('Imported Libraries')

title = "3photons_crossentropyloss"

# load data

num_events = 200000

rootfile1 = uproot.open('./stage5_clusters_1gamma.root')
ipd1 = hf.InputData(rootfile1, 1, numevents=num_events)
ipd1.train_test_split()

rootfile2 = uproot.open('./stage5_clusters_2gamma.root')
ipd2 = hf.InputData(rootfile2, 2, numevents=num_events)
ipd2.train_test_split()

rootfile3 = uproot.open('./stage5_clusters_3gamma.root')
ipd3 = hf.InputData(rootfile3, 3, numevents=num_events)
ipd3.train_test_split()

# prep data
trainings_data = np.concatenate((ipd1.num_photons_t-1, ipd2.num_photons_t-1,  ipd3.num_photons_t-1))
training = hf.one_hot(trainings_data)
data_veri = np.concatenate((ipd1.num_photons_v-1, ipd2.num_photons_v-1, ipd3.num_photons_v-1))

clus = np.concatenate((ipd1.shash_t.reshape(len(ipd1.E_truth_train),25, 49, 1), ipd2.shash_t.reshape(len(ipd2.E_truth_train),25, 49, 1), ipd3.shash_t.reshape(len(ipd3.E_truth_train),25, 49, 1)))
clus_v = np.concatenate((ipd1.shash_v.reshape(len(ipd1.E_truth_veri),25, 49, 1), ipd2.shash_v.reshape(len(ipd2.E_truth_veri),25, 49, 1), ipd3.shash_v.reshape(len(ipd3.E_truth_veri),25, 49, 1)))

mask = np.random.permutation(len(training))

# the network
model = tf.keras.models.Sequential() # parameter
layers = tf.keras.layers # alias to make it shorter to access the layers
model.add(layers.Input(shape=(25,49,1)))
model.add(layers.Normalization(mean=0.107, variance=0.429))
model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu', input_shape=(25, 49, 1), padding="same"))
model.add(layers.AveragePooling2D(pool_size=(2,2), strides=2))
model.add(layers.Conv2D(filters=32, kernel_size=(4, 4), activation='relu', input_shape=(12, 24, 6), padding="same"))
model.add(layers.AveragePooling2D(pool_size=(2,2), strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(units=256, activation = 'relu'))
model.add(layers.Dense(units=128, activation = 'relu'))
model.add(layers.Dense(units=64, activation = 'relu'))
model.add(layers.Dense(units=3, activation = 'softmax'))

model.summary()
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0001))

print("Start training the network with ", len(ipd1.shash_t)+len(ipd2.shash_t)+ +len(ipd3.shash_t), " clusters...")
a = time.time()
fit_hist = model.fit(clus[mask, :], training[mask, :], batch_size=64, epochs=50, validation_split=0.1) # sonst overfitting!!
b = time.time()
print('This took ', (b-a)/60, ' min')

model.save('./models/model_'+title+'_'+str(datetime.date.today()))

# evaluate 

output = model.predict(clus_v)
ev = hf.Evaluation(output) # create object to help with evaluation

ev.training_vs_validation_loss(fit_hist, save=True, title=title) # save loss functions
ev.show_confusion_matrix(data_veri, 3, save=True)

print("Succesfully evaluated ", title)