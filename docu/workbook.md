# ECAL with neural networks



## Description

Photons create clusters over multiple cells in the ECAL. To find the $x/y$-position and the energy of the inertial photon entering a fiting algorithm in coral is used. This fit doesn't work smoothly with two photons whose clusters overlap. This problem will be approched with a neural network.

## Overview

## Stage 1 - Monoenergetic photons E=140GeV

First everyting is kept simple. With a phast user event all cells with an energy deposit are saved and the MC truth and coral fit are also stored. The clusters are restored through python code and fit into 5x5 histograms where zero columns/rows are added (not completely) random. Clusters that are larger than 5x5 are cut out. By defining a new coordinate system in the middle of the lowest left cell that is involved the NN can learn relative coordinates to that corner as total position doesn't matter at the moment as the photons enter with no angle. 
Now use 80% of the data for training the network and 20% for validation. The input of the network are the 25 energy values of the 5x5 cluster which is read like a page row by row starting in the upper left corner. The output of the NN are the $x/y$-position and energy of the entering photon.
The first aim here is to get significantly better than the coral fit before moving on to the next stage.

**features investigated in stage 1:**
- [x] Stadardizing the input (standard score) 
- [x] learning rate ($\alpha = 0.00001$)
- [x] rouch epoch size (200)
- [x] dropout layers (not useful)

**What to do better the next time:**
- cutting non-usable clusters already in phast user event (e.g. clusters that are bigger than 5x5)
- adding the zero columns/rows completely random on each side of the clusters

### Summary stage 1

I finished stage 1 successfully. A basic network was set up and some hyperparameter tuning was done.  The network has the following architecture:

```
model1 = keras.Sequential([layers.Input(shape=(25)), 
                         layers.LayerNormalization(axis=1), 
                         layers.Dense(64, activation="relu"), 
                         layers.Dense(128, activation="relu"), 
                         layers.Dense(256, activation="relu"), 
                         layers.Dense(128, activation="relu"), 
                         layers.Dense(64, activation="relu"), 
                         layers.Dense(32, activation="relu"), 
                         layers.Dense(3, activation=None)]) 
```

The best result is 

| (x/y) | $\mu$ | $\sigma$ |
| --- | --- | --- |
| simplest model & standardization & $\alpha=0.00001$, 200 epochs| 0.00039 / 0.00893  | 0.045 / 0.046 |

when performing a gaussian fit over the absolute difference of the MC truth and the predicted values of the NN.

## Stage 2 - Photons with energies up to E=<200GeV

Now photons in the energy range of 2-200 GeV are used. Biggest issue solved: remove layers.LayerNormalization as it remove the information about energy. Instead make global standardization. The network is kept the same:

```
model1 = keras.Sequential([layers.Input(shape=(25)), 
                         layers.Dense(64, activation="relu"), 
                         layers.Dense(128, activation="relu"), 
                         layers.Dense(256, activation="relu"), 
                         layers.Dense(128, activation="relu"), 
                         layers.Dense(64, activation="relu"), 
                         layers.Dense(32, activation="relu"), 
                         layers.Dense(3, activation=None)]) 
```

I trained again with 200 epochs (batchsize now 64 instead of 50), validation split of 0.1.


**features investigated in stage 2:**
- [x] standardiation of input: global vs. none (not a big difference)
- [x] investigation of which E values are not trained well (lower worse than higher) 
- [x] change loss function from mean squared error to mean absolute percentage error (failed due to divergence)
- [x] give sum of all cluster cell energies also as an input (doesn't change much)
- [x] placing the clusters completely random in the 5x5 grid (improves precision but not resolution)
- [x] analysis of what exactly the network learned (values that are more than one $\sigma$ away have normally lower energy. There is a bias with x and y depending on how one gives the cluster as an input. With a random order of cells this can be bypassed. There is a weird peak for high energies, meaning that very high energies are not as good learned as photons with energy around 150GeV.)


### Summary stage 2

In stage 2 I showed that the network can not only learn the right $x/y$ position but also the correct energy when dealing with photons in an energy range of 2-200 GeV. The energy resolution is $\pm 2.2$ %. Furthermore some minor changes were investigated (see above). 

## Stage 3  - Photons with angle

Now we are looking at photons in the energy range of 2-200 GeV with angles that are realistic from the target. The frist dataset only illuminates the middle part of the shashlik cells in the ecal (dataset 1) so that the angles towards the $z$-axis are roughly $\in (0, 0.012)$ rad. The second dataset (dataset 2) illuminates the whole shashlik part, which is wider in $x$ than in $y$. I only investigate clusters that trigger shashlik cells exclusivly (all hit cells must be celltype 3!). The angles to the $z$-axis are $\in (0, 0.027)$ rad.

Let's have a look at the standard model I use for this stage. 

```
model = keras.Sequential([layers.Input(shape=(25)),
                         layers.Normalization(mean=3.9, variance=16.2), 
                         layers.Dense(64, activation="relu"),
                         layers.Dense(128, activation="relu"),
                         layers.Dense(256, activation="relu"),
                         layers.Dense(128, activation="relu"),
                         layers.Dense(64, activation="relu"),
                         layers.Dense(32, activation="relu"),
                         layers.Dense(3, activation=None)])
model.summary()
model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.00001))
```

One can see that is the same as in stage 2 with the only difference of global standardization with fixed values, so that this model can be applied to any dataset. I use the same trainingsparameter as in stage 2 (epochs=200, batchsize=64, validation split = 0.1). 
Note: I changed my data selection and preparation code into an object oriented code / I work with classes now :-) 

**features investigated in stage 3:**

The training on dataset 1 went so smoothly that there were no new corrections added. Dataset 2 with larger angles was more interesting:

- [x] Investigation of not well trained $x$-positions in datset 2. I could show that for dataset 2 the angles to the $x$-axis are much bigger than for dataset 1, as the whole shashlik part was illuminated. When looking at the ratio of clusters that lay more than $\pm \sigma$ away when looking at the relative position of $x$ one can see that these are the clusters with large angles. Conclusion: Only looking at the shape of the cluster doesn't give the NN enough information to learn the angle and therefore the right position correcly. 

- [x] Adding the absolute position. The idea behind this study is to give the NN more information for correct learning as the angles of the clusters correlate with the positions at the moment (all photons start from the target). So I changed the input shape from 25 to 27 and added the coordinates from the lower left corner of the 5 x 5 grid. The performance improved "back to normal"/ roughly same performance as without angles. 

### Summary stage 3

The network can learn small angles without any changes compared to stage 2 (relative difference in energy: $\mu$ = 0.004, $\sigma$ = 0.022. Relative difference in $x$- and $y$-position: $\mu$ = 0.002/0.000, $\sigma$ = 0.021/0.020). If one includes larger angles up to 0.027 rad the network performs worse (e.g. relative difference in $x$-position: $\sigma_{dataset1}$ = 0.021 vs. $\sigma_{dataset2}$ = 0.037) as it is not able to learn the correct position only from the cluster shape when dealing with larger angles. If ones includes the absolute position however the network can learn this correlation and performs well again (relative difference in $x$-position: $\sigma_{dataset2, abs. pos.}$ = 0.020). 

## Stage 4 - Overlapping photons

We level things up! Now we have two photons that can be placed in a 9x9 grid. 

**features investigated in stage 4:**
- [x] Labeling: How to label the photons as photon 1 and photon 2 as the solution of the NN to learn is like [x1, y1, E1, x2, y2, E2] now? I've tried to sort them after the magnitude of the energies but if one evaluates what the network learns then one can see that it mixes up with the correct labeling. Sorting after $x$ or $y$ is even worse. Therefore, I've tried an approach from Johannes Riekes [article](https://towardsdatascience.com/object-detection-with-neural-networks-a4e2c46b4491), that flips the dataset after each epoch. [This](https://github.com/jrieke/shape-detection/blob/master/two-rectangles.ipynb) is the git repository. It helped but takes ages to train. Therefore I started a new approach by modifying the loss function so that it returns the smaller MSE from [params1, params2] and [params2, params1]. This was satisfying enough for now. 
- [x] I've implemented a minimum distance of 4 cm between the two photon hits so that they are a bit more distinuighable and realistic to learn (Dominik said the distance is normally at least 5cm due to Physics(?)). 
- [x] The energy got a lot better learned than the position which makes sense as I mix two differnt scales in one loss function. What helped a lot is to device the values by the desired resolution ($pos_{AL}$ = 0.5 cm, $E_{AL}$ = 2.5 GeV).
- [x] I've tried to only train on data that the Lednev Fit is able to fit with 2 photons. But this didn't help the network, so one can conclude that it dosn't matter too much for the network how much the photon cluster overlap. The struggles lay somewhere else. 
- [x] only using monoenergetic photons also doesn't help with the resolution of (x,y).
- [x] I've added the total position which helped which the $x$-positions as we have bigger angles there (this effect was already shown in stage 3 too)
- [x] One convolutional layer helped a bit too although it doesn't help too much as hoped. 
- [x] By analizing with values are more than one $\sigma$ away, one could see that photons that hit close to the edge of the 9x9 gird are learned worse. Where this bias comes from is unclear. I therefore modified the clusters to not be placed randomly in the 9x9 gird but placed in the middle. This helped with the performance! In this study I also found out that the position is worse learned if the clusters hits in the middle of the ECAL cell (that makes sense). This can be seen in sin/cos-like values for the $x/y$-positions that are learned worse than one $\sigma$.

Although some changes helped to improve the network, it is still not as precise as the Lednev fit. The parameters are nicely centered around 0, but the width of the Gauß-Fit is still too big. The network is just not precise enough. 

The best result comes from a network with a weighted loss function that returns the smaller MSE and where the clusters are placed in the middle of the 9x9 grid and a minimum distance of 4cm. 

|    | $\mu$ | $\sigma$ |
| --- | --- | --- |
| $x_1$ | $-0,005\pm0,001$ | $-0,422\pm0,002$ |
| $x_2$ | $-0,003\pm0,002$ | $0,423\pm0,003$ |
| $y_1$ | $-0,056\pm0,002$ | $-0,332\pm0,002$ |
| $y_2$ | $-0,053\pm0,002$ | $-0,331\pm0,003$ |
| $E_1$ | $-0,001\pm0,000$ | $-0,035\pm0,000$ |
| $E_2$ | $-0,001\pm0,000$ | $-0,035\pm0,000$ |

where the ($x$ / $y$) is in cm and $E$ means the realtive energy. I'd like to achieve something like $\sigma_{pos} = 0.25$ cm and $\sigma_{E} = 0.025$.

```
model = keras.Sequential([layers.Input(shape=(81)), \
                         layers.Normalization(mean=2.5, variance=13.4), \
                         layers.Dense(128, activation="relu"), \
                         layers.Dense(256, activation="relu"), \
                         layers.Dense(512, activation="relu"), \
                         layers.Dense(256, activation="relu"), \
                         layers.Dense(128, activation="relu"), \
                         layers.Dense(64, activation="relu"), \
                         layers.Dense(6, activation=None)])
model.summary()
model.compile(loss=loss_flip_weighted, optimizer=Adam(learning_rate=0.00005))
fit_hist = model.fit(ipd.clusters_t, ipd.training, batch_size=64, epochs=150, validation_split=0.1)
```

With 

```
def loss_flip_weighted(y_true, y_pred):
    bs = int(tf.size(y_true)/6) # batchsize
    ort_AL = 0.5 # cm
    E_AL = 2.5 # GeV
    weights = [tf.ones(bs)*ort_AL, tf.ones(bs)*ort_AL, tf.ones(bs)*E_AL, tf.ones(bs)*ort_AL, tf.ones(bs)*ort_AL, tf.ones(bs)*E_AL]
    sq = k.square((y_true - y_pred) / tf.transpose(weights))
    mse = k.sum(sq, axis=1)
    
    y_pred_flipped = tf.roll(y_pred, 3, axis=1)
    sq_flipped = k.square((y_true - y_pred_flipped) / tf.transpose(weights))
    mse_flipped = k.sum(sq_flipped, axis=1)
    vec = tf.stack([mse, mse_flipped], axis=1)
    loss = k.min(vec, axis=1)
    return loss
```

The next idea is to use the whole ecal / the shashlik part of the ecal as an input and use a CNN. 

## Stage 4* - Overlapping photons: whole Shashlik Input

The input will be now (25, 49) as the whole shashlik part of the ECAL is given as an input for the NN. I rewrote the user event to also save column and row number so I can just assign the energy values to an 'empty ecal' (numpy array) to simplify the data making. The coordinate system lays now in the lower left corner of the ECAL as one need positive values to use the ReLu activation function.

The data input is now big enough to change to a CNN. Here one can see a simple try.

```
model = tf.keras.models.Sequential() # parameter
layers = tf.keras.layers # alias to make it shorter to access the layers
model.add(layers.Input(shape=(25,49,1)))
model.add(layers.Normalization(mean=0.107, variance=0.429))
model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(25, 49, 1), padding="same"))
model.add(layers.AveragePooling2D(pool_size=(2,2), strides=2))
model.add(layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu', input_shape=(12, 24, 6), padding="same"))
model.add(layers.AveragePooling2D(pool_size=(2,2), strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(units=256, activation = 'relu'))
model.add(layers.Dense(units=128, activation = 'relu'))
model.add(layers.Dense(units=64, activation = 'relu'))
model.add(layers.Dense(units=6, activation = 'relu'))

model.summary()
model.compile(loss=loss_flip_weighted, optimizer=Adam(learning_rate=0.0005))
```

Unfortunantly, this didn't help with the 2 photon problem. I tried some modifications that all didn't help neither.

- [x] adjust the learning rate. The loss function mostly has a sharp edge were it drops drastically and after that nothing happens/it's a bit bumpy. I've tried to varry the learning rate but could't find a solution.
- [x] I looked at the values that are more than one $\sigma$ away. The cluster on the edge of the ECAL as learned worse than in the middle as observed in stage 4! What's also weird: Showers with $x/y$ position worse than one $\sigma$ have low or high energies! Why is the position harder to learn if they have higher energies? 
- [x] Making 2D-histograms shows correlations. One can see a anti-correlation between the two energies. This means that the sum of the energies is well learned but the assignment of the energies to the two photons doesn't work too well.
- [x] changing the kernel size. I've changed the kernel size from 5 and 4 to 6 ad 3. This didn't help. 


## Stage 5 - Counting photons

As I was quite desperate on what to else to do with the 2 photons I've moved on to an other problem. I'd like to write a classification network that count's the number of photons in the picture. I rewrote the user event so I can apply it to any amought of photons. While doing this I've found a major mistake: I double saved energies of clusters if Lednev put two fits in it! This is bad! But I've reran the networks from 4 und 4* and it didn't change anything... this was apparently not the major problem and the network learned to correct the mistake.

I've started with 2 photons and without even applying a distance I've been better than the Lednev fit! Yeay! I then moved on to 3 photons. 

**3 photons**:

The dataset is created so that the frist 2 photons have a distance of (2-20)cm (I THINK) and the third photon is then again distrubuted the same way around the second. The energy is again 2-200 GeV. This is a simple NN: 

```
model = tf.keras.models.Sequential() # 640 000 parameter
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

```
that creates the Confusionmatrix of (1, 0.88, 0.75).
I played with the HP:
- [x] less parameter: 340 000 due to 8 and 16 kernels instead of 16 and 32. CM: (0.98, 0.78, 0.83). So it's a bit worse than before.
- [x] parameterplay 1: 196 000 due to other CNN structure (3 C-layers with 8,16, 32 kernels with 6x6, 5x5, 4x4). CM=(1, 0.87, 0.78). Very similar to the above example.
- [x] parameterplay 2: 638 000 due to other CNN structure (3 C-layers with 8,16, 32 kernels with 4x4, 4x4, 3x3) and only one AvergPool layern inbetween. CM=(1, 0.8, 0.8). A bit worse. 
- [x] adjust the learning rate: $\alpha = 0.001$ CM=(0.99, 0.81, 0.86). $\alpha = 0.00001$ CM=(0.99, 0.84, 0.76)/ (1, 0.82, 0.81). So a bit higher is better. I've tried the lower learning rate with Dropout (0.2, 0.1 in dense structure) and get a CM=(1, 0.82, 0.81).
- [x] use MaxPooling on the 'less parameter' network: (0.99, 0.77, 0.82). Not really better. Idea: do it on the network shown above?
- [x] let's combine some stuff: like parameterplay 1 and $\alpha = 0.0003$. CM = (1, 0.84, 0.79). Same but with two dropout layers with (0.2, 0.1) in the dense structure: CM=(1, 0.82, 0.82). Well this didn't change too much. 

A larger problem atm: strong overfitting after ~50 epochs. Trainingsloss goes down but the validation loss up. Maye that problem is just that simple? And ~80% is just the limit for the 3 photon classification? If one looks at the cluster distribution it's clear that 2 and 3 photons can look very similar. 

I've looked at the performance at differnet distances as well. One can see that at min. 5cm distance Lednev get's a lot better. But the network can cover a smaller distances as well!

**4 photons**:




