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
- [x] By analizing with values are more than one $\sigma$ away, one could see that photons that hit close to the edge of the 9x9 gird are learned worse. Where this bias comes from is unclear. I therefore modified the clusters to not be placed randomly in the 9x9 gird but placed in the middle. This helped with the performance! 

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

The input will be now (25, 49) as the whole shashlik part of the ECAL is given as an input for the NN. This is now big enough to change to a CNN.


## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.lrz.de/00000000014A9969/ecal_NN.git
git branch -M main
git push -uf origin main
```

