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

model1 = keras.Sequential([layers.Input(shape=(25)), $\newline$
                         layers.LayerNormalization(axis=1), $\newline$
                         layers.Dense(64, activation="relu"), $\newline$
                         layers.Dense(128, activation="relu"), $\newline$
                         layers.Dense(256, activation="relu"), $\newline$
                         layers.Dense(128, activation="relu"), $\newline$
                         layers.Dense(64, activation="relu"), $\newline$
                         layers.Dense(32, activation="relu"), $\newline$
                         layers.Dense(3, activation=None)]) $\newline$


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

This is exciting! 

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.lrz.de/00000000014A9969/ecal_NN.git
git branch -M main
git push -uf origin main
```

