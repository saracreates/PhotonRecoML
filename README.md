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

model1 = keras.Sequential([layers.Input(shape=(25)), $\newline$
                         layers.Dense(64, activation="relu"), $\newline$
                         layers.Dense(128, activation="relu"), $\newline$
                         layers.Dense(256, activation="relu"), $\newline$
                         layers.Dense(128, activation="relu"), $\newline$
                         layers.Dense(64, activation="relu"), $\newline$
                         layers.Dense(32, activation="relu"), $\newline$
                         layers.Dense(3, activation=None)]) $\newline$

I trained again with 200 epochs (batchsize now 64 instead of 50), validation split of 0.1.


**features investigated in stage 2:**
- [x] standardiation of input: global vs. none (not a big difference)
- [x] investigation of which E values are not trained well (lower worse than higher) 
- [ ] change loss function from mean squared error to mean absolute percentage error (failed due to divergence)
- [x] give sum of all cluster cell energies also as an input (doesn't change much)
- [x] placing the clusters completely random in the 5x5 grid (improves precision but not resolution)

**Open questions**

- [ ] How to normalize on other datasets?? Needed? Needed on ouput as well?

### Summary stage 2

In stage 2 I showed that the network can not only learn the right $x/y$ position but also the correct energy when dealing with photons in an energy range of 2-200 GeV. The energy resolution is $\pm 2.2$ %. Furthermore some minor changes were investigated (see above). 

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.lrz.de/00000000014A9969/ecal_NN.git
git branch -M main
git push -uf origin main
```

