# ECAL with neural networks



## Description

Photons create clusters over multiple cells in the ECAL. To find the $x/y$-position and the energy of the inertial photon entering a fiting algorithm in coral is used. This fit doesn't work smoothly with two photons whose clusters overlap. This problem will be approched with a neural network.

## Overview

### Stage 1 - Monoenergetic photons E=140GeV

First everyting is kept simple. With a phast user event all cells with an energy deposit are saved and the MC truth and coral fit are also stored. The clusters are restored through python code and fit into 5x5 histograms where zero columns/rows are added (not completely) random. Clusters that are larger than 5x5 are cut out. By defining a new coordinate system in the middle of the lowest left cell that is involved the NN can learn relative coordinates to that corner as total position doesn't matter at the moment as the photons enter with no angle. 
Now use 80% of the data for training the network and 20% for validation. The input of the network are the 25 energy values of the 5x5 cluster which is read like a page row by row starting in the upper left corner. The output of the NN are the $x/y$-position and energy of the entering photon.
The first aim here is to get significantly better than the coral fit before moving on to the next stage.

**features investigated in stage 1:**
- [x] Stadardizing the input
- [x] learning rate
- [x] epoch size
- [ ] dropout layers

**What to do better the next time:**
- cutting non-usable clusters already in phast user event (e.g. clusters that are bigger than 5x5)
- adding the zero columns/rows completely random on each side of the clusters

**summary**

we will see...

### Stage 2 - Photons with energies up to E<=200GeV

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.lrz.de/00000000014A9969/ecal_NN.git
git branch -M main
git push -uf origin main
```

