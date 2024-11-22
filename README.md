# Photon reconstruction with neural networks

This respository hosts the code for my bachelor thesis ["Neural Networks for Photon Reconstruction in Electromagnetic Calorimeters for the COMPASS and AMBER Experiments"](https://wwwcompass.cern.ch/compass/publications/theses/2023_bac_aumiller.pdf) submitted on the 07.07.2023 at the Technical University of Munich.


## Abstract of the thesis

As traditional methods for photon reconstruction in electromagnetic calorimeters at the
COMPASS and AMBER experiments at CERN are pushed to their limits, we consider an
alternative approach using artificial neural networks. This thesis is a proof of principle study
on whether neural networks can improve the reconstruction of photon positions and energies
detected with the calorimeter on simulated data. The network can determine the position of a
single photon with higher precision than the traditional fit method and shows roughly the same
performance in energy reconstruction. Although the network was trained on simulated data,
it succeeds in reconstructing real-data photons with the same accuracy as the traditional fit
method. In the case of two overlapping photons, the network still performs better in position
determination than the traditional method, however the energy prediction of the traditional
method remains superior. This lack of performance might be due to the network’s architecture
being chosen too simply. Examination of the network’s ability to count the amount of photons
detected on the calorimeter showed that using neural networks might improve the separation of
small-distance photon showers but should be trained sensibly to predict the correct number of
photons in all cases. In short, artificial neural networks show potential in photon reconstruction
in electromagnetic calorimeters but the network’s complexity and training process must be
chosen carefully to achieve better results than traditional methods.


## Structure of the repository 

To approach the task of reconstructing the position ($x$, $y$) and energy $E$ of photons hitting the ECAL, we build up the complexity step by step. 

1. Stage 1: Single, monoenergetic photons($E=1404 GeV)
2. Stage 2: Single photons ($E=<200$ GeV)
3. Stage 3: Single photons coming from different angles
4. Stage 4: Two photons, reconstruction locally (9x9 cells)
5. Stage 4, fullrange: Two photons
6. Stage 5: Counting the total amount of photons