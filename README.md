# Code for "A Machine Learning Framework for Event Detection and Analysis in Stratospheric Balloon Flight Data"

This repository provides the code associated with the paper:
"A Machine Learning Framework for Event Detection and Analysis in Stratospheric Balloon Flight Data." by Stott, A.E., Gerier, S., Garcia, R.F., Mimoun, D., Heydon, T. and Hertzog, A. https://doi.org/10.22541/essoar.15005338/v1

This code builds an autoencoder to reconstruct pressure data from the TSEN instrument on the CNES Strateole balloon flights. The data is a pre-requisite for this code and is available from: https://doi.org/10.14768/c417e612-015d-4812-9b59-294a6570c7c3

Contents:
- "PreProcessandTrainAutoencoder.py" provides the code to pre-process the raw data and train the autoencoder.
- "LoadTrainedAutoencoderandClusteringAnalysis.py" provides the code to pre-process the data and load in the already trained autoencoder as used in the manuscript as well as implement the cluster analysis.
- "TrainedModel_weeightsonly.h5" is the weights of the trained autoencoder used in the manuscript.


