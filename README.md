# Deep_KsuccSite
Deep_KsuccSite, a novel and effective Deep learning models for predicting Succinylation sites, which adopted CTDC, EGAAC, APAAC, and embedding methods to encode peptides, then constructed three base classifiers using 1D-CNN and 2D-CNN, and finally utilized voting method to get the final results. K-fold cross-validation and independent testing showed that Deep_KsuccSite can serve as a powerful tool to identify Ksucc sites.
# Requirement
Backend = Tensorflow(1.14.0)\
keras(2.3.1)\
Numpy(1.20.2)\
scikit-learn(1.0.2)\
pandas(1.3.5)\
matplotlib(3.5.2)
# Dataset
The dataset includes positive and negative examples with a window size of 33， and various feature representations obtained after the dataset is divided.
# Model
The best model of different feature can be downloaded in model directory, where best_modelCTDC.h5, best_modelEGAAC+APAAC.h5 are the best model of CTDC, the combination of EGAAC and APAAC. The model.part1-4.rar should be unzip before used, and it's used for embedding method.  
# Contact
Feel free to contact us if you nedd any help: flyinsky6@gmail.com
