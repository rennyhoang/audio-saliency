predict.py takes in a path to a directory containing spherical power maps along with a path to the model and produces corresponding saliency maps

model is a convolutional lstm trained on spherical power maps based on audio samples from the avs-odv data set as input features and pre-processed saliency maps from the avs-odv data set as labels

both were of resolution 19x36 (row x column), will experiment with higher resolution
