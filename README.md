predict.py takes in a path to a directory containing spherical power maps along with a path to the model and produces corresponding saliency maps

model is a convolutional lstm trained on spherical power maps based on audio samples from the avs-odv data set as input features and pre-processed saliency maps based on head movement from the avs-odv data set as labels

both were of resolution 19x36 (row x column), will experiment with higher resolution

[Link to example saliency map overlayed on top of video](https://drive.google.com/file/d/1zmIw1xIDSh_NetpsKjB6dpbEpbTtTRdb/view?usp=sharing)
(overlay makes less sense in this case because it's not a 360 video that's being projected)
