How to use prediction:
- Requires directory of sortable  19x36 (heightxwidth) PNGs of spatial audio map using inferno colors
- python -m venv env
- pip install -r requirements.txt (fix errors as needed)
- python new-predict.py --model best_model.keras --input [input dir] --output [output dir]

new-predict.py takes in a path to a directory containing spherical power maps along with a path to the model and produces corresponding saliency maps

model is a convolutional lstm trained on spherical power maps based on audio samples from the avs-odv data set as input features and pre-processed saliency maps based on head movement from the avs-odv data set as labels

[Link to example saliency map overlayed on top of video](https://drive.google.com/file/d/1zmIw1xIDSh_NetpsKjB6dpbEpbTtTRdb/view?usp=sharing)
(overlay makes less sense in this case because it's not a 360 video that's being projected)
