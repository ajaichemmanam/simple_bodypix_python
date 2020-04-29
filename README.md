# simple_bodypix_python
A simple and minimal bodypix inference in python


# Getting Started
1. Install requirements
2. Download tfjs models for bodypix.
```bash
# For example, download a ResNet50-based model to ./bodypix_resnet50_float_model-stride16
$ ./get-model.sh bodypix/resnet50/float/model-stride16
```
3. Set path to models and image for inference in .py files
```py
imagePath = './awesome_img.jpg'
modelPath = './bodypix_resnet50_float_model-stride16/'
```
4. python3 evalbody_singleposemodel.py (Image with single person)

# Observed Results

SINGLE POSE OUTPUT
![SinglePose Output](https://raw.githubusercontent.com/ajaichemmanam/simple_bodypix_python/master/assets/singlepose.png)
![SinglePose Part Heatmaps](https://raw.githubusercontent.com/ajaichemmanam/simple_bodypix_python/master/assets/singlepose_partheatmaps.png)

# Acknowledgement
1. https://github.com/patlevin for support functions
2. https://github.com/likeablob for download script

# Note:
Multipose is work in progress. Pull Requests are welcome.