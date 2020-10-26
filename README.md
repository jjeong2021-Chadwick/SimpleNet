# SimpleNet

This was based on the idea of Google's MobileNet regarding its use of depthwise-separable convolution to reduce the size of a neural network model

We've developed the techniques to reduce DNN layers to form SimpleNet

SimpleNet has been designed to detect hazardous objects that may exist on the sidewalk, which may include manholes, construction sites and fences, foundations that a pedestrian may trip over. The model has been developed into a mobile application called "PedGUARD" that alerts pedestrians, especially "smartphone zombies," whenever they are about to encounter such hazards while walking on the streets. The source code of this model primarily consits of a convolutional neural network model, which its size can be varied by adjusting the hyperparameters suggested:

- Feature Parameter (Determines the number of filters/features extracted from a single convolutional layer)
- Repeat Parameter (Determines the repeat of a single convolutional layer)

SimpleNet can still be applied to a dataset that has some degree of similarity among the features of the objects included in the images within the dataset.

Use reduced_model.py to use SimpleNet
