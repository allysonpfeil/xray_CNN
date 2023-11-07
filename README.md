# X-Ray Convolutional Neural Network
### By: [Allyson Pfeil](github.com/allysonpfeil)

#### note: I wrote this code on my Linux machine; thus, some variance in location and console will be noted.

## When to use a CNN:
CNNs are powerful tools often used for image recognition and classification tasks. A CNN might be a great solution for your problem if you have a large amount of image data to process and organize. CNNs are most applicable when a large amount of image data is already present, and more is anticipated. Such image data might need sorting for research, development, medical, engineering, or other purposes. A real-world example of CNN usgae would be algorithmic assistance of image digestion for a radiologist. The software can quickly, accurately, and efficiently classify x-rays as unremarkable or remarkable, essentially triaging the physician's efforts into more complex conditions. 

## What is a CNN:
A CNN is an artificial intelligence algorithm that maps features from images. CNNs often have a basic structure, and require hyperparameter/model engineering to properly learn and apply that learning to unseen data. CNNs can be engineered with respect to their layers, which have various filters to process features. Some layers average features, identify the max feature(s), and also purposefully "forget" (dropout) data, among others. 

## Code Review:

### [Main.Py](main.py)

This model is specifically engineered for binary classification machine learning problems. The skeleton code can be transformed for other solutions; however, it will likely require some programming skill to do so. This program uses my own propietary data that has been split manually into */train/* and */test/* folders. This may not be the most ideal solution for every programmer, especially those concerned with scalable and agile projects. However, for this specific example, that is what we are working with here. 

In brief, the program resizes and normalizes the images. Then, the image is passes through an 11 layer model:
  * four convolutional layers
  * four max pooling layers
  * two dense layers
  * one dropout layer

During training, early stopping is defined to help prevent overfitting. Otherwise, it is a pretty standard ~ although highly engineered ~ CNN model. 

## Results:

This model output an [accuracy of 96.5%](https://github.com/allysonpfeil/xray_CNN/blob/main/Screenshot%20from%202023-10-06%2019-29-05.png)

#### Breakdown of my propietary dataset:
#### n=1169 training images:
  * */experimental/* (n=760)
  * */control/* (n=409) and

#### n=200 testing images:
  * */experimental/* (n=100)
  * */control/* (n=100)
