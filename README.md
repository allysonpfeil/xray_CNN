CNN that determines whether a patient has a hip implant

This model output an accuracy of 96.5% on my propietary dataset of 1169 training images (split between two classes: /implant (#760) and /no_implant (#409) and 200 testing images (split between two classes: /implant (#100) and /no_implant (#100)).

this model should work quite well for binary classifcation machine learning problems.

First, I specify where the training and testing data is. Since this is my own data, I had already split the data into /train and /test folders. This is not necessarily typical, or even best fashion, but for this specific project it had been done. The script then resizes the images to 64 x 64. I was using my Linux machine for this project, so typical speed and efficacy concerns were not considered. 

After that, the script normalizes and augments the training data to be more robust and generalizable.

When defining this model, I elected for a very standard CNN with pooling layers after each convolutional layer. After the dense layer, a 30% dropout rate occurs, furthering the extensiveness of the model's training. 

The model utilizes early stopping so as to not overfit. Early stopping was defined with a patience of 10 epochs and will restore the best epoch in the event of training cessation. 

Lastly, the model evaluates its performance on the unseen training data. 
