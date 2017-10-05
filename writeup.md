## Follow Me Project ##

We train a deep neural network to identify and track a target in simulation. In particular, we want a drone to follow a "hero." There are other people in the environment so we must distinguish between people and our hero. The purpose of this exercise is to segment objects (aka people) from camera images.

## Technical Overview ##
Given a camera image, our goal is to assign each pixel one of each label (none, person, hero). We use a data driven approach via deep learning. 

** Architecture **
The network is a simple encoder-decoder network commonly used for semantic segmentation. The encoder is a single layer convolution transforming out image from 3 dimensions (RGB) to 32 dimensions. These are then sent through a 1x1 convolution layer to retain spatial information (as opposed to flattening via a fully connected layer). For the decoder step, we apply a transposed convolution by upsampling. Next, we concatenate the convolutional layer combined with the original input as a way to short-circuit information via skip connections. Finally, the decoder applies additional convolutional layers.

** NN Operators**
Batch normalization - normalizing layers of the neural net. Intuition is that given input normalization is valuable, we might also want to normalize information across NN layers


