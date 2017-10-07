[fcn]: ./docs/fcn.png
[nn]: ./docs/neural_net.jpeg
[curve]: ./docs/training_curve.png
[people]: ./docs/people.png
[hero]: ./docs/hero.png

## Follow Me Project ##

We train a deep neural network to identify and track a target in simulation. In particular, we want a drone to follow a "hero." There are other people in the environment so we must distinguish between people and our hero. The purpose of this exercise is to segment objects (aka people) from camera images.

## Technical Overview ##
Given a camera image, our goal is to assign each pixel one of each label (none, person, hero). We use a data driven approach via deep learning. 

**Architecture**
The network is a simple encoder-decoder network commonly used for semantic segmentation. 

![fcn] 

The encoder is a single layer convolution transforming out image from 3 dimensions (RGB) to 32 dimensions. These are then sent through a 1x1 convolution layer to retain spatial information (as opposed to flattening via a fully connected layer). For the decoder step, we apply a transposed convolution by upsampling. Next, we concatenate the convolutional layer combined with the original input as a way to short-circuit information via skip connections. Finally, the decoder applies additional convolutional layers.

![nn]

**NN Operators**
Batch normalization - normalizing layers of the neural net. Intuition is that given input normalization is valuable, we might also want to normalize information across NN layers. This generally avoids the need for dropout
Skip Connections - combine data from different layers so as to combine information from multiple levels of granularity
1x1 Convolution - Instead of a fully connected layer that aggregates all neurons into a single output array, we preserve the spatial context via 1x1 convolution but instead shrink down to smaller layers

**Network Parameters**
Generally, parameters were optimized via trial and error.

Epoch: needed enough iterations such that training accuracy improved alog with validation accuracy
Learning Rate: chose .001 as it improved faster than .0001 and more stable than .01
Batch Size: 26 as constrained by gpu memory available

The final model had the following accuracy scores. Notice that the training accuracy improves alongside validation accuracy. This means we aren't overfitting to our training data. In addition, the accuracy improves rapidly over the first 5 iterations and then improves slightly till 30 iterations. 

![curve]

We can evaluate the performance empirically with a few examples. In the examples below, we show the raw image (left), ground truth labeled with discrete colors (middle), and our predicted labels (right).

Our hero is highlighted in blue. Notice that our network is able to identify rough boundaries of these objects of interest.

![hero]

For images with normal people, notice that our classifier sometimes mislabels some pixels of people as the hero. This can possibly be improved with additional training data.

![people]

## Generalizability ##

If we wanted to segment pixels for other objects, only a few changes are required. First, we would need to add an additional output node for this new object. Next, we would need to add labels to training data with this label. Ultimately, there are few changes required for the network but it will likely require more data in order to discriminate this new object from the others.
