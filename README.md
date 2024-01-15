# Documentation

- [Machine Learning](#machine-learning)
  - [Perceptrons](#perceptrons)
  - [Neural Networks](#neural-networks)
  - [Deep Neural Networks](#deep-neural-networks)
  - [Examples](#examples)

### Credits:

- [W3School](https://www.w3schools.com/ai/default.asp)
- [Simplilearn](https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-epoch-in-machine-learning)

## Machine Learning

Traditional programming uses known algorithms to produce results from data:

Data + Algorithms = Results

Machine learning creates new algorithms from data and results:

Data + Results = Algorithms

![image info](./img/ml_basic.png)

Artificial Intelligence (AI) is an umbrella term for computer software that mimics human cognition in order to perform complex tasks and learn from them.

Machine learning (ML) is a subfield of AI that uses algorithms trained on data to produce adaptable models that can perform a variety of complex tasks.

Deep learning is a subset of machine learning that uses several layers within neural networks to do some of the most complex ML tasks without any human intervention.

In order to understand how ML works, firstly we need to learn how neural networks work.

### Perceptrons

The Perceptron defines the first step into Neural Networks. It represents a single neuron with only one input layer, and no hidden layers.

### Neural Networks

Neural Networks are Multi-Layer Perceptrons. Meaning each neural is connected to each other to perform more complex tasks.

![image info](https://www.w3schools.com/ai/img_nn_single_600.jpg)

Yellow nodes are the first perceptrons which is performing a simple decision. Once decision is made, its being passed through the next perceptron layer. Blue nodes will perform further decisions to have an accurate result.

### Deep Neural Networks

Deep Neural Networks are another layer in the network that performs even further decision in the network.

### Epoch and Batch in Machine Learning

Each time a dataset passes through an algorithm, it is said to have completed an epoch. Therefore, Epoch, in machine learning, refers to the one entire passing of training data through the algorithm. It's a hyperparameter that determines the process of training the machine learning model.

The training data is always broken down into small batches to overcome the issue that could arise due to storage space limitations of a computer system. These smaller batches can be easily fed into the machine learning model to train it. This process of breaking it down to smaller bits is called batch in machine learning. This procedure is known as an epoch when all the batches are fed into the model to train at once.

An epoch is when all the training data is used at once and is defined as the total number of iterations of all the training data in one cycle for training the machine learning model.

Another way to define an epoch is the number of passes a training dataset takes around an algorithm. One pass is counted when the data set has done both forward and backward passes.

Let's explain Epoch with an example. Consider a dataset that has 200 samples. These samples take 1000 epochs or 1000 turns for the dataset to pass through the model. It has a batch size of 5. This means that the model weights are updated when each of the 40 batches containing five samples passes through. Hence the model will be updated 40 times.

#### Neural Networks Examples

We will be using some JavaScript libraries to hide the complexity of the mathematics in order to use ML and AI. Yet understanding the basics will help to use those libraries.

- [How to build a Neural Network with brain.js](./ML/brainjs/app.js)
- [How to build a Neural Network with ml5js](./ML/ml5/app.js)
