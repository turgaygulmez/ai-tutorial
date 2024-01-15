# Documentation

- [Machine Learning](#machine-learning)
  - [Perceptrons](#perceptrons)
  - [Neural Networks](#neural-networks)
  - [Deep Neural Networks](#deep-neural-networks)
  - [Examples](#examples)

### Credits:

- [W3School](https://www.w3schools.com/ai/default.asp)

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

The Perceptron defines the first step into Neural Networks. It represents a single neuron with only one input layer, and no hidden layers. Consider them as a single node which is responsible to make a single decision.

### Neural Networks

Neural Networks are Multi-Layer Perceptrons. Meaning each node is connected to each other to perform more complex tasks.

![image info](https://www.w3schools.com/ai/img_nn_single_600.jpg)

Yellow nodes are the first perceptrons which is performing a simple decision. Once decision is made, its being passed through the next perceptron layer. Blue nodes will perform further decisions to have an accurate result.

### Deep Neural Networks

Deep Neural Networks are another layer in the network that performs even further decision in the network.

### Examples

We will be using some JavaScript libraries to hide the complexity of the mathematics in order to use ML and AI. Yet understanding the basics will help to use those libraries.

[How to build a Neural Network with brain.js](./ML/brainjs/app.js)
