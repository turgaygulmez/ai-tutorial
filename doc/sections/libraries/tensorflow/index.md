# TensorFlow - Machine Learning Library

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Getting Started](#getting-started)
- [Building Neural Networks](#building-neural-networks)
- [Data Pipeline](#data-pipeline)
- [Model Training](#model-training)
- [Model Deployment](#model-deployment)
- [Advanced Features](#advanced-features)

## Overview

TensorFlow is an open-source machine learning library developed by Google. It provides a comprehensive ecosystem for building and deploying machine learning models across various platforms, from mobile devices to large-scale distributed systems.

**Key Characteristics:**
- **Flexible Architecture**: Deploy computation to one or more CPUs or GPUs
- **Production Ready**: Robust tools for deploying models in production
- **Cross-Platform**: Works on desktop, mobile, web, and cloud
- **Large Ecosystem**: Extensive tools and extensions

## Key Features

### 1. Eager Execution
- Immediate evaluation of operations
- Intuitive debugging and development
- Dynamic computational graphs

### 2. Keras Integration
- High-level neural networks API
- User-friendly and modular design
- Fast prototyping capabilities

### 3. TensorFlow Hub
- Library of reusable machine learning modules
- Pre-trained models and embeddings
- Transfer learning capabilities

### 4. TensorFlow Serving
- Production-ready model serving system
- High-performance inference
- Model versioning and management

### 5. TensorFlow Lite
- Mobile and embedded devices deployment
- Model optimization and quantization
- Cross-platform mobile inference

## Installation

### Basic Installation
```bash
# Install TensorFlow (CPU version)
pip install tensorflow

# Install with GPU support
pip install tensorflow-gpu

# Install specific version
pip install tensorflow==2.15.0
```

### Development Installation
```bash
# Install with additional dependencies
pip install tensorflow[and-cuda]  # GPU support on Linux
pip install tensorflow-macos       # macOS optimized version
pip install tensorflow-metal       # Metal acceleration for macOS

# Install from source (advanced)
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
bazel build //tensorflow/tools/pip_package:build_pip_package
```

### Verification
```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())
```

## Core Concepts

### 1. Tensors
Tensors are multi-dimensional arrays, the core data structure in TensorFlow:

```python
import tensorflow as tf

# Scalar (0-D tensor)
scalar = tf.constant(3.14)
print("Scalar:", scalar)

# Vector (1-D tensor)
vector = tf.constant([1, 2, 3, 4])
print("Vector:", vector)

# Matrix (2-D tensor)
matrix = tf.constant([[1, 2], [3, 4]])
print("Matrix:", matrix)

# 3-D tensor
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("3D Tensor:", tensor_3d)

# Tensor properties
print("Shape:", matrix.shape)
print("Data type:", matrix.dtype)
print("Rank:", tf.rank(matrix))
```

### 2. Operations
Mathematical operations on tensors:

```python
# Basic arithmetic
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# Addition
c = tf.add(a, b)  # or a + b
print("Addition:", c)

# Matrix multiplication
d = tf.matmul(a, b)  # or a @ b
print("Matrix multiplication:", d)

# Element-wise operations
e = tf.multiply(a, b)  # or a * b
print("Element-wise multiplication:", e)

# Reduction operations
mean_a = tf.reduce_mean(a)
sum_a = tf.reduce_sum(a, axis=1)
print("Mean:", mean_a, "Sum along axis 1:", sum_a)
```

### 3. Variables
Mutable tensors for model parameters:

```python
# Create variables
weight = tf.Variable(tf.random.normal([784, 10]), name='weight')
bias = tf.Variable(tf.zeros([10]), name='bias')

print("Weight shape:", weight.shape)
print("Bias shape:", bias.shape)

# Assign new values
weight.assign(tf.random.normal([784, 10]))
bias.assign_add(tf.ones([10]))  # Add to existing values
```

## Getting Started

### Simple Linear Regression
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# Convert to TensorFlow tensors
X_tf = tf.constant(X)
y_tf = tf.constant(y)

# Define model parameters
W = tf.Variable(tf.random.normal([1, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# Define the model
def linear_model(x):
    return tf.matmul(x, W) + b

# Define loss function
def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Training
optimizer = tf.optimizers.SGD(learning_rate=0.01)

for epoch in range(1000):
    with tf.GradientTape() as tape:
        predictions = linear_model(X_tf)
        loss = mean_squared_error(y_tf, predictions)
    
    # Calculate gradients
    gradients = tape.gradient(loss, [W, b])
    
    # Update parameters
    optimizer.apply_gradients(zip(gradients, [W, b]))
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

print(f"Final W: {W.numpy()[0, 0]:.4f}, b: {b.numpy()[0]:.4f}")
```

## Building Neural Networks

### Using Keras Sequential API
```python
import tensorflow as tf
from tensorflow import keras

# Simple feedforward network
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()
```

### Using Keras Functional API
```python
# More flexible model definition
inputs = keras.layers.Input(shape=(784,))
x = keras.layers.Dense(128, activation='relu')(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Custom Model Class
```python
class CustomModel(keras.Model):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.dense1 = keras.layers.Dense(128, activation='relu')
        self.dropout1 = keras.layers.Dropout(0.2)
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.dropout2 = keras.layers.Dropout(0.2)
        self.dense3 = keras.layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.dense3(x)

# Create and compile custom model
model = CustomModel(num_classes=10)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Data Pipeline

### Using tf.data API
```python
# Create dataset from tensors
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

# Apply transformations
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Data preprocessing pipeline
def preprocess_image(image, label):
    # Normalize pixel values
    image = tf.cast(image, tf.float32) / 255.0
    # Data augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    return image, label

# Apply preprocessing
dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
```

### Loading Data from Files
```python
# Load images from directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    'path/to/train',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'path/to/train',
    validation_split=0.2,
    subset="validation", 
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

## Model Training

### Basic Training Loop
```python
# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {test_acc:.4f}')
```

### Training with Callbacks
```python
# Define callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
    keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
]

# Train with callbacks
history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=callbacks
)
```

### Custom Training Loop
```python
# Custom training for more control
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_accuracy.update_state(y, predictions)
    return loss

# Training loop
for epoch in range(epochs):
    for step, (x_batch, y_batch) in enumerate(train_ds):
        loss = train_step(x_batch, y_batch)
        
        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
    
    # Validation
    for x_val, y_val in val_ds:
        val_predictions = model(x_val, training=False)
        val_accuracy.update_state(y_val, val_predictions)
    
    print(f"Epoch {epoch} - Train Accuracy: {train_accuracy.result():.4f}, "
          f"Val Accuracy: {val_accuracy.result():.4f}")
    
    train_accuracy.reset_states()
    val_accuracy.reset_states()
```

## Model Deployment

### Saving and Loading Models
```python
# Save entire model
model.save('my_model.h5')

# Save in SavedModel format (recommended)
model.save('my_model_dir')

# Load model
loaded_model = keras.models.load_model('my_model.h5')

# Save only weights
model.save_weights('model_weights.h5')

# Load weights
model.load_weights('model_weights.h5')
```

### TensorFlow Serving
```python
# Export model for TensorFlow Serving
import tensorflow as tf

# Define serving signature
@tf.function
def serve_function(input_tensor):
    return model(input_tensor)

# Save with serving signature
tf.saved_model.save(
    model, 
    'serving_model',
    signatures={'serving_default': serve_function}
)
```

### Convert to TensorFlow Lite
```python
# Convert model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Apply optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Use TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Make prediction
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
```

## Advanced Features

### Custom Layers
```python
class CustomLayer(keras.layers.Layer):
    def __init__(self, units=32):
        super(CustomLayer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# Use custom layer
model = keras.Sequential([
    CustomLayer(64),
    keras.layers.ReLU(),
    keras.layers.Dense(10, activation='softmax')
])
```

### Custom Loss Functions
```python
def custom_loss(y_true, y_pred):
    # Custom loss implementation
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return mse + 0.1 * mae

# Use custom loss
model.compile(
    optimizer='adam',
    loss=custom_loss,
    metrics=['accuracy']
)
```

### Distributed Training
```python
# Multi-GPU training
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

# Create model within strategy scope
with strategy.scope():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# Train distributed model
model.fit(train_ds, epochs=10, validation_data=val_ds)
```

### TensorBoard Integration
```python
# Set up TensorBoard logging
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir, 
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# Train with TensorBoard callback
model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[tensorboard_callback]
)

# View in TensorBoard
# tensorboard --logdir logs/fit
```

### Model Optimization
```python
# Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Post-training quantization
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

quantized_model = converter.convert()

# Pruning
import tensorflow_model_optimization as tfmot

# Apply pruning
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50,
        final_sparsity=0.80,
        begin_step=0,
        end_step=1000
    )
}

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model, **pruning_params
)

model_for_pruning.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Computer Vision with TensorFlow

### Convolutional Neural Networks
```python
# CNN for image classification
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Transfer Learning
```python
# Use pre-trained model
base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom head
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(num_classes, activation='softmax')
])
```

## Natural Language Processing

### Text Processing
```python
# Text vectorization
vectorizer = keras.layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=100
)

# Adapt to text data
vectorizer.adapt(text_dataset)

# Create model for text classification
model = keras.Sequential([
    vectorizer,
    keras.layers.Embedding(10000, 128),
    keras.layers.LSTM(64),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])
```

### Transformer Model
```python
# Multi-head attention layer
class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)
        
        self.dense = keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights
```

## Best Practices

### 1. Model Development
- Start with simple models and gradually increase complexity
- Use appropriate data preprocessing and augmentation
- Monitor training with validation metrics
- Implement proper regularization techniques

### 2. Performance Optimization
- Use tf.data for efficient data pipelines
- Leverage GPU acceleration when available
- Apply model quantization for deployment
- Use mixed precision training for faster training

### 3. Code Organization
- Structure code with clear separation of concerns
- Use configuration files for hyperparameters
- Implement proper logging and monitoring
- Write unit tests for custom components

### 4. Production Deployment
- Use TensorFlow Serving for scalable inference
- Implement proper model versioning
- Monitor model performance in production
- Set up automated retraining pipelines

## Resources and Community

- **Official Website**: [tensorflow.org](https://tensorflow.org)
- **Documentation**: [TensorFlow Docs](https://www.tensorflow.org/api_docs)
- **Tutorials**: [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- **GitHub**: [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
- **Community**: TensorFlow Forums and Stack Overflow
- **Courses**: TensorFlow Developer Certificate Program

## Troubleshooting

### Common Issues
1. **GPU not recognized**: Check CUDA and cuDNN installation
2. **Out of memory errors**: Reduce batch size or model complexity
3. **Slow training**: Optimize data pipeline and use appropriate hardware
4. **Convergence issues**: Adjust learning rate and optimization settings

### Performance Tips
- Use tf.function for graph optimization
- Batch operations when possible
- Minimize data copying between CPU and GPU
- Use appropriate data types (float16 vs float32)

## Limitations

- Steep learning curve for beginners
- Large library size and dependencies
- Debugging can be challenging with graph execution
- Resource intensive for large models
- Frequent API changes between versions