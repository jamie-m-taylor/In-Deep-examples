# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:55:34 2024

@author: jamie.taylor
"""

import tensorflow as tf
import time 


# Define a function to compute the derivative of sin(x) using TensorFlow's GradientTape in Eager mode.
def deriv_sin(x):
    # GradientTape is used to record the computation for automatic differentiation
    with tf.GradientTape() as t1:
        # Watch the input tensor x to track its gradients
        t1.watch(x)
        # Compute sin(x)
        s = tf.math.sin(x)
    # Compute the gradient of sin(x) with respect to x
    ds = t1.gradient(s, x)
    return ds

# Define a function to compute the derivative of sin(x) in Graph mode by using the @tf.function decorator
@tf.function
def deriv_sin_graph(x):
    # Same computation as deriv_sin, but this function will be executed in Graph mode
    with tf.GradientTape() as t1:
        # Watch the input tensor x to track its gradients
        t1.watch(x)
        # Compute sin(x)
        s = tf.math.sin(x)
    # Compute the gradient of sin(x) with respect to x
    ds = t1.gradient(s, x)
    return ds

# Create a random tensor of size 10^8, which will be used for testing the functions
x = tf.random.uniform([10**8])

# Measure the time it takes to compute the derivative using the Eager execution mode
t0 = time.time()  # Start time before executing the function
ds = deriv_sin(x)  # Compute derivative in Eager mode
time_eager = time.time() - t0  # Calculate elapsed time for Eager execution

# Measure the time for the first execution of the graph-optimized function
t0 = time.time()  # Start time before executing the function
ds = deriv_sin_graph(x)  # Compute derivative in Graph mode (first call)
time_graph_1 = time.time() - t0  # Calculate elapsed time for the first Graph execution

# Measure the time for the second execution of the graph-optimized function
t0 = time.time()  # Start time before executing the function
ds = deriv_sin_graph(x)  # Compute derivative in Graph mode (second call)
time_graph_2 = time.time() - t0  # Calculate elapsed time for the second Graph execution

# Print the results for comparison between Eager and Graph mode executions
print("Time (Eager)", time_eager)  # Time for Eager mode (immediate execution)
print("Time (Graph, first call)", time_graph_1)  # Time for the first call in Graph mode (includes graph compilation overhead)
print("Time (Graph, second call)", time_graph_2)  # Time for the second call in Graph mode (faster since graph is reused)