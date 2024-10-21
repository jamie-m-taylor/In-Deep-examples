# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:27:05 2024

@author: jamie.taylor
"""

import tensorflow as tf
import matplotlib.pyplot as plt



x_input = tf.keras.layers.Input(shape=(1,), name="x_input") 
#Define the shape of the inputs

l1 = tf.keras.layers.Dense(40,activation="tanh")(x_input)
#Define a layer with 40 neurons and tanh activation.

u_output = tf.keras.layers.Dense(1)(l1)
#Define a layer with 1 neuron to be the output (no activation)

u_model = tf.keras.Model(inputs=x_input,outputs = u_output)
#Create the model 


x_data = tf.random.uniform([200,1],minval=-1,maxval=1) #Obtain training data from random sample
y_data = x_data**3 +tf.random.normal([200,1],stddev=0.1) #Noisy outputs

x_val = tf.random.uniform([20,1],minval=-1,maxval=1) #Obtain validation data from random sample
y_val = x_val**3 +tf.random.normal([20,1],stddev=0.1) #Noisy outputs

optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3)
##Define the optimiser, there are typically many parameters to choose from

u_model.compile(optimizer=optimizer,loss="mse")
##The model has to be compiled, specifying the loss, before training

history = u_model.fit(
    x = x_data,
    y = y_data,
    epochs=5000,
    batch_size=32,
    validation_data =[x_val,y_val]
    )
##This function defines the training loop. 




# Generate the x data
x_plot = tf.constant([[-1 + 2 * (i + 0.5) / 500] for i in range(500)])

# Create a larger figure size
plt.figure(figsize=(6, 4))

# Plot the model line
plt.plot(x_plot, u_model(x_plot), color='blue', linewidth=2, label="Model Prediction")

# Scatter plot for the data points
plt.scatter(x_data, y_data, color='red', marker='o', s=10, label="Data Points")

# Add title and labels
plt.title("Model vs Data", fontsize=16)
plt.xlabel("X-axis", fontsize=14)
plt.ylabel("Y-axis", fontsize=14)

# Add a grid
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend
plt.legend(loc='best')

# Show the plot
plt.show()



# Create a larger figure size
plt.figure(figsize=(6, 4))

# Plot training loss with a solid line
plt.plot(history.history["loss"], label="Training Loss", color='blue', linestyle='-', linewidth=2)

# Plot validation loss with a dashed line
plt.plot(history.history["val_loss"], label="Validation Loss", color='orange', linestyle='--', linewidth=2)

# Set logarithmic scale for both axes
plt.xscale("log")
plt.yscale("log")

# Add title and axis labels with more clarity
plt.title("Training and Validation Loss", fontsize=16)
plt.xlabel("Epoch (Log Scale)", fontsize=14)
plt.ylabel("Loss (Log Scale)", fontsize=14)

# Add a grid for better visual guidance
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend with a better location
plt.legend(loc='best', fontsize=12)

# Show the plot
plt.show()

