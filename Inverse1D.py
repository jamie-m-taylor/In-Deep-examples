# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:39:20 2024

@author: jamie.taylor
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



tf.random.set_seed(1234)


def rhs(x):
    return -4*np.pi**2*tf.math.sin(2*np.pi*x)

def u_exact(x):
    return tf.math.sin(2*np.pi*x)

class loss_layer_PDE(tf.keras.layers.Layer): ###Name the layer
    def __init__(self,u_model,n,w): #Initialisation with parameters, u_model
        super(loss_layer_PDE,self).__init__() ##Define objects contained in the class
        self.n=n #This will be the number of points for MC
        self.w = w #The weight for the boundary term
        self.u_model=u_model  #Load u_model into loss_layer
        self.x0 = tf.constant([[0.],[1.]]) # Boundary x        
    
    def build(self,inputs):
        self.k = self.add_weight(shape=[1],
                                 initializer=tf.keras.initializers.RandomUniform(minval=0,maxval=0))
        ##Here we build the new trainable variable k, 
        #we have to specify how to initialise it, I initialise at 0. 
        #Its shape must be specified. 
    def call(self,inputs):  #The input plays no role
        x = tf.random.uniform([self.n,1]) #Take MC Sample
        with tf.GradientTape() as t2:
            t2.watch(x)
            with tf.GradientTape() as t1:
                t1.watch(x)
                u=self.u_model(x)  #Evaluate u
            du = t1.gradient(u,x)  #Evaluate derivative
        ddu = t2.gradient(du,x)    #Evaluate second derivative. 
        loss_PDE = tf.reduce_mean((self.k*ddu-rhs(x))**2) #Evaluate ODE loss
        loss_bc = self.w*tf.reduce_sum(self.u_model(self.x0)**2)
        return loss_bc+loss_PDE


class loss_layer_interpolation(tf.keras.layers.Layer): ###Name the layer
    def __init__(self,u_model,x_data,y_data): #Initialisation with parameters, u_model
        super(loss_layer_interpolation,self).__init__() ##Define objects contained in the class
        
        self.u_model=u_model  #Load u_model into loss_layer   
        self.x_data = x_data
        self.y_data = y_data
        
    def call(self,inputs):  #The input plays no role
        interp_error = self.u_model(self.x_data)-self.y_data
        return tf.reduce_mean(interp_error**2)


#Make noisy data
x_data = tf.random.uniform([10,1])
y_data = tf.math.sin(2*np.pi*x_data)+tf.random.normal([10,1],stddev=0.1)


x_input = tf.keras.layers.Input(shape=(1,), name="x_input") 
#Define the shape of the inputs

l1 = tf.keras.layers.Dense(40,activation="tanh")(x_input)
#Define a layer with 40 neurons and tanh activation.

u_output = tf.keras.layers.Dense(1)(l1)
#Define a layer with 1 neuron to be the output (no activation)

u_model = tf.keras.Model(inputs=x_input,outputs = u_output)
#Create the model 


#Weight of the PINNs loss vs. interpolation
weight_PDE = 0.1

fake_input = tf.keras.layers.Input(shape=(1,), name="fake_input") 
#Define the shape of the inputs for the loss model

loss_output = weight_PDE*loss_layer_PDE(u_model,100,10)(fake_input)+loss_layer_interpolation(u_model,x_data,y_data)(fake_input)

loss_model = tf.keras.Model(inputs=fake_input,outputs = loss_output)
#Create the model 


optimizer = tf.keras.optimizers.Adam(learning_rate=10**-2)

def my_loss(y_true,y_pred):
    return y_pred

loss_model.compile(optimizer=optimizer,loss=my_loss)

history = loss_model.fit(x = tf.constant([1.]),
                         y = tf.constant([1.]),
                         epochs = 5000)



x = tf.constant([[(i+0.5)/200] for i in range(200)])

#Plot the neural network model prediction with a thicker line
plt.plot(x, u_model(x), label="$u_{NN}$", color="blue", linewidth=2)

# Plot the exact solution with a dashed line for distinction
plt.plot(x, u_exact(x), label="$u^*$", color="green", linestyle="--", linewidth=2)

# Scatter the training data with larger, distinct markers
plt.scatter(x_data, y_data, color="red", s=50, marker='o', label="Data", edgecolors='black')

# Add title and axis labels
plt.title("Comparison of Neural Network Prediction vs. Exact Solution", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("u(x)", fontsize=14)

# Add grid for better visual alignment
plt.grid(True, linestyle='--', alpha=0.7)

# Add a legend with larger font size and custom location
plt.legend(fontsize=12, loc="upper right")

# Display the plot
plt.show()



# Create a larger figure size
plt.figure(figsize=(6, 4))

# Plot training loss with a solid line
plt.plot(history.history["loss"], label="Training Loss", color='blue', linestyle='-', linewidth=2)
# Set logarithmic scale for both axes
plt.xscale("log")
plt.yscale("log")

# Add title and axis labels with more clarity
plt.title("Loss", fontsize=16)
plt.xlabel("Epoch (Log Scale)", fontsize=14)
plt.ylabel("Loss (Log Scale)", fontsize=14)

# Add a grid for better visual guidance
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend with a better location
plt.legend(loc='best', fontsize=12)

# Show the plot
plt.show()
