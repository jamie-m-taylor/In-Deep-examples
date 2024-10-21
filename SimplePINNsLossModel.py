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



class loss_layer(tf.keras.layers.Layer): ###Name the layer
    def __init__(self,u_model,n,w): #Initialisation with parameters, u_model
        super(loss_layer,self).__init__() ##Define objects contained in the class
        self.n=n #This will be the number of points for MC
        self.w = w #The weight for the boundary term
        self.u_model=u_model  #Load u_model into loss_layer
        self.x0 = tf.constant([[0.]]) # Boundary x        
    def call(self,inputs):  #The input plays no role
        x = tf.random.uniform([self.n,1]) #Take MC Sample
        with tf.GradientTape() as t1:
            t1.watch(x)
            u=self.u_model(x)  #Evaluate u
        du = t1.gradient(u,x)  #Evaluate derivative
        loss = self.w*self.u_model(self.x0)**2+tf.reduce_mean((du-2*x)**2) #Evaluate ODE loss
        return loss 
        

fake_input = tf.keras.layers.Input(shape=(1,), name="fake_input") 
#Define the shape of the inputs for the loss model

loss_output = loss_layer(u_model,100,1)(fake_input)

loss_model = tf.keras.Model(inputs=fake_input,outputs = loss_output)
#Create the model 


optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3)

def my_loss(y_true,y_pred):
    return y_pred

loss_model.compile(optimizer=optimizer,loss=my_loss)

history = loss_model.fit(x = tf.constant([1.]),
                         y = tf.constant([1.]),
                         epochs = 5000)



x_plot = tf.constant([[(i+0.5)/200] for i in range(200)])
plt.plot(x_plot,u_model(x_plot),label="u_{NN}")
plt.plot(x_plot,x_plot**2, label="Exact")
plt.legend()
plt.show()

plt.plot(x_plot,u_model(x_plot)-x_plot**2)
plt.title("Pointwise error")
plt.show()

plt.plot(history.history["loss"])
plt.xscale("log")
plt.yscale("log")
plt.show()

# First plot: u_{NN}(x) vs Exact solution
plt.figure(figsize=(6, 4))  # Set figure size
plt.plot(x_plot, u_model(x_plot), label=r"$u_{\text{NN}}(x)$", color="blue", linewidth=2)
plt.plot(x_plot, x_plot**2, label="Exact", color="orange", linestyle="--", linewidth=2)
plt.title("Neural Network Approximation vs Exact Solution", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.legend(loc="best", fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Add grid
plt.tight_layout()  # Improve layout
plt.show()

# Second plot: Pointwise error
plt.figure(figsize=(6, 4))  # Set figure size
plt.plot(x_plot, u_model(x_plot) - x_plot**2, color="red", linewidth=2)
plt.title("Pointwise Error", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("Error", fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Add grid
plt.tight_layout()  # Improve layout
plt.show()

# Third plot: Loss curve
plt.figure(figsize=(6, 4))  # Set figure size
plt.plot(history.history["loss"], color="green", linewidth=2)
plt.title("Training Loss (Log Scale)", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Add grid
plt.tight_layout()  # Improve layout
plt.show()
