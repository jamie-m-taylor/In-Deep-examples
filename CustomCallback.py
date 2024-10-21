# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:27:05 2024

@author: jamie.taylor
"""

import tensorflow as tf
import matplotlib.pyplot as plt


def u_exact(x):
    return x**2

x_input = tf.keras.layers.Input(shape=(1,), name="x_input") 
#Define the shape of the inputs

l1 = tf.keras.layers.Dense(40,activation="tanh")(x_input)
#Define a layer with 40 neurons and tanh activation.

u_output = tf.keras.layers.Dense(1)(l1)
#Define a layer with 1 neuron to be the output (no activation)

u_model = tf.keras.Model(inputs=x_input,outputs = u_output)
#Create the model 




# Define a custom callback class to measure the error at logarithmically spaced intervals during training
class measure_error(tf.keras.callbacks.Callback):
    
    # Initialization method for the callback
    def __init__(self, u_model, rate):
        # Call the base class constructor for Keras Callback
        super(measure_error, self).__init__()
        
        # Store the neural network model (u_model) passed as an argument
        self.u_model = u_model
        
        # Generate a test dataset (xtest) of 200 values evenly spaced between 0 and 1 
        # and store them as a TensorFlow constant
        self.xtest = tf.constant([[(i+0.5)/200] for i in range(200)])
        
        # Initialize the next iteration step for logging error at logarithmic intervals
        self.next_it = 1
        
        # Store the rate at which to increase the epoch interval for measuring error
        self.rate = rate
        
        # Initialize an empty list to store the calculated errors over time
        self.error_list = []
        
        # Initialize an empty list to store the epoch numbers where error is measured
        self.its_list = []

    # Method that runs at the end of each epoch during training
    def on_epoch_end(self, epoch, logs=None):
        # Check if the current epoch has reached or passed the next interval for error measurement
        if epoch > self.next_it:
            # Append the current epoch number to the list of iterations (its_list)
            self.its_list += [epoch]
            
            # Update next_it using the provided rate, to create logarithmic spacing for error measurement
            self.next_it = int(self.rate * self.next_it) + 1
            
            # Compute the model error and its gradient
            with tf.GradientTape() as t1:
                # Watch the test input for gradient calculation
                t1.watch(self.xtest)
                
                # Calculate the error between the model's predictions and the exact function (u_exact)
                u_err = self.u_model(self.xtest) - u_exact(self.xtest)
            
            # Compute the gradient of the error with respect to the test input
            du_err = t1.gradient(u_err, self.xtest)
            
            # Calculate the mean squared error and add it to the error list
            # The error is a combination of the error in the model's output (u_err)
            # and the error in the gradient (du_err)
            self.error_list += [tf.reduce_mean(du_err**2 + u_err**2)]
            

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

measure_callback = measure_error(u_model,1.1)


def my_loss(y_true,y_pred):
    return y_pred

loss_model.compile(optimizer=optimizer,loss=my_loss)

history = loss_model.fit(x = tf.constant([1.]),
                         y = tf.constant([1.]),
                         epochs = 5000,
                         callbacks = [measure_callback])





# Generate the x data
x_plot = tf.constant([[ (i + 0.5) / 500] for i in range(500)])

# Create a larger figure size
plt.figure(figsize=(6, 4))

# Plot the model line
plt.plot(x_plot, u_model(x_plot), color='blue', linewidth=2, label="$u_{NN}$")

plt.plot(x_plot, u_exact(x_plot), color='green',linestyle="--", linewidth=2, label="$u^*$")

# Add title and labels
plt.title("$u_{NN}$ vs exact soluiton", fontsize=16)
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


# Create a larger figure size
plt.figure(figsize=(6, 4))

# Plot training loss with a solid line
plt.scatter(measure_callback.its_list,measure_callback.error_list, label="Error",s=10)
# Set logarithmic scale for both axes
plt.xscale("log")
plt.yscale("log")

# Add title and axis labels with more clarity
plt.title("Error (callback)", fontsize=16)
plt.xlabel("Epoch (Log Scale)", fontsize=14)
plt.ylabel("Error (Log Scale)", fontsize=14)

# Add a grid for better visual guidance
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend with a better location
plt.legend(loc='best', fontsize=12)

# Show the plot
plt.show()

