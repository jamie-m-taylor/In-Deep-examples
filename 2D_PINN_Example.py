# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:27:05 2024

@author: jamie.taylor
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def rhs(x,y):
    return -8*tf.math.sin(2*x)*tf.math.cos(2*y)


def u_exact(x,y):
    return tf.math.sin(2*x)*tf.math.cos(2*y)

def make_u_model(neurons,activation="tanh"):
    
    
    xvals = tf.keras.layers.Input(shape=(2,), name="x_input")
    
    l1 = tf.keras.layers.Dense(neurons,activation=activation)(xvals)
    
    out = tf.keras.layers.Dense(1)(l1)
    
    
    u_model = tf.keras.Model(inputs=xvals,outputs = out)
    
    
    u_model.summary()
    
    return u_model

def sample_disc(n):
    r=tf.random.uniform([n,1])**0.5
    t = tf.random.uniform([n,1],maxval=2*np.pi)
    x = r*tf.math.cos(t)
    y = r*tf.math.sin(t)
    return x,y

def sample_circle(n):
    t = tf.random.uniform([n,1],maxval=2*np.pi)
    x = tf.math.cos(t)
    y = tf.math.sin(t)
    return x,y    


class loss_PDE_layer(tf.keras.layers.Layer):
    def __init__(self,u_model,n1):
        super(loss_PDE_layer,self).__init__()
        self.u_model = u_model
        self.n1 = n1
    def call(self,inputs):
        
        x,y = sample_disc(self.n1)
        
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            t1.watch(y)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                t2.watch(y)
                xy = tf.concat([x,y],axis=-1)
                u = self.u_model(xy)
            dux,duy = t2.gradient(u,[x,y])
        lapu = t1.gradient(dux,x)+t1.gradient(duy,y)
        return tf.reduce_mean((lapu-rhs(x,y))**2)


class loss_bc_layer(tf.keras.layers.Layer):
    def __init__(self,u_model,n2):
        super(loss_bc_layer,self).__init__()
        self.u_model = u_model
        self.n2 = n2
         
    def call(self,inputs):       
        x0,y0 = sample_circle(self.n2)
        xy0 = tf.concat([x0,y0],axis=-1)
        return tf.reduce_mean((self.u_model(xy0)-u_exact(x0,y0))**2)


class metric_h1_layer(tf.keras.layers.Layer):
    def __init__(self,u_model):
        super(metric_h1_layer,self).__init__()
        
        self.u_model = u_model
        
        rtest = tf.constant([(i+0.5)/50 for i in range(10)])
        ttest = tf.constant([(i+0.5)*2*np.pi/100 for i in range(50)])
        
        R,T = tf.meshgrid(rtest,ttest)
        R=tf.reshape(R,[50*10,1])
        T=tf.reshape(T,[50*10,1])
        
        self.xtest = tf.math.cos(T)*R
        self.ytest = tf.math.sin(T)*R
        self.Rtest = R
    def call(self,inputs):
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(self.xtest)
            t1.watch(self.ytest)
            xy = tf.concat([self.xtest,self.ytest],axis=-1)
            err = self.u_model(xy)-u_exact(self.xtest,self.ytest)
        derrx,derry = t1.gradient(err,[self.xtest,self.ytest])
        h1part = (derrx)**2+(derry)**2
        return tf.reduce_mean(self.Rtest*(h1part+err**2))**0.5

class my_stack_layer(tf.keras.layers.Layer):
    def call(self,inputs):
        return tf.stack(inputs,axis=-1)
        

u_model=make_u_model(200)

n1=1000 #MC points for PDE
n2 = 200 #MC points for BC
w=10 #Set weight for boundary term 


fake_input = tf.keras.layers.Input(shape=(1,), name="fake_input") 
#Define the shape of the inputs for the loss model

loss_PDE = loss_PDE_layer(u_model,n1)(fake_input)
#Define the PDE loss

loss_bc = loss_bc_layer(u_model,n2)(fake_input)
#Define the BC loss

metric_h1 = metric_h1_layer(u_model)(fake_input)
#Define the metric

loss_output = my_stack_layer()([loss_PDE,loss_bc,metric_h1])
#Final output

loss_model = tf.keras.Model(inputs=fake_input,outputs = loss_output)
#Create the model 


optimizer = tf.keras.optimizers.Adam(learning_rate=10**-3)



def my_loss(y_true,y_pred):  #The full loss, with the weight w
    return y_pred[0]+w*y_pred[1]

def pde_loss(y_true,y_pred): #Only measure the PDE component
    return y_pred[0]

def bc_loss(y_true,y_pred): #Only measure the bc component
    return y_pred[1]

def my_val(y_true,y_pred): #measure the H1-error
    return y_pred[2]

loss_model.compile(optimizer=optimizer,loss=my_loss,metrics=[pde_loss,bc_loss,my_val])

history = loss_model.fit(x = tf.constant([[1.]]),
                         y = tf.constant([1.]),
                         epochs = 5000)




plt.figure(figsize=(6, 4))  # Set figure size
plt.plot(history.history["loss"], color="green", linewidth=2,label="Full Loss")
plt.plot(history.history["pde_loss"], color="blue", linewidth=2,label="PDE Loss")
plt.plot(history.history["bc_loss"], color="black", linewidth=2,label="BC Loss")
plt.title("Loss (Log Scale)", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Add grid
plt.tight_layout()  # Improve layout
plt.show()


plt.figure(figsize=(6, 4))  # Set figure size
plt.plot(history.history["my_val"], color="green", linewidth=2,label="Error")
plt.title("$H^1$-error (Log Scale)", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Error", fontsize=12)
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Add grid
plt.tight_layout()  # Improve layout
plt.show()



ntest = 150

# Create test points in the range [-1, 1]
xtest = tf.constant([-1 + 2 * (i + 0.5) / ntest for i in range(ntest)])
Xtest, Ytest = tf.meshgrid(xtest, xtest)
xtest = tf.reshape(Xtest, [ntest**2])
ytest = tf.reshape(Ytest, [ntest**2])


# Predicted and exact values (using mask)
ua = tf.reshape( tf.squeeze(u_model(tf.stack([xtest, ytest], axis=-1))), [ntest, ntest])
ue = tf.reshape(u_exact(xtest, ytest), [ntest, ntest])

# Apply mask for the region inside the unit circle
mask = (Xtest**2 + Ytest**2 < 1)

# Plot the contour for ua in the region where x^2 + y^2 < 1
plt.contourf(Xtest, Ytest, np.where(mask, ua, np.nan), cmap='viridis')

# Add color bar and title for clarity
plt.colorbar()
plt.title("$u_{NN}$")
plt.legend()  # Add the legend for the scatter plot

# Set aspect ratio to ensure the plot is a circle
plt.gca().set_aspect('equal', adjustable='box')

# Tight layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()


# Plot the contour for ua in the region where x^2 + y^2 < 1
plt.contourf(Xtest, Ytest, np.where(mask, ue, np.nan), cmap='viridis')

# Add color bar and title for clarity
plt.colorbar()
plt.title("$u^*$")
plt.legend()  # Add the legend for the scatter plot

# Set aspect ratio to ensure the plot is a circle
plt.gca().set_aspect('equal', adjustable='box')

# Tight layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()




# Plot the contour for ua in the region where x^2 + y^2 < 1
plt.contourf(Xtest, Ytest, np.where(mask, ue-ua, np.nan), cmap='viridis')

# Add color bar and title for clarity
plt.colorbar()
plt.title("$u^*-u_{NN}$")
plt.legend()  # Add the legend for the scatter plot

# Set aspect ratio to ensure the plot is a circle
plt.gca().set_aspect('equal', adjustable='box')

# Tight layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()