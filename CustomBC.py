# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:27:05 2024

@author: jamie.taylor
"""

import tensorflow as tf
import matplotlib.pyplot as plt


class bc_layer(tf.keras.layers.Layer): ###Name the BC layer
    def __init__(self,a,b): ##Initialisation - include self, any parameters
        super(bc_layer,self).__init__() ##Define objects contained in the class
        self.a = a
        self.b = b
        
        #a and b will be the cutoff points, so u(a)=u(b)=0
        
    def call(self,inputs):  ##Define how constructs the output
        x,u1 = inputs # The inputs are a pair [x,u1]
        cut = (x-self.a)*(x-self.b) #We define the cutoff function
        output = cut*u1 #Apply the cutoff function
        return output 
        

x_input = tf.keras.layers.Input(shape=(1,), name="x_input") 
#Define the shape of the inputs

l1 = tf.keras.layers.Dense(40,activation="tanh")(x_input)
#Define a layer with 40 neurons and tanh activation.

u_no_cutoff = tf.keras.layers.Dense(1)(l1)
#Define a layer with 1 neuron to be the output (no activation)

u_output = bc_layer(0.,1.)([x_input,u_no_cutoff])
#Apply cutoff

u_model = tf.keras.Model(inputs=x_input,outputs = u_output)
#Create the model 


xtest = tf.constant([[i/100] for i in range(101)])


plt.plot(xtest,u_model(xtest))   #Evaluate the model