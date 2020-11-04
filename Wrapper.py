#%%
## -*- coding: utf-8 -*-
"""
Created on 16/04/2019 @author: HydeR
Basic wrapper to load data and run my Implementation of CEDAS
"""

# Initialise
import numpy as np
# import pandas as pd
import CEDAS as CEDAS
import matplotlib.pyplot as plt
# from timeit import default_timer as timer
from random import random as rand
import networkx as nx

# Close any open plot windows
plt.close('all')

def cross_data():
    # generate cross data
    d1 = []
    d2 = []          
    for idx1 in np.linspace(0.1, 0.9, 81):
            for idx2 in np.linspace(1,20,20):
                d1.append ( idx1 + rand()/30 )
                d2.append ( 1 - idx1 - rand()/30 )
    
    for idx1 in np.linspace(0.9, 0.1, 81):
            for idx2 in np.linspace(1,20,20):
                d1.append ( idx1 + rand()/30 )
                d2.append ( 1 - idx1 - rand()/30 )
    d = np.array([d1, d2])
    return d

data = cross_data() # generate synthetic data
print(data.shape)
data[:,0] = [-5,-5]
data[:,1] = [5,5]

# Start CEDAS
# Initialise
# CEDAS Parameters
radius = 0.05
decay = 500 # number of samples expected in time relevant period
decay = (1/decay)
min_thresh = (1,)
# Initialise cluster_graph
outliers = cluster_centre = np.array([]).reshape(0,2) # array of microC centre points
cluster_life = [] # value of microC life
cluster_count = cluster_kernel = [] # number of data in microC kernel
cluster_node = [] # list of graph nodes correspoding to microC
cluster_graph = nx.Graph()
idx1 = -1
cluster_parameters = [cluster_centre, cluster_life, cluster_count, cluster_kernel, outliers, radius, decay,\
     min_thresh, cluster_graph]

while idx1 < data.shape[1]-1:
    idx1 += 1
    # if idx1 == data.shape[1]-1: # restart loop if required
    #     idx1=0
    if idx1 > 5000:
        break
    
    sample = np.atleast_2d(data[:, idx1])
    # print(sample)
    # Run CEDAS Algorithm
    [ outliers, cluster_centre, cluster_life, cluster_count, cluster_kernel,cluster_graph]\
         = CEDAS.CEDAS(sample, radius, decay, min_thresh, cluster_node, cluster_centre, cluster_life,\
        cluster_count, cluster_kernel, outliers, cluster_graph)

    # Display cluster_graph

    # Display Clusters
    print(idx1)
