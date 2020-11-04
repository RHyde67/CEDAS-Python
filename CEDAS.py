#%%
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:33:07 2018

@author: HydeR
Copyright R Hyde 2019
Released under the GNU GPLver3.0
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/
If you use this file please acknowledge the author and cite as a
reference:
Hyde R, Angelov P, MacKenzie AR (2017) Fully online clustering of evolving
data streams into arbitrarily shaped clusters. Inf Sci (Ny) 382–383:96–114 
doi: 10.1016/j.ins.2016.12.004

"""

# Function initialization
import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt
from math import pi

#%%
def CEDAS(sample, radius, decay, min_thresh, cluster_node, cluster_centre, cluster_life,\
        cluster_count, cluster_kernel, outliers, cluster_graph):
#    print (cluster_centre.shape)
    # check if in current microC?
    dist_2_microC = spatial.distance.cdist(cluster_centre, sample, 'euclidean')
    inside = np.less(dist_2_microC, radius)
    if inside.any == 0 or inside.shape[0] == 0: # if not inside microC
        print("Not in microC, creating new")
        # create microC
        outliers = np.vstack((outliers,sample))
        outliers, cluster_centre, cluster_life, cluster_count, cluster_kernel,  cluster_graph\
            = create_microC(radius, decay, min_thresh, outliers, cluster_node, cluster_centre, cluster_life,\
            cluster_count, cluster_kernel, cluster_graph)
    else:
        print("In current microC, updating")
        # update microC

    #update graph

    return (outliers, cluster_centre, cluster_life, cluster_count, cluster_kernel, cluster_graph)

def create_microC(radius, decay, min_thresh, outliers, cluster_node, cluster_centre, cluster_life,\
            cluster_count, cluster_kernel, cluster_graph):
    distances = np.asarray( spatial.distance.cdist(outliers, outliers, 'euclidean') )
    inside = np.less(distances, radius)
    num_close = inside.sum(axis=1)
    number_in = np.amax( num_close , 0) # maximum number of data within radius of another
    
    if number_in > min_thresh: # if enough data to be a valid microC
        location = np.where(num_close == number_in) # list of locations of data within radius
        location_1st = location[0][0]
        # calculate mean to use as microC centre point
        cluster_centre = np.append(cluster_centre, np.atleast_2d(np.mean(outliers[location], axis=0)), axis=0)
        cluster_life.append(1)
        cluster_count.append(number_in)
        # find data within radius of location_1st
        distances = np.asarray( spatial.distance.cdist(np.asarray(cluster_centre), outliers, 'euclidean') )
        inside = np.less(distances, radius)
        num_close = inside.sum(axis=1)
        inside_kernel = np.less(distances, radius/2)
        kernel = inside.sum(axis=1)
        cluster_node.append(cluster_centre.shape[0])
        cluster_graph.add_node(cluster_node[-1]) # add node to graph
        # remove assigned data from outlier list
        outliers = outliers[np.logical_not(inside[0]),:]
   
    return[outliers, cluster_centre, cluster_life, cluster_count, cluster_kernel,  cluster_graph]

def StartCluster(cluster_centre, cluster_life, cluster_count, cluster_kernel, outliers, radius, min_thresh, cluster_graph):
    distances = np.asarray( spatial.distance.cdist(outliers, outliers, 'euclidean') )
    # print(distances)
    # inside = (distances < radius).sum()
    inside = np.less(distances, radius)
    print(inside)
    num_close = inside.sum(axis=1)
    print(num_close)
    number_in = np.amax( num_close , 0) # maximum number of data within radius of another
    location = np.where(num_close == number_in) # list of locations of data within radius
    location_1st = location[0][0] # array location of 1st datum with number_in neighbours, i.e. 1st microC centre candidate
   
    if number_in > min_thresh: # if enough data to be a valid microC
        N = 1 # this is the first microC
        # calculate mean to use as microC centre point
        cluster_centre = np.atleast_2d(np.mean(outliers[location], axis=0))
        cluster_life = 1
        cluster_count = number_in
        # find data within radius of location_1st
        distances = np.asarray( spatial.distance.cdist(np.asarray(cluster_centre), outliers, 'euclidean') )
        inside = np.less(distances, radius)
        print('Is inside?', inside)
        num_close = inside.sum(axis=1)
        print('Number close',num_close)
        inside_kernel = np.less(distances, radius/2)
        print('Is in kernel?',inside_kernel)
        kernel = inside.sum(axis=1)
        print('Number in Kernel',kernel)
        cluster_graph.add_node(1) # add node to graph
        # remove assigned data from outlier list
        outliers = np.delete(outliers, inside, 0)


    return [cluster_centre, cluster_life, cluster_count, cluster_kernel, outliers, cluster_graph]
#%%
def Assign(cluster_centre, cluster_life, cluster_count, cluster_kernel, outliers, sample, radius, min_thresh, cluster_graph):
    distances = np.asarray( spatial.distance.cdist(sample, cluster_centre, 'euclidean') )
    min_distance = np.amin(distances)
    assigned_cluster = np.where(min_distance == distances)[1][0] # reverse variable order?
    if min_distance < radius:
        print(assigned_cluster)
        print('Data assigned to mC ', assigned_cluster)



    return (cluster_centre, cluster_life, cluster_count, cluster_kernel, outliers, cluster_graph)
