#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:29:42 2019

@author: edwinagnew
"""

import network
import numpy as np


def prune_to(net, sparsity, training_data, test_data, validation_data = None):
    if validation_data == None:
        validation_data = test_data
    pnet = network.Network(net.sizes, return_vector = net.return_vector)
    pnet.weights = net.weights
    pnet.biases = net.biases
    """training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)
    validation_data = list(validation_data)"""
    
    all_weights = np.asarray([item for sublist in pnet.weights for subsub in sublist for item in subsub])
    thresh = np.percentile(np.absolute(all_weights), sparsity)
    
    
    for i in range(len(net.weights)):
        pnet.weights[i][(pnet.weights[i] > -thresh) & (pnet.weights[i] <= thresh)] = 0.0
   
    new_acc = pnet.evaluate(validation_data) / len(validation_data)
    print(new_acc, " around region ±", thresh)
    
    pnet.SGD(training_data, 2, 10, 3.0)
    
    sp, total = get_sparsity(pnet)
    print("sparsity: ", sp/total)
    print("Epoch with pruned weights around {}: {} / {}".format(thresh, pnet.evaluate(test_data),len(test_data))) 
    
    return pnet

def prune_retrain(net, region, training_data, test_data, threshold=0.001, verbose=False, validation_data = None):
    if validation_data == None:
        validation_data = test_data
    pnet = network.Network(net.sizes, return_vector=net.return_vector)
    pnet.weights = net.weights
    pnet.biases = net.biases
    """ training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)
    validation_data = list(validation_data)"""

    acc = pnet.evaluate(validation_data)
    d_acc = 0
    next_net = network.Network(net.sizes, return_vector=net.return_vector)
    print(next_net.sizes, pnet.sizes)
    next_net.weights = pnet.weights
    next_net.biases = pnet.biases

    print(acc, "/", len(validation_data) , " without prune")
    
    while(d_acc < threshold):
        pnet.weights = next_net.weights
        pnet.biases = next_net.biases
       

        """ weights10 = pnet.weights[0].copy()
        weights10[(weights10 > -region) & (weights10 <= region)] = 0.0
        weights11 = pnet.weights[1].copy()
        weights11[(weights11 > -region) & (weights11 <= region)] = 0.0
    
        fin = [weights10, weights11]
        if len(pnet.weights) > 2:
            weights12 = pnet.weights[2].copy()
            weights12[(weights12 > -region) & (weights12 <= region)] = 0.0
            fin.append(weights12)"""
        
        fin = []
        for i in range(len(pnet.weights)):
            temp_weights = pnet.weights[i]
            temp_weights[(temp_weights >- region) & (temp_weights <= region)] = 0.0
            fin.append(temp_weights)
            #print("pruned " ,len(temp_weights[temp_weights == 0.0]))
        
        next_net.weights = fin
        
        
        new_acc = next_net.evaluate(validation_data)
        d_acc = (acc - new_acc)/len(validation_data)
        sparsity = get_sparsity(next_net)
        if sparsity[0] == sparsity[1]: 
            print("warning you've reached 100% sparsity")
            break
        print(new_acc, "/", len(validation_data) , " around region ±", region, ', sparsity = ', 100 * sparsity[0]/sparsity[1] , '%')
        acc = new_acc
        region+=region
        
    
    
    pnet.SGD(training_data, 2, 10, 3.0)

    print("Epoch with pruned weights around {}: {} / {}".format(region/2, pnet.evaluate(test_data),len(test_data)));       
    r, total = get_sparsity(pnet)
    print("sparsity: " , r, " / " , total, " = "  , (r/total) * 100 , "%" )
    return pnet
    

def get_sparsity(netp):
    sparsity = 0
    total_weights = 0
    for i in range(len(netp.weights)):
        layer = netp.weights[i]
        sparsity += len(layer[layer == 0.0])
        total_weights += len(layer.flatten())

    return sparsity, total_weights
