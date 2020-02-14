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
    pnet = net.copy()
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
    print("Epoch with pruned weights around {}: {} / {}".format(thresh, pnet.evaluate(validation_data),len(validation_data))) 
    
    return pnet

def prune_retrain(net, region, training_data, test_data, threshold=0.001, verbose=False, validation_data = None):
    if validation_data == None:
        validation_data = test_data
    pnet = net.copy()
    """ training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)
    validation_data = list(validation_data)"""


    acc = net.evaluate(test_data) / len(test_data) #gonna use the same test data each time?
    d_acc = 0
    next_net = network.Network(net.sizes)
    print(next_net.sizes, pnet.sizes)
    next_net.weights = pnet.weights
    next_net.biases = pnet.biases

    print(acc, " without prune")
    
    while(d_acc < threshold):
        pnet.weights = next_net.weights
        pnet.biases = next_net.biases
       

        weights10 = pnet.weights[0].copy()
        weights10[(weights10 > -region) & (weights10 <= region)] = 0.0
        weights11 = pnet.weights[1].copy()
        weights11[(weights11 > -region) & (weights11 <= region)] = 0.0
    
        fin = [weights10, weights11]
        if len(pnet.weights) > 2:
            weights12 = pnet.weights[2].copy()
            weights12[(weights12 > -region) & (weights12 <= region)] = 0.0
            fin.append(weights12)
    
        next_net.weights = fin
    
        new_acc = next_net.evaluate(test_data) / len(test_data)
        d_acc = acc - new_acc
        print(new_acc, " around region ±", region, ', sparity = ', 100 * get_sparsity(next_net)[0]/23820 , '%')
        acc = new_acc
        region+=region
        
    
    
    pnet.SGD(training_data, 2, 10, 3.0)
    ''' f = open("wsp1.txt", "a")
    g = open("wsp2.txt", "a")
        
    h = open("bs1p.txt", "a")
    i = open("bs2p.txt", "a")
       
    np.savetxt(f, net.weights[0])
    np.savetxt(g, net.weights[1])
    
    np.savetxt(h, net.biases[0])
    np.savetxt(i, net.biases[1])
    
    f.close()
    g.close()
    h.close()
    i.close() '''
        
    print("Epoch with pruned weights around {}: {} / {}".format(region/2, pnet.evaluate(test_data),len(test_data)));       
    r, total = get_sparsity(pnet)
    print("sparsity: " , r, " / " , total, " = "  , (r/total) * 100 , "%" )
    return pnet
    

def get_sparsity(net):
    sparsity = 0
    total_weights = 0
    for i in range(len(net.weights)):
        layer = net.weights[i]
        sparsity += len(layer[layer == 0.0])
        total_weights += len(layer.flatten())

        
    return sparsity, total_weights
