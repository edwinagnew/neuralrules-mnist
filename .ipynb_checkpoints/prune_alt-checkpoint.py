#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 16:29:02 2019

@author: edwinagnew
"""

import network
import numpy as np

import mnist_loader





def prune_retrain_alt(net, start=0.3, threshold=0.001):
    pnet = net.copy()
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)


    region = start
    acc = net.evaluate(test_data) / len(test_data) #gonna use the same test data each time?
    d_acc = 0
    next_net = network.Network(net.sizes)
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
        print(new_acc, " around region ±", region)
        acc = new_acc
        region+=start
        
    
    
    pnet.SGD(training_data, 5, 10, 3.0)
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
        
    print("Epoch with pruned weights around {}: {} / {}".format(region - start, pnet.evaluate(test_data),len(test_data)));       
    weights10 = pnet.weights[0]
    weights11 = pnet.weights[1]
    r = len(weights10[weights10 == 0.0]) + len(weights11[weights11 == 0.0])
    if len(pnet.weights) > 2:
        weights12 = pnet.weights[2]
        r+= len(weights12[weights12 == 0.0])
    print("sparsity: " , r, " / 23820 = "  , (r/23820) * 100 , "%" )
    return pnet