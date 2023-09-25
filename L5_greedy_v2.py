#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 10:37:02 2023

@author: andreafranceschini
"""


# OPTIMIZATION AND GREEDY APPROACHES

# Travelling Salesman Problem - v0

import numpy as np
import matplotlib.pyplot as plt

class Cities(): 
    def __init__(self, n, seed=None): 
        # check if n is an int greater than 4
        if not (isinstance(n, int) and n >= 4): 
            raise Exception('n needs to be an int larger or equal than 4')
            
        if seed is not None: 
            np.random.seed(seed)
            
        # store n for later use
        self.n = n 
                        
        # create the coordinates of the cities
        x = np.random.rand(n)
        y = np.random.rand(n)
        
        # stores them in the object
        self.x, self.y = x, y
        
    # instead of having it as a function make it a method of the Cities object
    def dist(self, city1, city2): 
        """Computes the distance between two cities
        """
        x,y = self.x, self.y
        
        d = np.sqrt((x[city2] - x[city1])**2 + (y[city2] - y[city1])**2)
        
        return d

        
def init_config(cities: Cities): 
    """Create a first random hamiltonian path. Returns the indices of the cities. 
    """
    
    # gets the number of nodes (the number of cities)
    n = cities.n
    
    # create a random permutation of indices of the cities
    # np.random.permutation gives a permutation of numbers up to n 
    route = np.random.permutation(n)
    
    return route

def display(cities, route, color): 
    """Displays a route and the dots of the cities using matplotlib
    Uses .pause() method to show how the plot evolves for each iteration
    """
    
    # clear the plot always
    plt.clf()
    
    x,y = cities.x, cities.y 
    
    # plot the routes
    plt.plot(x[route], y[route], '-', color=color)
    
    # plot the last edge
    comeback = [route[-1], route[0]]
    plt.plot(x[comeback], y[comeback], color=color)
    
    # plot the nodes (after to put them on the top)
    plt.plot(x,y, 'o', color='black')
    
    plt.show()
    # plt.pause because otherwise it renders all the plots at the end. 
    # this way we can see the route evolve
    plt.pause(0.01)
    
def cost(route, cities): 
    # calc distances and sum them    
    n = cities.n
    d = 0.0
    
    for i in range(n):         
        city1 = route[i] 
        city2 = route[(i + 1) % n] 
                    
        d += cities.dist(city1, city2)
    
    return d
        

def propose_move(route, cities): 
    # we are going to implement the switch move
    n = cities.n
    
    # select the two edges randomly
    while True:
        e1 = np.random.randint(n)
        e2 = np.random.randint(n)
        
        if e1 > e2: 
            e1, e2 = e2, e1
            
        # we also want to avoid to choose the first and last index, otherwise it's just going to invert the route
        
        # we want to avoid e1 = e2 and that the two edges are already aadjacent (otherwise nothing will change)
        if e2 > e1 + 1 and not (e1 == 0 and e2 == n-1): 
            break        
    
    # print(f'e1 = {e1}, e2 = {e2}')
    # we create a copy of route to not affect the original arr
    new_route = route.copy()
    # we revert the order of the indices in that segment of the configuration
    new_route[e1+1:e2+1] = new_route[e2:e1:-1] # reason about the indices choice
    
    return new_route

def compute_delta_cost(cities, old_route, new_route): 
    """We compute the delta cost between the two routes without having to recalculate the whole cost
    """
    
    old_c = cost(old_route, cities)
    new_c = cost(new_route, cities)
    
    return new_c - old_c
    
    
def greedy(cities, num_iters=100, seed=None): 
    if seed is not None: 
        np.random.seed(seed)
        
    x = init_config(cities)
    cx = cost(x, cities)

    display(cities, x, 'orange')
    print(f'initial cost is {cx:.5f}, starting route is {x}')
    
    for t in range(num_iters):
        y = propose_move(x, cities)
        # cy = cost(y, cities)
        
        # delta_c = cy - cx
        delta_c = compute_delta_cost(cities, x, y)
        
        if delta_c <= 0: 
            # accepted!
            x = y
            # cx = cy
            cx += delta_c

            # print the new cost and display the route on the plot
            print(f'\tmove accepted, new cost = {cx}')
            display(cities, x, 'blue')
            
    # stopping criteria -> max number of iterations reached 
    print(f'final cost is {cost(x, cities)}')
    return x


# why is it so slow? we compute the total cost every time, we should just compute the delta

