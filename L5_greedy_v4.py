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
from copy import deepcopy

class TSP(): 
    def __init__(self, n, seed=None): 
        """Inits the TSP Problem
        
        * n: number of cities in the problem
        """
        
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
        
        self.route = np.zeros(n, dtype=int)
        self.init_config()
        
    # instead of having it as a function make it a method of the Cities object
    def dist(self, city1, city2): 
        """Computes the distance between two cities
        """
        x,y = self.x, self.y
        
        d = np.sqrt((x[city2] - x[city1])**2 + (y[city2] - y[city1])**2)
        
        return d
    
    def init_config(self): 
        """Create a first random hamiltonian path. Returns the indices of the cities. 
        """
        
        # gets the number of nodes (the number of cities)
        n = self.n
        
        # create a random permutation of indices of the cities
        # np.random.permutation gives a permutation of numbers up to n
        # assigning like this changes the pointer
        # self.route = np.random.permutation(n)
        
        # this reassigns the elements in the array instead 
        #   of having to find new space in memory for it
        self.route[:] = np.random.permutation(n)
        
        # we do this here and resassign so that when we repeat init_config 
        #   we use always the same sapce of memory
        # we call __init__ once but init_config many times, 
        #   so we are going to always target the same point in memory
                
    def display(self): 
        """Displays a route and the dots of the cities using matplotlib
        
        Uses .pause() method to show how the plot evolves for each iteration
        """
        
        # clear the plot always
        plt.clf()
        
        x,y,route = self.x, self.y, self.route
        
        # plot the routes
        plt.plot(x[route], y[route], '-', color='orange')
        
        # plot the last edge
        comeback = [route[-1], route[0]]
        plt.plot(x[comeback], y[comeback], color='orange')
        
        # plot the nodes (after to put them on the top)
        plt.plot(x,y, 'o', color='black')
        
        plt.show()
        # plt.pause because otherwise it renders all the plots at the end. 
        # this way we can see the route evolve
        plt.pause(0.01)
    
    def cost(self): 
        """Calculates all the distances and sums them to get to the final cost
        """
        
        d = 0.0
        route = self.route
        
        for i in range(self.n):         
            city1 = route[i] 
            city2 = route[(i + 1) % self.n] 
                        
            d += self.dist(city1, city2)
        
        return d
        
    # we split the propose_move in two functions 
    def propose_move(self): 
        n = self.n
        
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
            
        move = (e1, e2)
        return move
        
    def accept_move(self, move): 
        e1, e2 = move

        # we create a copy of route to not affect the original arr
        new_route = np.copy(self.route)
        # we revert the order of the indices in that segment of the configuration
        new_route[e1+1:e2+1] = new_route[e2:e1:-1] # reason about the indices choice
        
        self.route = new_route

    def compute_delta_cost(self, move): 
        """We compute the delta cost between the two routes without having to recalculate the whole cost
        """
        
        old_c = self.cost()
        new_route = self.accept_move(move)
        new_c = self.cost(new_route)
        
        return new_c - old_c
    
    def copy(self): 
        # we could optimize it more, i.e. the coordinates of the cities will not change so
        #   we just need to copy the reference to the route
        return deepcopy(self)
    
# we now generalize the greedy function
# need to define
# * .init_config()
# * .cost(x)
# * .display(x)
# * .compute_delta_cost(x, y)
# * .propose_move(x)

def greedy(probl, repeats=1, num_iters=100, seed=None): 
    """Greedy algorithm
    * probl: the problem to solve
    * repeates: how many times you repeat the algorithm
    * num_iters: how many iterations per run
    * seed: for consistent results
    """
    
    best_config = None
    best_cost = np.infty
    
    for i in range(repeats): 
        if seed is not None: 
            np.random.seed(seed)
        
        # store the configuration inside the object and not in the greedy function 
        probl.init_config()
        cx = probl.cost()
        
        probl.display()

        print(f'initial cost is {cx:.5f}, starting route is {probl.route}')
        
        for t in range(num_iters):
            # we make propose move a method of the problem so that you can specify it in the problem object
            # y = probl.propose_move(x)
            move = probl.propose_move()
            
            # now we pass move so that we optimize the calc of the delta cost
            delta_c = probl.compute_delta_cost(move)
            
            if delta_c <= 0: 
                # accepted!
                probl.accept_move(move)
                # cx = cy
                cx += delta_c
    
                # print the new cost
                print(f'\tmove accepted, c = {cx}, t = {t}')
                
        # stopping criteria -> max number of iterations reached 
        print(f'final cost: {cx}')
        
        if cx < best_cost: 
            best_cost = cx
            # copying the object straight away is not correct though!
            # this assigns the reference to the object, and when later on the
            #   probl gets initialized again, the best_probl will point to the new 
            #   problem
            # but by just doing copy,there are arrays in the object that are references themselves, 
            #   so those references will still point to the arrays from the iniitla obj (which will chang ) 
            
            # hence there are different types of copy, shallow (depth = 1, leads to the problem described above)
            #   we are going to use deep copy which solves the problem we've described 
            best_probl = probl.copy() # use the method defined in the object
    
    best_probl.display(best_config)
    print(f'Best cost: {best_cost}')
    
    return best_cost, best_config