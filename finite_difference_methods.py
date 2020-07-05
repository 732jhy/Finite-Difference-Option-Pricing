# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 00:07:27 2020

@author: Justin Yu, M.S. Financial Engineering, Stevens Institute of Technology

Finite difference methods for pricing European call and put options
"""

import numpy as np

def explicit_fin_diff(S,K,T,sigma,r,q,N,Nj,CallPut):
    '''
    Explicit finite difference method for pricing European call and put options
    
    Args:
        S - intial price of underlying asset
        K - strike price
        T - time to maturity
        sigma - volatility
        r - risk-free rate
        q - dividend rate
        N - number of spacing points along the time partition (horizontal)
        Nj - number of partition points from 0 to the upper/lower boundary (vertical)
        CallPut - 'Call' or 'Put'
        
    Returns the option price estimated by the finite difference grid
    
    On the choice of N and Nj:
        The following condition must be met to guarantee convergence of the explicit finite difference method:
            dx >= sigma*sqrt(3*dt)
        
        In fact, the best choice for dx is:
            dx = sigma*sqrt(3*dt)
        
        Therefore, finding the number N of time intervals is quite simple. For a given error epsilon, we set:
            (sigma^2)*3*dt + dt = epsilon       or      dt = epsilon/(1 + 3*sigma^2)
            
        Since dt = T/N, we can easily solve for N.
    '''
    dt = T/N;   dx = sigma*np.sqrt(3*dt)
    nu = r - q - 0.5*sigma**2
    pu = 0.5*dt*((sigma/dx)**2 + nu/dx);    pm = 1.0 - dt*(sigma/dx)**2 - r*dt;     pd = 0.5*dt*((sigma/dx)**2 - nu/dx)
    grid = np.zeros((N+1,2*Nj+1))
    
    # Asset prices at maturity:
    St = [S*np.exp(-Nj*dx)]
    for j in range(1, 2*Nj+1):
        St.append(St[j-1]*np.exp(dx))
    
    # Option value at maturity:
    for j in range(2*Nj+1):
        if CallPut == 'Call':
            grid[N,j] = max(0, St[j] - K)
        elif CallPut == 'Put':
            grid[N,j] = max(0, K - St[j])
    
    # Backwards computing through grid:
    for i in range(N-1, -1, -1):
        for j in range(1, 2*Nj):
            grid[i,j] =pu*grid[i+1,j+1] + pm*grid[i+1,j] + pd*grid[i+1,j-1]
    
        # Boundary conditions 
        grid[i,0] = grid[i,1]
        grid[i,2*Nj] = grid[i,2*Nj-1] + (St[2*Nj]-St[2*Nj-1])
    
    return grid[0,Nj]


def implicit_fin_diff(S,K,T,sigma,r,q,N,Nj,CallPut):
    '''
    Implicit finite difference method for pricing European call and put options
        
    Args:
        S - intial price of underlying asset
        K - strike price
        T - time to maturity
        sigma - volatility
        r - risk-free rate
        q - dividend rate
        N - number of time intervals (horizontal)
        Nj - number of partition points from 0 to the upper/lower boundary (vertical)
        CallPut - 'Call' or 'Put'
        
    Returns the option price estimated by the finite difference grid
    
    On the choice of N and Nj:
        Similar to the case of the explicit method, we again choose dt and dx such that:
            (dx)^2 + dt = epsilon
            
        We find that setting each term to 0.5*epsilon and solving for N and Nj provides pretty good results.
    '''
    dt = T/N;    #dx = sigma*np.sqrt(3*dt)
    dx = 1.0/(2*Nj+1)
    nu = r - q - 0.5*sigma**2
    pu = -0.5*dt*((sigma/dx)**2 + nu/dx);   pm = 1.0 + dt*(sigma/dx)**2 + r*dt;    pd = -0.5*dt*((sigma/dx)**2 - nu/dx)
    grid = np.zeros(2*Nj+1)
    
    # Asset prices at maturity:
    St = [S*np.exp(-Nj*dx)]
    for j in range(1, 2*Nj+1):
        St.append(St[j-1]*np.exp(dx))
    
    # Option value at maturity:
    for j in range(2*Nj+1):
        if CallPut == 'Call':
            grid[j] = max(0, St[j] - K)
        elif CallPut == 'Put':
            grid[j] = max(0, K - St[j])
    
    # Boundary Conditions:
    if CallPut == 'Call':
        lambdaU = St[2*Nj] - St[2*Nj-1];    lambdaL = 0.0;
    elif CallPut == 'Put':
        lambdaU = 0.0;  lambdaL = -1.0*(St[1] - St[0])
    
    # Backwards computing through grid
    def tridiagonal(C,pU,pM,pD,lambda_L,lambda_U,nj):
        '''
        Helper function for solving the tridiagonal matrix system specified by the 
        implicit finite difference method
        '''
        C1 = np.zeros(2*nj+1)     
        pmp = [pM+pD]
        pp = [C[1]+pD*lambda_L]
        for j in range(2,2*nj):
            pmp.append(pM - pU*pD/pmp[j-2])
            pp.append(C[j] - pp[j-2]*pD/pmp[j-2])
        C1[2*nj] = (pp[len(pp)-1] + pmp[len(pmp)-1]*lambda_U)/(pU + pmp[len(pmp)-1])
        C1[2*nj-1] = C1[2*nj] - lambda_U
        for j in range(2*nj-2, -1, -1):
            C1[j] = (pp[j-1] - pU*C1[j+1])/pmp[j-1]
        C1[0] = C1[1] - lambda_L
        return C1
    
    for i in range(N):  
        grid = tridiagonal(grid,pu,pm,pd,lambdaL,lambdaU,Nj)
    
    return grid[Nj]


def crank_nicolson(S,K,T,sigma,r,q,N,Nj,CallPut):
    '''
    Crank-Nicolson finite difference method for pricing European calls and puts
    
    Args:
        S - intial price of underlying asset
        K - strike price
        T - time to maturity
        sigma - volatility
        r - risk-free rate
        q - dividend rate
        N - number of spacing points along the time partition (horizontal)
        Nj - number of partition points from 0 to the upper/lower boundary (vertical)
        CallPut - 'Call' or 'Put'
        
    Returns the option price estimated by the finite difference grid
    
    On the choice of N and Nj:
        dt and dx should be chosen specifically such that the following condition holds:
            (dx)^2 + (0.5*dt)^2 = epsilon
            
        Again, we find that setting each term to 0.5*epsilon and solving for N and Nj gives pretty good results
    '''
    dt = T/N;   
    dx= 1.0/(2*Nj+1)
    nu = r - q - 0.5*sigma**2
 
    pu = -0.25*dt*((sigma/dx)**2 + nu/dx)
    pm = 1.0 + 0.5*dt*((sigma/dx)**2) + 0.5*r*dt
    pd = -0.25*dt*((sigma/dx)**2 - nu/dx)    
    
    grid = np.zeros(2*Nj+1)
    
    # Asset prices at maturity:
    St = [S*np.exp(-Nj*dx)]
    for j in range(1, 2*Nj+1):
        St.append(St[j-1]*np.exp(dx))
        
    # Option value at maturity:
    for j in range(2*Nj+1):
        if CallPut == 'Call':
            grid[j] = max(0, St[j] - K)
        elif CallPut == 'Put':
            grid[j] = max(0, K - St[j])
    
    # Boundary Conditions:
    if CallPut == 'Call':
        lambdaU = St[2*Nj] - St[2*Nj-1]
        lambdaL = 0.0
    elif CallPut == 'Put':
        lambdaU = 0.0
        lambdaL = -1.0*(St[1] - St[0])
    
    # Backwards computing through grid:
    def tridiagonal(C,pU,pM,pD,lambda_L,lambda_U,nj):
        '''
        Helper function for solving the tridiagonal matrix system specified by the 
        Crank-Nicolson finite difference method
        '''
        C1 = np.zeros(2*nj+1)
        pmp = [pM+pD]
        pp = [-pU*C[2]-(pM-2)*C[1]-pD*C[0]+pD*lambda_L]
        
        for j in range(2,2*nj):
            pmp.append(pM - pU*pD/pmp[j-2])
            pp.append(-pU*C[j+1] - (pM-2)*C[j] - pD*C[j-1] - pp[j-2]*pD/pmp[j-2])

        # Boundary conditions:
        C1[2*nj] = (pp[len(pp)-1] + pmp[len(pmp)-1]*lambda_U)/(pU + pmp[len(pmp)-1])
        C1[2*nj-1] = C1[2*nj] - lambda_U
        
        # Back substitution
        for j in range(2*nj-2, 0, -1):
            C1[j] = (pp[j-1] - pU*C1[j+1])/pmp[j-1]
        C1[0] = C[0]
        return C1
    
    for i in range(N):
        grid = tridiagonal(grid,pu,pm,pd,lambdaL,lambdaU,Nj)
    
    return grid[Nj]




















