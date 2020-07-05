# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 00:47:07 2020

@author: Justin Yu, M.S. Financial Engineering, Stevens Institute of Technology

Uses the explicit finite difference pricing method to compute the Greeks
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


def exp_fin_diff_delta(S,K,T,sigma,r,q,N,Nj,CallPut):
    '''
    Uses the explicit finite difference method to approximate the option's delta
    
    Args:
        S - initial asset price          K - strike price
        T - time to maturity             sigma - volatility
        r - risk-free rate               q - dividend rate
        N - number of spacing points along the time partition (horizontal)
        Nj - number of partition points from 0 to the upper/lower boundary (vertical)
        CallPut - 'Call' or 'Put'
        
    Returns the estimated delta of the specified option
    '''    
    C1 = explicit_fin_diff(S, K, T, sigma, r, q, N, Nj, CallPut)
    A = S*0.01
    S1 = S+A
    C2 = explicit_fin_diff(S1, K, T, sigma, r, q, N, Nj, CallPut)
    delta = (C2-C1)/A
    return delta


def exp_fin_diff_gamma(S,K,T,sigma,r,q,N,Nj,CallPut):
    '''
    Uses the explicit finite difference method to approximate the option's gamma
    
    Args:
        S - initial asset price         K - strike price
        T - time to maturity            sigma - volatility
        r - risk-free rate              q - dividend rate
        N - number of spacing points along the time partition (horizontal)
        Nj - number of partition points from 0 to the upper/lower boundary (vertical)
        CallPut - 'Call' or 'Put'
        
    Returns the estimated gamma of the specified option
    ''' 
    C1 = explicit_fin_diff(S, K, T, sigma, r, q, N, Nj, CallPut)
    A = S*0.01
    S1 = S+A
    S2 = S-A
    C2 = explicit_fin_diff(S1, K, T, sigma, r, q, N, Nj, CallPut)
    C3 = explicit_fin_diff(S2, K, T, sigma, r, q, N, Nj, CallPut)
    gamma = (((C2-C1)/(S1-S)) - ((C1-C3)/(S-S2)))/(0.5*(S1-S2))
    return gamma


def exp_fin_diff_theta(S,K,T,sigma,r,q,N,Nj,CallPut):
    '''
    Uses the explicit finite difference method to approximate the option's theta
    
    Args:
        S - initial asset price         K - strike price
        T - time to maturity            sigma - volatility
        r - risk-free rate              q - dividend rate
        N - number of spacing points along the time partition (horizontal)
        Nj - number of partition points from 0 to the upper/lower boundary (vertical)
        CallPut - 'Call' or 'Put'
        
    Returns the estimated theta of the specified option
    ''' 
    C1 = explicit_fin_diff(S, K, T, sigma, r, q, N, Nj, CallPut)
    A = T*0.01
    T1 = T+A
    C2 = explicit_fin_diff(S, K, T1, sigma, r, q, N, Nj, CallPut)
    theta = (C2-C1)/A
    return theta


def exp_fin_diff_vega(S,K,T,sigma,r,q,N,Nj,CallPut):
    '''
    Uses the explicit finite difference method to approximate the option's vega
    
    Args:
        S - initial asset price         K - strike price
        T - time to maturity            sigma - volatility
        r - risk-free rate              q - dividend rate
        N - number of spacing points along the time partition (horizontal)
        Nj - number of partition points from 0 to the upper/lower boundary (vertical)
        CallPut - 'Call' or 'Put'
        
    Returns the estimated vega of the specified option
    ''' 
    C1 = explicit_fin_diff(S, K, T, sigma, r, q, N, Nj, CallPut)
    A = sigma*0.01
    sigma1 = sigma+A
    C2 = explicit_fin_diff(S, K, T, sigma1, r, q, N, Nj, CallPut)
    vega = (C2-C1)/A
    return vega


