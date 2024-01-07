"""
run as:
mpirun -n 2 python3 mcmc.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from numpy.polynomial.legendre import leggauss
import pandas as pd
from pandas.plotting import scatter_matrix
import pickle
from mpi4py import MPI




def pw_linear_fem(k,f,N,a=1,b=1):
    """
    given log-permeability field k:[0,1]\to\R
    source term f:[0,1]\to\R
    return FEM coefficients (c_j)_{j=1}^n of pw linear solution to
    -(exp(k(x))u'(x))'=f(x)
    u'(0)=a
    u(1)=b
    """
    # meshwidth
    h = 1./N

    # stiffness matrix is tridiagonal
    Al = np.zeros(N-1) # lower diagonal
    Ad = np.zeros(N) # diagonal
    Au = np.zeros(N-1) # upper diagonal
    # RHS
    F = np.zeros(N)

    # Composite simpson rule over [0,1]
    n = 10
    q0 = np.linspace(0,1,n+1)
    m = (q0[:-1]+q0[1:])/2. # midpoints
    q0 = np.concatenate((q0,m)) # quadrature points
    w0 = [1]+[2]*(n-1)+[1]+[4]*n
    w0 = np.array(w0)/(6.*n) # quadrature weights
    w = w0*h
    
    # assemble stiffness matrix and RHS by looping over elements
    m = 1./(h**2)
    for j in range(0,N-1):
        # stiffness matrix
        x = j/N
        q = x+h*q0
        I = np.exp(k(q)).dot(w)
        Ad[j] += I*m
        Ad[j+1] += I*m
        Al[j] += -I*m
        Au[j] += -I*m        
        
        # RHS
        F[j] += np.dot(f(q)*(1-x),w)
        F[j+1] += np.dot(f(q)*x,w)

    # contribution from last element
    x = (N-1)/N
    q = x+h*q0
    I = np.exp(k(q)).dot(w)
    Ad[N-1] += I/(h**2)
    F[N-1] += np.dot(f(q)*x,w)

    # boundary condition terms in RHS
    F[0] += -a
    F[N-1] += I*b/(h**2)

    # use sparse solver for efficiency
    A = diags([Al,Ad,Au],[-1,0,1],format='csr')
    u = spsolve(A,F)
    # boundary condition at x=1
    u = np.concatenate((u,[b]))
    
    return u

def fem(k,f,a=1,b=1):
    """
    given log-permeability field k:[0,1]\to\R
    source term f:[0,1]\to\R
    return FEM coefficients (c_j)_{j=1}^n of pw linear solution to
    -(exp(k(x))u'(x))'=f(x)
    u'(0)=a
    u(1)=b
    """
    N = len(k)-1
    
    # meshwidth
    h = 1./N

    # stiffness matrix is tridiagonal
    Al = np.zeros(N-1) # lower diagonal
    Ad = np.zeros(N) # diagonal
    Au = np.zeros(N-1) # upper diagonal
    # RHS
    F = np.zeros(N)

    # Composite simpson rule over [0,1]
    n = 5
    q0 = np.linspace(0,1,n+1)
    m = (q0[:-1]+q0[1:])/2. # midpoints
    q0 = np.concatenate((q0,m)) # quadrature points
    w0 = [1]+[2]*(n-1)+[1]+[4]*n
    w0 = np.array(w0)/(6.*n) # quadrature weights
    w = w0*h
    
    # assemble stiffness matrix and RHS by looping over elements
    m = 1./(h**2)
    for j in range(0,N-1):
        # stiffness matrix
        I = h*(np.exp(k[j])+np.exp(k[j+1]))/2. # trapezoidal rule
        Ad[j] += I*m
        Ad[j+1] += I*m
        Al[j] += -I*m
        Au[j] += -I*m        
        
        # RHS
        x = j/N        
        q = x+h*q0
        F[j] += np.dot(f(q)*(1-x),w)
        F[j+1] += np.dot(f(q)*x,w)

    # contribution from last element
    I = h*(np.exp(k[-2])+np.exp(k[-1]))/2.
    Ad[N-1] += I/(h**2)
    x = (N-1.)/N    
    q = x+h*q0    
    F[N-1] += np.dot(f(q)*x,w)

    # boundary condition terms in RHS    
    F[0] += -a
    F[N-1] += I*b/(h**2)

    # use sparse solver for efficiency
    A = diags([Al,Ad,Au],[-1,0,1],format='csr')
    u = spsolve(A,F)
    # boundary condition at x=1
    u = np.concatenate((u,[b]))
    
    return u

def nystrom(N):
    """
    compute eigenvalues and eigenmodes of kernel at (i/N) for i=0,...,N

    input:

    output:
    w
    V
    """
    x = np.linspace(0,1,N+1)
    oneN = np.ones(N+1)
    C = np.outer(x,oneN)-np.outer(oneN,x)
    C = 0.3**2*np.exp(-np.abs(C)/0.3)
    w,V = np.linalg.eig(C/N)
    return w,V*np.sqrt(N+1)

def evaluate_u(Z,w,V,f):
    """
    input:
    Z       d-dim array
    w
    V
    f       function, RHS of the equation

    return (u(i/10))_{i=0}^9
    """
    assert(len(Z)==len(w))
    # log-permeability
    k = 1+np.dot(V,np.sqrt(w)*Z)
    # ode solution
    u = fem(k,f,a=1,b=1)
    # extract ode solution at points i/10
    N = V.shape[0]-1
    idx = np.round(N*np.arange(10)/10).astype(int)
    # idx = int(N*np.arange(10,dtype=int)/10)
    return u[idx]

def unnormalized_log_posterior(Z,w,V,f,obs,ssq=1e-4):
    p = -Z.dot(Z)/2
    u = evaluate_u(Z,w,V,f)
    n = np.dot(u-obs,u-obs)
    p += -n/(2*ssq)
    return p,u[0]

def unnormalized_log_posterior_hyper(Z,s,w,V,f,obs,a=2,b=1e-4):
    """
    use Gamma(a,b) as a prior for noise variance
    """
    if s > 0:
        p = -Z.dot(Z)/2
        p += -(a+1+d/2.)*np.log(s)-b/s
        u = evaluate_u(Z,w,V,f)
        n = np.dot(u-obs,u-obs)
        p += -n/(2*s)
    else:
        p = 0
        u = np.array([0])
    return p,u[0]

def mcmc(log_density,d,m,s=1):
    """
    f is the log density
    """
    # matrix to store mcmc samples
    samples = np.zeros((m,d))
    # vector to store values of solution at 0
    u = np.zeros(m)
    
    current = np.random.randn(d)
    logd_current,u_current = log_density(current)
    n = 0
    for j in range(m):
        print(j)
        proposal = current+np.random.randn(d)*s
        logd_proposal,u_proposal = log_density(proposal)
        a = logd_proposal-logd_current
        r = np.random.rand(1)
        if r < np.exp(a):
            samples[j,:] = proposal
            u[j] = u_proposal
            current = proposal
            logd_current = logd_proposal
            u_current = u_proposal
            n += 1
        else:
            samples[j,:] = current
            u[j] = u_current
    print("acception rate:",n/m)
    return samples,u

def mcmc_hyper(log_density,d,m,s1=1,s2=1e-4):
    """
    f is the log density
    """
    # matrix to store mcmc samples
    samples = np.zeros((m,d+1))
    # vector to store values of solution at 0
    u = np.zeros(m)
    
    current_Z = np.random.randn(d)
    current_s = 1e-4
    logd_current,u_current = log_density(current_Z,current_s)
    n = 0
    for j in range(m):
        print(j)
        proposal_s = current_s+np.random.randn(1)*s2
        if proposal_s > 0:
            proposal_Z = current_Z+np.random.randn(d)*s1
            logd_proposal,u_proposal = log_density(proposal_Z,proposal_s)
            a = logd_proposal-logd_current
            r = np.random.rand(1)
            if r < np.exp(a):
                samples[j,:] = np.append(proposal_Z,proposal_s)
                u[j] = u_proposal
                current_Z = proposal_Z
                current_s = proposal_s
                logd_current = logd_proposal
                u_current = u_proposal
                n += 1
            else:
                samples[j,:] = np.append(current_Z,current_s)
                u[j] = u_current
        else:
            samples[j,:] = np.append(current_Z,current_s)
            u[j] = u_current
    print("acception rate:",n/m)
    return samples,u

if __name__=="__main__":
    # nr of elements on [0,1] (spatial discretization)
    N = 1000
    # nr of mcmc samples
    m = 100000
    # nr of parameters (truncation dimension of KL expansion)
    d = 14
    
    # observation data
    obs = np.array([1.33324463,1.36464171,1.40744323,1.40555567,1.38851428,1.39799451,1.31587775,1.23031611,1.15017067,1.06664471])

    # compute eigenvalues and eigenmodes of KL expansion
    w,V = nystrom(N)
    # truncate KL expansion after d terms
    w = w[:d]
    V = V[:,:d]

    # RHS of equation
    theta = 0.8
    delta = 0.05
    f = lambda t: (theta/(delta*np.sqrt(2*np.pi)))*np.exp(-((t-0.2)**2)/(2*delta**2))\
    +(theta/(delta*np.sqrt(2*np.pi)))*np.exp(-((t-0.4)**2)/(2*delta**2))\
    +(theta/(delta*np.sqrt(2*np.pi)))*np.exp(-((t-0.6)**2)/(2*delta**2))\
    +(theta/(delta*np.sqrt(2*np.pi)))*np.exp(-((t-0.8)**2)/(2*delta**2))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # ---------------------
        # known noise covariance
        # ---------------------
        ssq=1e-4

        # log posterior
        log_post = lambda X: unnormalized_log_posterior(X,w,V,f,obs,ssq=ssq)

        # vanilla mcmc
        samples,u = mcmc(log_post,d,m,s=0.18)
        data = {"samples":samples,"KL_eigenvalues":w,"KL_eigenmodes":V,"u_values":u}
        pickle.dump(data, open("mcmc.p","wb"))

    if rank == 1:
        # ---------------------
        # unknown noise covariance
        # ---------------------
        # prior for 1/\sigma^2: Gamma(a,b)
        a = 2
        b = 1e-2

        # log posterior
        log_post = lambda Z,s: unnormalized_log_posterior_hyper(Z,s,w,V,f,obs,a=a,b=b)

        # vanilla mcmc
        samples,u = mcmc_hyper(log_post,d,m,s1=0.18,s2=1e-4)
        data = {"samples":samples,"KL_eigenvalues":w,"KL_eigenmodes":V,"u_values":u}
        pickle.dump(data, open("mcmc_hyper.p","wb"))
