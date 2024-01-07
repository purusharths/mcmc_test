import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from numpy.polynomial.legendre import leggauss
import pandas as pd
from pandas.plotting import scatter_matrix
import pickle
import pylab as pl

# observed data y = u(t)+eps
y = np.array([[1.33324463],
       [1.36464171],
       [1.40744323],
       [1.40555567],
       [1.38851428],
       [1.39799451],
       [1.31587775],
       [1.23031611],
       [1.15017067],
       [1.06664471]])
t = np.array([[0. ],
       [0.1],
       [0.2],
       [0.3],
       [0.4],
       [0.5],
       [0.6],
       [0.7],
       [0.8],
       [0.9]])

# true value of a
atrue = np.array([[4.60851933],
       [4.86913698],
       [4.62032248],
       [5.00739558],
       [4.82792527],
       [5.07901564],
       [4.66138568],
       [4.17793583],
       [4.12539241],
       [4.44239649],
       [3.92333781],
       [4.54533171],
       [4.60912071],
       [4.5259571 ],
       [4.07893867],
       [3.82307833],
       [3.32917918],
       [3.01300182],
       [2.84411211],
       [2.8700658 ],
       [2.8988206 ],
       [3.00884548],
       [2.9126627 ],
       [2.81452943],
       [2.49228003],
       [2.71739596],
       [3.03644913],
       [2.57126956],
       [3.34661707],
       [2.94035042],
       [2.76450419],
       [2.62201507],
       [2.67758311],
       [3.12653884],
       [2.82006727],
       [2.92333828],
       [2.96095777],
       [3.02618524],
       [2.86938906],
       [2.70856279],
       [2.7873468 ],
       [3.10554301],
       [2.69887846],
       [2.65547557],
       [2.5444871 ],
       [2.28154079],
       [2.05414577],
       [2.15512303],
       [2.02513576],
       [1.871455  ],
       [1.62145223],
       [1.43214563],
       [1.54012049],
       [1.57796945],
       [1.60947965],
       [1.47991658],
       [1.43280284],
       [1.42363837],
       [1.27337928],
       [1.24573075],
       [1.17312198],
       [1.31585193],
       [1.44015136],
       [1.66857903],
       [1.50962174],
       [1.54635945],
       [1.57589885],
       [1.70268497],
       [1.48816589],
       [1.43441179],
       [1.30716586],
       [1.3044823 ],
       [1.3594924 ],
       [1.45074973],
       [1.31924084],
       [1.42582943],
       [1.44731166],
       [1.3159029 ],
       [1.54561862],
       [1.14077717],
       [1.16094735],
       [1.23679487],
       [1.29243735],
       [1.09482575],
       [1.01743298],
       [1.28626862],
       [1.22763471],
       [1.24686001],
       [1.16866417],
       [1.11627212],
       [1.1575002 ],
       [1.21062568],
       [1.09896521],
       [1.18421265],
       [1.18059405],
       [1.17768529],
       [1.22455198],
       [1.04759883],
       [1.11647043],
       [1.33203748],
       [1.81313704],
       [1.60459249],
       [1.83411925],
       [1.86578565],
       [1.85751417],
       [1.74414425],
       [1.73943692],
       [1.65779697],
       [1.8353933 ],
       [1.6650141 ],
       [1.77233392],
       [2.12184255],
       [2.08316077],
       [2.56829458],
       [2.89616947],
       [2.76305599],
       [2.83952733],
       [2.84136602],
       [3.30459311],
       [3.7174832 ],
       [3.39326459],
       [3.72997251],
       [3.99814054],
       [3.56505218],
       [2.97146782],
       [2.53762826],
       [2.60267064],
       [2.80376325],
       [2.8883067 ],
       [2.63016047],
       [2.96651061],
       [2.84645723],
       [2.31591604],
       [2.33897006],
       [2.54556737],
       [2.49416021],
       [2.58507404],
       [2.92341952],
       [3.03701111],
       [3.13978838],
       [2.68511724],
       [2.82033378],
       [3.00537275],
       [3.09752033],
       [3.37058892],
       [2.81517596],
       [2.62448796],
       [2.27282371],
       [2.12617721],
       [2.00544201],
       [1.95385076]])
x_atrue = np.array([[0.        ],
       [0.00666667],
       [0.01333333],
       [0.02      ],
       [0.02666667],
       [0.03333333],
       [0.04      ],
       [0.04666667],
       [0.05333333],
       [0.06      ],
       [0.06666667],
       [0.07333333],
       [0.08      ],
       [0.08666667],
       [0.09333333],
       [0.1       ],
       [0.10666667],
       [0.11333333],
       [0.12      ],
       [0.12666667],
       [0.13333333],
       [0.14      ],
       [0.14666667],
       [0.15333333],
       [0.16      ],
       [0.16666667],
       [0.17333333],
       [0.18      ],
       [0.18666667],
       [0.19333333],
       [0.2       ],
       [0.20666667],
       [0.21333333],
       [0.22      ],
       [0.22666667],
       [0.23333333],
       [0.24      ],
       [0.24666667],
       [0.25333333],
       [0.26      ],
       [0.26666667],
       [0.27333333],
       [0.28      ],
       [0.28666667],
       [0.29333333],
       [0.3       ],
       [0.30666667],
       [0.31333333],
       [0.32      ],
       [0.32666667],
       [0.33333333],
       [0.34      ],
       [0.34666667],
       [0.35333333],
       [0.36      ],
       [0.36666667],
       [0.37333333],
       [0.38      ],
       [0.38666667],
       [0.39333333],
       [0.4       ],
       [0.40666667],
       [0.41333333],
       [0.42      ],
       [0.42666667],
       [0.43333333],
       [0.44      ],
       [0.44666667],
       [0.45333333],
       [0.46      ],
       [0.46666667],
       [0.47333333],
       [0.48      ],
       [0.48666667],
       [0.49333333],
       [0.5       ],
       [0.50666667],
       [0.51333333],
       [0.52      ],
       [0.52666667],
       [0.53333333],
       [0.54      ],
       [0.54666667],
       [0.55333333],
       [0.56      ],
       [0.56666667],
       [0.57333333],
       [0.58      ],
       [0.58666667],
       [0.59333333],
       [0.6       ],
       [0.60666667],
       [0.61333333],
       [0.62      ],
       [0.62666667],
       [0.63333333],
       [0.64      ],
       [0.64666667],
       [0.65333333],
       [0.66      ],
       [0.66666667],
       [0.67333333],
       [0.68      ],
       [0.68666667],
       [0.69333333],
       [0.7       ],
       [0.70666667],
       [0.71333333],
       [0.72      ],
       [0.72666667],
       [0.73333333],
       [0.74      ],
       [0.74666667],
       [0.75333333],
       [0.76      ],
       [0.76666667],
       [0.77333333],
       [0.78      ],
       [0.78666667],
       [0.79333333],
       [0.8       ],
       [0.80666667],
       [0.81333333],
       [0.82      ],
       [0.82666667],
       [0.83333333],
       [0.84      ],
       [0.84666667],
       [0.85333333],
       [0.86      ],
       [0.86666667],
       [0.87333333],
       [0.88      ],
       [0.88666667],
       [0.89333333],
       [0.9       ],
       [0.90666667],
       [0.91333333],
       [0.92      ],
       [0.92666667],
       [0.93333333],
       [0.94      ],
       [0.94666667],
       [0.95333333],
       [0.96      ],
       [0.96666667],
       [0.97333333],
       [0.98      ],
       [0.98666667],
       [0.99333333],
       [1.        ]])

def KL(Z,w,V,n=1):
    """
    input:
    w    d array, eigenvalues of KL expansion
    V    (N+1)xd matrix, values of eigenmodes in KL expansion
    Z    if n==1: d array, KL coefficients in R^d where to evaluate KL exp.
         if n>1 : nxd matrix, n KL coefficients in R^d where to evaluate KL exp.

    return values of KL expansion at N+1 points in [0,1] for n different 
    KL coefficients
    if n==1: N+1 array
    if n>1 : (N+1)xn matrix
    """
    
    d = len(w)
    if n == 1:
        return 1+np.dot(V,Z*np.sqrt(w))
    else:
        tmp = np.outer(np.sqrt(w),np.ones(n))
        return 1+np.dot(V,Z.T*tmp)

def autocorrelation(x):
    n = len(x)
    var = x.var()
    x = x-x.mean()
    r = np.correlate(x,x,mode='full')[-n:]
    return r/(var*np.arange(n,0,-1))
    
if __name__=="__main__":
    # mpl.use('pgf')
    
    dmax = 6
    burnin = 0.1
    
    # read data
    
    # ---------------
    # known variance
    # ---------------    
    data = pickle.load(open("mcmc.p","rb"))
    # unpack data:
    # samples     mxd matrix, mcmc samples
    # m           nr of mcmc samples
    # d           dimension of parameter
    # N           nr of elements on [0,1] (spatial resolution)
    # w           d array, eigenvalues of KL expansion
    # V           (N+1)xd matrix, values of
    #             eigenmodes in KL expansion
    # u           m array, u(0) at mcmc samples
    samples = data['samples']
    w = data['KL_eigenvalues']
    V = data['KL_eigenmodes']
    u = data['u_values']
    m,d = samples.shape
    N = V.shape[0]-1
    samples = samples[int(m*burnin):,:]
    u = u[int(m*burnin):]
    m = samples.shape[0]

    # transform samples from KL coefficients to samples from k
    k_samples = KL(samples,w,V,n=m) # (N+1)xm matrix
    # compute mean and standard deviation
    k_mean = np.mean(k_samples,axis=1)
    k_std = np.sqrt(np.var(k_samples,axis=1))

    # read data
    # ---------------
    # unknown variance
    # ---------------    
    data_h = pickle.load(open("mcmc_hyper.p","rb"))
    # unpack data:
    # samples     mxd matrix, mcmc samples
    # m           nr of mcmc samples
    # d           dimension of parameter
    # N           nr of elements on [0,1] (spatial resolution)
    # w           d array, eigenvalues of KL expansion
    # V           (N+1)xd matrix, values of
    #             eigenmodes in KL expansion
    # u           m array, u(0) at mcmc samples
    samples_h = data_h['samples']
    m_h,d_h = samples_h.shape    
    u_h = data_h['u_values']
    samples_s = samples_h[int(m_h*burnin):,-1]
    samples_h = samples_h[int(m_h*burnin):,:-1]
    u_h = u_h[int(m*burnin):]

    # transform samples from KL coefficients to samples from k
    k_samples_h = KL(samples_h,w,V,n=m) # (N+1)xm matrix
    # compute mean and standard deviation
    k_mean_h = np.mean(k_samples_h,axis=1)
    k_std_h = np.sqrt(np.var(k_samples_h,axis=1))
    
    # -------------------
    # -------------------
    # Plots
    # -------------------
    # -------------------
    
    # -------------------
    # posterior mean of k
    # -------------------
    plt.figure(figsize=(8.0, 5.0))
    ktrue = np.log(atrue)    
    plt.plot(x_atrue,ktrue,color='tab:orange',Linewidth=2,label='ktrue')    
    x = np.linspace(0,1,N+1)    
    plt.plot(x,k_mean,color='tab:blue',Linewidth=2,label='mean')
    plt.fill_between(x,k_mean+k_std,k_mean-k_std,alpha=0.2,label='std dev')
    plt.legend()
    plt.title('posterior mean of $k$')
    plt.savefig('k_mean.pdf',bbox_inches='tight',dpi=300)
    plt.close()

    plt.figure(figsize=(8.0, 5.0))    
    plt.plot(x_atrue,ktrue,color='tab:orange',Linewidth=2,label='ktrue')    
    x = np.linspace(0,1,N+1)    
    plt.plot(x,k_mean_h,color='tab:blue',Linewidth=2,label='mean')
    plt.fill_between(x,k_mean_h+k_std_h,k_mean_h-k_std_h,alpha=0.2,label='std dev')
    plt.legend()
    plt.title('posterior mean of $k$ with unknown noise variance')
    plt.savefig('k_mean_h.pdf',bbox_inches='tight',dpi=300)
    plt.close()

    # -------------------    
    # marginal of posterior variance (posterior variance of \sigma_\eps^2)
    # -------------------
    plt.figure(figsize=(8.0, 5.0))    
    plt.hist(samples_s, bins=None, range=None, density=True)    
    plt.savefig('s_distribution.pdf',bbox_inches='tight',dpi=300)
    plt.close()
    
    # -------------------    
    # traces of chain
    # -------------------
    dp = min(d,dmax)    
    if dp > 3:
        n1 = int(np.sqrt(dp))
        n2 = int(np.ceil(dp/n1))
        fig, axes = plt.subplots(n1, n2,figsize=(8.0, 5.0))
        x =np.arange(1,m+1)
        for i in range(n1):
            for j in range(n2):
                k = i+j*n1
                if k<d:
                    trace = samples[:,i+j*n1]
                    axes[i,j].plot(x,trace,Linewidth=1)
    else:
        fig, axes = plt.subplots(d,figsize=(8.0, 5.0))
        x =np.arange(1,m+1)
        for i in range(d):
            trace = samples[:,i]
            axes[i].plot(x,trace,Linewidth=1)
    pl.suptitle('traces')
    plt.savefig('traces.pdf',bbox_inches='tight',dpi=300)
    plt.close()

    # -------------------    
    # marginals
    # -------------------
    plt.figure(figsize=(8.0, 5.0))    
    df = pd.DataFrame(samples[:,:min(d,dmax)],columns=['z'+str(i) for i in range(1,1+min(d,dmax))])
    pd.plotting.scatter_matrix(df,rasterized=True)
    pl.suptitle('marginals')
    plt.savefig('marginals.pdf',bbox_inches='tight',dpi=300)
    plt.close()

    # -------------------    
    # distribution of u[0]
    # -------------------
    plt.figure(figsize=(8.0, 5.0))    
    plt.hist(u, bins=None, range=None, density=True,ls='dashed',lw=3,label='known variance',fc=(0,0,1,0.5))
    plt.hist(u_h, bins=None, range=None, density=True,ls='dotted',lw=3,label='unknown variance',fc=(1,0,0,0.5))
    
    plt.title('posterior distribution of $u(0)$')
    plt.legend()    
    plt.savefig('u0_dist.pdf',bbox_inches='tight',dpi=300)
    plt.close()

    # -------------------    
    # autocorrelation
    # -------------------
    dp = min(d,dmax)
    if dp > 3:
        n1 = int(np.sqrt(dp))
        n2 = int(np.ceil(dp/n1))
        fig, axes = plt.subplots(n1, n2,figsize=(8.0, 5.0))
        x =np.arange(1,int(m/2)+1)
        for i in range(n1):
            for j in range(n2):
                k = i+j*n1
                if k<d:
                    r = autocorrelation(samples[:,i+j*n1])
                    axes[i,j].plot(x,r[:int(m/2)],Linewidth=1)
    else:
        fig, axes = plt.subplots(d,figsize=(8.0, 5.0))
        x =np.arange(1,int(m/2)+1)
        for i in range(d):
            r = autocorrelation(samples[:,i])
            axes[i].plot(x,r[:int(m/2)],Linewidth=1)
    pl.suptitle('autocorrelation')            
    plt.savefig('autocorrelation.pdf',bbox_inches='tight',dpi=300)
    plt.close()

    # -------------------    
    # prior and posterior covariance functions
    # -------------------
    plt.figure(figsize=(8.0, 5.0))    
    x = np.linspace(0,1,N+1)
    one = np.ones(N+1)
    A = np.outer(x,one)
    C = (0.3**2)*np.exp(-0.3*np.abs(A-A.T))
    plt.imshow(C[::-1,:],interpolation='bicubic',extent=[0,1,0,1])
    plt.colorbar()
    plt.title('Prior covariance')    
    plt.savefig('prior_covariance.pdf',bbox_inches='tight',dpi=300)
    plt.close()

    plt.figure(figsize=(8.0, 5.0))    
    C = np.cov(k_samples)
    plt.imshow(C[::-1,:],interpolation='bicubic',extent=[0,1,0,1])
    plt.colorbar()    
    plt.title('Posterior covariance')
    plt.savefig('posterior_covariance.pdf',bbox_inches='tight',dpi=300)
    plt.close()
