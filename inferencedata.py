import numpy as np
import matplotlib.pyplot as plt

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
x = np.array([[0.        ],
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

if __name__=="__main__":
    plt.plot(x,np.log(atrue))
    plt.title('ktrue')
    plt.show()
