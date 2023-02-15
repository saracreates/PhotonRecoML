import numpy as np
import random
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fill_zeros(cluster, ex, ey):
    '''make clusters to random 5x5 shape with zeros and return 5x5 shape, coordinate system and edges (for visualizing)'''
    # csys = array mit [x, y] der unteren linken ecke
    csys = np.array([ex[0], ey[0]])
    exdiff = np.diff(ex)
    eydiff = np.diff(ey)

    valuex, countx = np.unique(np.around(exdiff, decimals=4), return_counts=True)
    valuey, county = np.unique(np.around(eydiff, decimals=4), return_counts=True)

    if(countx[0]!=len(exdiff) or county[0]!=len(eydiff)):
        print("Celles don't have same distance - This is bad", xdiff, ydiff)
    diff_x = np.mean(valuex)
    diff_y = np.mean(valuey)
    # IST NICHT GANZ RANDOM?!!!! 
    if(cluster.shape[0]>5 or cluster.shape[1]>5):
        print("Well, this is bad - clusters are bigger than 5 celles.")
        return np.zeros((5,5))
    else:
        while cluster.shape != (5,5): 
            if cluster.shape[0] <5: # horizontal
                # an random Position die null einfuegen
                if random.random() > 0.5:
                    ind = 0
                else:
                    ind = cluster.shape[0]
                    csys[1] = csys[1] - diff_y
                    ey = np.insert(ey, 0, ey[0]-diff_y)
                cluster = np.insert(cluster, ind, 0, axis=0)
            if cluster.shape[1] <5: # vertikal
                # an random Position die null einfuegen
                if random.random() > 0.5:
                    ind = 0
                    csys[0] = csys[0] - diff_x
                    ex = np.insert(ex, 0, ex[0]-diff_x)
                else:
                    ind = cluster.shape[1]
                cluster = np.insert(cluster, ind, 0, axis=1)
    return cluster, csys, ex, ey

def form_cluster(xMC, yMC, EMC):
    '''make 5x5 shape from phast data - return cluster, coordinate points and edges of histogram'''
    t1 = time.time()
    arr_cluster = np.zeros((len(xMC), 5, 5))
    edge_x = []
    edge_y = []
    c_sys = []
    for i in range(len(xMC)):
        x = xMC[i]
        y = yMC[i]
        E = EMC[i]
        val_x, count_x = np.unique(x, return_counts=True)
        val_y, count_y = np.unique(y, return_counts=True)
        histo, ex, ey = np.histogram2d(x, y, bins=[count_y.max(),count_x.max()], weights=E, density=False)
        cluster = np.flip(histo.T, axis=0) # make shape more intuitive
        cluster, csys, ex, ey = fill_zeros(cluster, ex, ey)
        edge_x.append(ex)
        edge_y.append(ey)
        arr_cluster[i] = cluster
        c_sys.append(csys)
    t2 = time.time()
    print("This took ", t2-t1, "s")
    return arr_cluster, np.array(c_sys), np.array(edge_x), np.array(edge_y)

def prep_trainingsdata(x_truth, y_truth, E_truth, coordin):
    # returns [x relative pos, y relative pos, E]
    coord = coordin.T
    return np.array([x_truth-coord[0], y_truth-coord[1], E_truth]).T

def gaus(x, mu, sigma, A):
    return (A/np.sqrt(2*np.pi * sigma**2))* np.exp(-(x-mu)**2 / (2*sigma**2))

def histo_output(arr_NN, arr_fit, arr_truth, name='', figsave=False, range_x = (-1,1), bins=300):
    '''takes output of neural network (either x,y OR E), the fit values (either x,y, OR E) and the true values of e,y or E. 
    Define with name which parameter (x,y,E) you gave as input'''
    
    plt.rcParams["figure.figsize"] = (10,6)
    plt.subplot(2,1,1)
    n_counts, bins, patches = plt.hist(arr_NN-arr_truth, bins=bins, range=range_x)
    plt.title(name + " difference NN - truth")
    # fit
    maxv = n_counts.max()
    ind_fit = np.where(n_counts > 0.3*maxv)
    liml = ind_fit[0].min()
    limu = ind_fit[0].max() + 1
    
    x_centers = 0.5*(bins[:-1]+bins[1:])
    popt1, pcov1 = curve_fit(gaus, x_centers[liml:limu], n_counts[liml:limu], p0=[0,1, 100], sigma=1/np.sqrt(n_counts)[liml:limu])
    perr1 = np.sqrt(np.diag(pcov1))
    x_fit = np.linspace(x_centers[liml:limu][0], x_centers[liml:limu][-1], 500)
    y_fit = gaus(x_fit, *popt1)
    plt.plot(x_fit, y_fit, 'r--', label="Gaussian fit with " + r"$\mu = ({:.3f} \pm {:.3f})$".format(popt1[0], perr1[0]) + ", " + r"$\sigma = ({:.3f} \pm {:.3f})$".format(popt1[1], perr1[1]))
    plt.legend()
    
    plt.subplot(2,1,2)
    n_counts, bins, patches = plt.hist(arr_fit-arr_truth, bins=bins, range=range_x)
    plt.title(name + " difference fit - truth")

    # fit
    maxv = n_counts.max()
    ind_fit = np.where(n_counts > 0.3*maxv)
    liml = ind_fit[0].min()
    limu = ind_fit[0].max() + 1
    
    x_centers = 0.5*(bins[:-1]+bins[1:])
    popt2, pcov2 = curve_fit(gaus, x_centers[liml:limu], n_counts[liml:limu], p0=[0,1, 100], sigma=1/np.sqrt(n_counts)[liml:limu])
    perr2 = np.sqrt(np.diag(pcov2))
    x_fit = np.linspace(x_centers[liml:limu][0], x_centers[liml:limu][-1], 500)
    y_fit = gaus(x_fit, *popt2)
    plt.plot(x_fit, y_fit, 'r--', label="Gaussian fit with " + r"$\mu = ({:.3f} \pm {:.3f})$".format(popt2[0], perr2[0]) + ", " + r"$\sigma = ({:.3f} \pm {:.3f})$".format(popt2[1], perr2[1]))
    plt.legend()
    
    plt.tight_layout()
    if figsave == True:
        plt.savefig(name+"_histo.pdf")
    plt.show()
    
    return popt1, perr1, popt2, perr2 
