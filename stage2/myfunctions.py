import numpy as np
import random
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fill_zeros(cluster, ex, ey, cshape):
    '''make clusters to (not completely) random 5x5 shape with zeros and return 5x5 shape, coordinate system and edges (for visualizing)'''
    # csys = array mit [x, y] der unteren linken ecke, liegt genau in getroffener Zellmitte!
    csys = np.array([ex[0], ey[0]])
    # cover case of only one cell in one dimension... np.histogram2D used +/- 0.5 for the bin then
    if len(ex) == 2:
        csys[0] = ex[0]+0.5
    if len(ey) == 2:
        csys[1] == ey[0] + 0.5
    
    cellsize = 3.83
    ret = True
    if(cluster.shape[0]>cshape[0] or cluster.shape[1]>cshape[1]):
        #print("Clusters are bigger than 5 celles: ", cluster.shape[0], cluster.shape[1])
        ret = False
    else:
        while cluster.shape != cshape: 
            if cluster.shape[0] <cshape[0]: # horizontal
                # an random Position die null einfuegen
                if random.random() > 0.5:
                    ind = 0
                else:
                    ind = cluster.shape[0]
                    csys[1] = csys[1] - cellsize
                cluster = np.insert(cluster, ind, 0, axis=0)
            if cluster.shape[1] < cshape[1]: # vertikal
                # an random Position die null einfuegen
                if random.random() > 0.5:
                    ind = 0
                    csys[0] = csys[0] - cellsize
                else:
                    ind = cluster.shape[1]
                cluster = np.insert(cluster, ind, 0, axis=1)
    return cluster, csys, ret

def fill_zeros_random(cluster, ex, ey, cshape):
    '''make clusters to random NxN shape with zeros'''
    # csys = array with [x, y] at lower left corner, exactly in the center of the ecal cell!
    csys = np.array([ex[0], ey[0]])
    # cover case if only one cell in one dimension... np.histogram2D used +/- 0.5 for the binning then
    if len(ex) == 2:
        csys[0] = ex[0]+0.5
    if len(ey) == 2:
        csys[1] == ey[0] + 0.5
    
    cellsize = 3.83
    # cut clusters that are bigger than cshape! -> use bool return variable
    ret = True
    if(cluster.shape[0]>cshape[0] or cluster.shape[1]>cshape[1]):
        #print("Clusters are bigger than 5 celles: ", cluster.shape[0], cluster.shape[1])
        ret = False
        cluster_new = cluster
    else:
        dely = cshape[0] - cluster.shape[0]
        delx = cshape[1] - cluster.shape[1]
        # randomly choose how much upper left corner of cluster should be moved in NxN zero cluster 
        x = int(np.random.choice(np.arange(0, delx+1), 1))
        y = int(np.random.choice(np.arange(0, dely+1), 1))
        cluster_new = np.zeros(cshape[0]*cshape[1]).reshape(cshape) # create empty/zero cluster
        cluster_new[y:(cluster.shape[0]+y), x:(cluster.shape[1]+x)] = cluster # insert ecal cluster into the zero cluster
        # new coordinate system
        csys[0] = csys[0] - x*cellsize
        csys[1] = csys[1] - (cshape[0] - cluster.shape[0] - y)*cellsize # as one also needs to shift from upper left to lower left corner
       
    return cluster_new, csys, ret

def form_cluster(xMC, yMC, EMC, clustershape=(5,5)):
    '''make nxn shape from phast data - return cluster, coordinate points and edges of histogram'''
    t1 = time.time()
    arr_cluster = np.zeros((len(xMC), clustershape[0], clustershape[1])) # space holder of clusters 
    c_sys = np.zeros((len(xMC), 2)) # space holder for coordinate system
    ind_i = [] # add all indecies of not used clusters
    for i in range(len(xMC)):
        x = xMC[i]
        y = yMC[i]
        E = EMC[i]
        cellsize = 3.83
        binx = round((x.max() - x.min())/ cellsize) +1
        biny = round((y.max() - y.min()) / cellsize) +1
        histo, ex, ey = np.histogram2d(x, y, bins=[binx,biny], weights=E, density=False)
        cluster = np.flip(histo.T, axis=0) # make shape more intuitive - looks like scattered x,y now when array is printed!
        cluster, csys, ret = fill_zeros(cluster, ex, ey, clustershape) # bring into right clustershape
        if ret == True:
            arr_cluster[i] = cluster
            c_sys[i] = csys
        else:
            ind_i.append(i)
    arr_cluster = np.delete(arr_cluster, ind_i, axis=0) # delete unfilled zeros clusters
    c_sys = np.delete(c_sys, ind_i, axis=0) # delete same indices in coord system
    t2 = time.time()
    print("This took ", t2-t1, "s")
    return arr_cluster, c_sys, np.array(ind_i)

def prep_trainingsdata(x_truth, y_truth, E_truth, coordin):
    # returns [x relative pos, y relative pos, E]
    coord = coordin.T
    return np.array([x_truth-coord[0], y_truth-coord[1], E_truth]).T

def gaus(x, mu, sigma, A):
    return (A/np.sqrt(2*np.pi * sigma**2))* np.exp(-(x-mu)**2 / (2*sigma**2))

def histo_output(arr_NN, arr_fit, arr_truth, name='', figsave=False, range_x = (-1,1), bins=300, fit=True):
    '''takes output of neural network (either x,y OR E), the fit values (either x,y, OR E) and the true values of e,y or E. 
    Define with name which parameter (x,y,E) you gave as input'''
    
    plt.rcParams["figure.figsize"] = (10,6)
    plt.subplot(2,1,1)
    n_counts, bins, patches = plt.hist(arr_NN-arr_truth, bins=bins, range=range_x)
    plt.title(name + " difference NN - truth")
    # fit
    if fit==True:
        maxv = n_counts.max()
        ind_fit = np.where(n_counts > 0.3*maxv)
        liml = ind_fit[0].min()
        limu = ind_fit[0].max() + 1

        x_centers = 0.5*(bins[:-1]+bins[1:])
        popt1, pcov1 = curve_fit(gaus, x_centers[liml:limu], n_counts[liml:limu], p0=[0,1, 100], sigma=1/np.sqrt(n_counts)[liml:limu], maxfev=10000)
        perr1 = np.sqrt(np.diag(pcov1))
        x_fit = np.linspace(x_centers[liml:limu][0], x_centers[liml:limu][-1], 500)
        y_fit = gaus(x_fit, *popt1)
        plt.plot(x_fit, y_fit, 'r--', label="Gaussian fit with " + r"$\mu = ({:.3f} \pm {:.3f})$".format(popt1[0], perr1[0]) + ", " + r"$\sigma = ({:.3f} \pm {:.3f})$".format(popt1[1], perr1[1]))
        plt.legend()
    
    plt.subplot(2,1,2)
    n_counts, bins, patches = plt.hist(arr_fit-arr_truth, bins=bins, range=range_x)
    plt.title(name + " difference fit - truth")
    
    if fit==True:
        # fit
        maxv = n_counts.max()
        ind_fit = np.where(n_counts > 0.3*maxv)
        liml = ind_fit[0].min()
        limu = ind_fit[0].max() + 1

        x_centers = 0.5*(bins[:-1]+bins[1:])
        popt2, pcov2 = curve_fit(gaus, x_centers[liml:limu], n_counts[liml:limu], p0=[0,1, 100], sigma=1/np.sqrt(n_counts)[liml:limu], maxfev=10000)
        perr2 = np.sqrt(np.diag(pcov2))
        x_fit = np.linspace(x_centers[liml:limu][0], x_centers[liml:limu][-1], 500)
        y_fit = gaus(x_fit, *popt2)
        plt.plot(x_fit, y_fit, 'r--', label="Gaussian fit with " + r"$\mu = ({:.3f} \pm {:.3f})$".format(popt2[0], perr2[0]) + ", " + r"$\sigma = ({:.3f} \pm {:.3f})$".format(popt2[1], perr2[1]))
        plt.legend()
    
    plt.tight_layout()
    if figsave == True:
        plt.savefig(name+"_histo.pdf")
    plt.show()
    
    if fit==True:
        return popt1, perr1, popt2, perr2 

def training_and_validation_data(xMC, yMC, EMC, x_truth, y_truth, E_truth, x_fit, y_fit, E_fit, ind_del, clusters, coord, per=0.8):
    '''cut data into useful peaces'''
    
    # delete clusters that are bigger than 5 -> next time: do it in phast user event?
    xMC = np.delete(xMC, ind_del, axis=0)
    yMC = np.delete(yMC, ind_del, axis=0)
    EMC = np.delete(EMC, ind_del, axis=0)
    x_truth = np.delete(x_truth, ind_del)
    y_truth = np.delete(y_truth, ind_del)
    E_truth = np.delete(E_truth, ind_del)
    x_fit = np.delete(x_fit, ind_del)
    y_fit = np.delete(y_fit, ind_del)
    E_fit = np.delete(E_fit, ind_del)
    
    # devide into training and verification 
    xMC_train = xMC[:round(len(E_truth)*0.8)]
    xMC_veri = xMC[round(len(E_truth)*0.8):]

    yMC_train = yMC[:round(len(E_truth)*0.8)]
    yMC_veri = yMC[round(len(E_truth)*0.8):]

    EMC_train = EMC[:round(len(E_truth)*0.8)]
    EMC_veri = EMC[round(len(E_truth)*0.8):]

    x_truth_train = x_truth[:round(len(E_truth)*0.8)]
    x_truth_veri = x_truth[round(len(E_truth)*0.8):]

    y_truth_train = y_truth[:round(len(E_truth)*0.8)]
    y_truth_veri = y_truth[round(len(E_truth)*0.8):]

    E_truth_train = E_truth[:round(len(E_truth)*0.8)]
    E_truth_veri = E_truth[round(len(E_truth)*0.8):]

    x_fit_veri = x_fit[round(len(E_truth)*0.8):]
    y_fit_veri = y_fit[round(len(E_truth)*0.8):]
    E_fit_veri = E_fit[round(len(E_truth)*0.8):]

    clusters_t =  clusters[:round(len(E_truth)*0.8)]
    clusters_v = clusters[round(len(E_truth)*0.8):]

    coord_t = coord[:round(len(E_truth)*0.8)]
    coord_v = coord[round(len(E_truth)*0.8):]
    
    return xMC_train, xMC_veri, yMC_train, yMC_veri, EMC_train, EMC_veri, x_truth_train, x_truth_veri, y_truth_train, y_truth_veri, E_truth_train, E_truth_veri, x_fit_veri, y_fit_veri, E_fit_veri, clusters_t, clusters_v, coord_t, coord_v 


def training_vs_validation_loss(fit_hist, log=True):
    plt.plot(fit_hist.history['loss'])
    plt.plot(fit_hist.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(["Training Loss", "Validation Loss"])
    if log==True:
        plt.yscale('log')
    plt.show()