# Helper File 
'''
Ok here is the sample of how to use this script, as there might occur error, if the right functions are not called in the right order:

import helperfile as hf
rootfile = uproot.open('./stage4_clusters.root')
ipd = hf.InputData_2photon(rootfile)
ipd.form_cluster()
ipd.train_test_split()
ipd.prep_trainingsdata()
ipd.prep_verificationdata()

keras.model.fit(ipd.clusters_t, ipd.training, ...)
keras.model.predict(ipd.clusters_v) 
'''

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import unumpy as unp
from matplotlib.colors import LogNorm


class InputData_1photon():
    # initialize with all raw data
    def __init__(self, rootfile, numevents=-1):
        # extract data from rootfile
        self.event = rootfile["user202302;1"]
        self.xMC = self.event["x_MC"].array(library="np")[:numevents]
        self.yMC = self.event["y_MC"].array(library="np") [:numevents]
        self.EMC = self.event["E_MC"].array(library="np") [:numevents]
        self.celltype = self.event["celltype"].array(library="np") [:numevents]
        self.x_truth = np.array(self.event["x_truth"].array()) [:numevents]
        self.y_truth = np.array(self.event["y_truth"].array()) [:numevents]
        self.E_truth = np.array(self.event["E_truth"].array()) [:numevents]
        self.momentum = np.array(self.event["momentum"].array()) [:numevents]
        self.x_fit = self.event["x_fit"].array(library="np") [:numevents] # fit refers to Lednev / coral fit 
        self.y_fit = self.event["y_fit"].array(library="np") [:numevents]
        self.E_fit = self.event["E_fit"].array(library="np") [:numevents]
        self.chi2_fit = self.event["chi2_fit"].array(library="np") [:numevents]
        self.ndf_fit = self.event["ndf_fit"].array(library="np") [:numevents]
        self.num_fit = self.event["num_fit"].array(library="np") [:numevents]
        # additional info
        self.cellsize = 3.83 #cm
        print('Initialized object')
    
    def delete_bad_events(self, ind_del):
        # delete clusters that are bigger than n x n
        self.xMC = np.delete(self.xMC, ind_del, axis=0)
        self.yMC = np.delete(self.yMC, ind_del, axis=0)
        self.EMC = np.delete(self.EMC, ind_del, axis=0)
        self.celltype = np.delete(self.celltype, ind_del, axis=0)
        self.x_truth = np.delete(self.x_truth, ind_del, axis=0)
        self.y_truth = np.delete(self.y_truth, ind_del, axis=0)
        self.E_truth = np.delete(self.E_truth, ind_del, axis=0)
        self.momentum = np.delete(self.momentum, ind_del, axis=0)
        self.x_fit = np.delete(self.x_fit, ind_del, axis=0)
        self.y_fit = np.delete(self.y_fit, ind_del, axis=0)
        self.E_fit = np.delete(self.E_fit, ind_del, axis=0)
        self.chi2_fit = np.delete(self.chi2_fit, ind_del, axis=0)
        self.ndf_fit = np.delete(self.ndf_fit, ind_del, axis=0)
        self.num_fit = np.delete(self.num_fit, ind_del)
        try: # might be not defined yet
            self.clusterNxN = np.delete(self.clusterNxN, ind_del, axis=0) # delete unfilled zeros clusters
            self.coordinates = np.delete(self.coordinates, ind_del, axis=0) # delete same indices in coord system
        except:
            pass
        
    def fill_zeros_random(self, cluster, ex, ey, cshape, random):
        '''place the cluster random in the n x n grid'''
        # csys = array with [x, y] at lower left corner, exactly in the center of the ecal cell!
        csys = np.array([ex[0], ey[0]])
        # cover case if only one cell in one dimension... np.histogram2D used +/- 0.5 for the binning then
        if len(ex) == 2:
            csys[0] = ex[0]+0.5
        if len(ey) == 2:
            csys[1] == ey[0] + 0.5

        # cut clusters that are bigger than cshape! -> use bool return variable
        ret = True
        if(cluster.shape[0]>cshape[0] or cluster.shape[1]>cshape[1]):
            #print("Clusters are bigger than 5 celles: ", cluster.shape[0], cluster.shape[1])
            ret = False
            cluster_new = cluster
        else:
            dely = cshape[0] - cluster.shape[0]
            delx = cshape[1] - cluster.shape[1]
            if random == True:
                # randomly choose how much upper left corner of cluster should be moved in NxN zero cluster 
                x = int(np.random.choice(np.arange(0, delx+1), 1))
                y = int(np.random.choice(np.arange(0, dely+1), 1))
            else:
                # place in the middle
                x_mid = np.mean(np.arange(0, delx+1))
                y_mid = np.mean(np.arange(0, dely+1))
                x = int(np.random.choice(np.array([np.floor(x_mid), np.ceil(x_mid)]), 1))
                y = int(np.random.choice(np.array([np.floor(y_mid), np.ceil(y_mid)]), 1))
            cluster_new = np.zeros(cshape[0]*cshape[1]).reshape(cshape) # create empty/zero cluster
            cluster_new[y:(cluster.shape[0]+y), x:(cluster.shape[1]+x)] = cluster # insert ecal cluster into the zero cluster
            # new coordinate system
            csys[0] = csys[0] - x*self.cellsize
            csys[1] = csys[1] - (cshape[0] - cluster.shape[0] - y)*self.cellsize # as one also needs to shift from upper left to lower left corner

        return cluster_new, csys, ret
        
    def form_cluster(self, clustershape=(5,5), random=True):
        '''make nxn shape from phast data, extract coordinate system and find useless events'''
        print('Start shaping clusters in a ', clustershape, ' grid...')
        t1 = time.time()
        arr_cluster = np.zeros((len(self.xMC), clustershape[0], clustershape[1])) # space holder of clusters 
        c_sys = np.zeros((len(self.xMC), 2)) # space holder for coordinate system
        ind_i = [] # add all indecies of not used clusters
        for i in range(len(self.xMC)):
            x = self.xMC[i]
            y = self.yMC[i]
            E = self.EMC[i]
            ct = self.celltype[i]
            binx = round((x.max() - x.min())/ self.cellsize) +1
            biny = round((y.max() - y.min()) / self.cellsize) +1
            histo, ex, ey = np.histogram2d(x, y, bins=[binx,biny], weights=E, density=False)
            cluster = np.flip(histo.T, axis=0) # make shape more intuitive - looks like scattered x,y now when array is printed!
            cluster, csys, ret = self.fill_zeros_random(cluster, ex, ey, clustershape, random) # bring into right clustershape
            if ret == True and len(np.unique(ct)) == 1 and np.all(self.E_truth[i]>=1): # nutze nur events, die groesser als n x n sind und im Shashilik bereich liegen und ueber dem threshold von 1 GeV liegen fuer jedes Photon.
                arr_cluster[i] = cluster
                c_sys[i] = csys
            else:
                ind_i.append(i)
                
        t2 = time.time()
        print("Shaping the clusters took ", t2-t1, "s")
        # define new attritubes of class:
        self.clusterNxN = arr_cluster
        self.coordinates = c_sys
        # cut all useless events
        self.delete_bad_events(np.array(ind_i))
        
    def train_test_split(self, percentage=0.8):
        
        # reshape clusters:
        self.clusters = self.clusterNxN.reshape((self.clusterNxN.shape[0], self.clusterNxN.shape[1]*self.clusterNxN.shape[2]))
        
        # devide into training and verification
        cut=round(len(self.E_truth)*percentage)
        
        self.xMC_train = self.xMC[:cut]
        self.xMC_veri = self.xMC[cut:]

        self.yMC_train = self.yMC[:cut]
        self.yMC_veri = self.yMC[cut:]

        self.EMC_train = self.EMC[:cut]
        self.EMC_veri = self.EMC[cut:]

        self.x_truth_train = self.x_truth[:cut]
        self.x_truth_veri = self.x_truth[cut:]

        self.y_truth_train = self.y_truth[:cut]
        self.y_truth_veri = self.y_truth[cut:]

        self.E_truth_train = self.E_truth[:cut]
        self.E_truth_veri = self.E_truth[cut:]

        self.x_fit_veri = self.x_fit[cut:]
        self.y_fit_veri = self.y_fit[cut:]
        self.E_fit_veri = self.E_fit[cut:]
        
        self.clusters_t =  self.clusters[:cut]
        self.clusters_v = self.clusters[cut:]

        self.coord_t = self.coordinates[:cut]
        self.coord_v = self.coordinates[cut:]
        
        self.momentum_t = self.momentum[:cut]
        self.momentum_v = self.momentum[cut:]
        
        self.numfit_v = self.num_fit[cut:]
        print('Splitted data into training and test set!')

        # prep trainings and evaluation data
        self.prep_trainingsdata()
        self.prep_verificationdata()
        
    def prep_trainingsdata(self):
         # returns [x relative pos, y relative pos, E] where relative means relative to lower left corner
        self.training = np.array([self.x_truth_train-self.coord_t.T[0], self.y_truth_train-self.coord_t.T[1], self.E_truth_train]).T
        print("Prepared 'training' data")
        
    def prep_verificationdata(self):
        # creates [x rel, y rel, E] with coordinate system with respect to lower left corner
        self.veri_truth = np.array([self.x_truth_veri-self.coord_v.T[0], self.y_truth_veri-self.coord_v.T[1], self.E_truth_veri]).T
        print("Prepared 'veri_truth' data")
        self.veri_fit = np.array([self.x_fit_veri-self.coord_v.T[0], self.y_fit_veri-self.coord_v.T[1], self.E_fit_veri]).T
        print("Prepared 'veri_fit' data (Lednev fit from coral)")
        
class InputData_2photon(InputData_1photon):

    # initialize with all raw data
    def __init__(self, rootfile, numevents=-1, sort_cond='E', min_dist=0):
        super().__init__(rootfile, numevents=numevents)
        self.prep_coralfit_data() # bring into uniform shape
        self.sort_rightsolution(sort_cond) # bring 2 photons in right order
        if min_dist>0:
            self.cut_min_distance(min_dist) #cut events that are too close together. Distance in cm.

    def form_cluster(self, clustershape=(9,9), random=True):
        super().form_cluster(clustershape=clustershape, random=random)

    def prep_trainingsdata(self):
        # returns [x1 rel pos, y1 rel pos, E1, x2 rel pos, y2 rel pos, E2] where relative means relative to lower left corner
        self.training = np.array([self.x_truth_train.T[0]-self.coord_t.T[0], self.y_truth_train.T[0]-self.coord_t.T[1], self.E_truth_train.T[0], \
                                self.x_truth_train.T[1]-self.coord_t.T[0], self.y_truth_train.T[1]-self.coord_t.T[1], self.E_truth_train.T[1]]).T

    def deal_with_coral_fit_format(self, arr, num_fit):
        
        nice_arr = np.ones((len(arr), 2))*(-1000)
        good_ind = np.where(num_fit==2)
        for i in good_ind[0]:
                nice_arr[i] = arr[i]
        return nice_arr

    def prep_coralfit_data(self):
        # modify coral fit data into an uniform shape so that I can use numpy arrays. Bad events (not correct number of fits) have the value -1000.
        self.x_fit = self.deal_with_coral_fit_format(self.x_fit, self.num_fit) 
        self.y_fit = self.deal_with_coral_fit_format(self.y_fit, self.num_fit) 
        self.E_fit = self.deal_with_coral_fit_format(self.E_fit, self.num_fit) 
        self.chi2_fit = self.deal_with_coral_fit_format(self.chi2_fit, self.num_fit)
        self.ndf_fit = self.deal_with_coral_fit_format(self.ndf_fit, self.num_fit)

    def roll_data_MSE(self, data):
        '''bring Lednev or NN data into right order. 'data' must have shape (n, 6)'''
        data_flipped = np.roll(data, 3, axis=1)
        mse = np.sum(np.square(data-self.veri_truth), axis=1) 
        mse_flipped = np.sum(np.square(data_flipped-self.veri_truth), axis=1)

        ind_flip = np.where(mse_flipped<mse)
        data[ind_flip] = np.roll(data[ind_flip], 3, axis=1)
        return data

    def prep_verificationdata(self):
        # creates [x1 rel, y1 rel, E1, x2 rel, y2 rel, E2] with coordinate system with respect to lower left corner
        self.veri_truth = np.array(\
        [self.x_truth_veri.T[0]-self.coord_v.T[0], self.y_truth_veri.T[0]-self.coord_v.T[1], self.E_truth_veri.T[0], \
        self.x_truth_veri.T[1]-self.coord_v.T[0], self.y_truth_veri.T[1]-self.coord_v.T[1], self.E_truth_veri.T[1]]).T
        print("Prepared 'veri_truth' data")

        # Lednev
        self.veri_fit = np.array(\
        [self.x_fit_veri.T[0]-self.coord_v.T[0], self.y_fit_veri.T[0]-self.coord_v.T[1], self.E_fit_veri.T[0], \
        self.x_fit_veri.T[1]-self.coord_v.T[0], self.y_fit_veri.T[1]-self.coord_v.T[1], self.E_fit_veri.T[1]]).T
        # make label right:
        self.veri_fit = self.roll_data_MSE(self.veri_fit)
        print("Prepared 'veri_fit' data (Lednev fit from coral)")

    def sort_rightsolution(self, type):
        '''sort shower through condition. The shower with the higher value is the frist, the one with lower value the second.'''
        # sort condition
        if type == 'E': # sort through energy
            order = np.argmax(self.E_truth, axis=1) # retruns index of higher value, so either 0 or 1
            order_fit = np.argmax(self.E_fit, axis=1)
        elif type == 'x': # sort through x position
            order = np.argmax(self.x_truth, axis=1) 
            order_fit = np.argmax(self.x_fit, axis=1) # MACHT DAS SINN??? 
        elif type == 'y': # sort through y position
            order = np.argmax(self.y_truth, axis=1)
            order_fit = np.argmax(self.y_fit, axis=1)
        elif type == 'none':
            order = np.ones(len(self.E_truth)) # will not change anything
            order_fit = np.ones(len(self.E_truth)) # will not change anything
        else:
            raise(ValueError('type must be either x, y, E or none'))
        # true values
        ind_flip = np.where(order==0)
        self.E_truth[ind_flip] = np.fliplr(self.E_truth[ind_flip])
        self.x_truth[ind_flip] = np.fliplr(self.x_truth[ind_flip])
        self.y_truth[ind_flip] = np.fliplr(self.y_truth[ind_flip])
        self.momentum[ind_flip] = np.roll(self.momentum[ind_flip], 3, axis=1) #change (0,1,2,3,4,5) -> (4,5,6,0,1,2)
        # fit values
        ind_flip_fit = np.where(order_fit==0)
        self.E_fit[ind_flip_fit] = np.fliplr(self.E_fit[ind_flip_fit])
        self.x_fit[ind_flip_fit] = np.fliplr(self.x_fit[ind_flip_fit])
        self.y_fit[ind_flip_fit] = np.fliplr(self.y_fit[ind_flip_fit])
        self.chi2_fit[ind_flip_fit] = np.fliplr(self.chi2_fit[ind_flip_fit])
        self.ndf_fit[ind_flip_fit] = np.fliplr(self.ndf_fit[ind_flip_fit])

    def cut_min_distance(self, dis):
        '''cuts all photons that are less than the given distance apart! NEEDS TO BE PERFORMED BEFORE DIVIDING INTO TRAINING AND TEST SET'''
        acc_dist = np.sqrt(self.x_truth.T[0]**2 + self.y_truth.T[0]**2) - np.sqrt(self.x_truth.T[1]**2 + self.y_truth.T[1]**2) # pos 1 - pos 2
        ind_cut = np.where(abs(acc_dist)<dis) # scheide alle events raus bei dene die photonen zu nah sind
        self.delete_bad_events(ind_cut)
        print("Cutted ", len(ind_cut[0]), " clusters due to photon pair with distance smaller than ", dis, " cm.") 

    def cut_onlycoraldata(self):
        ind_cut = np.where(self.x_fit==-1000)
        self.delete_bad_events(ind_cut)
        print("Cutted ", len(ind_cut[0]), " to only use data with correct Lednev identification.") 

class Evaluation_2photon:
    def __init__(self, ipd, output):
        '''first parameter is the dataset we are dealing with'''
        self.ipd = ipd
        # NN output
        self.x1 = output.T[0]
        self.y1 = output.T[1]
        self.E1 = output.T[2]
        self.x2 = output.T[3]
        self.y2 = output.T[4]
        self.E2 = output.T[5]
        # true values
        self.x1_t = ipd.veri_truth.T[0]
        self.y1_t = ipd.veri_truth.T[1]
        self.E1_t = ipd.veri_truth.T[2]
        self.x2_t = ipd.veri_truth.T[3]
        self.y2_t = ipd.veri_truth.T[4]
        self.E2_t = ipd.veri_truth.T[5]
        # coral values
        self.x1_c = ipd.veri_fit.T[0]
        self.y1_c = ipd.veri_fit.T[1]
        self.E1_c = ipd.veri_fit.T[2]
        self.x2_c = ipd.veri_fit.T[3]
        self.y2_c = ipd.veri_fit.T[4]
        self.E2_c = ipd.veri_fit.T[5]

# Data visualisation

    def show_cluster(self, ind, lednev=True):
        ''' show cluster and points of cluster "ind" (integer)'''
        xlim = ipd.coordinates[ind][0]-ipd.cellsize/2 #left 
        ylim = ipd.coordinates[ind][1]-ipd.cellsize/2 #bottom

        ax1 = plt.subplot(1,2, 1)
        cluster = ipd.clusterNxN[ind]
        plt.imshow(cluster, norm=LogNorm(), extent=[xlim, xlim+ipd.cellsize*9, ylim, ylim+ipd.cellsize*9]) # extent (left, right, bottom, top)
        im_ratio = cluster.shape[0]/cluster.shape[1]
        plt.colorbar(fraction=0.047*im_ratio)
        plt.xlabel("$x$ [cm]")
        plt.ylabel("$y$ [cm]")

        ax2 = plt.subplot(1,2,2)
        plt.scatter(ipd.xMC[ind], ipd.yMC[ind], label="center of ECAL cells")
        plt.scatter(ipd.x_truth[ind], ipd.y_truth[ind], label="MC coordinates")
        if lednev==True:
            plt.scatter(ipd.x_fit[ind], ipd.y_fit[ind], label="Lednev coordinates")
        plt.xlim(xlim, xlim+ipd.cellsize*9)
        plt.ylim(ylim, ylim+ipd.cellsize*9)
        plt.legend()
        plt.xlabel("$x$ [cm]")
        plt.ylabel("$y$ [cm]")
        ax2.set_aspect('equal', 'box')

        plt.tight_layout()
        plt.show()

    def show_cluster_NNpred(self, ind, lednev=True, lim=True):
        ''' show cluster and points of cluster "ind" (integer)'''
        plt.style.use('standard_style.mplstyle')
        plt.rcParams["figure.figsize"] = (10,5)
        xlim = self.ipd.coord_v[ind][0]-self.ipd.cellsize/2 #left 
        ylim = self.ipd.coord_v[ind][1]-self.ipd.cellsize/2 #bottom

        # energy deposition of cluster
        ax1 = plt.subplot(1,2, 1)
        cluster = self.ipd.clusters_v[ind].reshape(9,9)
        plt.imshow(cluster, norm=LogNorm(), extent=[xlim, xlim+self.ipd.cellsize*9, ylim, ylim+self.ipd.cellsize*9]) # extent (left, right, bottom, top)
        im_ratio = cluster.shape[0]/cluster.shape[1]
        plt.colorbar(fraction=0.047*im_ratio)
        plt.xlabel("$x$ [cm]")
        plt.ylabel("$y$ [cm]")

        # scatter plot and photon positions
        ax2 = plt.subplot(1,2,2)
        plt.scatter(self.ipd.xMC_veri[ind], self.ipd.yMC_veri[ind], label="center of ECAL cells")
        # idee: nicht scattern sondern graue Quadrate zeichnen?
        plt.scatter(self.ipd.x_truth_veri[ind], self.ipd.y_truth_veri[ind], label="MC coordinates")
        if lednev==True:
            plt.scatter(self.ipd.x_fit_veri[ind], self.ipd.y_fit_veri[ind], label="Lednev coordinates")
        plt.scatter(np.array([self.x1[ind], self.x2[ind]])+ self.ipd.coord_v[ind, 0], np.array([self.y1[ind], self.y2[ind]])+ self.ipd.coord_v[ind, 1], label="NN prediction")
        if lim == True:
            plt.xlim(xlim, xlim+self.ipd.cellsize*9)
            plt.ylim(ylim, ylim+self.ipd.cellsize*9)
        plt.legend()
        plt.xlabel("$x$ [cm]")
        plt.ylabel("$y$ [cm]")
        ax2.set_aspect('equal', 'box')

        plt.tight_layout()
        plt.show()

    def training_vs_validation_loss(self, fit_hist, log=False, save=False, title=""):
        plt.style.use('standard_style.mplstyle')
        plt.plot(fit_hist.history['loss'])
        plt.plot(fit_hist.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(["Training Loss", "Validation Loss"])
        if log==True:
                plt.yscale('log')
        if save==True:
            plt.savefig(title + "_loss.pdf")
        else:
            plt.show()

# 1D Histogramme mit Gaus Fit

    def show_hist_NN(self, figsize=(12,10), r_x = (-4, 4), r_y = (-4, 4), r_E = (-0.3, 0.3), fit_x=True, fit_y=True, fit_E=True, figsave=(False, 'network1')):
        '''
        shows plots of all relative differences of NN and Lednev/coral fit like
        (x1, x2)
        (y1, y2)
        (E1, E2)

        First entry in r_x refers to NN and second to Lednev 
        '''
        plt.style.use('standard_style.mplstyle')
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Performence of " + figsave[1])
        plt.rcParams['figure.figsize'] = figsize
        alpha = 0.55

        # x1 PLOT
        fig.add_subplot(3,2, 1)
        fit_x1 = self.make_subplot(self.x1, self.x1_t, r_x, alpha, 'x_1', fit_x)

        # x2 PLOT
        fig.add_subplot(3,2,2)
        fit_x2 = self.make_subplot(self.x2, self.x2_t, r_x, alpha, 'x_2', fit_x)

        # y1 PLOT
        fig.add_subplot(3,2,3)
        fit_y1 = self.make_subplot(self.y1, self.y1_t, r_y, alpha, 'y_1', fit_y)

        # y2 PLOT
        fig.add_subplot(3,2,4)
        fit_y2 = self.make_subplot(self.y2, self.y2_t, r_y, alpha, 'y_2', fit_y)

        # E1 PLOT
        fig.add_subplot(3,2,5)
        fit_E1 = self.make_subplot(self.E1, self.E1_t, r_E, alpha, 'E_1', fit_E, rel=True)

        # E2 PLOT
        fig.add_subplot(3,2,6)
        fit_E2 = self.make_subplot(self.E2, self.E2_t, r_E, alpha, 'E_2', fit_E, rel=True)
        
        plt.tight_layout()
        if figsave[0] == True:
            plt.savefig(figsave[1] +"histo.pdf")
        else:
            plt.show()
        # returns (mu_x1, sigma_x1, mux2, sigma_x2, mu_y1, sigma_y1, mu_y2 , ....) with uncertainties 
        return unp.uarray(np.concatenate((fit_x1[0], fit_x2[0], fit_y1[0], fit_y2[0], fit_E1[0], fit_E2[0])), np.concatenate((fit_x1[1], fit_x2[1], fit_y1[1], fit_y2[1], fit_E1[1], fit_E2[1])))

    def show_hist_NN_withLednev(self, figsize=(12,10), r_x = (-4, 4), r_y = (-4, 4), r_E = (-0.3, 0.3), fit_x=(True, False), fit_y=(True, False), fit_E=(True, True), figsave=(False, 'network1')):
        '''
        shows plots of all relative differences of NN and Lednev/coral fit like
        (x1, x2)
        (y1, y2)
        (E1, E2)

        First entry in r_x refers to NN and second to Lednev 
        '''
        plt.style.use('standard_style.mplstyle') # this is needed later on!
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Performence of " + figsave[1] + " compared to Lednev")
        plt.rcParams['figure.figsize'] = figsize
        alpha = 0.55

        # x1 PLOT
        fig.add_subplot(3,2, 1)
        fit_x1 = self.make_subplot_withLednev(self.x1, self.x1_t, self.x1_c, r_x, alpha, 'x_1', fit_x)

        # x2 PLOT
        fig.add_subplot(3,2,2)
        fit_x2 = self.make_subplot_withLednev(self.x2, self.x2_t, self.x2_c, r_x, alpha, 'x_2', fit_x)

        # y1 PLOT
        fig.add_subplot(3,2,3)
        fit_y1 = self.make_subplot_withLednev(self.y1, self.y1_t, self.y1_c, r_y, alpha, 'y_1', fit_y)

        # y2 PLOT
        fig.add_subplot(3,2,4)
        fit_y2 = self.make_subplot_withLednev(self.y2, self.y2_t, self.y2_c, r_y, alpha, 'y_2', fit_y)

        # E1 PLOT
        fig.add_subplot(3,2,5)
        fit_E1 = self.make_subplot_withLednev(self.E1, self.E1_t, self.E1_c, r_E, alpha, 'E_1', fit_E, rel=True)

        # E2 PLOT
        fig.add_subplot(3,2,6)
        fit_E2 = self.make_subplot_withLednev(self.E2, self.E2_t, self.E2_c, r_E, alpha, 'E_2', fit_E, rel=True)
        
        plt.tight_layout()
        if figsave[0] == True:
            plt.savefig(figsave[1] +"histo_withLednev.pdf")
        plt.show()
        return unp.uarray(np.concatenate((fit_x1[0], fit_x2[0], fit_y1[0], fit_y2[0], fit_E1[0], fit_E2[0])), np.concatenate((fit_x1[1], fit_x2[1], fit_y1[1], fit_y2[1], fit_E1[1], fit_E2[1])))

    def make_subplot_withLednev(self, val, val_t, val_c, r, alpha, param, fit, rel=False):
        fit_values = np.zeros(8) # mu, simga, mu_c, sigma_c and then uncertainties
        if rel == True:
            n_counts, containers, patches = plt.hist((val-val_t)/val_t, range=r, alpha=alpha, label="NN")
            n_counts_c, containers_c, patches_c = plt.hist((val_c-val_t)/val_t, range=r, alpha=alpha, label="Lednev")
            plt.xlabel("$({}$".format(param)+"$_{\mathrm{, pred}}$" + "$- {}$".format(param)+"$_{\mathrm{, MC}})$" + "$ / {}$".format(param)+"$_{\mathrm{, MC}}$ [cm]")
        else:
            n_counts, containers, patches = plt.hist((val-val_t), range=r, alpha=alpha, label="NN")
            n_counts_c, containers_c, patches_c = plt.hist((val_c-val_t), range=r, alpha=alpha, label="Lednev")
            plt.xlabel("${}$".format(param)+"$_{\mathrm{, pred}}$" + "$- {}$".format(param)+"$_{\mathrm{, MC}}$ [cm]")
        if fit[0]==True:
            popt, perr = self.gaus_fit(containers, n_counts, label='{NN}')
            fit_values[np.array([0,1, 4, 5])] = np.array([popt[0], popt[1], perr[0], perr[1]])
        if fit[1]==True:
            popt_c, perr_c = self.gaus_fit(containers_c, n_counts_c, label='{L}')
            fit_values[np.array([2,3, 6, 7])] = np.array([popt_c[0], popt_c[1], perr_c[0], perr_c[1]])
        plt.ylabel("counts")
        plt.legend()
        #plt.title('${}$'.format(param))
        return fit_values.reshape((2,4))

    def make_subplot(self, val, val_t, r, alpha, param, fit, rel=False):
        fit_values = np.zeros(4) # mu, simga and then uncertainties
        if rel == True:
            n_counts, containers, patches = plt.hist((val-val_t)/val_t, range=r, alpha=alpha)
            plt.xlabel("$({}$".format(param)+"$_{\mathrm{, pred}}$" + "$- {}$".format(param)+"$_{\mathrm{, MC}})$" + "$ / {}$".format(param)+"$_{\mathrm{, MC}}$ [cm]")
        else:
            n_counts, containers, patches = plt.hist((val-val_t), range=r, alpha=alpha)
            plt.xlabel("${}$".format(param)+"$_{\mathrm{, pred}}$" + "$- {}$".format(param)+"$_{\mathrm{, MC}}$ [cm]")
        if fit==True:
            popt, perr = self.gaus_fit(containers, n_counts, label='')
            fit_values = np.array([popt[0], popt[1], perr[0], perr[1]])
        plt.ylabel("counts")
        plt.legend()
        #plt.title('${}$'.format(param))
        return fit_values.reshape((2,2))

    def gaus_fit(self, containers, n_counts, label):
        # label should be '{L}' or '{NN}'
        x_centers = 0.5*(containers[:-1]+containers[1:])
        maxv = n_counts.max()
        ind_fit = np.where(n_counts > 0.3*maxv)
        liml = ind_fit[0].min()
        limu = ind_fit[0].max() + 1

        popt, pcov = curve_fit(self.gaus, x_centers[liml:limu], n_counts[liml:limu], p0=[0,1, 100], sigma=1/np.sqrt(n_counts)[liml:limu], maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        a1_fit = np.linspace(x_centers[liml:limu][0], x_centers[liml:limu][-1], 500)
        a2_fit = self.gaus(a1_fit, *popt)
        plt.plot(a1_fit, a2_fit, '--', label= r"$\mu_{} = ({:.3f} \pm {:.3f})$".format(label, popt[0], perr[0]) + "\n" + r"$\sigma_{} = ({:.3f} \pm {:.3f})$".format(label, popt[1], perr[1]))
        return popt, perr

    def gaus(self, x, mu, sigma, A):
        return (A/np.sqrt(2*np.pi * sigma**2))* np.exp(-(x-mu)**2 / (2*sigma**2))

# Tabellen aus Gaus Fit

    def make_Latex_tab(self, uarr, lednev=True):
        '''
        uarr is a unp array from show_hist() function. Forms:
        x1 x2
        y1 y2
        E1 E2
        '''
        if lednev == False:
            uarr = uarr.reshape(6, 2)
        else:
            uarr = uarr.reshape(12, 2)
            uarr = uarr[::2]

        for n in range(6):
            print(self.to_latex(uarr[n, 0]), '&', self.to_latex(uarr[n, 1]), r"\\")

    def to_latex(self, w):
        "nimmt ufloat Wert und formatiert es in die korrekte Schreibweise für Latex Tabellen (nur eine Zeile)"
        wert = w.nominal_value
        fehler = w.std_dev
        tup = ("$", format(wert, '.3f').replace(".", ","), "\pm", format(fehler, '.3f').replace(".", ","), "$" )
        str = ''.join(tup)
        return str

    def make_markdown_tab(self, uarr, lednev=True):
        print('| | $\mu$ | $\sigma$ |')
        print('| --- | --- | --- |')
        if lednev == False:
            uarr = uarr.reshape(6, 2)
        else:
            uarr = uarr.reshape(12, 2)
            uarr = uarr[::2]
        names = ['$x_1$', '$x_2$','$y_1$','$y_2$','$E_1$','$E_2$']
        for n in range(6):
            print('|', names[n], '|', self.to_latex(uarr[n, 0]), '|', self.to_latex(uarr[n, 1]), '|')

# 2D Histogramme 

    def show_2d_hist(self, figsize=(15,15), title="", figsave=False):
        plt.style.use('standard_style.mplstyle')
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Relations of performence of network " + title)

        diff_x = self.x1_t-self.x2_t
        diff_y = self.y1_t-self.y2_t

        # ersten 4: gegen x1_t - x2_t
        fig.add_subplot(4,4, 1)
        hist = plt.hist2d(self.x1 - self.x1_t, diff_x, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{1, NN} - x_{1, MC}$ [cm]")
        plt.ylabel("$x_{1, MC} - x_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(4,4, 2)
        hist = plt.hist2d(self.x2 - self.x2_t, diff_x, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{2, NN} - x_{2, MC}$ [cm]")
        plt.ylabel("$x_{1, MC} - x_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(4,4, 3)
        hist = plt.hist2d(self.y1 - self.y1_t, diff_x, norm=LogNorm(), bins=100)
        plt.xlabel("$y_{1, NN} - y_{1, MC}$ [cm]")
        plt.ylabel("$x_{1, MC} - x_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(4,4, 4)
        hist = plt.hist2d(self.y2 - self.y2_t, diff_x, norm=LogNorm(), bins=100)
        plt.xlabel("$y_{2, NN} - y_{2, MC}$ [cm]")
        plt.ylabel("$x_{1, MC} - x_{2, MC}$ [cm]")
        plt.colorbar()

        # zweiten 4: gegen y1_t - y2_t
        fig.add_subplot(4,4, 5)
        hist = plt.hist2d(self.x1 - self.x1_t, diff_y, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{1, NN} - x_{1, MC}$ [cm]")
        plt.ylabel("$y_{1, MC} - y_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(4,4, 6)
        hist = plt.hist2d(self.x2 - self.x2_t, diff_y, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{2, NN} - x_{2, MC}$ [cm]")
        plt.ylabel("$y_{1, MC} - y_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(4,4, 7)
        hist = plt.hist2d(self.y1 - self.y1_t, diff_y, norm=LogNorm(), bins=100)
        plt.xlabel("$y_{1, NN} - y_{1, MC}$ [cm]")
        plt.ylabel("$y_{1, MC} - y_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(4,4, 8)
        hist = plt.hist2d(self.y2 - self.y2_t, diff_y, norm=LogNorm(), bins=100)
        plt.xlabel("$y_{2, NN} - y_{2, MC}$ [cm]")
        plt.ylabel("$y_{1, MC} - y_{2, MC}$ [cm]")
        plt.colorbar()

        # dritten 4 +2: nicht gegen truth
        fig.add_subplot(4,4, 9)
        hist = plt.hist2d(self.x1 - self.x1_t, self.x2 - self.x2_t, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{1, NN} - x_{1, MC}$ [cm]")
        plt.ylabel("$x_{2, NN} - x_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(4,4, 10)
        hist = plt.hist2d(self.y1 - self.y1_t, self.y2 - self.y2_t, norm=LogNorm(), bins=100)
        plt.xlabel("$y_{1, NN} - y_{1, MC}$ [cm]")
        plt.ylabel("$y_{2, NN} - y_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(4,4, 11)
        hist = plt.hist2d(self.x1 - self.x1_t, self.y2 - self.y2_t, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{1, NN} - x_{1, MC}$ [cm]")
        plt.ylabel("$y_{2, NN} - y_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(4,4, 12)
        hist = plt.hist2d(self.x2 - self.x2_t, self.y1 - self.y1_t, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{2, NN} - x_{2, MC}$ [cm]")
        plt.ylabel("$y_{1, NN} - y_{1, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(4,4, 13)
        hist = plt.hist2d(self.x2 - self.x2_t, self.y2 - self.y2_t, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{2, NN} - x_{2, MC}$ [cm]")
        plt.ylabel("$y_{2, NN} - y_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(4,4, 14)
        hist = plt.hist2d(self.x1 - self.x1_t, self.y1 - self.y1_t, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{1, NN} - x_{1, MC}$ [cm]")
        plt.ylabel("$y_{1, NN} - y_{1, MC}$ [cm]")
        plt.colorbar()

        # zuletzt truth gegen truth

        fig.add_subplot(4,4, 15)
        hist = plt.hist2d(diff_x, diff_y, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{1, MC} - x_{2, MC}$ [cm]")
        plt.ylabel("$y_{1, MC} - y_{2, MC}$ [cm]")
        plt.colorbar()

        plt.tight_layout()
        if figsave==True:
            plt.savefig(title + "_2D_histos.pdf")
        else:
            plt.show()