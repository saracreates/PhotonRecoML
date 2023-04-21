import numpy as np
import random
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import unumpy as unp
from matplotlib.colors import LogNorm

# other helpfull functions...

def standardscore(cluster):
    '''input should be input of NN use this right before feeding NN'''
    mu = np.mean(cluster, axis=1)
    sigma = np.std(cluster, axis=1) 
    print("mean: ", np.mean(mu))
    print("standard deviation: ", np.mean(sigma))



class InputData():
    # initialize with all raw data
    def __init__(self, rootfile, numevents=-1, min_dist=0, coord='positive'):
        # extract data from rootfile
        self.event = rootfile["user202302;1"]
        self.xMC = self.event["x_MC"].array(library="np")[:numevents]
        self.yMC = self.event["y_MC"].array(library="np") [:numevents]
        self.EMC = self.event["E_MC"].array(library="np") [:numevents]
        self.celltype = self.event["celltype"].array(library="np") [:numevents]
        self.icol = self.event["icol"].array(library="np") [:numevents] # vertical spalte
        self.irow = self.event["irow"].array(library="np") [:numevents] # horizontal zeile
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
        self.x_0 = -117.585 # ecal links unten in der Ecke aka (0, 0)
        self.y_0 = -89.855
        self.coord = coord
        print('Loaded data')

        # modify data
        self.form_cluster() # forms cluster in shashlik part of ecal
        self.prep_coralfit_data() # bring into uniform shape
        if min_dist>0:
            self.cut_min_distance(min_dist) #cut events that are too close together. Distance in cm.
        if self.coord == 'positive':
            self.shift_coord()


    def shift_coord(self):
        '''shift coordinate system so that only positive values for (x, y) exist'''
        self.x_truth = self.x_truth - self.x_0
        self.y_truth = self.y_truth - self.y_0
        self.x_fit = self.x_fit - self.x_0
        self.y_fit = self.y_fit - self.y_0
        self.xMC = self.xMC - self.x_0
        self.yMC = self.yMC - self.y_0

    def show_ecal(self, i, shashlik=True):
        ''' show picture of ecal with cluster i'''
        if self.coord=='middle':
            x_0 = self.x_0
            y_0 = self.y_0
            delx = 0
            dely = 0
        elif self.coord=='positive':
            x_0 = 0
            y_0 = 0
            delx = self.x_0
            dely = self.y_0
        plt.imshow(self.ecal[i], origin='lower', norm=LogNorm(), extent=[x_0-self.cellsize/2, x_0-self.cellsize/2 + 64*self.cellsize, y_0-self.cellsize/2, y_0-self.cellsize/2 + 48*self.cellsize]) # extent (left, right, bottom, top)
        plt.scatter(x_truth[i], self.y_truth[i], c='r', label="MC coordinates")
        plt.xticks(np.linspace(x_0+self.cellsize/2, x_0+64*self.cellsize+self.cellsize/2, (32+1)))
        plt.yticks(np.linspace(y_0+self.cellsize/2, y_0+48*self.cellsize+self.cellsize/2, (24+1)))
        plt.legend()
        if shashlik==True:
            plt.xlim(-86.945 - delx, 96.895 - delx) # shashlik
            plt.ylim(-43.895- dely, 48.025- dely) # shashlik
        plt.tight_layout() 
        plt.xlabel('$x$ [cm]')
        plt.ylabel('$y$ [cm]')
        plt.grid(visible=True)
        plt.show()

    def form_cluster(self):
        '''make empty ECAL and shashlik and fill with cluster'''
        self.ecal = np.zeros((len(self.x_truth), 48, 64)) # empty ecal
        for i in range(len(self.x_truth)):
            self.ecal[i, self.icol[i], self.irow[i]] = self.EMC[i] # fill with energy values
        self.shash = self.ecal[:,12:12+24+1, 8:8+48+1] # cut out shashlik part! 
        print('Formed cluster')

    def train_test_split(self, percentage=0.8):
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
        
        self.shash_t =  self.shash[:cut]
        self.shash_v = self.shash[cut:]

        self.ecal_t =  self.ecal[:cut]
        self.ecal_v = self.ecal[cut:]
        
        self.momentum_t = self.momentum[:cut]
        self.momentum_v = self.momentum[cut:]
        
        self.numfit_v = self.num_fit[cut:]
        print('Splitted data into training and test set!')

        # prep trainings and evaluation data
        self.prep_trainingsdata()
        self.prep_verificationdata()

    def prep_trainingsdata(self):
        # creates [x1, y1, E1, x2, y2, E2] 
        self.training = np.array([self.x_truth_train.T[0], self.y_truth_train.T[0], self.E_truth_train.T[0], \
                                self.x_truth_train.T[1], self.y_truth_train.T[1], self.E_truth_train.T[1]]).T
        print("Prepared trainings data")

    def prep_verificationdata(self):
        # creates [x1 , y1 , E1, x2 , y2 , E2] 
        self.veri_truth = np.array(\
        [self.x_truth_veri.T[0], self.y_truth_veri.T[0], self.E_truth_veri.T[0], \
        self.x_truth_veri.T[1], self.y_truth_veri.T[1], self.E_truth_veri.T[1]]).T

        # Lednev
        self.veri_fit = np.array(\
        [self.x_fit_veri.T[0], self.y_fit_veri.T[0], self.E_fit_veri.T[0], \
        self.x_fit_veri.T[1], self.y_fit_veri.T[1], self.E_fit_veri.T[1]]).T
        # make label right:
        self.veri_fit = self.roll_data_MSE(self.veri_fit)

        print("Prepared test data")

    def roll_data_MSE(self, data):
        '''bring Lednev or NN data into right order. 'data' must have shape (n, 6)'''
        data_flipped = np.roll(data, 3, axis=1)
        mse = np.sum(np.square(data-self.veri_truth), axis=1) 
        mse_flipped = np.sum(np.square(data_flipped-self.veri_truth), axis=1)

        ind_flip = np.where(mse_flipped<mse)
        data[ind_flip] = np.roll(data[ind_flip], 3, axis=1)
        return data

    def cut_min_distance(self, dis):
        '''cuts all photons that are less than the given distance apart! NEEDS TO BE PERFORMED BEFORE DIVIDING INTO TRAINING AND TEST SET'''
        acc_dist = np.sqrt(self.x_truth.T[0]**2 + self.y_truth.T[0]**2) - np.sqrt(self.x_truth.T[1]**2 + self.y_truth.T[1]**2) # pos 1 - pos 2
        ind_cut = np.where(abs(acc_dist)<dis) # scheide alle events raus bei dene die photonen zu nah sind
        self.delete_bad_events(ind_cut)
        print("Cutted ", len(ind_cut[0]), " clusters due to photon pairs with distance smaller than ", dis, " cm.")

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
            self.ecal = np.delete(self.ecal, ind_del, axis=0) 
            self.shash = np.delete(self.shash, ind_del, axis=0) 
        except:
            pass
        
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


class Evaluation:
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

        '''
        if self.ipd.coord == 'positive':
            self.x_0 = 0
            self.y_0 = 0
        elif self.ipd.coord == 'middle':
            self.x_0 = ipd.x_0
            self.y_0 = ipd_y_0
        '''

# Data visualisation

    def show_cluster_NNpred(self, i, shashlik=True):
        ''' show picture of ecal with cluster i and NN pred'''
        plt.style.use('standard_style.mplstyle')
        plt.rcParams["figure.figsize"] = (10,5)
        if self.ipd.coord=='middle':
            x_0 = self.ipd.x_0
            y_0 = self.ipd.y_0
            delx = 0
            dely = 0
        elif self.ipd.coord=='positive':
            x_0 = 0
            y_0 = 0
            delx = self.ipd.x_0
            dely = self.ipd.y_0
        plt.imshow(self.ipd.ecal_v[i], origin='lower', norm=LogNorm(), extent=[x_0-self.ipd.cellsize/2, x_0-self.ipd.cellsize/2 + 64*self.ipd.cellsize, y_0-self.ipd.cellsize/2, y_0-self.ipd.cellsize/2 + 48*self.ipd.cellsize]) # extent (left, right, bottom, top)
        plt.colorbar()
        plt.scatter(np.array([self.x1_t[i], self.x2_t[i]]), np.array([self.y1_t[i], self.y2_t[i]]), c='r', s=12., label="MC coordinates")
        plt.scatter(np.array([self.x1[i], self.x2[i]]), np.array([self.y1[i], self.y2[i]]), s=12., c='black', label="NN coordinates")
        plt.xticks(np.linspace(x_0+self.ipd.cellsize/2, x_0+64*self.ipd.cellsize+self.ipd.cellsize/2, (32+1)))
        plt.yticks(np.linspace(y_0+self.ipd.cellsize/2, y_0+48*self.ipd.cellsize+self.ipd.cellsize/2, (24+1)))
        plt.legend()
        if shashlik==True:
            plt.xlim(-86.945 - delx, 96.895- delx) # shashlik
            plt.ylim(-43.895- dely, 48.025- dely) # shashlik
        plt.tight_layout() 
        plt.xlabel('$x$ [cm]')
        plt.ylabel('$y$ [cm]')
        plt.grid(visible=True)
        plt.tight_layout()
        plt.show()

    def training_vs_validation_loss(self, fit_hist, log=True, save=False, title=""):
        plt.style.use('standard_style.mplstyle')
        plt.plot(fit_hist.history['loss'])
        plt.plot(fit_hist.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(["Training Loss", "Validation Loss"])
        if log==True:
                plt.yscale('log')
        if save==True:
            plt.savefig("./bilder/" + title + "_loss.pdf")
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
            plt.savefig("./bilder/" + figsave[1] +"histo.pdf")
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
            plt.savefig("./bilder/" + figsave[1] +"histo_withLednev.pdf")
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
        diff_E = abs(self.E1_t-self.E2_t)

        # erste Spalte gegen x1_t - x2_t
        fig.add_subplot(6,5, 1)
        hist = plt.hist2d(self.x1 - self.x1_t, diff_x, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{1, NN} - x_{1, MC}$ [cm]")
        plt.ylabel("$x_{1, MC} - x_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 6)
        hist = plt.hist2d(self.x2 - self.x2_t, diff_x, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{2, NN} - x_{2, MC}$ [cm]")
        plt.ylabel("$x_{1, MC} - x_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 11)
        hist = plt.hist2d(self.y1 - self.y1_t, diff_x, norm=LogNorm(), bins=100)
        plt.xlabel("$y_{1, NN} - y_{1, MC}$ [cm]")
        plt.ylabel("$x_{1, MC} - x_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 16)
        hist = plt.hist2d(self.y2 - self.y2_t, diff_x, norm=LogNorm(), bins=100)
        plt.xlabel("$y_{2, NN} - y_{2, MC}$ [cm]")
        plt.ylabel("$x_{1, MC} - x_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 21)
        hist = plt.hist2d(self.E1 - self.E1_t, diff_x, norm=LogNorm(), bins=100)
        plt.xlabel("$E_{1, NN} - E_{1, MC}$ [cm]")
        plt.ylabel("$x_{1, MC} - x_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 26)
        hist = plt.hist2d(self.E2 - self.E2_t, diff_x, norm=LogNorm(), bins=100)
        plt.xlabel("$E_{2, NN} - E_{2, MC}$ [cm]")
        plt.ylabel("$x_{1, MC} - x_{2, MC}$ [cm]")
        plt.colorbar()

 

        # zweite Spalte: gegen y1_t - y2_t
        fig.add_subplot(6,5, 2)
        hist = plt.hist2d(self.x1 - self.x1_t, diff_y, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{1, NN} - x_{1, MC}$ [cm]")
        plt.ylabel("$y_{1, MC} - y_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 7)
        hist = plt.hist2d(self.x2 - self.x2_t, diff_y, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{2, NN} - x_{2, MC}$ [cm]")
        plt.ylabel("$y_{1, MC} - y_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 12)
        hist = plt.hist2d(self.y1 - self.y1_t, diff_y, norm=LogNorm(), bins=100)
        plt.xlabel("$y_{1, NN} - y_{1, MC}$ [cm]")
        plt.ylabel("$y_{1, MC} - y_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 17)
        hist = plt.hist2d(self.y2 - self.y2_t, diff_y, norm=LogNorm(), bins=100)
        plt.xlabel("$y_{2, NN} - y_{2, MC}$ [cm]")
        plt.ylabel("$y_{1, MC} - y_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 22)
        hist = plt.hist2d(self.E1 - self.E1_t, diff_y, norm=LogNorm(), bins=100)
        plt.xlabel("$E_{1, NN} - E_{1, MC}$ [cm]")
        plt.ylabel("$y_{1, MC} - y_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 27)
        hist = plt.hist2d(self.E2 - self.E2_t, diff_y, norm=LogNorm(), bins=100)
        plt.xlabel("$E_{2, NN} - E_{2, MC}$ [cm]")
        plt.ylabel("$y_{1, MC} - y_{2, MC}$ [cm]")
        plt.colorbar()

        # dritte spalte: gegen  E truth
        fig.add_subplot(6,5, 3)
        hist = plt.hist2d(self.x1 - self.x1_t, diff_E, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{1, NN} - x_{1, MC}$ [cm]")
        plt.ylabel("$E_{1, MC} - E_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 8)
        hist = plt.hist2d(self.x2 - self.x2_t, diff_E, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{2, NN} - x_{2, MC}$ [cm]")
        plt.ylabel("$E_{1, MC} - E_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 13)
        hist = plt.hist2d(self.y1 - self.y1_t, diff_E, norm=LogNorm(), bins=100)
        plt.xlabel("$y_{1, NN} - y_{1, MC}$ [cm]")
        plt.ylabel("$E_{1, MC} - E_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 18)
        hist = plt.hist2d(self.y2 - self.y2_t, diff_E, norm=LogNorm(), bins=100)
        plt.xlabel("$y_{2, NN} - y_{2, MC}$ [cm]")
        plt.ylabel("$E_{1, MC} - E_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 23)
        hist = plt.hist2d(self.E1 - self.E1_t, diff_E, norm=LogNorm(), bins=100)
        plt.xlabel("$E_{1, NN} - E_{1, MC}$ [cm]")
        plt.ylabel("$E_{1, MC} - E_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 28)
        hist = plt.hist2d(self.E2 - self.E2_t, diff_E, norm=LogNorm(), bins=100)
        plt.xlabel("$E_{2, NN} - E_{2, MC}$ [cm]")
        plt.ylabel("$E_{1, MC} - E_{2, MC}$ [cm]")
        plt.colorbar()

        # vierte Spalte: nicht gegen truth (Position)
        fig.add_subplot(6,5, 4)
        hist = plt.hist2d(self.x1 - self.x1_t, self.x2 - self.x2_t, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{1, NN} - x_{1, MC}$ [cm]")
        plt.ylabel("$x_{2, NN} - x_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 9)
        hist = plt.hist2d(self.y1 - self.y1_t, self.y2 - self.y2_t, norm=LogNorm(), bins=100)
        plt.xlabel("$y_{1, NN} - y_{1, MC}$ [cm]")
        plt.ylabel("$y_{2, NN} - y_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 14)
        hist = plt.hist2d(self.x1 - self.x1_t, self.y2 - self.y2_t, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{1, NN} - x_{1, MC}$ [cm]")
        plt.ylabel("$y_{2, NN} - y_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 19)
        hist = plt.hist2d(self.x2 - self.x2_t, self.y1 - self.y1_t, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{2, NN} - x_{2, MC}$ [cm]")
        plt.ylabel("$y_{1, NN} - y_{1, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 24)
        hist = plt.hist2d(self.x2 - self.x2_t, self.y2 - self.y2_t, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{2, NN} - x_{2, MC}$ [cm]")
        plt.ylabel("$y_{2, NN} - y_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 29)
        hist = plt.hist2d(self.x1 - self.x1_t, self.y1 - self.y1_t, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{1, NN} - x_{1, MC}$ [cm]")
        plt.ylabel("$y_{1, NN} - y_{1, MC}$ [cm]")
        plt.colorbar()

        # fuenfte Spalte: nicht gegen truth (Energie)
        fig.add_subplot(6,5, 5)
        hist = plt.hist2d(self.E1 - self.E1_t, self.E2 - self.E2_t, norm=LogNorm(), bins=100)
        plt.xlabel("$E_{1, NN} - E_{1, MC}$ [cm]")
        plt.ylabel("$E_{2, NN} - E_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 10)
        hist = plt.hist2d(self.E1 - self.E1_t, self.x1 - self.x1_t, norm=LogNorm(), bins=100)
        plt.xlabel("$E_{1, NN} - E_{1, MC}$ [cm]")
        plt.ylabel("$x_{1, NN} - x_{1, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 15)
        hist = plt.hist2d(self.E1 - self.E1_t, self.y1 - self.y1_t, norm=LogNorm(), bins=100)
        plt.xlabel("$E_{1, NN} - E_{1, MC}$ [cm]")
        plt.ylabel("$y_{1, NN} - y_{1, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 20)
        hist = plt.hist2d(self.E2 - self.E2_t, self.x2 - self.x2_t, norm=LogNorm(), bins=100)
        plt.xlabel("$E_{2, NN} - E_{2, MC}$ [cm]")
        plt.ylabel("$x_{2, NN} - x_{2, MC}$ [cm]")
        plt.colorbar()

        fig.add_subplot(6,5, 25)
        hist = plt.hist2d(self.E2 - self.E2_t, self.y2 - self.y2_t, norm=LogNorm(), bins=100)
        plt.xlabel("$E_{2, NN} - E_{2, MC}$ [cm]")
        plt.ylabel("$y_{2, NN} - y_{2, MC}$ [cm]")
        plt.colorbar()


        '''
        # zuletzt truth gegen truth

        fig.add_subplot(6,5, 15)
        hist = plt.hist2d(diff_x, diff_y, norm=LogNorm(), bins=100)
        plt.xlabel("$x_{1, MC} - x_{2, MC}$ [cm]")
        plt.ylabel("$y_{1, MC} - y_{2, MC}$ [cm]")
        plt.colorbar()
        '''
        plt.tight_layout()
        if figsave==True:
            plt.savefig("./bilder/" + title + "_2D_histos.pdf")
        plt.show()

# Was liegt ein sigma entfernt?

    def one_sigma_away(self, fit_param, condition, figsave=False, title="network1"):
        '''fit_param ist aus ev.show_hist_NN()'''
        if condition=='x':
            index = [1,3]
            a1 = self.x1
            a2 = self.x2
            a1_t = self.x1_t
            a2_t = self.x2_t
            relative = False
        elif condition=="y":
            index = [5,7]
            a1 = self.y1
            a2 = self.y2
            a1_t = self.y1_t
            a2_t = self.y2_t
            relative = False
        elif condition=="E":
            index = [9,11]
            a1 = self.E1
            a2 = self.E2
            a1_t = self.E1_t
            a2_t = self.E2_t
            relative = True
        else:
            raise ValueError("condition must be 'x', 'y' or 'E', not {}.".format(condition))
        sig1 = unp.nominal_values(fit_param[index[0]])
        sig2 = unp.nominal_values(fit_param[index[1]])

        ind1_bad, ind2_bad = self.find_indicies(a1, a2, a1_t, a2_t, sig1, sig2, rel=relative)

        plt.rcParams["figure.figsize"] = (15, 15)
        plt.style.use('standard_style.mplstyle')

        # plots mit Werten die mehr als 1 sigma von a1 abweichen
        plt.subplot(3,2,1)
        self.subplot_sigma(self.E1_t, ind1_bad, condition + "_1", "E_1", einheit='GeV')
        plt.subplot(3,2,3)
        self.subplot_sigma(self.x1_t, ind1_bad, condition + "_1", "x_1")
        plt.subplot(3,2,5)
        self.subplot_sigma(self.y1_t, ind1_bad, condition + "_1", "y_1")
        # plots mit Werten die mehr als 1 sigma von a2 abweichen
        plt.subplot(3,2,2)
        self.subplot_sigma(self.E2_t, ind2_bad, condition + "_2", "E_2", einheit='GeV')
        plt.subplot(3,2,4)
        self.subplot_sigma(self.x2_t, ind2_bad, condition + "_2", "x_2")
        plt.subplot(3,2,6)
        self.subplot_sigma(self.y2_t, ind2_bad, condition + "_2", "y_2")

        plt.tight_layout()
        if figsave==True:
            plt.savefig("./bilder/" + title + "_sigma_"+condition+".pdf")
        plt.show()

    def find_indicies(self, a1, a2, a1_t, a2_t, sig1, sig2, rel=False):
        ''' finds indicies of values (= x, y, E) that are more than one sigma away'''
        if rel==True:  
            diff_a1 = (a1 - a1_t) / a1_t
            diff_a2 = (a2 - a2_t) / a2_t
        else:
            diff_a1 = (a1 - a1_t) 
            diff_a2 = (a2 - a2_t)

        ind1_bad = np.where(abs(diff_a1) > abs(sig1))
        ind2_bad = np.where(abs(diff_a2) > abs(sig2))
        return ind1_bad, ind2_bad

    def subplot_sigma(self, a_t, ind_a_bad, condition, label, einheit='cm', b=100):
        hist_a_t, bin_a_t = np.histogram(a_t, bins=b)#, range=(0, 200))
        hist_a_bad, bins = np.histogram(a_t[ind_a_bad], bins=b)#, range=(0, 200))
        plt.bar(bin_a_t[:len(bin_a_t)-1], hist_a_bad/hist_a_t, align='edge', width=bin_a_t[1]-bin_a_t[0]) 
        plt.xlabel("${}$ [{}]".format(label, einheit))
        plt.ylabel("ratio of ${}$ values with $\Delta {} > \sigma_{} $".format(label, condition, "{"+condition+"}"))