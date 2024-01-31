# pylint: disable=C0103

####### This is the MCMC plotting code for the DiskFM data products and models #######
####### Authors: Johan Mazoyer, Justin Hom
import sys
import os
#Use these to override enivornment-path packages
#sys.path.insert(0,'/home/jrhom/pyklip/')
sys.path.insert(0,'/home/jrhom/anadisk_model')
basedir = os.environ["PWD"]  # the base directory where is
# your data (using OS environnement variable allow to use same code on
# different computer without changing this).

#Input the file name of your yaml parameter file. Assumes that your parameter file is in a directory called "initialization_files"

default_parameter_file = "GPI_Hband_MCMC_ADI_4796K1.yaml"


import glob
import socket
import warnings

from datetime import datetime

import math as mt
import numpy as np
from scipy.ndimage import rotate

import astropy.io.fits as fits
from astropy.convolution import convolve

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle, Ellipse
import matplotlib.lines as mlines
import matplotlib.colors as colors
import yaml

import corner
from emcee import backends

from numba.core.errors import NumbaWarning

import pyklip.instruments.GPI as GPI
import pyklip.instruments.SPHERE as SPHERE
from pyklip.fmlib.diskfm import DiskFM

from anadisk_model.anadisk_sum_mask import phase_function_spline, generate_disk


from disk_models import hg_1g, hg_2g, hg_3g, log_hg_2g, log_hg_3g

import astro_unit_conversion as convert
#from kowalsky import kowalsky
import make_gpi_psf_for_disks as gpidiskpsf


import diskfit_mcmc as diskfit_mcmc

plt.switch_backend('agg')

# There is a conflict when I import
# matplotlib with pyklip if I don't use this line


def chains_to_params(chain, flatten = False):
    #Grabs the MCMC chains and turns them back into your free parameters

    chain_param = chain*0.
    
    n_iter = chain.shape[0]
    nwalkers = chain.shape[1]
    n_dim_mcmc = chain.shape[2]

    for i in range(n_iter):
        for j in range(nwalkers):
            _ , chain_param[i,j,:] = diskfit_mcmc.from_theta_to_params(chain[i,j,:])


    if flatten:
        return_chain = np.zeros((n_iter*nwalkers,n_dim_mcmc))
        for i in range(n_dim_mcmc):
            return_chain[:,i] = chain_param[:,:,i].flatten()
    else:
        return_chain = chain_param

    return return_chain

########################################################
def crop_center_odd(img, crop):
    #Crops/zooms in on images
    y, x = img.shape
    startx = (x - 1) // 2 - crop // 2
    starty = (y - 1) // 2 - crop // 2
    return img[starty:starty + crop, startx:startx + crop]


########################################################
def offset_2_RA_dec(dx, dy, inclination, principal_angle, distance_star):
    """ right ascension and declination of the ellipse centre with respect to the star
        location from the offset in AU in the disk plane define by the max disk code

    Args:
        dx: offsetx of the star in AU in the disk plane define by the max disk code
            au, + -> NW offset disk plane Minor Axis
        dy: offsety of the star in AU in the disk plane define by the max disk code
            au, + -> SW offset disk plane Major Axis
        inclination: inclination in degrees
        principal_angle: prinipal angle in degrees

    Returns:
        [right ascension, declination]
    """

    dx_disk_mas = convert.au_to_mas(dx * np.cos(np.radians(inclination)),
                                    distance_star)
    dy_disk_mas = convert.au_to_mas(-dy, distance_star)

    dx_sky = np.cos(np.radians(principal_angle)) * dx_disk_mas - np.sin(
        np.radians(principal_angle)) * dy_disk_mas
    dy_sky = np.sin(np.radians(principal_angle)) * dx_disk_mas + np.cos(
        np.radians(principal_angle)) * dy_disk_mas

    dAlpha = -dx_sky
    dDelta = dy_sky

    return dAlpha, dDelta


########################################################
def make_chain_plot(params_mcmc_yaml):
    """ make_chain_plot reading the .h5 file from emcee

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        None
    """

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']
    quality_plot = params_mcmc_yaml['QUALITY_PLOT']
    labels = params_mcmc_yaml['LABELS']
    names = params_mcmc_yaml['NAMES']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    name_h5 = file_prefix + '_backend_file_mcmc'

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    
    iter = reader.iteration
    if iter < burnin - 1:
        burnin =0
        params_mcmc_yaml['BURNIN'] = 0 

    chain = reader.get_chain(discard=0, thin=thin)
    log_prob_samples_flat = reader.get_log_prob(discard=burnin,
                                                flat=True,
                                                thin=thin)
    # print(log_prob_samples_flat)
    tau = reader.get_autocorr_time(tol=0)
    if burnin > reader.iteration - 1:
        raise ValueError(
            "the burnin cannot be larger than the # of iterations")
    print("")
    print("")
    print(name_h5)
    print("# of iteration in the backend chain initially: {0}".format(
        reader.iteration))
    print("Max Tau times 50: {0}".format(50 * np.max(tau)))
    print("")

    print("Maximum Likelyhood: {0}".format(np.nanmax(log_prob_samples_flat)))

    print("burn-in: {0}".format(burnin))
    print("chain shape: {0}".format(chain.shape))

    n_dim_mcmc = chain.shape[2]
    nwalkers = chain.shape[1]

    diskfit_mcmc.SPF_MODEL = params_mcmc_yaml['SPF_MODEL']  #Type of description for the SPF
    
    chain = chains_to_params(chain)

    _, axarr = plt.subplots(n_dim_mcmc,
                            sharex=True,
                            figsize=(6.4 * quality_plot, 4.8 * quality_plot))

    for i in range(n_dim_mcmc):
        axarr[i].set_ylabel(labels[names[i]], fontsize=5 * quality_plot)
        axarr[i].tick_params(axis='y', labelsize=4 * quality_plot)
        if i == 0:
            axarr[i].set_ylim(bottom=73.5,top=74.5)

        for j in range(nwalkers):
            axarr[i].plot(chain[:, j, i], linewidth=quality_plot)

        axarr[i].axvline(x=burnin, color='black', linewidth=1.5 * quality_plot)

    axarr[n_dim_mcmc - 1].tick_params(axis='x', labelsize=6 * quality_plot)
    axarr[n_dim_mcmc - 1].set_xlabel('Iterations', fontsize=10 * quality_plot)

    plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_chains.jpg'))
    plt.close()

    
########################################################
def make_corner_plot(params_mcmc_yaml):
    """ make corner plot reading the .h5 file from emcee

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file


    Returns:
        None
    """

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']
    labels = params_mcmc_yaml['LABELS']
    names = params_mcmc_yaml['NAMES']
    sigma = params_mcmc_yaml['sigma']
    nwalkers = params_mcmc_yaml['NWALKERS']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    name_h5 = file_prefix + '_backend_file_mcmc'

    band_name = params_mcmc_yaml['BAND_NAME']
    diskfit_mcmc.SPF_MODEL = params_mcmc_yaml['SPF_MODEL']  #Type of description for the SPF


    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    
    chain = reader.get_chain(discard=burnin, thin=thin)
    chain_flat = chains_to_params(chain, flatten = True)
    n_dim_mcmc = chain_flat.shape[1]


    for j in range(n_dim_mcmc):
        chain4thatparam = chain_flat[:, j]
        wherenotnan = np.where(~np.isnan(chain4thatparam))
        chainflatnonan = np.zeros((len(chain4thatparam[wherenotnan]),n_dim_mcmc))
        for i in range(n_dim_mcmc):
            chainflatnonan[:,i] = chain_flat[wherenotnan,i]
        chain_flat = chainflatnonan

    rcParams['axes.labelsize'] = 19
    rcParams['axes.titlesize'] = 14

    rcParams['xtick.labelsize'] = 13
    rcParams['ytick.labelsize'] = 13

    ### cumulative percentiles
    ### value at 50% is the center of the Normal law
    ### value at 50% - value at 15.9% is -1 sigma
    ### value at 84.1%% - value at 50% is 1 sigma
    if sigma == 1:
        quants = (0.159, 0.5, 0.841)
    if sigma == 2:
        quants = (0.023, 0.5, 0.977)
    if sigma == 3:
        quants = (0.001, 0.5, 0.999)

    #### Check truths = bests parameters


    shouldweplotalldatapoints = False
    
    
    labels_hash = [labels[names[i]] for i in range(n_dim_mcmc)]
    fig = corner.corner(chain_flat,
                        labels=labels_hash,
                        quantiles=quants,
                        show_titles=False,
                        plot_datapoints=shouldweplotalldatapoints,
                        verbose=False)



        green_line = mlines.Line2D([], [],
                                   color='red',
                                   label='True injected values')
        plt.legend(handles=[green_line],
                   loc='center right',
                   bbox_to_anchor=(0.5, 8),
                   fontsize=30)

        # Extract the axes
        axes = np.array(fig.axes).reshape((n_dim_mcmc, n_dim_mcmc))

        # Loop over the diagonal
        for i in range(n_dim_mcmc):
            ax = axes[i, i]
            ax.axvline(initial_values[i], color="r")

        # Loop over the histograms
        for yi in range(n_dim_mcmc):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(initial_values[xi], color="r")
                ax.axhline(initial_values[yi], color="r")

    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    fig.gca().annotate(band_name,
                       xy=(0.55, 0.99),
                       xycoords="figure fraction",
                       xytext=(-20, -10),
                       textcoords="offset points",
                       ha="center",
                       va="top",
                       fontsize=44)

    fig.gca().annotate("{0:,} iterations (+ {1:,} burn-in)".format(
                           reader.iteration - burnin, burnin),
                       xy=(0.55, 0.95),
                       xycoords="figure fraction",
                       xytext=(-20, -10),
                       textcoords="offset points",
                       ha="center",
                       va="top",
                       fontsize=44)

    fig.gca().annotate("with {0:,} walkers: {1:,} models".format(
        nwalkers, reader.iteration * nwalkers),
                       xy=(0.55, 0.91),
                       xycoords="figure fraction",
                       xytext=(-20, -10),
                       textcoords="offset points",
                       ha="center",
                       va="top",
                       fontsize=44)

    plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_pdfs.pdf'))
    plt.close()


########################################################
def create_header(params_mcmc_yaml):
    """ measure all the important parameters and exctract their error bars
        and print them and save them in a hdr file

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file


    Returns:
        header for all the fits
    """

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']

    comments = params_mcmc_yaml['COMMENTS']
    names = params_mcmc_yaml['NAMES']

    distance_star = params_mcmc_yaml['DISTANCE_STAR']

    sigma = params_mcmc_yaml['sigma']
    nwalkers = params_mcmc_yaml['NWALKERS']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    name_h5 = file_prefix + '_backend_file_mcmc'

    diskfit_mcmc.SPF_MODEL = params_mcmc_yaml['SPF_MODEL']  #Type of description for the SPF


    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    log_prob_samples_flat = reader.get_log_prob(discard=burnin,
                                                flat=True,
                                                thin=thin)

    chain = reader.get_chain(discard=burnin, thin=thin)
    chain_flat = chains_to_params(chain, flatten = True)

    n_dim_mcmc = chain_flat.shape[1]

    for j in range(n_dim_mcmc):
        chain4thatparam = chain_flat[:, j]
        wherenotnan = np.where(~np.isnan(chain4thatparam))
        chainflatnonan = np.zeros((len(chain4thatparam[wherenotnan]),n_dim_mcmc))
        for i in range(n_dim_mcmc):
            chainflatnonan[:,i] = chain_flat[wherenotnan,i]
        chain_flat = chainflatnonan
        log_prob_samples_flat = log_prob_samples_flat[wherenotnan]

    samples_dict = dict()
    comments_dict = comments
    MLval_mcmc_val_mcmc_err_dict = dict()

    for i, key in enumerate(names[:n_dim_mcmc]):
        samples_dict[key] = chain_flat[:, i]

    for i, key in enumerate(names[n_dim_mcmc:]):
        samples_dict[key] = chain_flat[:, i] * 0.

    # measure of 2 other parameters:  eccentricity and argument
    # of the perihelion
    for modeli in range(chain_flat.shape[0]):
        r1_here = samples_dict['R1'][modeli]
        dx_here = samples_dict['dx'][modeli]
        #dx_here = 0.
        dy_here = samples_dict['dy'][modeli]
        a = r1_here
        c = np.sqrt(dx_here**2 + dy_here**2)
        eccentricity = c / a
        samples_dict['ecc'][modeli] = eccentricity
        samples_dict['Argpe'][modeli] = np.degrees(np.arctan2(
            dx_here, dy_here))

        samples_dict['R1mas'][modeli] = convert.au_to_mas(
            r1_here, distance_star)

        # dAlpha, dDelta = offset_2_RA_dec(dx_here, dy_here, inc_here, pa_here,
        #                                  distance_star)

        # samples_dict['RA'][modeli] = dAlpha
        # samples_dict['Decl'][modeli] = dDelta

        # semimajoraxis = convert.au_to_mas(r1_here, distance_star)
        # ecc = np.sin(np.radians(inc_here))
        # semiminoraxis = semimajoraxis*np.sqrt(1- ecc**2)

        # samples_dict['Smaj'][modeli] = semimajoraxis
        # samples_dict['ecc'][modeli] = ecc
        # samples_dict['Smin'][modeli] = semiminoraxis

        # true_a, true_ecc, argperi, inc, longnode = kowalsky(
        #     semimajoraxis, ecc, pa_here, dAlpha, dDelta)

        # samples_dict['Rkowa'][modeli] = true_a
        # samples_dict['ekowa'][modeli] = true_ecc
        # samples_dict['ikowa'][modeli] = inc
        # samples_dict['Omega'][modeli] = longnode
        # samples_dict['Argpe'][modeAli] = argperi

    wheremin = np.where(log_prob_samples_flat == np.max(log_prob_samples_flat))
    wheremin0 = np.array(wheremin).flatten()[0]

    if sigma == 1:
        quants = [15.9, 50., 84.1]
    if sigma == 2:
        quants = [2.3, 50., 97.77]
    if sigma == 3:
        quants = [0.1, 50., 99.9]
    quants3 = [0.1,50.,99.9]

    for key in samples_dict.keys():
        MLval_mcmc_val_mcmc_err_dict[key] = np.zeros(6)

        percent = np.percentile(samples_dict[key], quants)
        percent3sig = np.percentile(samples_dict[key], quants3)

        MLval_mcmc_val_mcmc_err_dict[key][0] = samples_dict[key][wheremin0]
        MLval_mcmc_val_mcmc_err_dict[key][1] = percent[1]
        MLval_mcmc_val_mcmc_err_dict[key][2] = percent[0] - percent[1]
        MLval_mcmc_val_mcmc_err_dict[key][3] = percent[2] - percent[1]
        MLval_mcmc_val_mcmc_err_dict[key][4] = percent3sig[0] - percent3sig[1]
        MLval_mcmc_val_mcmc_err_dict[key][5] = percent3sig[2] - percent3sig[1]


    # MLval_mcmc_val_mcmc_err_dict['RAp'] = convert.mas_to_pix(
    #     MLval_mcmc_val_mcmc_err_dict['RA'], PIXSCALE_INS)
    # MLval_mcmc_val_mcmc_err_dict['Declp'] = convert.mas_to_pix(
    #     MLval_mcmc_val_mcmc_err_dict['Decl'], PIXSCALE_INS)

    # MLval_mcmc_val_mcmc_err_dict['R2mas'] = convert.au_to_mas(
    #     MLval_mcmc_val_mcmc_err_dict['R2'], distance_star)

    # print(" ")
    # for key in MLval_mcmc_val_mcmc_err_dict.keys():
    #     print(key +
    #           '_ML: {0:.3f}, MCMC {1:.3f}, -/+1sig: {2:.3f}/+{3:.3f}'.format(
    #               MLval_mcmc_val_mcmc_err_dict[key][0],
    #               MLval_mcmc_val_mcmc_err_dict[key][1],
    #               MLval_mcmc_val_mcmc_err_dict[key][2],
    #               MLval_mcmc_val_mcmc_err_dict[key][3]) + comments_dict[key])
    # print(" ")

    print(" ")
    if (diskfit_mcmc.SPF_MODEL  == 'hg_1g') or (diskfit_mcmc.SPF_MODEL  == 'hg_2g') or (
            diskfit_mcmc.SPF_MODEL  == 'hg_3g'):
        just_these_params = ['g1', 'g2', 'Alph1']
        for key in just_these_params:
            print(key + ' MCMC {0:.3f}, -/+1sig: {1:.3f}/+{2:.3f}'.format(
                MLval_mcmc_val_mcmc_err_dict[key][1],
                MLval_mcmc_val_mcmc_err_dict[key][2],
                MLval_mcmc_val_mcmc_err_dict[key][3]))
        print(" ")

    hdr = fits.Header()
    hdr['COMMENT'] = 'Best model of the MCMC reduction'
    hdr['COMMENT'] = 'PARAM_ML are the parameters producing the best LH'
    hdr['COMMENT'] = 'PARAM_MM are the parameters at the 50% percentile in the MCMC'
    hdr['COMMENT'] = 'PARAM_M and PARAM_P are the -/+ sigma error bars (16%, 84%)'
    hdr['KL_FILE'] = name_h5
    hdr['FITSDATE'] = str(datetime.now())
    hdr['BURNIN'] = burnin
    hdr['THIN'] = thin

    hdr['TOT_ITER'] = reader.iteration

    hdr['n_walker'] = nwalkers
    hdr['n_param'] = n_dim_mcmc

    hdr['MAX_LH'] = (np.max(log_prob_samples_flat),
                     'Max likelyhood, obtained for the ML parameters')

    for key in samples_dict.keys():
        hdr[key + '_ML'] = (MLval_mcmc_val_mcmc_err_dict[key][0],
                            comments_dict[key])
        hdr[key + '_MC'] = MLval_mcmc_val_mcmc_err_dict[key][1]
        hdr[key + '_M'] = MLval_mcmc_val_mcmc_err_dict[key][2]
        hdr[key + '_P'] = MLval_mcmc_val_mcmc_err_dict[key][3]
        hdr[key + '_3M'] = MLval_mcmc_val_mcmc_err_dict[key][4]
        hdr[key + '_3P'] = MLval_mcmc_val_mcmc_err_dict[key][5]

    return hdr

########################################################
def best_model_plot(params_mcmc_yaml, hdr):
    """ Make the best models plot and save fits of
        BestModel
        BestModel_Conv
        BestModel_FM
        BestModel_Res
        BestModel_SNRRes

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
        hdr: the header obtained from create_header

    Returns:
        None
    """

    # I am going to plot the model, I need to define some of the
    # global variables to do so

    # global ALIGNED_CENTER, PIXSCALE_INS, DISTANCE_STAR, WHEREMASK2GENERATEDISK, DIMENSION, SPF_MODEL

    diskfit_mcmc.DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    diskfit_mcmc.PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']

    quality_plot = params_mcmc_yaml['QUALITY_PLOT']
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    band_name = params_mcmc_yaml['BAND_NAME']
    name_h5 = file_prefix + '_backend_file_mcmc'

    numbasis = [params_mcmc_yaml['KLMODE_NUMBER']]

    diskfit_mcmc.ALIGNED_CENTER = params_mcmc_yaml['ALIGNED_CENTER']
    diskfit_mcmc.SPF_MODEL = params_mcmc_yaml['SPF_MODEL']  #Type of description for the SPF

    thin = params_mcmc_yaml['THIN']
    burnin = params_mcmc_yaml['BURNIN']

    logr2_init = np.log(params_mcmc_yaml['r2_init'])
    global r2_fix
    r2_fix = mt.exp(logr2_init)

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))
    chain_flat = reader.get_chain(discard=burnin, thin=thin, flat=True)
    log_prob_samples_flat = reader.get_log_prob(discard=burnin,
                                                flat=True,
                                                thin=thin)

    
    wheremin = np.where(log_prob_samples_flat == np.nanmax(log_prob_samples_flat))
    wheremin0 = np.array(wheremin).flatten()[0]
    print(chain_flat[wheremin0,4])
    theta_ml = chain_flat[wheremin0, :]
    print('THETA ML')
    print(theta_ml)
    #stop()

    if (diskfit_mcmc.SPF_MODEL == 'spf_fix'):

        n_points = 21  # odd number to ensure that scattangl=pi/2 is in the list for normalization
        scatt_angles = np.linspace(0, np.pi, n_points) 
        uniSPF = fits.getdata('/home/jrhom/debrisdisk_mcmc_fit_and_plot/spf_generic_new.fits') #Need to specify proper path for your SPF

        uniSPFdata = uniSPF

        diskfit_mcmc.F_SPF = phase_function_spline(np.radians(uniSPFdata[0]),uniSPFdata[1])


    psf = fits.getdata(os.path.join(klipdir, file_prefix + '_SmallPSF.fits'))

    mask2generatedisk = fits.getdata(
        os.path.join(klipdir, file_prefix + '_mask2generatedisk.fits'))


    mask2generatedisk[np.where(mask2generatedisk == 0.)] = np.nan
    diskfit_mcmc.WHEREMASK2GENERATEDISK = (mask2generatedisk != mask2generatedisk)

    #Load in minimization mask
    mask2minimize = fits.getdata(
        'masks/hr4796_mask2minimize.fits') #Load in your minimization mask

    mask2minimize[np.where(mask2minimize==0)] = np.nan
    
    instrument = params_mcmc_yaml['INSTRUMENT']

    # load the raw data (necessary to create the DiskFM obj)
    # this is the only part different for SPHERE and GPI

    if instrument == 'SPHERE':
         # only for SPHERE. 
        data_files_str = params_mcmc_yaml['DATA_FILES_STR']
        psf_files_str = params_mcmc_yaml['PSF_FILES_STR']
        angles_str = params_mcmc_yaml['ANGLES_STR']
        band_name = params_mcmc_yaml['BAND_NAME']

        dataset = SPHERE.Irdis(data_files_str, psf_files_str, angles_str, band_name, psf_cube_size=31)
        #collapse the data spectrally
        dataset.spectral_collapse(align_frames=True,
                                  aligned_center=diskfit_mcmc.ALIGNED_CENTER)
    elif instrument == 'GPI':

        #only for GPI
        filelist = sorted(glob.glob(os.path.join(DATADIR,
                                                 "*_distorcorr.fits")))

        # load the bad slices and bad files in the psf header
        hdr_psf = fits.getheader(
            os.path.join(klipdir, file_prefix + '_SmallPSF.fits'))

        # We can choose to remove completely from the correction
        # the angles where the disk intersect the disk (they are exlcuded
        # from the PSF measurement by defaut).
        # We can removed those if rm_file_disk_cross_satspots=True
        if params_mcmc_yaml['RM_FILE_DISK_CROSS_SATSPOTS']:

            excluded_files = []
            if hdr_psf['N_BADFIL'] > 0:
                for badfile_i in range(hdr_psf['N_BADFIL']):
                    excluded_files.append(hdr_psf['BADFIL' +
                                                  str(badfile_i).zfill(2)])

            for excluded_filesi in excluded_files:
                if excluded_filesi in filelist:
                    filelist.remove(excluded_filesi)

        # in IFS mode, we always exclude the slices with too much noise. We
        # chose the criteria as "SNR(mean of sat spot)< 3""
        excluded_slices = []
        if hdr_psf['N_BADSLI'] > 0:
            for badslice_i in range(hdr_psf['N_BADSLI']):
                excluded_slices.append(hdr_psf['BADSLI' +
                                               str(badslice_i).zfill(2)])

        # load the raw data without the bad slices
        dataset = GPI.GPIData(filelist, quiet=True, skipslices=excluded_slices)

        #collapse the data spectrally
        dataset.spectral_collapse(align_frames=True, numthreads=1)

    diskfit_mcmc.DIMENSION = dataset.input.shape[1]

    # load the data
    reduced_data = fits.getdata(
        os.path.join(klipdir, file_prefix + '-klipped-KLmodes-all.fits'))[
            0]  ### we take only the first KL mode



    # load the noise
    noise = fits.getdata(os.path.join(klipdir,
                                      file_prefix + '_noisemap.fits'))

    disk_ml = diskfit_mcmc.call_gen_disk(theta_ml)

    fits.writeto(os.path.join(mcmcresultdir, name_h5 + '_BestModel.fits'),
                 disk_ml,
                 header=hdr,
                 overwrite=True)

    # find the position of the pericenter in the model
    argpe = hdr['ARGPE_MC']
    pa = hdr['PA_MC']

    model_rot = np.clip(
        rotate(disk_ml, argpe + pa, mode='wrap', reshape=False), 0., None)

    argpe_direction = model_rot[int(diskfit_mcmc.ALIGNED_CENTER[0]):,
                                int(diskfit_mcmc.ALIGNED_CENTER[1])]
    radius_argpe = np.where(argpe_direction == np.nanmax(argpe_direction))[0]

    x_peri_true = radius_argpe * np.cos(
        np.radians(argpe + pa + 90))  # distance to star, in pixel
    y_peri_true = radius_argpe * np.sin(
        np.radians(argpe + pa + 90))  # distance to star, in pixel

    #convolve by the PSF
    disk_ml_convolved = convolve(disk_ml, psf, boundary='wrap')

    fits.writeto(os.path.join(mcmcresultdir, name_h5 + '_BestModel_Conv.fits'),
                 disk_ml_convolved,
                 header=hdr,
                 overwrite=True)

    # load the KL numbers
    diskobj = DiskFM(dataset.input.shape,
                     numbasis,
                     dataset,
                     disk_ml_convolved,
                     basis_filename=os.path.join(klipdir,
                                                 file_prefix + '_klbasis.h5'),
                     load_from_basis=True)

    #do the FM
    diskobj.update_disk(disk_ml_convolved)
    disk_ml_FM = diskobj.fm_parallelized()[0]
    ### we take only the first KL modemode

    fits.writeto(os.path.join(mcmcresultdir, name_h5 + '_BestModel_FM.fits'),
                 disk_ml_FM,
                 header=hdr,
                 overwrite=True)

    fits.writeto(os.path.join(mcmcresultdir, name_h5 + '_BestModel_Res.fits'),
                 reduced_data - disk_ml_FM,
                 header=hdr,
                 overwrite=True)

    #Mesaure the residuals
    residuals = reduced_data - disk_ml_FM
    snr_residuals = (reduced_data - disk_ml_FM) / noise

    fits.writeto(os.path.join(mcmcresultdir, name_h5 + '_BestModel_SNRRes.fits'),
                 snr_residuals,
                 header=hdr,
                 overwrite=True)

    #Set the colormap
    vmin = 0.3 * np.nanmin(reduced_data)
    vmax = 0.9 * np.nanmax(reduced_data)

    

    #Multiply by the likelihood mask for the best model image and residuals
    residuals = residuals*mask2minimize
    snr_residuals = snr_residuals*mask2minimize

    #Calculate the reduced chi square
    Chisquare = np.nansum(snr_residuals*snr_residuals)
    ndimMCMC = 10 #You must manually set your ndimMCMC specifically
    DOF = np.count_nonzero(~np.isnan(snr_residuals)) - ndimMCMC

    print('DOF',DOF)

    reducedChisquare = Chisquare/DOF

    print('Chisquare',Chisquare)

    print('Reduced Chisquare',reducedChisquare)



    dim_crop_image = int(4 * params_mcmc_yaml['OWA'] // 2) + 1

    disk_ml_crop = crop_center_odd(disk_ml, dim_crop_image)
    disk_ml_convolved_crop = crop_center_odd(disk_ml_convolved, dim_crop_image)
    disk_ml_FM_crop = crop_center_odd(disk_ml_FM, dim_crop_image)
    reduced_data_crop = crop_center_odd(reduced_data, dim_crop_image)
    nanindex = np.argwhere(np.isnan(reduced_data_crop))
    for i in nanindex:
        disk_ml_FM_crop[i[0]][i[1]] = np.nan
    residuals_crop = crop_center_odd(residuals, dim_crop_image)
    snr_residuals_crop = crop_center_odd(snr_residuals, dim_crop_image)
    mask2minimize_crop = crop_center_odd(mask2minimize, dim_crop_image)

    caracsize = 40 * quality_plot / 2.

    fig = plt.figure(figsize=(6.4 * 2 * quality_plot, 4.8 * 2 * quality_plot))
    #The data
    ax1 = fig.add_subplot(235)
    cax = plt.imshow(reduced_data_crop + 0.1,
                     origin='lower',
                     vmin=int(np.round(vmin)),
                     vmax=int(np.round(vmax)),
                     cmap=plt.cm.get_cmap('inferno'))

    #circle2 = plt.Circle((100, 100), 10, color='y', fill=False)
    #ax1.add_patch(circle2)
    ax1.contour(mask2minimize_crop,[0.5],colors=['yellow'])

    if file_prefix == 'Hband_hd48524_fake':
        ax1.set_title("Injected Disk (KLIP)",
                      fontsize=caracsize,
                      pad=caracsize / 3.)
    else:
        ax1.set_title("KLIP reduced data",
                      fontsize=caracsize,
                      pad=caracsize / 3.)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)
    
    #ax1.set_title("Residuals", fontsize=caracsize, pad=caracsize / 3.)
    #plt.axis('off')

    #The residuals
    ax1 = fig.add_subplot(233)
    cax = plt.imshow(residuals_crop,origin='lower',vmin=-5,vmax=5,
                     cmap=plt.cm.get_cmap('seismic')) 
    ax1.contour(mask2minimize_crop,[0.5],colors=['yellow'])
    ax1.set_title("Residuals", fontsize=caracsize, pad=caracsize / 3.)

    # make the colobar ticks integer only for gpi
    if instrument == 'SPHERE':
        tick_int = list(np.arange(int(np.round(vmax) // 2) + 1))
        tick_int_st = [str(i) for i in tick_int]
        cbar = fig.colorbar(cax, ticks=tick_int, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)
        cbar.ax.set_yticklabels(tick_int_st)
    else:
        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)
    #plt.axis('off')

    #The SNR of the residuals
    ax1 = fig.add_subplot(236)
    cax = plt.imshow(snr_residuals_crop,
                     origin='lower',
                     vmin=-5,
                     vmax=5, #Original is -2 to 2
                     cmap=plt.cm.get_cmap('seismic'))
    ax1.contour(mask2minimize_crop,[0.5],colors=['yellow'])
    ax1.set_title("SNR Residuals", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)
    #plt.axis('off')

    # The model
    ax1 = fig.add_subplot(231)
    vmax_model = int(np.round(np.nanmax(disk_ml_crop) / 1.5))
    if instrument == 'SPHERE':

        vmax_model = 433
    cax = plt.imshow(disk_ml_crop,
                     origin='lower',
                     vmin=-2,
                     vmax=vmax_model,
                     cmap=plt.cm.get_cmap('inferno'))
    ax1.set_title("Best Model", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)

    pos_argperi = plt.Circle(
        (x_peri_true + dim_crop_image // 2, y_peri_true + dim_crop_image // 2),
        3,
        color='g',
        alpha=0.8)
    pos_star = plt.Circle((dim_crop_image // 2, dim_crop_image // 2),
                          2,
                          color='r',
                          alpha=0.8)
    ax1.add_artist(pos_argperi)
    ax1.add_artist(pos_star)
    #plt.axis('off')

    rect = Rectangle((9.5, 9.5),
                     psf.shape[0],
                     psf.shape[1],
                     edgecolor='white',
                     facecolor='none',
                     linewidth=2)

    disk_ml_convolved_crop[10:10 + psf.shape[0],
                           10:10 + psf.shape[1]] = 2 * vmax * psf

    ax1 = fig.add_subplot(234)
    cax = plt.imshow(disk_ml_convolved_crop,
                     origin='lower',
                     vmin=int(np.round(vmin)),
                     vmax=int(np.round(vmax * 2)),
                     cmap=plt.cm.get_cmap('inferno'))
    ax1.add_patch(rect)

    ax1.set_title("Model Convolved", fontsize=caracsize, pad=caracsize / 3.)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)
    plt.axis('off')

    #The FM convolved model
    ax1 = fig.add_subplot(232)
    cax = plt.imshow(disk_ml_FM_crop,
                     origin='lower',
                     vmin=int(np.round(vmin)),
                     vmax=int(np.round(vmax)),
                     cmap=plt.cm.get_cmap('inferno'))
    ax1.contour(mask2minimize_crop,[0.5],colors=['yellow'])
    ax1.set_title("Model Convolved + FM",
                  fontsize=caracsize,
                  pad=caracsize / 3.)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=caracsize * 3 / 4.)
    #plt.axis('off')

    fig.subplots_adjust(hspace=-0.4, wspace=0.2)

    fig.suptitle(band_name + ': Best Model and Residuals',
                 fontsize=5 / 4. * caracsize,
                 y=0.985)

    fig.tight_layout()

    plt.savefig(os.path.join(mcmcresultdir, name_h5 + '_BestModel_Plot.jpg'))
    plt.close()

########################################################
def print_geometry_parameter(params_mcmc_yaml, hdr):
    """ Print some of the important values from the header to put in
        excel

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
        hdr: the header obtained from create_header

    Returns:
        None
    """

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    distance_star = params_mcmc_yaml['DISTANCE_STAR']

    name_h5 = file_prefix + '_backend_file_mcmc'

    reader = backends.HDFBackend(os.path.join(mcmcresultdir, name_h5 + '.h5'))

    f1 = open(
        os.path.join(mcmcresultdir, name_h5 + '_fit_geometrical_params.txt'),
        'w+')
    f1.write("\n'{0} / {1}".format(reader.iteration, reader.iteration * 192))
    f1.write("\n")

    to_print_str = 'R1'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    to_print = convert.au_to_mas(to_print, distance_star)
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'R2'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    to_print = convert.au_to_mas(to_print, distance_star)
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'PA'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'RA'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'Decl'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'dx'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    to_print = convert.au_to_mas(to_print, distance_star)
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'dy'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    to_print = convert.au_to_mas(to_print, distance_star)
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    f1.write("\n")
    f1.write("\n")

    to_print_str = 'Rkowa'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'eKOWA'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'ikowa'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'Omega'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    to_print_str = 'Argpe'
    to_print = [
        hdr[to_print_str + '_MC'], hdr[to_print_str + '_M'],
        hdr[to_print_str + '_P']
    ]
    f1.write("\n'{0:.3f} {1:.3f} +{2:.3f}".format(to_print[0], to_print[1],
                                                  to_print[2]))

    f1.close()


def measure_spf_errors(params_mcmc_yaml,
                       Number_rand_mcmc,
                       Norm_90_inplot=1.,
                       median_or_max='median',
                       save=False):
    """
    take a set of scatt angles and a set of HG parameter and return
    the log of a 2g HG SPF (usefull to fit from a set of points) Note: This functionality is untested in the updated model framework

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
        Number_rand_mcmc: number of randomnly selected psf we use to
                        plot the error bars
        Norm_90_inplot: the value at which you want to normalize the spf
                        in the plot ay 90 degree (to re-measure the error
                        bars properly)
        median_or_max: 'median' or 'max' use 50% percentile 
                        or maximum of likelyhood as "best model". default 'median'
        save: 

    Returns:
        a dic that contains the 'best_spf', 'errorbar_sup',
                                'errorbar_sup', 'errorbar'
    """

    dico_return = dict()

    burnin = params_mcmc_yaml['BURNIN']
    nwalkers = params_mcmc_yaml['NWALKERS']
    DATADIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    mcmcresultdir = os.path.join(DATADIR, 'results_MCMC')
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    diskfit_mcmc.SPF_MODEL  = params_mcmc_yaml['SPF_MODEL']  #Type of description for the SPF

    name_h5 = file_prefix + "_backend_file_mcmc"
    chain_name = os.path.join(mcmcresultdir, name_h5 + ".h5")
    reader = backends.HDFBackend(chain_name)

    min_scat = 90 - params_mcmc_yaml['inc_init']
    max_scat = 90 + params_mcmc_yaml['inc_init']

    scattered_angles = np.arange(np.round(max_scat - min_scat)) + np.round(
        np.min(min_scat))


    #we only exctract the last itearations, assuming it converged
    chain_flat = reader.get_chain(discard=burnin, flat=True)
 
    #if we use the argmax(chi2) as the 'best model' we need to find this maximum 
    if median_or_max == 'max':
        log_prob_samples_flat = reader.get_log_prob(discard=burnin,
                                                    flat=True)
        wheremax = np.where(log_prob_samples_flat == np.max(log_prob_samples_flat))
        wheremax0 = np.array(wheremax).flatten()[0]

    if (diskfit_mcmc.SPF_MODEL  == 'hg_1g'):
        norm_chain = np.exp(chain_flat[:, 7])
        g1_chain = chain_flat[:, 8]

        if median_or_max == 'median':
            bestmodel_Norm = np.percentile(norm_chain, 50)
            bestmodel_g1 = np.percentile(g1_chain, 50)
        elif median_or_max == 'max':
            bestmodel_Norm = norm_chain[wheremax0]
            bestmodel_g1 = g1_chain[wheremax0]

        Normalization = Norm_90_inplot

        if save == True:
            Normalization = bestmodel_Norm

        best_hg_mcmc = hg_1g(scattered_angles, bestmodel_g1, Normalization)

    elif diskfit_mcmc.SPF_MODEL  == 'hg_2g':
        norm_chain = np.exp(chain_flat[:, 7])
        g1_chain = chain_flat[:, 8]
        g2_chain = chain_flat[:, 9]
        alph1_chain = chain_flat[:, 10]

        if median_or_max == 'median':
            bestmodel_Norm = np.percentile(norm_chain, 50)
            bestmodel_g1 = np.percentile(g1_chain, 50)
            bestmodel_g2 = np.percentile(g2_chain, 50)
            bestmodel_alpha1 = np.percentile(alph1_chain, 50)

        elif median_or_max == 'max':
            bestmodel_Norm = norm_chain[wheremax0]
            bestmodel_g1 = g1_chain[wheremax0]
            bestmodel_g2 = g2_chain[wheremax0]
            bestmodel_alpha1 = alph1_chain[wheremax0]


        Normalization = Norm_90_inplot

        if save == True:
            Normalization = bestmodel_Norm

        best_hg_mcmc = hg_2g(scattered_angles, bestmodel_g1, bestmodel_g2,
                             bestmodel_alpha1, Normalization)

    elif diskfit_mcmc.SPF_MODEL == 'hg_3g':

        norm_chain = np.exp(chain_flat[:, 7])
        g1_chain = chain_flat[:, 8]
        g2_chain = chain_flat[:, 9]
        alph1_chain = chain_flat[:, 10]
        g3_chain = chain_flat[:, 11]
        alph2_chain = chain_flat[:, 12]


        if median_or_max == 'median':
            bestmodel_Norm = np.percentile(norm_chain, 50)
            bestmodel_g1 = np.percentile(g1_chain, 50)
            bestmodel_g2 = np.percentile(g2_chain, 50)
            bestmodel_alpha1 = np.percentile(alph1_chain, 50)
            bestmodel_g3 = np.percentile(g3_chain, 50)
            bestmodel_alpha2 = np.percentile(alph2_chain, 50)

        elif median_or_max == 'max':
            bestmodel_Norm = norm_chain[wheremax0]
            bestmodel_g1 = g1_chain[wheremax0]
            bestmodel_g2 = g2_chain[wheremax0]
            bestmodel_alpha1 = alph1_chain[wheremax0]
            bestmodel_g3 = g3_chain[wheremax0]
            bestmodel_alpha2 = alph2_chain[wheremax0]

        
        Normalization = Norm_90_inplot
        

        # we normalize the best model at 90 either by the value found
        # by the MCMC if we want to save or by the value in the
        # Norm_90_inplot if we want to plot

        Normalization = Norm_90_inplot
        if save == True:
            Normalization = bestmodel_Norm

        best_hg_mcmc = hg_3g(scattered_angles, bestmodel_g1, bestmodel_g2,
                             bestmodel_g3, bestmodel_alpha1, bestmodel_alpha2,
                             Normalization)

    dico_return['best_spf'] = best_hg_mcmc

    random_param_number = np.random.randint(1,
                                            len(g1_chain) - 1,
                                            Number_rand_mcmc)

    if (diskfit_mcmc.SPF_MODEL  == 'hg_1g') or (diskfit_mcmc.SPF_MODEL  == 'hg_2g') or (
            diskfit_mcmc.SPF_MODEL  == 'hg_3g'):
        g1_rand = g1_chain[random_param_number]
        norm_rand = norm_chain[random_param_number]

        if (diskfit_mcmc.SPF_MODEL  == 'hg_2g') or (diskfit_mcmc.SPF_MODEL  == 'hg_3g'):
            g2_rand = g2_chain[random_param_number]
            alph1_rand = alph1_chain[random_param_number]

            if diskfit_mcmc.SPF_MODEL  == 'hg_3g':
                g3_rand = g3_chain[random_param_number]
                alph2_rand = alph2_chain[random_param_number]

    hg_mcmc_rand = np.zeros((len(best_hg_mcmc), len(random_param_number)))

    errorbar_sup = scattered_angles * 0.
    errorbar_inf = scattered_angles * 0.
    errorbar = scattered_angles * 0.

    for num_model in range(Number_rand_mcmc):

        norm_here = norm_rand[num_model]

        # we normalize the random SPF at 90 either by the value of
        # the SPF by the MCMC if we want to save or around the
        # Norm_90_inplot if we want to plot

        Normalization = norm_here * Norm_90_inplot / bestmodel_Norm
        if save == True:
            Normalization = norm_here

        if (diskfit_mcmc.SPF_MODEL  == 'hg_1g'):
            g1_here = g1_rand[num_model]

        if (diskfit_mcmc.SPF_MODEL  == 'hg_2g'):
            g1_here = g1_rand[num_model]
            g2_here = g2_rand[num_model]
            alph1_here = alph1_rand[num_model]
            hg_mcmc_rand[:,
                         num_model] = hg_2g(scattered_angles, g1_here, g2_here,
                                            alph1_here, Normalization)

        if diskfit_mcmc.SPF_MODEL  == 'hg_3g':
            g3_here = g3_rand[num_model]
            alph2_here = alph2_rand[num_model]
            hg_mcmc_rand[:, num_model] = hg_3g(scattered_angles, g1_here,
                                               g2_here, g3_here, alph1_here,
                                               alph2_here, Normalization)

    for anglei in range(len(scattered_angles)):
        errorbar_sup[anglei] = np.max(hg_mcmc_rand[anglei, :])
        errorbar_inf[anglei] = np.min(hg_mcmc_rand[anglei, :])
        errorbar[anglei] = (np.max(hg_mcmc_rand[anglei, :]) -
                            np.min(hg_mcmc_rand[anglei, :])) / 2.

    dico_return['errorbar_sup'] = errorbar_sup
    dico_return['errorbar_inf'] = errorbar_inf
    dico_return['errorbar'] = errorbar

    dico_return['all_rando_spfs'] = hg_mcmc_rand

    dico_return['scattered_angles'] = scattered_angles

    return dico_return


def compare_injected_spfs_plot(params_mcmc_yaml):
    ####################################################################################
    ## injected spf plot. NOTE: This functionality is untested in the updated model framework
    ####################################################################################

    fill_or_all = 'all'
    Number_rand_mcmc = 50

    color0 = 'black'
    color1 = '#3B73FF'
    color2 = '#ED0052'
    color3 = '#00AF64'
    color4 = '#FFCF0B'
    
    band_name = params_mcmc_yaml['BAND_NAME']

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    name_pdf = file_prefix + '_comparison_spf.pdf'
    plt.figure()

    spf_fake_recovered = measure_spf_errors(params_mcmc_yaml, Number_rand_mcmc, median_or_max='max')

    scattered_angles = spf_fake_recovered['scattered_angles']

    injected_hg = hg_2g(scattered_angles, params_mcmc_yaml['g1_init'],
                        params_mcmc_yaml['g2_init'],
                        params_mcmc_yaml['alpha1_init'], 1.0)

    if fill_or_all== 'fill':
        plt.fill_between(scattered_angles,
                        spf_fake_recovered['errorbar_sup'],
                        spf_fake_recovered['errorbar_inf'],
                        facecolor=color3,
                        alpha=0.1)
    elif fill_or_all == 'all': 
        
        for num_model in range(Number_rand_mcmc): 
            plt.plot(scattered_angles,
                spf_fake_recovered['all_rando_spfs'][:,num_model],
                linewidth=1,
                color=color3,
                alpha=0.1
                )

    plt.plot(scattered_angles,
             spf_fake_recovered['best_spf'],
             linewidth=3,
             color=color3,
             label="SPF Recoreved After MCMC")

    plt.plot(scattered_angles,
             injected_hg,
             linewidth=2,
             linestyle='-.',
             color=color2,
             label="Fiducial 'Zodiacal light' SPF (Hong et al. 1985)")

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    plt.yscale('log')

    plt.ylim(bottom=0.3, top=30)
    plt.xlim(left=0, right=180)
    plt.xlabel('Scattering angles')
    plt.ylabel('Normalized total intensity')
    plt.title(band_name + ' SPF')

    plt.tight_layout()

    if "32297" in band_name:
        # for HD32297 add grey parts on the plots where the disk is hidden
        # behind the FP mask or for the back scattering part
        plt.axvspan(0, 7, alpha=0.2, facecolor='grey')
        plt.text(18, 0.4, 'Behind FPM', fontsize=10)
        plt.arrow(17,0.42, -8, 0,head_width=0.03,head_length=2, fc='k', ec='k')

        plt.axvspan(90, 180, alpha=0.2, facecolor='grey')
        plt.text(110, 0.4, 'HD 32297 Back Side', fontsize=10)

    plt.savefig(os.path.join(mcmcresultdir, name_pdf))

    plt.close()


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.simplefilter('ignore', NumbaWarning)

    if len(sys.argv) == 1:
        str_yalm = default_parameter_file
    else:
        str_yalm = sys.argv[1]

    with open(os.path.join('initialization_files', str_yalm),
              'r') as yaml_file:
        params_mcmc_yaml = yaml.load(yaml_file)

    params_mcmc_yaml['BAND_NAME'] = params_mcmc_yaml['BAND_NAME'] + ' (KL#: ' + str(params_mcmc_yaml['KLMODE_NUMBER']) +')'
    print(params_mcmc_yaml['BAND_NAME'])
    DATADIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    klipdir = os.path.join(DATADIR, 'klip_fm_files')
    mcmcresultdir = os.path.join(DATADIR, 'results_MCMC')
    
    

    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    name_h5 = file_prefix + '_backend_file_mcmc'

    if not os.path.isfile(os.path.join(mcmcresultdir, name_h5 + '.h5')):
        raise ValueError("the mcmc h5 file does not exist")

    # Plot the chain values
    make_chain_plot(params_mcmc_yaml)

    # compare SPF with injected
    # compare_injected_spfs_plot(params_mcmc_yaml)

    # # Plot the PDFs
    make_corner_plot(params_mcmc_yaml)
    
    # measure the best likelyhood model and excract MCMC errors
    hdr = create_header(params_mcmc_yaml)

    # save the fits, plot the model and residuals
    best_model_plot(params_mcmc_yaml, hdr)

    # print some of the best parameter values to put in excel/latex easily(not super clean)
    # print_geometry_parameter(params_mcmc_yaml, hdr)
