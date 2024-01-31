##### May 6, 2018 #####
# The goal of this effort is to re-generalize things. In particular, I want only one function, that will generate x disks, based on x scattering phase functions. 


# Analytic disk model
# This version is being developed to create a version that is 3D and can be summed along the line of sight instead of intergrated. 

##### Jan 31st, 2024 #####
#Notes from J. Hom: This modified version of the code retains the original functions utilized in addition to the modified functionality used in Hom et al. (2024)

import matplotlib.pyplot as plt
import numpy as np
import math as mt
from datetime import datetime
from numba import jit
from numba import vectorize,float64
from scipy.interpolate import interp1d
import scipy.ndimage.filters as snf
import copy
from scipy import stats

###################################################################################
####################### Some Built-In Scattering Functions ########################
###################################################################################
@jit
def hgg_phase_function(phi,g):
    #Inputs:
    # g - the g 
    # phi - the scattering angle in radians
    g = g[0]

    cos_phi = np.cos(phi)
    g2p1 = g**2 + 1
    gg = 2*g
    k = 1./(4*np.pi)*(1-g*g)
    return k/(g2p1 - (gg*cos_phi))**1.5

# This function will accept a vector of scattering angles, and a vector of scattering efficiencies 
# and then compute a cubic spline that fits through them all. 
@jit
def phase_function_spline(angles, efficiency):
    #Input arguments: 
    #     angles - in radians
    #     efficiencies - from 0 to 1
    return interp1d(angles, efficiency, kind='cubic')

@jit
def rayleigh(phi, args):
    #Using Rayleigh scattering (e.g. Eq 9 in Graham2007+)
    pmax = args[0]
    return pmax*np.sin(phi)**2/(1+np.cos(phi)**2)

@jit
def modified_rayleigh(phi, args):
    #Using a Rayleigh scattering function where the peak is shifted by args[1]
    pmax = args[0] #The maximum scattering phase function
    return pmax*np.sin(phi-np.pi/2+args[1])**2/(1+np.cos(phi-np.pi/2+args[1])**2)
##########################################################################################
############ Gen disk and integrand for a 1 scattering function disk #####################
##########################################################################################

@jit
def calculate_disk(xci,zpsi_dx,yy_dy2,x2,z2,x,zpci,xsi,a_r,R1,R2, beta_in, beta_out,scattering_function_list):
    '''
    # compute the brightness in each pixel
    # see analytic-disk.nb - originally by James R. Graham
    '''
    
    #The 'x' distance in the disk plane
    xx=(xci + zpsi_dx)

    #Distance in the disk plane
    # d1 = np.sqrt(yy_dy2 + np.square(xx))
    # d1 = np.sqrt(yy_dy2 + xx*xx)
    d1_2 = yy_dy2 + xx*xx
    d1 = np.sqrt(d1_2)

    #Total distance from the center 
    d2 = x2 + yy_dy2 + z2

    #The line of sight scattering angle
    cos_phi=x/np.sqrt(d2)
    phi = np.arccos(cos_phi)

    #The scale height exponent
    zz = (zpci - xsi)
    # hh = (a_r*d1)
    # expo = np.square(zz)/np.square(hh)
    # expo = (zz*zz)/(hh*hh)
    expo = (zz*zz)/(d1_2)
    # expo = zz/hh

    #The power law here has been moved from previous versions of anadisk so that we only calculate it once
    # int2 = np.exp(0.5*expo) / np.power((R1/d1),beta) 
    # int1 = np.piecewise(d1,[(d1 < R1),(d1 >=R1),(d1 > R2)],[(R1/d1)**-7,(R1/d1)**beta, 0.])
    # int1 = (R1/d1)**beta
    # int1 = np.piecewise(d1,[ d1 < R1, d1 >=R1, d1 > R2],[lambda d1:(R1/d1)**(-7.5),lambda d1:(R1/d1)**beta, 0.])
    int1 = (R1/d1)**beta_in
    int1[d1 >= R1] = (R1/d1[d1 >= R1])**beta_out
    int1[d1 > R2]  = 0.

    #Get rid of some problematic pixels
    d2_no = d2 == 0. 
    int1[d2_no] = 0.
    print(int1)

    int2 = np.exp( (0.5/a_r**2)*expo) / int1
    print(int2.shape)
    #stop()
    int3 = int2 * d2 
    #print(np.where(np.isinf(int2) == False))
    #print(len(int2))
    #print(len(d1))
    #np.savetxt('int2.txt',int2)
    #intreplace = np.copy(int2)
    #intreplace[np.where(np.isinf(intreplace)==True)] = 0.0
    #np.savetxt('intreplace.txt',intreplace)
    #print(np.where(np.isinf(intreplace)==False))
    #print(intreplace)
    #print(d1)
    #print(zz)
    plt.figure(1)
    plt.semilogy(d1,int2)
    plt.xlabel('Distance from Star [AU]')
    plt.ylabel('Dust Density')
    plt.xlim([50,120])

    #plt.figure(2)
    #plt.plot(zz,1./int2)
    #plt.xlabel('Height')
    #plt.ylabel('Dust Density')
    plt.show()
    stop()

    

    #This version is faster than the integration version because of the vectorized nature of the 
    # #scattering functions.
    # if sf1_args is not None:
    #     sf1 = scattering_function1(phi,sf1_args)
    # else: 
    #     sf1 = scattering_function1(phi)

    # if sf2_args is not None:
    #     sf2 = scattering_function2(phi,sf2_args)
    # else: 
    #     sf2 = scattering_function2(phi)

    out = []
    sfout = []

    for scattering_function in scattering_function_list:
        sf = scattering_function(phi)
        sfout.append(sf)
        out.append(sf/int3)
    sfout = np.array(sfout)
    #print(phi)
    #print(len(phi))
    #print(sfout)
    #print(len(sfout))
    #np.savetxt('hr4796_phi.txt',[phi,sfout[0]])
    #stop()
    
    out = np.array(out)
    # print(out.shape)
    # out = np.rollaxis(np.array(out),0,5)
    return out.T

@jit
def calculate_disk2(xci,zpsi_dx,yy_dy2,x2,z2,x,zpci,xsi,a_r,R1,RC, R2, beta_in, beta_out, gamvert, scattering_function_list):
    '''
    # compute the brightness in each pixel
    # see analytic-disk.nb - originally by James R. Graham
    #This version utilizes the Augereau et al. 1999 density function
    '''
    
    #The 'x' distance in the disk plane
    xx=(xci + zpsi_dx)


    #Distance in the disk plane

    d1_2 = yy_dy2 + xx*xx
    d1 = np.sqrt(d1_2)

    #Total distance from the center 
    d2 = x2 + yy_dy2 + z2

    #The line of sight scattering angle
    cos_phi=x/np.sqrt(d2)
    phi = np.arccos(cos_phi)


    #The scale height exponent
    zz = (zpci - xsi)
    expo = zz/(a_r*d1)

    #The power law here has been moved from previous versions of anadisk so that we only calculate it once
    int1 = (R1/d1)**beta_in

    #Denominator for Augereau's density profile
    term1 = (d1/RC)**(-2*beta_in)
    term2 = (d1/RC)**(-2*beta_out)
    denom = np.sqrt(term1 + term2)

    #Get rid of some problematic pixels
    d2_no = d2 == 0. 

    int2 = (np.exp(-expo**gamvert) / denom)


    int2[d2_no] = 0.

    #Need to move limits here!!!!!
    int2[d1 < R1] = 0. #CHECK HERE
    int2[d1 > R2]  = 0.



    int3 = d2/int2 #CHECK HERE

    #This version is faster than the integration version because of the vectorized nature of the 
    # #scattering functions.
    # if sf1_args is not None:
    #     sf1 = scattering_function1(phi,sf1_args)
    # else: 
    #     sf1 = scattering_function1(phi)

    # if sf2_args is not None:
    #     sf2 = scattering_function2(phi,sf2_args)
    # else: 
    #     sf2 = scattering_function2(phi)

    out = []

    for scattering_function in scattering_function_list:
        sf = scattering_function_list(phi)
        out.append(sf/int3)
    
    out = np.array(out)
    return out.T

@jit
def generate_disk2(scattering_function_list, scattering_function_args_list=None,
    R1=74.42, RC=82.45, beta_in=-7.5,beta_out=1.0, aspect_ratio=0.1, inc=76.49, pa=30, distance=72.8, 
    psfcenx=140,psfceny=140, sampling=1, mask=None, dx=0, dy=0., los_factor = 4, dim = 281.,pixscale=0.01414, R2=80.0, gamvert=2.0):
    '''
    #NOTE: This version utilizes the Augereau et al. 1999 density function
    Keyword Arguments:
    pixscale    -   The pixelscale to be used in "/pixel. Defaults to GPI's pixel scale (0.01414)
    dim         -   The final image will be dim/sampling x dim/sampling pixels. Defaults to GPI datacube size.

    '''

    #The number of input scattering phase functions and hence the number of disks to generate
    n_sf = len(scattering_function_list) 

    ###########################################
    ### Setup the initial coordinate system ###
    ###########################################
    #If you set sampling < 1 need to rebin to original grid size. The fraction should be set to 1/integer
    npts=int(np.floor(dim/sampling)) #The number of pixels to use in the final image directions
    npts_los = int(los_factor*npts) #The number of points along the line of sight 

    factor = (pixscale*distance)*sampling # A multiplicative factor determined by the sampling. Physical distance per pixel, controlled by the sampling. Used to be *sampling
    # In all of the following we only want to do calculations in part of the non-masked part of the array
    # So we need to replicate the mask along the line of sight.  
    #if mask is not None:
    #    mask = np.dstack([~mask]*npts_los)
    #else: 
    #    mask = np.ones([npts,npts])
    #    mask = np.dstack([~mask]*npts_los)


    #Set up the coordiname arrays
    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight 
    # +ve y is going right from the center
    # +ve z is going up from the center
    z = np.arange(npts)
    y = np.arange(npts)
    x = np.arange(npts_los)
    #print('Finished setting up indices')

    #Center the line-of-sight coordinates on the disk center. 

    ## THIS WAY DOESN'T WORK. IT CREATES INCONCISTENT RESULTS. 
    # x[mask] = x[mask]/(npts_los/(2*R2)) - R2 #We only need to calculate this where things aren't masked. 

    #THIS WAY IS A BIT SLOWER, BUT IT WORKS.
    #Here we'll try just a set pixel scale equal to the y/z pixel scale divided by the los_factor
    x = x.astype('float')
    #x[mask] = x[mask] - npts_los/2. #We only need to calculate this where things aren't masked.
    x = x - npts_los/2. #Not sure if this step is necessary here?
    x *= factor/los_factor
    #print('Finished x masking')

    #Setting up the output array
    #threeD_disk = np.zeros([npts,npts,npts_los,n_sf]) + np.nan
    #print('Finished setting up the output array')
    
    #####################################
    ### Set up the coordinate system ####
    #####################################

    #Inclination Calculations
    #print('Finished grid, starting coordinate system')
    incl = np.radians(90-inc)
    ci = mt.cos(incl) #Cosine of inclination
    si = mt.sin(incl) #Sine of inclination
    #Position angle calculations
    pa_rad=np.radians(90-pa) #The position angle in radians
    cos_pa=mt.cos(pa_rad) #Calculate these ahead of time
    sin_pa=mt.sin(pa_rad)
    a_r=aspect_ratio

    image = np.zeros([npts,npts]) + np.nan
    xci = x*ci
    xsi = x*si
    x2 = np.square(x)
    #print('Starting pixelization')
    for zc in z:
        for yc in y:
            
            yy = yc*(cos_pa*factor) - zc * (sin_pa*factor) - ((cos_pa*psfcenx*factor)-sin_pa*psfceny*factor)
            zz = yc*(sin_pa*factor) + zc * (cos_pa*factor) - ((cos_pa*psfceny*factor)+sin_pa*psfcenx*factor)
            z2 = np.square(zz)
            zpci = zz*ci
            zpsi = zz*si
            zpsi_dx = zpsi - dx
            yy_dy = yy-dy
            yy_dy2 = np.square(yy_dy)

            #Can keep x as an array
            #The 'x' distance in the disk plane
            xx=(xci + zpsi_dx)
            #Distance in the disk plane
            d1_2 = yy_dy2 + xx*xx
            d1 = np.sqrt(d1_2)

            zz = (zpci - xsi)
            expo = zz/(a_r*d1)
            #Radius conditions here
            #z conditions here
            indi = np.where((d1 >= R1) & (d1 <= R2) & (np.abs(zz)<(10*np.sqrt(a_r*d1*np.log(2)))))#some scale height: Where function if dealing with arrays
            if len(x[indi]) != 0:
                #Total distance from the center
                #print(x2[indi])

                d2 = x2[indi] + yy_dy2 + z2
                #print('d2 complete')
                term1 = (d1[indi]/RC)**(-2*beta_in)
                #print('term1 complete')
                term2 = (d1[indi]/RC)**(-2*beta_out)
                #print('term2 complete')
                denom = np.sqrt(term1 + term2)
                #print('denom complete')
                cos_phi=x[indi]/np.sqrt(d2)
                #print('cos_phi complete')
                phi = np.arccos(cos_phi)
    
                int2 = (np.exp(-expo[indi]**gamvert) / denom)#*int1
                #print('int2 complete')
                int3 = d2/int2

                #print('Density calculation complete')

                for scattering_function in scattering_function_list:
                    sf = scattering_function(phi)

                image[zc,yc] = np.sum(sf/int3)
            else:
                pass

    #print('Completed pixel calculations')

    return image

@jit
def generate_disk3(scattering_function_list, scattering_function_args_list=None,
    R1=74.42, RC=82.45, beta_in=-7.5,beta_out=1.0, aspect_ratio=0.1, inc=76.49, pa=30, distance=72.8, 
    psfcenx=140,psfceny=140, sampling=1, mask=None, dx=0, dy=0., los_factor = 4, dim = 281.,pixscale=0.01414, R2=80.0, gamvert=2.0):
    '''
    #NOTE: This is the function that works with the current model framework. It is slow, but great at handling large 3D arrays
    Keyword Arguments:
    pixscale    -   The pixelscale to be used in "/pixel. Defaults to GPI's pixel scale (0.01414)
    dim         -   The final image will be dim/sampling x dim/sampling pixels. Defaults to GPI datacube size.

    '''

    #The number of input scattering phase functions and hence the number of disks to generate
    n_sf = len(scattering_function_list) 

    ###########################################
    ### Setup the initial coordinate system ###
    ###########################################
    #If you set sampling < 1 need to rebin to original grid size. The fraction should be set to 1/integer
    npts=int(np.floor(dim/sampling)) #The number of pixels to use in the final image directions
    #print(npts)
    npts_los = int(los_factor*npts) #The number of points along the line of sight 

    factor = (pixscale*distance)*sampling # A multiplicative factor determined by the sampling. Physical distance per pixel, controlled by the sampling. Used to be *sampling
    z = np.arange(npts)
    

    y = np.arange(npts)
    coordlen = np.linspace(0,npts-1,npts)
    y,z = np.meshgrid(coordlen,coordlen)
    x = np.arange(npts_los)
    #finalimagearray = np.zeros((npts,npts))
    #print('Finished setting up indices')

    #Center the line-of-sight coordinates on the disk center. 

    ## THIS WAY DOESN'T WORK. IT CREATES INCONCISTENT RESULTS. 
    # x[mask] = x[mask]/(npts_los/(2*R2)) - R2 #We only need to calculate this where things aren't masked. 

    #THIS WAY IS A BIT SLOWER, BUT IT WORKS.
    #Here we'll try just a set pixel scale equal to the y/z pixel scale divided by the los_factor
    x = x.astype('float')
    #x[mask] = x[mask] - npts_los/2. #We only need to calculate this where things aren't masked.
    x = x - npts_los/2. #Not sure if this step is necessary here?
    x *= factor/los_factor
    #print('Finished x masking')
    
    #####################################
    ### Set up the coordinate system ####
    #####################################

    #Inclination Calculations
    #print('Finished grid, starting coordinate system')
    incl = np.radians(90-inc)
    ci = mt.cos(incl) #Cosine of inclination
    si = mt.sin(incl) #Sine of inclination
    #Position angle calculations
    #pa_rad=np.radians(90-pa) #The position angle in radians. Be wary of PA conventions
    pa_rad=np.radians(270-pa)
    cos_pa=mt.cos(pa_rad) #Calculate these ahead of time
    sin_pa=mt.sin(pa_rad)
    a_r=aspect_ratio
    image = np.zeros([npts,npts]) + np.nan
    xci = x*ci
    xsi = x*si
    x2 = np.square(x)
    yy = y*(cos_pa*factor) - z * (sin_pa*factor) - ((cos_pa*psfcenx*factor)-sin_pa*psfceny*factor)
    zz = y*(sin_pa*factor) + z * (cos_pa*factor) - ((cos_pa*psfceny*factor)+sin_pa*psfcenx*factor)
    z2 = np.square(zz)
    zpci = zz*ci
    zpsi = zz*si
    zpsi_dx = zpsi - dx
    yy_dy = yy-dy
    yy_dy2 = np.square(yy_dy)
    print('Starting pixelization')
    for xpos in np.arange(len(x)):
        xx =  (xci[xpos] + zpsi_dx)
        d1_2 = yy_dy2 + xx*xx
        d1 = np.sqrt(d1_2)

        zz = (zpci - xsi[xpos])
        expo = zz/(a_r*d1)

        indi = np.where((d1 >= R1) & (d1 <= R2) & (np.abs(zz)<(10*np.sqrt(a_r*d1*np.log(2)))))
        int3 = np.zeros([npts,npts])
        sf = np.zeros([npts,npts])

        if len(d1[indi]) != 0: # & len(zz[indi]) != 0:

            int3[indi] = (x2[xpos] + yy_dy2[indi] + z2[indi])/(np.exp(-expo[indi]**gamvert) / (np.sqrt((d1[indi]/RC)**(-2*beta_in)+(d1[indi]/RC)**(-2*beta_out))))

            for scattering_function in scattering_function_list:
                sf[indi] = scattering_function(np.arccos(x[xpos]/np.sqrt(x2[xpos] + yy_dy2[indi] + z2[indi])))

            image[indi] = np.nansum(np.array([image[indi],(sf[indi]/int3[indi])]),axis=0)





    



    #print('Completed pixel calculations')

    return image


@jit
def generate_disk(scattering_function_list, scattering_function_args_list=None,
    R1=74.42, RC=82.45, beta_in=-7.5,beta_out=1.0, aspect_ratio=0.1, inc=76.49, pa=30, distance=72.8, 
    psfcenx=140,psfceny=140, sampling=1, mask=None, dx=0, dy=0., los_factor = 4, dim = 281.,pixscale=0.01414, R2=80.0, gamvert=2.0):
    '''

    Keyword Arguments:
    pixscale    -   The pixelscale to be used in "/pixel. Defaults to GPI's pixel scale (0.01414)
    dim         -   The final image will be dim/sampling x dim/sampling pixels. Defaults to GPI datacube size.

    '''

    #The number of input scattering phase functions and hence the number of disks to generate
    n_sf = len(scattering_function_list) 

    ###########################################
    ### Setup the initial coordinate system ###
    ###########################################
    #If you set sampling < 1 need to rebin to original grid size. The fraction should be set to 1/integer
    npts=int(np.floor(dim/sampling)) #The number of pixels to use in the final image directions
    print(npts)
    npts_los = int(los_factor*npts) #The number of points along the line of sight 

    factor = (pixscale*distance)*sampling # A multiplicative factor determined by the sampling. Physical distance per pixel, controlled by the sampling. Used to be *sampling
    #print(factor)
    #print(pixscale)
    # In all of the following we only want to do calculations in part of the non-masked part of the array
    # So we need to replicate the mask along the line of sight.  
    if mask is not None:
        mask = np.dstack([~mask]*npts_los)
    else: 
        mask = np.ones([npts,npts])
        mask = np.dstack([~mask]*npts_los)


    #Set up the coordiname arrays
    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight 
    # +ve y is going right from the center
    # +ve z is going up from the center
    z,y,x = np.indices([npts,npts,npts_los])
    #print(len(z))

    #print('Finished setting up indices')

    #Center the line-of-sight coordinates on the disk center. 

    ## THIS WAY DOESN'T WORK. IT CREATES INCONCISTENT RESULTS. 
    # x[mask] = x[mask]/(npts_los/(2*R2)) - R2 #We only need to calculate this where things aren't masked. 

    #THIS WAY IS A BIT SLOWER, BUT IT WORKS.
    #Here we'll try just a set pixel scale equal to the y/z pixel scale divided by the los_factor
    x = x.astype('float')
    x[mask] = x[mask] - npts_los/2. #We only need to calculate this where things aren't masked. 
    x[mask] *=factor/los_factor
    #print('Finished x masking')

    #Setting up the output array
    threeD_disk = np.zeros([npts,npts,npts_los,n_sf]) + np.nan
    #print('Finished setting up the output array')
    
    #####################################
    ### Set up the coordinate system ####
    #####################################

    #Inclination Calculations
    #print('Finished grid, starting coordinate system')
    incl = np.radians(90-inc)
    ci = mt.cos(incl) #Cosine of inclination
    si = mt.sin(incl) #Sine of inclination

    # x*cosine i and x*sin i 
    xci = x[mask] * ci
    xsi = x[mask] * si
    print(len(xci))
    #print('XcosI and XsinI done')

    #Position angle calculations
    pa_rad=np.radians(90-pa) #The position angle in radians
    cos_pa=mt.cos(pa_rad) #Calculate these ahead of time
    sin_pa=mt.sin(pa_rad)
    a_r=aspect_ratio
    #print('Position angle calc done')

    # Rotate the coordinates in the image frame for the position angle
    # yy=y[mask]*(cos_pa*factor) - z[mask] * (sin_pa*factor) - ((cos_pa*npts/2*factor)-sin_pa*npts/2*factor) #Rotate the y coordinate by the PA
    # zz=y[mask]*(sin_pa*factor) + z[mask] * (cos_pa*factor) - ((cos_pa*npts/2*factor)+sin_pa*npts/2*factor) #Rotate the z coordinate by the PA

    yy=y[mask]*(cos_pa*factor) - z[mask] * (sin_pa*factor) - ((cos_pa*psfcenx*factor)-sin_pa*psfceny*factor) #Rotate the y coordinate by the PA
    zz=y[mask]*(sin_pa*factor) + z[mask] * (cos_pa*factor) - ((cos_pa*psfceny*factor)+sin_pa*psfcenx*factor) #Rotate the z coordinate by the PA
    print(yy.shape)
    print(yy)
    print(len(zz))
    #print('rotating frames')
    #del y
    #del z
    #The distance from the center in each coordiate squared
    #y2 = np.square(yy)
    z2 = np.square(zz)
    x2 = np.square(x[mask])

    #This rotates the coordinates in and out of the sky
    zpci=zz*ci #Rotate the z coordinate by the inclination. 
    zpsi=zz*si
    print('rotating in and out of sky')
    
    #Subtract the stellocentric offset
    zpsi_dx = zpsi - dx
    yy_dy = yy - dy
    #del zpsi
    #del yy
    #del ci
    #del si
    #del zz
    #del cos_pa
    #del sin_pa
    #del pa_rad
    #del incl

    #The distance from the stellocentric offset squared
    yy_dy2=np.square(yy_dy)

    # ########################################################
    # ### Now calculate the actual brightness in each bin ####
    # ######################################################## 

    #threeD_disk[:,:,:][mask] = calculate_disk(xci,zpsi_dx,yy_dy2,x2,z2,x[mask],zpci,xsi,aspect_ratio,R1,R2,beta_in,beta_out,scattering_function_list)
    #For Gaspard's density profile
    print('Finished coordinates, calculating the flux')
    threeD_disk[:,:,:][mask] = calculate_disk2(xci,zpsi_dx,yy_dy2,x2,z2,x[mask],zpci,xsi,aspect_ratio,R1,RC,R2,beta_in,beta_out,gamvert,scattering_function_list)
    #print('This is the one')
    #print(threeD_disk[:,:,:,0].shape)
    #Re-bin the disk here
    #print('Bin the data')
    #rebinned = np.reshape(threeD_disk[:,:,:,0], (281,2,281,2,281,2,1))
    #rebinnedcube = np.sum(rebinned,axis=(1,3,5))
    #print(rebinnedcube.ndim)
    #print(rebinnedcube.shape)
    #print(threeD_disk.ndim)
    #print(threeD_disk.shape)
    #for index in np.arange(281):
    #    print(len(rebinnedcube[:,index]))
    #print(np.where(len(rebinnedcube[:,:])<281))
    #print(np.where(rebinnedcube=float("inf")))

    #hist,binedges = np.histogramdd(np.array([threeD_disk[:,:,:,0],3]),bins=(281,281,281*los_factor),normed=False)
    #print(hist.shape)

    return np.sum(threeD_disk,axis=2)
    #return np.sum(rebinnedcube,axis=2)




########################################################################################
########################################################################################
########################################################################################
if __name__ == "__main__":

    sampling = 1

    #With two HG functions
    # sf1 = hgg_phase_function
    # sf1_args = [0.8]

    # sf2 = hgg_phase_function
    # sf2_args = [0.3]

    # im = gen_disk_dxdy_2disk(sf1, sf2,sf1_args=sf1_args, sf2_args=sf2_args, sampling=2)

    #With splines fit to HG function + rayleigh
    n_points = 20
    angles = np.linspace(0,np.pi,n_points)
    g = 0.8
    pmax = 0.3

    hg = hgg_phase_function(angles,[g])
    f = phase_function_spline(angles,hg)

    pol = hg*rayleigh(angles, [pmax])
    f_pol = phase_function_spline(angles,pol)

    y,x = np.indices([281,281])
    rads = np.sqrt((x-140)**2+(y-140)**2)
    mask = (rads > 90)

    start1 = datetime.now()
    im = generate_disk([f,f_pol],los_factor=1,mask=mask)
    end = datetime.now()
    print(end-start1)

    # f = lambda x: hgg_phase_function(x,[g])
    start1 = datetime.now()
    im = generate_disk([f,f_pol],los_factor=1,mask=mask)
    end = datetime.now()
    print(end-start1)

    im1 = im[:,:,0]
    im2 = im[:,:,1]
    # before_gauss = datetime.now()
    # #Testing 
    sigma = 1.3/sampling
    im1 = snf.gaussian_filter(im1,sigma)
    im2 = snf.gaussian_filter(im2,sigma)
    # print("Time to smooth with Gaussian: {}".format(datetime.now() - before_gauss))
    # #No smoothing
    # im1 = im[0]
    # im2 = im[1]

    fig = plt.figure()

    fig.add_subplot(121)
    plt.imshow(im1)
    fig.add_subplot(122)
    plt.imshow(im2)
    # plt.imshow(im)
    # plt.show()
# return x

