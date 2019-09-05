#In[]:
#Importing packages
import numpy as np 
import matplotlib.pyplot as plt 
import random as ran
import cartopy.crs as ccrs
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib import style
import collections
import emcee
import scipy.optimize as op
import datetime as dt
import time as tyme
import itertools
from netCDF4 import Dataset
import math
from astropy.time import Time
import cloud_killer_lib as ck_lib
import corner

# RETRIEVE DATA
data = Dataset("dscovr_single_light_timeseries.nc") # netCDF4 module used here
data.dimensions.keys()
radiance = data.variables["normalized"][:] # lightcurves for 10 wavelengths

# Constants used throughout
SOLAR_IRRAD_780 = 1.190 # Units: W m^-2 nm^-1

# Constant arrays used throughout
RAD_780 = radiance[9] # lightcurves for 780 nm
#time in seconds since June 13, 2015 00:00:00 UTC
TIME_SECS = radiance[10]
#time in days since June 13, 2015  00:00:00 UTC
TIME_DAYS = TIME_SECS/86148.0 #86148 = 23.93h

#longitude at SOP/SSP: convert UTC at SOP/SSP to longitude 
#longitude is 2pi at t=0 and decreases with time
SOP_LONGITUDE = [(2*np.pi-(t%86148.0)*(2*np.pi/86148.0))%(2*np.pi) for t in TIME_SECS]
#longitude in degrees rather than radians
#SOP_LONGITUDE_DEG = [l*180.0/np.pi for l in SOP_LONGITUDE]
SOP_LONGITUDE_DEG = np.rad2deg(SOP_LONGITUDE)

#In[]
#Loading EPIC data
def EPIC_data(day, plot=True):
    """
    Input: a date (int) after 13 June 2015 00:00:00, a boolean indicating 
    whether or not to plot the data
    Output: time, longitude (deg), apparent albedo, error on apparent albedo, 
    a bool indicating if dataset contains NaNs
    """
    # starting on the desired day
    n=0
    while (TIME_DAYS[n] < day):
        n += 1 # this n is where we want to start
    # EPIC takes data between 13.1 to 21.2 times per day
    # need to import 22 observations and then truncate to only one day
    t = TIME_DAYS[n:n+22]
    longitude = SOP_LONGITUDE_DEG[n:n+22]
    flux_rad = RAD_780[n:n+22] # Units: W m^-2 nm^-1
    
    # conversion to "reflectance" according to Jiang paper
    reflectance = flux_rad*np.pi/SOLAR_IRRAD_780 

    # truncate arrays to span one day only
    while ((t[-1] - t[0]) > 1.0):   # while t spans more than one day
        t = t[0:-1]                 # truncate arrays 
        longitude = longitude[0:-1]
        flux_rad = flux_rad[0:-1]
        reflectance = reflectance[0:-1]

    # error on reflectance
    reflectance_err = 0.02*reflectance # assuming 2% error     
    # add gaussian noise to the data with a variance of up to 2% mean reflectance
    gaussian_noise = np.random.normal(0, 0.02*np.mean(reflectance), len(reflectance))
    reflectance += gaussian_noise
    
    # check for nans in the reflectance data
    contains_nan = False 
    number_of_nans = 0
    for f in flux_rad:
        if math.isnan(f) == True:
            number_of_nans += 1
            contains_nan = True     
    #if contains_nan: # data not usable
    #    print("CAUTION: "+str(number_of_nans)+" points in this set are NaN")
       # return t, longitude, reflectance, reflectance_err, contains_nan
    
    # if we want to plot the raw data
    if (plot):
        # plotting reflectance over time
        fig = plt.figure()
        ax1 = fig.add_subplot(111)    
        ax1.errorbar((t - t[0])*24, reflectance, yerr=reflectance_err, fmt='.', 
                     markerfacecolor="cornflowerblue", 
                     markeredgecolor="cornflowerblue", color="black")
        ax1.set_ylabel("Apparent Albedo "+r"$A^*$", size=18)
        ax1.set_xlabel("T-minus 13 June 2015 00:00:00 UTC [Days]", size=18)
        title = r"EPIC data [$d$ = {}, $\phi_0$ = {}] ".format(date_after(day),np.round(np.deg2rad(longitude[0]),3))
        plt.title(title)
        plt.rcParams.update({'font.size':14})
        plt.show()

    return t, longitude, reflectance, reflectance_err, contains_nan

#In[]:
#Intializing the model
def timeToLongitude(time):
    """
    Function that converts time elapsed since an arbitrary start 
    of the simulation to longitude. 
    Input(s): 
        time: time elapsed in SECONDS
    Ouput(s):
        longitude: if input is a list, returns a list of longitudes
        else, returns a value. Both cases in DEGREES.
    """
    if not isinstance(time,list):
        longitude = np.rad2deg((2*np.pi-(time%86400.0)*(2*np.pi/86400.0)))
        return longitude
    longitude = [np.rad2deg((2*np.pi-(t%86400.0)*(2*np.pi/86400.0))) for t in time]
    return longitude

def initialPlanet(numOfSlices,plot=True):
    """
    Initializes surface map of an arbitrary planet with random values
    of albedos. The surface map will be divided into the given number
    of slices. 
    
    Input(s):
        numOfSlices: number of slices of the simulation
        plot (Default to TRUE): if TRUE, will plot a Eckert projection map of the generated map
    Output(s): 
        A dictionary of albedo and slice number. See github documentation for how the slices are defined.  
    """
    planet = {}
    sliceNum = range(numOfSlices)
    albedo = [0.9*ran.random() for i in range(numOfSlices)] 
    #albedo = [1 if i<numOfSlices/2 else 0 for i in range(numOfSlices)]
    #albedo = [0.34,0.67,0.89,0.12]
    albedo = [1,1,0,0]
    if plot:
        Eckert(albedo,len(albedo))
        plt.show()
    for i in sliceNum:
        planet[i] = albedo[i]
    return planet

def cloudCoverage(numOfSlices): 
    """
    Function that generates an initial cloud coverage over the slices. 
    Input(s):
        nmOfSlices: Number of slices
    Output(s): 
        Random values of cloud coverage ranging from 0 to 1. "1" being
        complete cloud cover and 0 having no clouds. 
    """
    #If boolean Slice is 1, it will generate a random value between 0 and 1
    #on that slice, else it wont generate cloud over that slice. 
    booleanSlice = [ran.randint(0,1) for i in range(numOfSlices)]
    clouds = np.zeros(numOfSlices)
    for i in range(numOfSlices):
        if booleanSlice[i]==1:
            clouds[i] = ran.random()
        else:
            clouds[i] = 0
    clouds = [0,0,0,0]
    return clouds

#In[]:
#Forward model
#NEED TO WORK ON THIS SHIT
def bounds(t,bounds,longitudes):
    "returns the longitudal bound slice for a given time and limit "
    Two = bounds[0]
    One = bounds[1]
    slices = len(longitudes)-1
    longitudes = list(longitudes)
    longitudes.extend([One,Two])
    longitudes = list(dict.fromkeys(sorted(longitudes)))
    pairLon = list(pairwise(iter(longitudes)))
    #If the region is including the middle
    if (One>Two):
        finalList = [x if (pairLon[i][0] < Two or pairLon[i][1]>One) else (0,0) for i,x in enumerate(pairLon)][::-1]
        while (len(finalList) !=slices):
            finalList.remove((0,0))
        if (len(finalList)!= slices):
            print (t,"here")
        return finalList
    #When the region is not including the middle
    elif (Two>One):
        finalList = [x if (pairLon[i][0] >= One and pairLon[i][1] <= Two) else (0,0) for i,x in enumerate(pairLon)][::-1]
        if (len(finalList)== slices+1):
            if Two>=np.pi and One>=np.pi:
                del finalList[0]
            else: 
                del finalList[-1]
        if (len(finalList)!=slices):
            del finalList[0]
            del finalList[-1]
        return finalList

def integral(time,VisAngles,w_Earth,phi_obs_0,longitudes):
    """
    Calculates the integral of cosine squared, which is analytically the 
    variable "integral" below. 
    Input(s): 
        time: a time array (in HOURS. If need to change to MINUTES OR SECONDS, 
              need to change the "w_Earth" variable.)
        VisAngles: The two East and West terminators given as a TUPLE. This is 
                   calculated from "visibleLong" function defined below.
        w_Earth: Angular frequency of the Planet (in units RAD/HOUR. Again needs
                 to be the same units as "time").
        phi_obs_0 (Default to 0): initial sub-observer point
        longitudes: Longitudes defined when slicing the Planet (in RAD).
    Output(s): 
        The integral result predicted by the forward model multiplied by 
        the 4/3*pi constant at a given time t. (See documentation for 
        derivation). 
    """
    phi_obs = phi_obs_0 - 1.0*w_Earth*time # SOP longitude at each time
    
    #Longitude bounds for each slice
    limits = [bounds(t,VisAngles[x],longitudes) for x, t in enumerate(time)]
    #print (limits)
    lenTime,slices,Z = np.shape(limits) #Just need the number of slices

    #Transposing the longitude bounds for a given time t such that the upper
    #bounds are in one element and the lower bound are in the other
    limits = np.transpose(limits,[2,0,1])
    C = (4/(3*np.pi))

    #Fixing the size of the SOP longitude at each time in order to get it to 
    #the same size as "limits".
    x = np.array([phi_obs,]*slices).transpose()
    #print (x)
    #The final integral given a time t.
    #print (limits[1]-limits[0],time)
    integral = ((1/2)*(np.asarray(limits[1]-limits[0]))+
        (1/4)*np.sin(2*limits[1]-2*x)-
        (1/4)*np.sin(2*limits[0]-2*x)) 
    return C*integral

def visibleLong(hour):
    """
    Computes the East and West terminators given the hour.
    Input(s):
        hour: hour elapsed in the simulation (just a value). 
    Output(s):
        the East and West terminators (as a TUPLE).
    """

    #The current longitude at time "hour"
    currentLon = np.deg2rad(timeToLongitude(hour*60*60)) 
    
    #The two terminators visible at time "hour". This is assuming half
    #the planet is visible at any given point in time 
    OneBound = currentLon + np.pi/2
    TwoBound = currentLon - np.pi/2
    
    #Special cases for the two bounds:
    #   If TwoBound is negative, just change that into the positive 
    #       equivalent
    #   Since the max is 2pi, if OneBound is above it, we restart 
    #       the longitude back to 0
    if TwoBound < 0:
        TwoBound = 2*np.pi + TwoBound

    if OneBound>2*np.pi and OneBound != 2*np.pi:
        OneBound = OneBound%(2*np.pi)
    
    return (OneBound,TwoBound)

def apparentAlbedo(albedos, time_days=1.0, long_frac=1.0, n=1000, phi_obs_0=0.0, 
               plot=False, alb=False): 
    """ 
    Input: an array of albedos, the time in days which the model should span
    (default: 1.0), the longitude as a fraction of 2pi which the model should 
    span (default: 1.0), the no. of points n to generate (default: 10000), 
    the initial sub-observer longitude (default: 0.0), a boolean indicating 
    whether or not to plot the lightcurve (default: False), and a boolean 
    indicating whether to return the reflectance or to apply the 
    multiplicative factor of 3/2 such that the lightcurve's units match those 
    of EPIC data.
    
    Computes the lightcurve A*(t) predicted by the forward model.
    
    Output: the lightcurve, in units of reflectance or apparent albedo 
    """
    #Gridlines generated depending on the number of slices
    longitudes = np.linspace(0,2*np.pi*long_frac,numOfSlices+1)

    #Time elapsed in hours  
    time = np.linspace(0, 24*time_days , n*time_days , False)
    w_Earth = 2.0*np.pi/24 # Earth's angular velocity in RAD/HOURS 
    
    #Calculates the extreme longitudes visible at each time 
    VisLongs = list(map(lambda t : visibleLong(t),time))

    #print (VisLongs)

    #Computes the kernel multiplied by the albedo
    kern = albedos*integral(time,VisLongs,w_Earth,phi_obs_0,longitudes)
    lightcurveR = sum(kern.T)
    if alb:
        lightcurveR *= 3/2
    
    #Plotting the result if the plot variable is TRUE.
    if plot:
        fig,ax = plt.subplots(1,1,gridspec_kw={'height_ratios':[1]},figsize=(10,8))
        ax.plot(time,lightcurveR,6,'.',color='red')
        if alb:
            ax.set_ylabel("Apparent Albedo $(A^*)$")
        else: 
            ax.set_ylabel("Reflectance")
        ax.set_xlabel("time (h)")
        plt.show()

    return time, lightcurveR

#NEED TO COMMENT ON THIS SHITTY FUNCTION
def Eckert(albedo,numOfSlices,nlats=400,nlons=400,fig=None,bar=None,
    plot=True,data=False):
    """
    General function to plot the Eckert map based off the number of 
    slices
    Input(s):
        albedo: takes in the albedo value at each longitudinal slices.
        numOfSlices: number of longitudinal slices.
        nlats, nlons = Used for the contour function. Basically, how 
                       many points to plot in the Eckert projection. 
                       (Default to 400, I do not recommend changing it!)
        fig (Default to None): This is for animation, since the animation function updates
             the fig, it needs the previous iteration. 
        bar (Default to None): The color bar on the side. Again for the animation to keep 
             the bar constant and not changing between frames.
        plot (Default to TRUE): if TRUE, plots the Eckert project, else
             returns longitudes, gridlines and the albedos. 
    Output(s): 
        EckertIV map projection of the results for a general number 
        of slices.
    """
    mod = nlons%numOfSlices 
    if (mod==0):
        interval = int(nlons/numOfSlices)
    else: 
        interval = int(nlons/numOfSlices)
        nlons=nlons-mod
        nlons = nlons-mod
    longitude = np.rad2deg(np.linspace(2*np.pi,0,nlons))
    lattitude = np.rad2deg(np.linspace(-np.pi/2,np.pi/2,nlats))
    lattitude, longitude = np.meshgrid(lattitude,longitude)
    w = 0
    A = []
    a_dumb = []
    gridlines=[360]
    for i in range(nlons):
        if (w == len(albedo)):
            break
        temp = [albedo[w] for j in range(nlats)]
        temp_dum = [np.random.randint(0,2) for j in range(nlats)]
        if i%interval==0 and i!=0:
            w=w+1
            gridlines.append(longitude[i][0])
        A.append(temp)
        a_dumb.append(temp_dum)
    gridlines.append(0)
    if plot == False:
        return longitude,lattitude,A,gridlines
    if type(fig)==type(None):
        fig = plt.figure(figsize=(12,6))
    else:
        fig = fig
    ax = fig.add_subplot(1,1,1,projection = ccrs.EckertIV())
    #if (len(A)!=nlats):
    #    for i in range(nlats-len(A)):
    #        A.append(A[len(A)-1])
    ax.clear()
    cs = ax.contourf(longitude,lattitude,A,transform=ccrs.PlateCarree(),
        cmap='gist_gray',alpha=0.3)
    
    if data: 
        return cs
    #SET MAX OF COLORBAR TO 1
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r'Albedo $A$')
    ax.coastlines()
    gl = ax.gridlines(crs=ccrs.PlateCarree(),color='black',linewidth=1.5, xlocs = gridlines) 
    gl.ylines = False
    ax.set_global()
    #plt.show()
    return longitude,lattitude,A,cs,fig,ax,gridlines

def effectiveAlbedo(numOfSlices,Acloud,plot=True,calClouds=None,calsurf=None):
    """
    Computes the effective albedo of a longitudinal slice with cloud 
    coverage taken into account. 
    Input(s):
        numOfSlices: number of longitudal slices
        Acloud: the average albedo of clouds (usually around 0.8)
        plot (Default to TRUE): if TRUE, will plot an Eckert projection
             of the effective albedo.  
        calClouds (Default to NONE): if None, will generate a cloud map 
        calsurf (Default to NONE): if None, will generate a surface map
        
        For both inputs above, if they are given, will use those instead 
        of generating new ones (both inputs as arrays)
    Output(s):
        An array of effective albedo.
    """
    #If no maps are given as an inputs, generate them. 
    if type(calClouds)==type(None):
        clouds = cloudCoverage(numOfSlices)
        surfAlb = initialPlanet(numOfSlices,False)
        
        #Function that describes the effective albedo given a surface albedo
        #and the cloud coverage over that slice. 
        effAlb = [clouds[i]*(1-(1-Acloud)*(1-surfAlb[i]))+(1-clouds[i])*surfAlb[i] for i in range(numOfSlices)]
        
        #Plotting the Eckert projection
        if plot: 
            Eckert(effAlb,numOfSlices)
            plt.show()
        return surfAlb,clouds,effAlb
    else: 
        effAlb = [calClouds[i]*(1-(1-Acloud)*(1-calsurf[i]))+(1-calClouds[i])*calsurf[i] for i in range(numOfSlices)] 
        return effAlb

# In[]:
#Cloud model & earth rotation
def dissipate(cloudIn,rate):
    """
    Function that dissipates the intensity of cloud (linearly, as an assumption) 
    with each time the function is called.
    Input(s):
        cloudIn: The input cloud array that will recursively get updated.  
        rate: rate at which the clouds dissipate over time. On average, clouds 
              take about 3 hours to dissipate, so the rate will be 1/3. This 
              of course depends on the type of clouds, composition,etc. but this 
              is a very simple model.
    Output(s): 
        None. Updates the given cloudIn array. 
    """  
    for i in range(len(cloudIn)):
        if cloudIn[i] <= 0:
            cloudIn[i] = 0 
        else:
            cloudIn[i] = cloudIn[i] - rate
            if cloudIn[i] <= 0:
                cloudIn[i] = 0

def form(cloudIn):
    """
    Forms clouds with each function call. The cloud generation process
    takes about from 1 minute to several hours. Therefore, as a temporary 
    phenomenon, only if the clouds in a slice have reached 0 will it then 
    form a random cloud coverage.
    Input(s):
        cloudIn: the cloud coverage array.
    Output(s):
        None. Updates the cloud array at each iteration. 
    """
    for i in range(len(cloudIn)):
        if cloudIn[i] <= 0.2:
            u = ran.randint(0,1)
            if u > 0.3:
                cloudIn[i] = cloudIn[i] + 0.8*ran.random()

def move(cloudIn,time,speed):
    """
    Function that moves the clouds. What this means in code is that it 
    will shift all the elements of the array to the left (first element 
    becomes last element), since clouds on average move from right
    to left (from East to West). See documentation for more details
    Input(s):
        cloudIn: the cloud array
        time: time elapsed since start of simulation
        speed: average speed of clouds.
    The last two inputs HAVE to be in the same units of time with units 
    of KM for distance. 
    Output(s):
        the cloud array.
    """
    disSlice = 40070/len(cloudIn)        #lenght of each slice 
    timeSlice = int(disSlice/speed)      #time it takes to traverse each slice 
    if timeSlice==0 or ((time%timeSlice) == 0 and time !=0):
        cloudIn = np.roll(cloudIn,1)
    return cloudIn

def dynamicMap(time,numOfSlices,Acloud,surfAlb,cloudIn,rateDiss,speedCloud,forming=True):
    """
    Dynamically changes the effective albedo everytime the function is called.
    Uses the previously defined functions in the "Cloud model & Earth rotation".
    First, it dissipates the clouds, then check if new ones can be formed. After, 
    it moves the clouds. 
    Input(s):
        time: time elapsed since the start of the simulation.
        numOfSlices: number of slices 
        Acloud: average albedo of clouds (usually around 0.8)
        surfAlb: the surface albedo array 
        cloudIn: the cloud coverage array
        rateDiss: rate of dissipation of clouds (should be the same units as 
                  the time array.)
        speedCloud: the speed of clouds (again should be the same units as the 
                  time array.)
        forming (Default to TRUE): a boolean variable. If TRUE, clouds will form. 
                else, no clouds will form.  
    Output(s):
        the effective albeod and cloud array.
    
    """
    dissipate(cloudIn,rateDiss)
    if (forming):
        form(cloudIn)
    cloudIn = move(cloudIn,time,speedCloud)
    effAlb=effectiveAlbedo(numOfSlices,Acloud,False,calClouds=cloudIn,calsurf=surfAlb)
    return effAlb,cloudIn

def rotate(albedo):
    """
    Same concept as the "move" function. However, will shift the array from left to 
    right instead. Since the planet rotates from West to East. This will shift and
    return the array everytime it is called. 
    Input(s):
        albedo: surface albedo array 
    Output(s):
        updated surface alebdo array with the shift.
    """
    albedo = np.roll(albedo,-1)
    return albedo 

def rotateEarth(w,albedo,numberOfSlices,t,Days):
    """
    Function that rotates the albedo array based on the angular velocity. 
    Inputs:
        w: angular velocity (RAD/h, RAD/m, RAD/S, in the same units as t. )
        albedo: surface albedo array 
        numberOfSlices: number of longitudinal slices 
        t: time (in HOURS, MINUTES OR SECONDS)
    Outputs: a shifted array of albedos 
    """
    #shift index by -1 since the Earth rotates from West to East 
    diff = 2*np.pi/numberOfSlices
    cond = diff/w
    
    time_array = np.linspace(cond,cond*numberOfSlices*Days,numberOfSlices*Days)

    if any(s == t  for s in time_array):
        albedo = np.roll(albedo,-1)
    return albedo

#In[]:
#Stats stuff
def lnlike(alpha, time,ref, ref_err,timespan,phispan):
    """
    Input: array of albedos A(phi) to feed into the forward model and the time, 
    lightcurve, and error on the lightcurve of the data being fit
    
    Feeds the albedos into the forward model and produces a model, compares 
    the model to the data, then assesses the likelihood of a set of 
    given observations. Likelihood assessed using chisq.
    
    Output: ln(likelihood)
    """
    
    # time/longitude spanned by forward model
    timepts = len(time) # no. of time points
    #timespan = (time[-1]-time[0]) # time spanned, in days
    #phispan = timespan # longitude spanned, as a fraction of 2pi
    
    # obtain model prediction, in units of apparent albedo
    model_time, model_ref = apparentAlbedo(alpha, timespan, phispan, timepts, 
                                       0, plot=False, alb=True) 
    
    # compute ln(likelihood)
    chisq_num = np.power(np.subtract(ref,model_ref), 2) # (data-model)**2
    chisq_denom = np.power(ref_err, 2) # (error)**2
    res = -0.5*sum(chisq_num/chisq_denom + np.log(2*np.pi) + np.log(np.power(
            ref_err,2))) #lnlike
    
    return res

def opt_lnlike(alpha, time, ref, ref_err,timespan,phispan):
    """
    Input: guesses for the fit parameters (alpha, an array of albedos, 
    representing A(phi)) and the time, lightcurve, and error on the lightcurve 
    of the data being fit 
    
    Maximizes the ln(likelihood).
    
    Output: The values of albedos with maximum likelihood
    """
    nll = lambda *args: -lnlike(*args) # return -lnlike of args
    # boundaries on the possible albedos:
    bound_alb = tuple((0.000001,0.999999) for i in range(len(alpha))) 
    # minimize (-ln(like)) to maximimize the likelihood 
    result = op.minimize(nll, alpha, args=(time,ref,ref_err,timespan,phispan), bounds=bound_alb)
    
    return result['x'] # the optimized parameters

def Aic(model,exp,npara):
    """
    Function that calculates the AIC.
    Inputs: 
        model: predicted data shape
        exp: given data with MCMC
        npara: Number of parameters in the model
    """
    res = model - exp
    sse = sum(res**2)
    return (2*npara-2*np.log(sse))

def Bic(model,exp,npara,ndata):
    """
    Function that calculates the BIC.
    Inputs: 
        model: predicted data shape
        exp: given data with MCMC
        npara: Number of parameters in the model
    """ 
    res = model - exp
    sse = sum(res**2)
    return (ndata*np.log(sse/ndata)+npara*np.log(ndata))

def lnprior(alpha):
    """
    Input: guesses for the fit parameters (alpha, an array of albedos,
    representing A(phi))
    Output: The ln(prior) for a given set of albedos 
    """
    if np.all(alpha>0.0) and np.all(alpha<1.0): # if valid albedos
        return 0.0
    return -np.inf # if not, probability goes to 0 

def lnpost(alpha, time, ref, ref_err,timespan,phispan):
    """
    Input: guesses for the fit parameters (alpha, an array of albedos,
    representing A(phi)) and the time, lightcurve, and error on the lightcurve 
    of the data being fit 
    Output: ln(posterior)
    """
    lp = lnprior(alpha)
    if not np.isfinite(lp): # if ln(prior) is -inf (prior->0) 
        return -np.inf      # then ln(post) is -inf too
    return lp + lnlike(alpha, time, ref, ref_err,timespan,phispan)

#In[]:
#MCMC Yo Boi's job
#knowing the turth and initialize the ball there and let them wander
#no for loops, for loops in fortan

def Lnprior(scale_t,scale_phi,movie):
    """
    Function that calculates the prior probability. The difference between 
    this function and lnprior() defined above is the fact that the smoothness 
    of the change of albedo in phi and time space is taken into account. 

    To do so, the following function is defined: 
    scale_phi*sum((difference of neighboring longitudinal albedos)**2)
        + scale_t*sum((difference of neighboring time albedos)**2)  
    
    Inputs:
        scale_t & scale_phi: "smoothness factor". 
        movie: a NxM matrix, where N represents the length of the time axis 
               and M is the lenght of the phi axis (the number of slices).   
    """
    N,M = np.shape(movie)
    diff_phi = 0
    diff_T = 0

    for i in range(N):
        for j in range(M):
            if j == 0:
                diff_phi = diff_phi + (movie[i][j] - movie[i][j+1])**2
            if j == M-1:
                diff_phi = diff_phi + (movie[i][j] - movie[i][j-1])**2
            else: 
                diff_phi = diff_phi + (movie[i][j] - movie[i][j-1])**2 + (movie[i][j] - movie[i][j+1])**2 
        if i == 0:
            diff_T = diff_T + (movie[i][j] - movie[i+1][j])**2
        if i == N-1:
            diff_T = diff_T + (movie[i][j] - movie[i-1][j])**2
        else:
            diff_T = diff_T + (movie[i][j] - movie[i-1][j])**2 + (movie[i][j] - movie[i+1][j])**2
    
    term_phi = scale_phi*diff_phi
    term_T = scale_t*diff_T
    final_term = term_phi + term_T
    return np.log(final_term)

def Lnpost(alpha,time,ref,ref_err,lon,timespan,phispan,s_T,s_phi):
    """
    Modified posterior probability. This is equal to the likelihood times the prior. 
    However, since we are taking the ln. It is just the addition of both ln of the terms
    """
    lp = Lnprior(s_T,s_phi,movie)
    return lp + lnlike(alpha,time,ref,ref_err,timespan,phispan)

#In[]:
#MCMC functions

def mcmc_results(chain, burnin):
    """
    Input: an emcee sampler chain and the steps taken during the burnin
    
    Averages the position of all walkers in each dimension of parameter space 
    to obtain the mean MCMC results 
    
    Output: an array representing the mean albedo map found via MCMC
    """
    ndims = len(chain[0][0]) # obtain no. of dimensions
    flat = flatten_chain(chain, burnin) # flattened chain, post-burnin
    
    mcmc_params = []
    for n in range(ndims): # for each dimension
        param_n_temp = []
        for w in range(len(flat)):
            param_n_temp.append(flat[w][n])
        mcmc_params.append(np.mean(param_n_temp)) # append the mean
    return mcmc_params

def init_walkers(alpha, time, ref, ref_err, ndim, nwalkers,timespan,phispan):
    """
    Input: guesses for the fit parameters (alpha, an array of albedos,
    representing A(phi)), the time, lightcurve, and error on the lightcurve of 
    the data being fit, the number of dimensions (i.e., albedo  slices to be 
    fit), and the number of walkers to initialize
    
    Initializes the walkers in albedo-space in a Gaussian "ball" centered 
    on the parameters which maximize the likelihood.
    
    Output: the initial positions of all walkers in the ndim-dimensional 
    parameter space
    """
    opt_albs = opt_lnlike(alpha, time, ref, ref_err,timespan,phispan) # mazimize likelihood
    # generate walkers in Gaussian ball
    pos = [opt_albs + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    return pos

def make_chain(nwalkers, nsteps, ndim, t,r,timespan,phispan,alb=True):
    """
    Input: the number of albedo slices (parameters) being fit, the number of 
    walkers, and the number of steps to take in the chain, and either the day
    of interest in the EPIC data or an array of artificial albedos 
    
    Runs MCMC on either EPIC data for the given day of interest to see if MCMC 
    can obtain the map A(phi) which produced the lightcurve, OR, runs MCMC with
    some artificial albedo map A(phi) to see if MCMC can recover the input map.
    
    Output: an emcee sampler object's chain
    """
    # if making chain for real EPIC data
    # if both a day and synthetic albedos are supplied, array is ignored 
    if alb: 
        t = np.asarray(t)
        r = np.asarray(r)
        r_err = 0.02*r # assuming 2% error     
        # add gaussian noise to the data with a variance of up to 2% mean app alb
        gaussian_noise = np.random.normal(0, 0.02*np.mean(r), len(r))
        r += gaussian_noise
    # if neither a day nor an articial albedo map is supplied
    else:
        print("Error: please supply either a day of interest in the EPIC data \
              or a synthetic array of albedo values.")
        return
    print ("Got my albedo, Thank you!")
    # guess: alb is 0.25 everywhere
    init_guess = np.asarray([0.25 for n in range(ndim)])
    # better guess: maximize the likelihood
    opt_params  = opt_lnlike(init_guess, t, r, r_err,timespan,phispan) 
    
    # initialize nwalkers in a gaussian ball centered on the opt_params
    print ("Intializing Walkers...")
    init_pos = init_walkers(opt_params, t, r, r_err, ndim, nwalkers,timespan,phispan)
    print ("Walkers initialized, ready for destruction!")

    # set up the sampler object and run MCMC 
    print ("Setting up chain")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(t, r, r_err,timespan,phispan))
    sampler.run_mcmc(init_pos, nsteps)
    print ("chain completed")
    return sampler.chain

def flatten_chain(chain, burnin):
    """
    Input: an emcee sampler chain and the steps taken during the burnin
    Output: a flattened chain, ignoring all steps pre-burnin
    """
    ndim = len(chain[0][0]) # number of params being fit 
    return chain[:,burnin:,:].reshape(-1, ndim)

def MCMC(nwalkers,nsteps,numOfSlices,time,app,lon,timespan,phispan,burning,hPerDay,chainArray,i,ax,plot):
    """
    """
    chain  = make_chain(nwalkers,nsteps,numOfSlices,time,app,timespan,phispan,alb=True)
    chainArray.append(chain)
    print ("Well call me a slave, because I just made some chains for day {}...".format(i))
    mean_mcmc_params = mcmc_results(chain,burning)
    mean_mcmc_time, mean_mcmc_ref = apparentAlbedo(mean_mcmc_params,time_days=timespan,
        long_frac=phispan,n=5000,plot=False,alb=True)
    print ("Got the mean MCMC results for day {}. Ya YEET!".format(i))

    flat = flatten_chain(chain,burning)
    sample_params = flat[np.random.randint(len(flat),size=nsamples)]
    for s in sample_params:
        sample_time,sample_ref = apparentAlbedo(s,time_days=timespan,long_frac=phispan,n=1000,plot=False,alb=True)
        sample_time = np.asarray(sample_time)
        mean_mcmc_params = np.asarray(mean_mcmc_params)
        plotting_x = np.asarray(sample_time)+(i-1)*hPerDay
        if (plot):
            ax.plot(plotting_x,sample_ref,color='k',alpha=0.1)
    if (plot):     
        ax.plot(plotting_x,sample_ref,color='k',alpha=0.1)
        ax.plot(mean_mcmc_time+(i-1)*hPerDay,mean_mcmc_ref,color='red',label="Mean MCMC")
        plt.show()
    return mean_mcmc_ref   

def plot_walkers_all(chain,expAlb=None):
    """
    Input: an emcee sampler chain
    
    Plots the paths of all walkers for all dimensions (parameters). Each 
    parameter is represented in its own subplot.
    
    Output: None
    """
    nsteps = len(chain[0]) # number of steps taken
    ndim = len(chain[0][0]) # number of params being fit
    step_number = [x for x in range(1, nsteps+1)] # steps taken as an array
    
    # plot the walkers' paths
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.1)
    for n in range(ndim):   # for each param
        paths = walker_paths_1dim(chain, n) # obtain paths for the param
        fig.add_subplot(ndim,1,n+1) # add a subplot for the param
        for p in paths:
            if n is not ndim-1:
                plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
            else:
                plt.xlabel("Steps")
            plt.ylabel(r"$A$"+"[%d]"%(n))
            plt.plot(step_number, p,color='k',alpha=0.3) # all walker paths
            if type(expAlb)!=type(None):
                plt.axhline(expAlb[n],color='red',linewidth=1) #Draw the expected value
            plt.xlim([0,nsteps])

def walker_paths_1dim(chain, dimension):
    """
    Input: an emcee sampler chain and the dimension (parameter, beginning 
    at 0 and ending at ndim-1) of interest
    
    Builds 2D array where each entry in the array represents a single walker 
    and each subarray contains the path taken by a particular walker in 
    parameter space. 
    
    Output: (nwalker x nsteps) 2D array of paths for each walker
    """
    
    ndim = len(chain[0][0])
    # if user asks for a dimension larger than the number of params we fit
    if (dimension >  (ndim-1)): 
        print("\nWarning: the input chain is only %d-dimensional. Please \
              input a number between 0 and %d. Exiting now."%(ndim,(ndim-1)))
        return
        
    nwalkers = len(chain)  # number of walkers
    nsteps = len(chain[0]) # number of steps taken

    # obtain the paths of all walkers for some dimension (parameter)
    walker_paths = []
    for n in range(nwalkers): # for each walker
        single_path = [chain[n][s][dimension] for s in range(nsteps)] # 1 path
        walker_paths.append(single_path) # append the path
    return walker_paths

def cornerplot(chain, burnin):
    """
    Input: an emcee sampler chain and the steps taken during the burnin
    
    Produces a corner plot for the fit parameters. 
    
    Output: None
    """
    ndim = len(chain[0][0]) # number of params being fit
    samples = flatten_chain(chain, burnin) # flattened chain, post-burnin
    
    label_albs = [] # setting the labels for the corner plot
    for n in range(ndim):
        label_albs.append(r"$A$"+"[%d]"%(n)) # A[0], A[1], ...
    
    plt.rcParams.update({'font.size':12}) # increased font size
    
    # include lines denoting the 16th, 50th (median) and 84th quantiles     
    corner.corner(samples, labels=label_albs, quantiles=(0.16, 0.5, 0.84), 
                  levels=(1-np.exp(-0.5),),show_titles=True,truth_color='#FFD43B')

#PROBLEM WITH 10 SLICE ALBEDO for dataAlbedoDynamic
#In[]:
#Versions of simulations: runShatellite --> VERSION 1: fits normally
#                         runShatelite  --> VERSION 2: "dynamic fit" (NOT DONE)
#                         runSatellowan --> VERSION 3: static map each day (NEEDS TWEAKING) 
#Need to take out for loops from this section. Making the code relatively slow. 
#Function below needs a bit of cleaning. 
def dataAlbedoDynamic(numOfSlices,Days,w,Acloud,surf,clouds,rateDiss,speedCloud,fastforward=1.0,
        Animation=True,hourlyChange=True,repeat=False,forming=True):
    """
    Function that simulates the satellite taking ndata amount of data points per day, depending on the surface 
    albedo and cloud coverage. It also simulates the Earth rotating and the clouds moving. It was assumed that 
    the clouds moved from East to West (so the cloud coverage array was getting shifted accordingly) and the
    planet itself is rotating West to East (the surface albedo is the one getting shifted).
    Input(s): 
        numOfSlices: Number of slices. 
        Days: How many days the satellites takes data for. 
        w: rotational angular velocity of the planet.
        Acloud (Default 0.8): The average albedo of clouds.
        surf: An array of surface albedo for each slice.
        clouds: An array of cloud coverage for each slice. 
        rateDiss: The rate of dissipation of clouds (units of hours, default 1/3)
        speedCloud: The average velocity of clouds in the simulation (in km/h, default 126)
        fastforward (Default 1.0): How sped up the model is. 
        Animation: If True, will display an animation of how the slices change.
    Output(s):
        Updated effective array with the shift and the dynamic mapping occuring.  
    """
    condition = False
    if (Animation):
        plt.ion()
        fig = plt.figure(1,figsize=(12,6))
        dum_alb = [np.random.randint(0,2) for i in range(numOfSlices)]
        dum_lon, dum_lat, dum_A, dum_grid = Eckert(dum_alb,numOfSlices,fig=fig,plot=False)
        css = plt.contourf(dum_lon,dum_lat,dum_A,cmap='gist_gray',alpha=0.3)
        condition = True

    hPerDay = int((w/(2*np.pi))**(-1))

    #longitudeF = np.linspace(2*np.pi*Days,0,(numOfSlices+1)*Days)
    #ndata = []

    #Don't know if the bottom two lines are necessary. Ill keep them for now.
    dum_alb = [np.random.randint(0,2) for i in range(numOfSlices)]
    dum_lon, dum_lat, dum_A, dum_grid = Eckert(dum_alb,numOfSlices,plot=False)
    eff = effectiveAlbedo(numOfSlices,Acloud,plot=False,calClouds=clouds,calsurf=surf)
    daysPassed = 0

    #Loop that loops through the 24 hour cycle (each i is 1 hour)

    #Change 22 for ndata
    #diff = int(hPerDay/(numOfSlices)) 
    #indices = np.linspace(0,diff*numOfSlices*Days,(numOfSlices*Days)+1)
    #indices = [i*diff for i in range(0,numOfSlices*Days)]
    #print (indices)
    #rotations = 0
    eff_final = []
    for i in range(hPerDay*Days): 
        #Calculates the days passed depending on how many hours per day.
        if (i%hPerDay == 0 and i!=0):
            daysPassed += 1        
        
        #Changes the map  
        if (hourlyChange):
            eff,clouds = dynamicMap(i,numOfSlices,Acloud,surf,clouds,rateDiss,speedCloud*fastforward,forming=forming)
        elif(i%hPerDay == 0 and hourlyChange == False and repeat==False):
            #sprint (clouds)
            eff,clouds = dynamicMap(i,numOfSlices,Acloud,surf,clouds,rateDiss,speedCloud*fastforward,forming=forming)
            eff_final.append(eff)
        #Rotates the earth 
        #if (hourlyChange and repeat==False):
        #    surf = rotateEarth(w,surf,numOfSlices,i,Days)
        #    rotations += 1
        #elif(hourlyChange==False and i%hPerDay==0 and i!=0):
        #    eff,clouds = rotate(surf) 
        #    eff = effectiveAlbedo(numOfSlices,Acloud,plot=False,calClouds=clouds,calsurf=surf)
        #    rotations += 1
        #else: 
        #    eff = rotateEarth(w,eff,numOfSlices,i,Days)
        #    rotations += 1
        
        #if (i%hPerDay == 0 or i%hPerDay == hPerDay or any(c == i%(hPerDay) for c in indices)):
            #longitudeF.append(360*(Days-daysPassed-1)+longitude[i%len(longitude)])
        #    ndata.append(eff[0])
        plt.clf()

        #If Animation is TRUE, it will draw an Eckert Projection animation. The plot
        #settings can be found starting from 482 to 488.
        if Animation:
            Eckert(eff,numOfSlices,fig=fig,bar=css,plot=condition)
            plt.pause(0.01)
    eff_final = np.asarray(eff_final).flatten()
    plt.ioff() 

    return eff_final
    #longitudeF,
    #,rotations

def extractN(time,apparent,n,Day):
    """
    Function that extracts N evenly spaced data from the apparent albedo array given. 
    It also makes sure that the first (0th hour) and the last (23th hour) is included.
    """
    limit = len(apparent)
    print (limit)
    diff = int(limit/(n*Day))
    indicesFinal = []
    for j in range(Day):
        indices = [i*diff+j*int((limit/Day)) for i in range(n) if i*diff<limit/Day]
        indicesFinal.append(indices)
    
    indicesFinal= np.asarray(indicesFinal).flatten()
    t = [time[i] for i in indicesFinal]
    a = [apparent[i] for i in indicesFinal]
    return t,a

#TEST this with the new MCMC
def runShatellite(numOfSlices,Acloud,rateDiss,speedCloud,w,ndata,fastFoward,Days,
        nwalkers,nsamples,nsteps,timespan,phispan,burning,plot=True,mcmc=True):
    
    #Maximum number of slices is hPerDayHours 
    hPerDay = int((w/(2*np.pi))**(-1))
    if (numOfSlices>hPerDay):
        print ("Cannot have more than 24 number of slices for now")
        return 0,0
    #Generate the initial condition of the planet
    surf = initialPlanet(numOfSlices,False)
    clouds = cloudCoverage(numOfSlices)
    finalTime = []
    apparentTime = []
    l,d=dataAlbedoDynamic(numOfSlices,Days,w,Acloud,surf,clouds,rateDiss,speedCloud,Animation=False)
        
    print ("Got sum fake data. YOHO, SCIENCE BITCH!")

    for i in range(1,Days+1):
        #Seperates the effective albedo and longitude per day.
        effective = d[(i-1)*numOfSlices:(i)*(numOfSlices)] 
        lon = l[(i-1)*numOfSlices:(i)*(numOfSlices)]
        #Calculates the apparent albedo with the forward model. 
        time, apparent = apparentAlbedo(effective,time_days=timespan,
                long_frac=phispan,n=5000,plot=False,alb=True)
        finalTime.append(time+(hPerDay*(i-1)))
        apparentTime.append(apparent)
        
    finalTime= np.asarray(finalTime).flatten()
    apparentTime = np.asarray(apparentTime).flatten()
    t,a = extractN(finalTime,apparentTime,ndata*Days)
    print ("Done extracting {}".format(numOfSlices))
    #Plotting
    if plot:
        fig,ax = plt.subplots(1,1,gridspec_kw={'height_ratios':[1]},figsize=(10,8))
        for i in range(Days+1):
            ax.axvline((i)*hPerDay,color='orange',alpha=1,zorder=10)
        ax.plot(finalTime,apparentTime,'-',color='black',linewidth=5,label="Simulated curve")
        ax.errorbar(t,a,fmt='.',color='green',yerr = np.asarray(a)*0.02,markersize=8,solid_capstyle='projecting', capsize=4,
                    label= "selected {} data".format(ndata))
        ax.set_xlabel("Time (h)",fontsize=22)
        ax.set_ylabel("Apparent Albedo ($A^*$)",fontsize = 22)
        ax.tick_params(labelsize=22)
        ax.legend(fontsize=15)
    chainArray = []
    alb = []
    if (mcmc):
        #Implement the MCMC running stuff in a seperate function
        for i in range(1,Days+1):
            time = t[(i-1)*ndata:i*ndata]
            app = a[(i-1)*ndata:i*ndata]
            #Maybe this is wrong, check this, fix this stuff
            lon = np.asarray(l[(i-1)*numOfSlices:(i)*(numOfSlices)])
            lon = [l%(360) for l in lon]
            lon[lon==0] = 360
            MCMC(nwalkers,nsteps,numOfSlices,time,app,lon,timespan,phispan,burning,hPerDay,chainArray,i,ax,plot)
            print ("done MCMC for day {}".format(i))
    for chain in chainArray:
        alb.append(mcmc_results(chain,burning))
    return alb

def runShatelite(numOfSlices,Acloud,rateDiss,speedCloud,w,ndata,fastFoward,Days,
        nwalkers,nsamples,nsteps,timespan,phispan,burning,plot=True,mcmc=True):
    #Need to make this a bit more efficient)
    if (numOfSlices>24):
        print ("Cannot have more than 24 number of slices for now")
        return 0,0
    surf = initialPlanet(numOfSlices,False)
    clouds = cloudCoverage(numOfSlices)
    hPerDay = int((w/(2*np.pi))**(-1))
    finalTime = []
    apparentTime = []
    ndim = numOfSlices
    l,d = dataAlbedoDynamic(numOfSlices,Days,w,Acloud,surf,clouds,rateDiss,speedCloud,Animation=False,hourlyChange=False)

def runSatellowan(numOfSlices,Acloud,npara,rateDiss,speedCloud,w,ndata,fastFoward,Days,
        nwalkers,nsamples,nsteps,timespan,phispan,burning,plot=True,mcmc=True,repeat=False,walkers=False,forming=True,Epic=None):
    #Need to make this a bit more efficient)
    ndim = numOfSlices
    hPerDay = int((w/(2*np.pi))**(-1))
    if plot: 
        fig,ax = plt.subplots(1,1,gridspec_kw={'height_ratios':[1]},figsize=(10,4))

    if type(Epic)==type(None):
        #if (numOfSlices>24):
        #    print ("Cannot have more than 24 number of slices for now")
        #    return 0,0
        surfDict = initialPlanet(numOfSlices,plot=True)
        surf = np.fromiter(surfDict.values(),dtype=float)
        print ("The planet's surface albedo is theoritically", surf)
        clouds = cloudCoverage(numOfSlices)
        print ("The effective albedo is ", effectiveAlbedo(numOfSlices,Acloud,plot=False,calClouds=clouds,calsurf=surf))
        finalTime = []
        apparentTime = []
        print (clouds, "Cloud coverage is ")
        d = dataAlbedoDynamic(numOfSlices,Days,w,Acloud,surf,clouds,rateDiss,speedCloud,Animation=False,hourlyChange=False,repeat=repeat,forming=forming)
        for i in range(1,Days+1):
            effective = d[(i-1)*numOfSlices:(i)*(numOfSlices)] 
            print("The albedo map for day {} is ".format(i-1), effective)
            #start = tyme.time()
            time, apparent = apparentAlbedo(effective,time_days=timespan,
                    long_frac=phispan,phi_obs_0=0,n=10000,plot=False,alb=True)
            #tim_comp, app_comp = ck_lib.lightcurve(effective,time_days=timespan,
            #        long_frac=phispan,phi_obs_0=0,n=10000,plot=False,alb=True)
            #end = tyme.time()
            #print (end-start)
            finalTime.append(time+(hPerDay*(i-1)))
            apparentTime.append(apparent)
            
        finalTime= np.asarray(finalTime).flatten()
        apparentTime = np.asarray(apparentTime).flatten()

        t,a = extractN(finalTime,apparentTime,ndata,Days)
        if plot: 
            fig,ax = plt.subplots(1,1,gridspec_kw={'height_ratios':[1]},figsize=(10,3))
            tim, albedo = drawAlbedo(d,Days,5000)
            ax.plot(finalTime,np.round(apparentTime,6),'.',color='black',linewidth=4,label="Simulated lightcurve")
            #ax.plot(tim_comp,app_comp,'.',color='orange',linewidth=4,label="Early Version of F.M.")
            #ax.plot(tim,albedo,'--',color='purple',label='Albedo Generated for {} slices'.format(numOfSlices),alpha=0.3)
            ax.set_xlabel("Time (h)",fontsize=22)
            ax.set_ylabel("Apparent Albedo ($A^*$)",fontsize = 22)

            #ax.legend(fontsize=17)
            ax.tick_params(labelsize=22)
    else:
        t = Epic[0]
        a = Epic[1]
        t = (t - t[0])*24 

    print ("Done extracting {}".format(numOfSlices))
    chainArray=[]
    alb=[]
    if plot:
        #ax.errorbar(t,a,fmt='.',color='blue',yerr = np.asarray(a)*0.02,markersize=10,solid_capstyle='projecting', capsize=4,
        #            label= "Raw Data from EPIC")
        ax.set_xlabel("Time (h)",fontsize=22)
        ax.set_ylabel("Apparent Albedo ($A^*$)",fontsize = 22)
        if type(Epic)!=type(None):
            title = r"EPIC data [$d$ = {}] ".format(date_after(Epic[2]))
            title = r"Forward model for {} slice albedo map".format(numOfSlices)
            ax.set_title(title,fontsize=22)
        ax.legend(fontsize=17)
        ax.tick_params(labelsize=22)    
    if (mcmc):
        #Implement the MCMC running stuff in a seperate function
        for i in range(1,Days+1):
            time = t[(i-1)*ndata:i*ndata]
            app = a[(i-1)*ndata:i*ndata]
            chain  = make_chain(nwalkers,nsteps,ndim,time,app,timespan,phispan,alb=True)
            chainArray.append(chain)
            print ("Well call me a slave, because I just made some chains for day {}...".format(i))
            mean_mcmc_params = mcmc_results(chain,burning)
            mean_mcmc_time, mean_mcmc_ref = apparentAlbedo(mean_mcmc_params,time_days=timespan,
                long_frac=phispan,n=10000,plot=False,alb=True)
            print ("Got the mean MCMC results for day {}. Ya YEET!".format(i))

            flat = flatten_chain(chain,burning)
            sample_params = flat[np.random.randint(len(flat),size=nsamples)]
            for s in sample_params:
                sample_time,sample_ref = apparentAlbedo(s,time_days=timespan,long_frac=phispan,n=10000,plot=False,alb=True)
                sample_time = np.asarray(sample_time)
                mean_mcmc_params = np.asarray(mean_mcmc_params)
                plotting_x = np.asarray(sample_time)+(i-1)*hPerDay
                if (plot):
                    ax.plot(plotting_x,sample_ref,color='k',alpha=0.1)
            if (plot):     
                ax.plot(plotting_x,sample_ref,color='k',alpha=0.1, label='50 samples from MCMC')
                ax.plot(mean_mcmc_time+(i-1)*hPerDay,mean_mcmc_ref,color='red',label="Mean MCMC")
                ax.legend(fontsize=15)

            #aic = Aic(app,mean_mcmc_ref,npara)
            #bic = Bic(app,mean_mcmc_ref,npara,ndata)
            print ("done MCMC for day {}".format(i))
    if plot:
        plt.show()

    counter = 0
    effmean = []
    #hour = dt.datetime.now().time()
    #fileName = hour.strftime("%H:%M:%S").replace(":","_",2)
    #f= open("MCMC_{}__{}.csv".format(fileName,numOfSlices),"w+") 
    #f.write(''.join(str(effectiveAlbedo(numOfSlices,Acloud,plot=False,calClouds=clouds,calsurf=surf)).replace("[","",1).replace("]","",1))+"\n")
    for chainy in chainArray:
        results = mcmc_results(chainy,burning)
        if (walkers):
            plot_walkers_all(chainy,expAlb=None)
            cornerplot(chainy,burning)
            plt.show()
        alb.append(results)
        print ("The result for day {} is".format(counter),results)
        #f.write(''.join(str(results).replace("[","",1).replace("]","",1))+"\n")
        effmean.append(np.mean(results))
        counter += 1
    #f.close()
    if (plot):
        x = np.arange(counter)
        mean = np.mean(effmean)
        error = np.std(effmean)/np.sqrt(len(effmean))
        fig,ax = plt.subplots(1,1,gridspec_kw={'height_ratios':[1]},figsize=(10,8))
        ax.plot(x,effmean,'.',markersize=8,color='purple')
        ax.axhline(np.mean(effectiveAlbedo(numOfSlices,Acloud,plot=False,calClouds=clouds,calsurf=surf)),linewidth=8,color='orange',label='Expected Mean')
        ax.axhline(mean+error,linewidth=8,color='purple',alpha=0.3)
        ax.axhline(mean-error,linewidth=8,color='purple',alpha=0.3)
        ax.axhline(mean,linewidth=8,color='purple',label='Found Mean')
        ax.set_xlabel("Day",fontsize=22)
        ax.set_ylabel("Average albedo from MCMC",fontsize=22)
        ax.legend(fontsize=15)
        plt.show()
    minAlb = minimumAlb(alb)
    return alb,minAlb,surf

#I might not need this function
def fixAlbedo(albedos,rotations):
    """
    """
    N,M  = np.shape(albedos)
    r = rotations
    for i in range(N):
        albedos[i] = np.ndarray.tolist(np.roll(albedos[i],r))
        print ("The fixed result for day {} is".format(i), albedos[i])

        r -= 1
    return albedos

def drawAlbedo(albedo,Days,numdata):
    """
    Function that outputs the rate of change of albedo as a function of time
    """
    slices = len(albedo)
    hPerDay = 24*Days
    time = hPerDay/slices 
    x = np.linspace(0,24*Days,numdata)
    albhour= []
    cond = True
    i = 1
    j = 0 
    indices = [i*time for i in range(slices)]

    while (cond==True):
        albhour.append(albedo[i-1])
        j += 1
        if (j==numdata):
            break
        if ((i)==len(albedo)):
            if (not (indices[i-1]<x[j])):
                i+=1 
        else:
            if (not (indices[i-1]<x[j]<indices[i])):
                i+=1 
    
    final = np.asarray(albhour)*Days

    return x,final

def minimumAlb(albedos):
    """
    Function that returns the minimum albedo of each slice 
    """
    try:
        nslice = len(albedos[0])
    except:
        return 
    albedoSlice = np.transpose(albedos)
    minimum = [min(albedoSlice[i]) for i in range(nslice)]
    return minimum

#In[]
#Utilities
def roll(Dict,shift):
    slices = np.fromiter(Dict.keys(), dtype=int)
    albedo = np.fromiter(Dict.values(),dtype=float)
    albedo = np.roll(albedo,shift)
    slices = np.roll(slices,shift)
    Dict.clear()
    for i in slices:
        Dict[i] = albedo[i]
    return Dict

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def date_after(d):
    """
    Input: an integer d
    
    Quickly find out the actual calendar date of some day in the EPIC dataset. 
    
    Output: the date, d days after 2015-06-13 00:00:00.000
    """
    
    t_i = Time("2015-06-13", format='iso', scale='utc')  # make a time object
    t_new_MJD = t_i.mjd + d # compute the Modified Julian Day (MJD) of the new date
    t_new = Time(t_new_MJD, format='mjd') # make a new time object
    t_new_iso = t_new.iso # extract the ISO (YY:MM:DD HH:MM:SS) of the new date
    t_new_iso = t_new_iso.replace(" 00:00:00.000", "") # truncate after DD
    return t_new_iso

#In[]:
#Main method
if __name__ == "__main__":
    #VARY CLOUDS VERY SLOWLY such that the time map is not varying that much.
    #SEE IF I CAN EXTRACT THE TIME CHANGE dissipation. HOW I CAN FIT FOR THAT
    #Add a verbose option
    #A lot of parameters eh 
    numOfSlices = 4                         #Number of slices 
    Acloud = 0.8                            #The average albedo of clouds
    #surf = initialPlanet(numOfSlices,False)#Surface albedo of Planet
    
    #clouds = cloudCoverage(numOfSlices)    #Cloud coverage for each slice 
    #NEED TO FIT FOR THIS
    rateDiss = 1/3                          #Average Rate of dissipation of clouds in hours 
    speedCloud = 126                        #Speed at which clouds move. I put it real high for effects for now (km/h)
    w = 2*np.pi/(24)                        #Angular frequency of the Earth (rad/h^(-1))
    ndim = numOfSlices                      #Number of dimensions for MCMC
    nsamples = 25                           #Number of samples to graph 
    nwalkers = 100                          #Number of walkers for MCMC
    #nwalkers = [100,200,300,400,500,600,700,800,900,1000]
    #nwalkers = [1500]
    ntrials = 1
    nsteps = [500]                         #Number of steps for MCMC
    fastForward = 1                        #Speed of how much the dynamic cloud
    Days = 1                              #Number of days that the model spans 
    timespan = 1                           #Time span of data (default of 1.0)
    phispan = 1                            #Fraction of 2pi for the longitude (1 for a full rotation)
    burnin = 150                           #Burning period
    McMc = False                           #Run MCMC on the simulation or not
    PlOt = True                            #Whether to plot things or not
    ndata = 22                             #Nnumber of data taken by the satellite
    npara = numOfSlices
    repeat = False
    cloudForming = True
    walkers = False
    
    daySim = 790
    #t, longitude, reflectance, reflectance_err, contains_nan = EPIC_data(daySim,plot=True)
    #Eckert([0.23616174792007924, 0.21081179505525094, 0.24803993899567978, 0.15060034674753922, 0.27145193081592317, 0.26167097423359004],6)
    #plt.show()
    batch1 = np.arange(77,101)
    batch2 = np.arange(365,376)
    batch3 = np.arange(407,435)
    batch4 = np.arange(708,726)
    
    #Data = [EPIC_data(i,plot=False) for i in range(861)]
    #Data = np.transpose(np.asarray(Data))
    #lengths = [len(Data[0][i]) for i in range(861)]
    #lengths = np.asarray(lengths)
    #good_indices = [i for i,l in enumerate(lengths) if l >20]
    #print (good_indices)

    #Testing version 1 of the satellite
    """
    #Running the simulation, extract albedo from MCMC
    alb =  runShatellite(numOfSlices,Acloud,rateDiss,speedCloud,w,ndata,fastForward,Days,nwalkers,nsamples,nsteps,timespan,phispan,burnin,plot=True,mcmc=McMc) 
    #res = minimumAlb(alb)
    print ("For each day, albedos from MCMC are",alb)
    #print ("The surface albedo from the minimum function is", res)
    print ("The effective albedo is ", effectiveAlbedo(numOfSlices,Acloud,plot=False,calClouds=clouds,calsurf=surf))
    #print ("The difference is", np.asarray(surf)-np.asarray(res))
    #Need to implement albedo minimum now
    """
    
    #dumAlb,aic,bic = runSatellowan(numOfSlices,Acloud,npara,rateDiss,speedCloud,w,ndata,fastForward,Days,nwalkers,nsamples,nsteps,timespan,phispan,burnin,plot=False,mcmc=True,)
    #res = minimumAlb(dumAlb)
    #print ("For each day, albedos from MCMC are",dumAlb)
    #print ("The surface albedo from the minimum function is", res)
    
    #print ("The difference is", np.asarray(surf)-np.asarray(res))
    #Testing version 3 of the satellite 
    #Takes about 6 minutes and 30sec for one trial. 
    #AT HOME: fix plots, save them as pdfs instead of showing them. 
    #print aic and bic along with the minimum value to show if it improves the fit or not 
    #do the very slow varying cloud map and see if i can fit the dissipation rate. 
    #fig, ax = plt.subplots(1,1,gridspec_kw={'height_ratios':[1]},figsize=(10,8))
    #[t,reflectance,daySim,longitude]
    #[t,reflectance,daySim,longitude]
    for steps in nsteps:
        aics = []
        bics = []
        for j in range(ntrials):
            dumAlb,dumMin,eff= runSatellowan(numOfSlices,Acloud,npara,rateDiss,speedCloud,w,ndata,fastForward,Days,
                    nwalkers,nsamples,steps,timespan,phispan,burnin,plot=PlOt,mcmc=McMc,repeat=repeat,walkers=walkers,forming=cloudForming,Epic=None)
    print ("The surface albedo from MCMC is " ,dumMin," compared to the inputted map of ", eff)
    """     
            #,aic,bic #print (aic,bic)
            #aics.append(aic)
            #bics.append(bic)
     
        #np.savetxt('aic_with_{}_steps.csv'.format(steps),aics,delimiter=',')
        #np.savetxt('bic_with_{}_steps.csv'.format(steps),bics,delimiter=',')
        #ax.hist(aics,bins=20,alpha=0.7,label='AIC values with {} walkers'.format(walk))
        #ax.hist(bics,bins=20,alpha=0.7,label='BIC values with {} walkers'.format(walk))
    #ax.legend()
    
    #plt.show()
    time, albedo =drawAlbedo([0.2,0.3,0.4,0.5,0.3,0.3,0.4,0.2],w,50)
    print (albedo)
    plt.plot(time,albedo,'.')
    plt.show()
    """
