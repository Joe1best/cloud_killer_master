import numpy as np 
import matplotlib.pyplot as plt 
import random as ran 
import cartopy.crs as ccrs
import init as var
import Model_Init as M_init


#Model initialization
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
    return clouds


#In[]
#Forward model defined in the paper 
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
    currentLon = np.deg2rad(M_init.timeToLongitude(hour*60*60)) 
    
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
    longitudes = np.linspace(0,2*np.pi*long_frac,var.NUMOFSLICES+1)

    #Time elapsed in hours  
    time = np.linspace(0, 24*time_days , n*time_days , False)
    w_Earth = var.WW # Earth's angular velocity in RAD/HOURS 
    
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

def effectiveAlbedo(numOfSlices,Acloud,plot=True,calClouds=None,calsurf=None):
    """
    Computes the effective albedo of a longitudinal slice with cloud 
    coverage taken into account. 
    Input(s):
        numOfSlices: number of longitudal slices
        Acloud: the average albedo of clouds (usually around 0.8)
        plot (Default to TRUE): if TRUE, will plot an M_init.Eckertprojection
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
        clouds = M_init.cloudCoverage(numOfSlices)
        surfAlb = M_init.initialPlanet(numOfSlices,False)
        
        #Function that describes the effective albedo given a surface albedo
        #and the cloud coverage over that slice. 
        effAlb = [clouds[i]*(1-(1-Acloud)*(1-surfAlb[i]))+(1-clouds[i])*surfAlb[i] for i in range(numOfSlices)]
        
        #Plotting the M_init.Eckertprojection
        if plot: 
            Eckert(effAlb,numOfSlices)
            plt.show()
        return surfAlb,clouds,effAlb
    else: 
        effAlb = [calClouds[i]*(1-(1-Acloud)*(1-calsurf[i]))+(1-calClouds[i])*calsurf[i] for i in range(numOfSlices)] 
        return effAlb
