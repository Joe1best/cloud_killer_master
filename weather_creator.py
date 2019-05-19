import numpy as np 
import matplotlib.pyplot as plt 
import random as ran
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.animation as animation
from matplotlib import style

def initialPlanet(numOfSlices,plot=True,nlons=400,nlats=400):
    """
    Function that takes an input of the number of longitudal slices used and outputs the 
    corresponding longitudes. It also generates a random albedo for the surface at each 
    slice. 
    Also takes a boolean input plot (default to True) depending on whether plotting is 
    desired or not. 
    """
    #time = np.linspace(0,86400,nlons)
    #longitude = np.rad2deg(((2*np.pi-time%86400)*(2*np.pi/86400))%2*np.pi)
    albedo = [0.4*ran.random() for i in range(numOfSlices)] 
    if plot:
        Eckert(albedo,len(albedo))
    return albedo

def kernel(longitude,phi_obs):
    """
    Input: an array of longitudes and the sub-observer longitude phi_obs
    
    Computes the kernel K(theta,phi,t) predicted by the forward model.
    
    Output: the kernel of the forward model
    """
    # I=V in this case since the SOP and SSP are the same at L1, and we choose 
    # to fix sub-observer/sub-stellar latitude at pi/2 
    V = np.cos(longitude[...,np.newaxis] - phi_obs)
    V[V<0] = 0 #set values <0 to be = 0

    return V*V # K=I*V=V*V

def apparentAlbedo(albedos, time_days=1.0, long_frac=1.0, n=10000, phi_obs_0=0.0, 
               plot=False, alb=False): 
    """
    Input: an array of albedos, the time in days which the model should span
    (default: 1.0), the longitude as a fraction of 2pi which the model should 
    span (default: 1.0), the no. of points n to generate (default: 10000), 
    the initial sub-observer longitude (default: 0.0), a boolean indicating 
    whether or not to plot the lightcurve (default: False), and a boolean 
    indicating whether to return the reflectance or to apply the 
    multiplicative factor of 3/2 such that the lightcurve's units match those 
    of EPIC data
    
    Computes the lightcurve A*(t) predicted by the forward model.
    
    Output: the lightcurve, in units of reflectance or apparent albedo 
    """
    C = 4.0/(3.0*np.pi) # integration constant
    
    # n times between 0 and 23.93 h, in h
    time = np.linspace(0.0, time_days*23.93, n, False)
    # len(albedos) longitudes
    phi = np.linspace(2*np.pi, 0, len(albedos), False) 
    
    w_Earth = 2.0*np.pi/23.93 # Earth's angular velocity 
    phi_obs = phi_obs_0 - 1.0*w_Earth*time # SOP longitude at each time
    # phi decreases before returning to 0 in this convention
    
    albedos = np.asarray(albedos) # convert to numpy array
    
    kern = kernel(phi, phi_obs) # compute the kernel  
    
    reflectance = np.sum(albedos[...,np.newaxis]*kern, axis=0)
    reflectance = C*reflectance*(2*np.pi)/len(albedos)
    
    if alb: # if we want units in agreement with EPIC data
        reflectance *= 3.0/2.0 # multiply by 3/2
        
    # if we want to plot the lightcurve
    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)    
        ax1.plot(time, reflectance, color='red')
        if alb: # if we applied the 3/2 factor
            ax1.set_ylabel("Apparent Albedo "+r"$A^*$")
        else: 
            ax1.set_ylabel("Reflectance")
        ax1.set_xlabel("Time [h]")
        plt.show()
    
    return time, reflectance


def Eckert(albedo,numOfSlices,nlats=400,nlons=400,fig=None):
    """
    General function to plot the Eckert map based off the number of slices
    Inputs:
        albedo: takes in the albedo value at each longitudinal slices
        numOfSlices: number of longitudinal slices
    Output: EckertIV map projection of the results for a general number of slices.
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
    if type(fig)==type(None):
        fig = plt.figure(figsize=(12,6))
    else:
        fig = fig
    ax = fig.add_subplot(1,1,1,projection = ccrs.EckertIV())
    w = 0
    A = []
    gridlines=[360]
    for i in range(nlons):
        if (w == len(albedo)):
            break
        temp = [albedo[w] for j in range(nlats)]
        if i%interval==0 and i!=0:
            w=w+1
            gridlines.append(longitude[i][0])
        A.append(temp)
    gridlines.append(0)
    #if (len(A)!=nlats):
    #    for i in range(nlats-len(A)):
    #        A.append(A[len(A)-1])
    ax.clear()
    cs = ax.contourf(longitude,lattitude,A,transform=ccrs.PlateCarree(),
        cmap='gist_gray',alpha=0.3)

    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r'Albedo $A$')
    ax.coastlines()
    ax.gridlines(color='grey',linewidth=1.5,xlocs = gridlines,ylocs=None) 
    ax.set_global()
    #plt.show()
    #ax.clear()
    return longitude,lattitude,A,cs,fig,ax

def cloudCoverage(numOfSlices):
    """
    Function that generates a cloud coverage, and calculates its effect on the albedo generated
    using initialPlanet.

    Outputs: random values for each slice that ranges from 0 to 1. 1 being total cloud cover 
    and 0 having no clouds. 
    """
    booleanSlice = [ran.randint(0,1) for i in range(numOfSlices)]
    clouds = np.zeros(numOfSlices)
    for i in range(numOfSlices):
        if booleanSlice[i]==1:
            clouds[i] = ran.random()
        else:
            clouds[i] = 0
    return clouds

def effectiveAlbedo(numOfSlices,Acloud,plot=True,calClouds=None,calsurf=None):
    """
    Function that calculates the effective albedo of a longitudinal slice with cloud 
    coverage taken into account. 

    If the cloud coverage is 0, then the albedo is just the albedo of the surface. If 
    there is cloud coverage, then we weight the cloud albedo with the surface albedo. 
    """
    if type(calClouds)==type(None):
        clouds = cloudCoverage(numOfSlices)
        surfAlb = initialPlanet(numOfSlices,False)
        effAlb = [clouds[i]*(1-(1-Acloud)*(1-surfAlb[i]))+(1-clouds[i])*surfAlb[i] for i in range(numOfSlices)]
        if plot: 
            Eckert(effAlb,numOfSlices)
        return surfAlb,clouds,effAlb
    else: 
        effAlb = [calClouds[i]*(1-(1-Acloud)*(1-calsurf[i]))+(1-calClouds[i])*calsurf[i] for i in range(numOfSlices)] 
        return effAlb

def dissipate(cloudIn,time,rate,scale="minutes"):
    """
    Function that dissipates the intensity of cloud with each iteration of time. I am 
    assuming that on average, the clouds dissipate from 100 to 0 in three hours 
    (10 800s or 180min). ASSUMING A LINEAR MODEL FOR NOW (WHICH IS PROBABLY INACCURATE).

    The rate should be given in the same units as scale
    """  
    for i in range(len(cloudIn)):
        if cloudIn[i] <= 0:
            cloudIn[i] = 0 
        else:
            if scale=="minutes":
                cloudIn[i] = cloudIn[i] - rate
                #np.abs(ran.gauss(rate,30))
            elif scale=="hours":
                cloudIn[i] = cloudIn[i] - rate
                #np.abs(ran.gauss(rate,0.5))
            else:
                cloudIn[i] = cloudIn[i] - rate
                #np.abs(ran.gauss(rate,1800))
            if cloudIn[i] <= 0:
                cloudIn[i] = 0

def form(cloudIn,time,scale="minutes"):
    """
    Function that forms clouds with each iteration of time. The cloud generation process
    takes about from 1 minute to sevral hours. So I will just generate a rate gaussian 
    with a mean of 1 hour and a standard deviation 
    
    Need to make sure that cloud coverage in each slice does not exceed 1.

    NEED TO IMPLEMENT FORMATION EVEN IF THERE ARE NO CLOUDS.
    
    """
    for i in range(len(cloudIn)):
        if cloudIn[i] == 0:
            u = ran.randint(0,1)
            if u > 0.5:
                cloudIn[i] = ran.random()

def move(cloudIn,time,speed,scale="minutes"):
    """
    Function that moves the clouds from East to West. Important to note, that with the
    definition of longitude of the eckert maps, the first array value starts at the 
    left middle slice and goes westerwards, wraps around the other side and back to 
    the right middle slice. However, clouds move the other way, hence depending on the 
    speed, the index will shift upwards. 
    """
    disSlice = 40070/len(cloudIn)        #distance of each slice
    timeSlice = int(disSlice/speed)    #time of each slice 
    print(timeSlice)
    if timeSlice==0 or ((time%timeSlice) == 0 and time !=0):
        cloudIn = np.roll(cloudIn,1)
    return cloudIn

def dynamicMap(time,numOfSlices,Acloud,surfAlb,cloudIn,rateDiss,speedCloud):
    """
    Function that changes the cloud dynamically per iteration. Changes every minute
    """
    dissipate(cloudIn,time,rateDiss)
    form(cloudIn,time)
    cloudIn = move(cloudIn,time,speedCloud)
    effAlb=effectiveAlbedo(numOfSlices,Acloud,False,cloudIn,surfAlb)
    return effAlb,cloudIn


numOfSlices = 20
"""
clouds = cloudCoverage(numOfSlices)
print (clouds, "initial")
for i in range(100):
    dissipate(clouds,i,1/180)
    form(clouds,i)
    print (clouds)
    
"""

Acloud = 0.8
surf = initialPlanet(numOfSlices,False)
clouds = cloudCoverage(numOfSlices)
rateDiss = 1/180
speedCloud=80000
plt.ion()
fig = plt.figure(figsize=(12,6))


for i in range(150):
    eff,clouds=dynamicMap(i,numOfSlices,Acloud,surf,clouds,rateDiss,speedCloud)
    plt.clf()
    Eckert(eff,numOfSlices,fig=fig)
    print("t=",i)
    plt.pause(0.01)
#if Animation:
#    ani = animation.FuncAnimation(fig,animate,fargs=(numOfSlices,),interval=1000,blit=True)
#    plt.show()