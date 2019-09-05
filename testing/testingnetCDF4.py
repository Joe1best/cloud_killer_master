from netCDF4 import Dataset
import sys
sys.path.insert(0, 'C:\\Users\\joebe\\Desktop\\Research Project 2019\\cloud_killer-master\\cloud_killer_master')
import cloud_killer_lib as ck
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import random as ran
import time as tyme
import itertools
import cartopy.crs as ccrs


#test = Dataset("dscovr_single_light_timeseries.nc")
#print (test.dimensions['y'])
#print (test.variables['normalized'][:])

#print (test.variables['normalized'][:][9])
#ck.EPIC_data(695,True)
t = 1
test = [] 
test.append([[0,1],[0,2]])
time_array = [np.random.randint(0,2) for i in range(1,8)]
#print (time_array)
#if (any(s == 1 for s in time_array)):
#    print ("see")

t = [1,4,6,7,8,4,3,9]
s = [2,5,6,7,8,9,1,3,4,5,6]
#s = s[x for x in t]

s = [s[x] for x in t]

#print (s)

x = np.arange(0,22)
#print (x)
#print (test)

def shift(l, n): 
    #print (l[n:])
    #print (l[:n])
    #print (l[n:]+l[:n])
    return np.concatenate((l[n:],l[:n]))

#test = np.array([2,3,4,5])
#test=np.roll(test,2)
#print (test)

"""
a = [[2,3,4],[1,6,7],[9,8,2]]
N,M = np.shape(a)
for i in range(N):
    rows = [row[i] for row in a]
print (rows)
"""
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

#kernel(np.asarray([270,180,90,45,0]),np.pi/2)

def apparentAlbedo(albedos, time_days=1.0, long_frac=1.0, n=10000, phi_obs_0=0.0, 
               plot=False, alb=False,lon=None,ndata=None): 
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
    C = 4.0/(3.0*np.pi) # integration constant

    # n times between 0 and 23.93 h, in h
    #print (time_days,"time_days")
    # len(albedos) longitudes
    if type(lon) == type(None):
        time = np.linspace(0.0, time_days*24, n,False)
        phi = np.linspace(2*np.pi, 0, len(albedos),False) 
    else:
        phi = np.asarray(np.deg2rad(lon))
        time = np.linspace(0.0, time_days*24, n,False)

        
    w_Earth = 2.0*np.pi/24 # Earth's angular velocity 
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
        ax1.plot(time, reflectance,'.', color='red')
        if alb: # if we applied the 3/2 factor
            ax1.set_ylabel("Apparent Albedo "+r"$A^*$")
        else: 
            ax1.set_ylabel("Reflectance")
        ax1.set_xlabel("Time [h]")
        plt.show()
    
    return time, reflectance


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
    planet = {}
    sliceNum = range(numOfSlices)
    albedo = [0.4*ran.random() for i in range(numOfSlices)] 
    if plot:
        Eckert(albedo,len(albedo))
        plt.show()
    for i in sliceNum:
        planet[i] = albedo[i]
    return planet


#Need to build a function that calculates what angles are visible at each hour.(Done) 
#Make this general for multiple days
def visibleAngle(hour,Days):
    #start = tyme.time()
    """
    Function that calculates the angle visible for the satellite depending on the hour 
    Input(s):
        hour: hour elapsed in the simulation. 
        longitudes: longitudes slices that are pre-defined beforehand. 
    Output(s):

    """
    hour = hour*60*60
    
    currentLon = np.deg2rad(timeToLongitude(hour)) + 2*np.pi*(Days-1)
    #print (currentLon, "current longitude is")
    """
    if np.pi*(Days) < currentLon < 2*np.pi*Days:
        TwoBound = (currentLon - np.pi/2) + 2*np.pi*(Days-1)
        diff = (2*np.pi-currentLon) + 2*np.pi*(Days-1)
        OneBound = (np.pi/2 - diff)
        if OneBound == 0: 
            OneBound = 2*np.pi 
    elif np.pi*(Days-1) < currentLon < np.pi*(Days):
        OneBound = currentLon + np.pi/2
        diff = np.pi/2 - currentLon
        TwoBound = 2*np.pi - diff
    elif currentLon == 0 or currentLon == 2*np.pi:
        TwoBound = 2*np.pi - np.pi/2
        OneBound = np.pi/2
    elif currentLon == np.pi:
        TwoBound = np.pi/2
        OneBound = 2*np.pi - np.pi/2
    else: 
        print ("why you here nibba")
    """
    
    OneBound = currentLon + np.pi/2
    TwoBound = currentLon - np.pi/2
    
    if TwoBound < 0:
        TwoBound = 2*np.pi + TwoBound
    #elif OneBound < 0 : 
    #    OneBound = 2*np.pi + OneBound
    
    #if TwoBound>2*np.pi*(Days-1) and TwoBound != 2*np.pi :
    #    TwoBound = TwoBound%(2*np.pi)

    if OneBound>2*np.pi*(Days-1) and OneBound != 2*np.pi:
        OneBound = OneBound%(2*np.pi)
    #print (OneBound,TwoBound)
    
    return [OneBound,TwoBound]

#do this without for loops
#Dont need this function!
def sliceFinder(numOfSlices,bounds,albedo):
    """
    Given the bounds, finds which slices exists within those bounds.
    Input: 
        numOfSlices: number of slices in the model
        bounds: the TWO bounds of the integral 
        albedo: albedo map
    """
    borders = np.linspace(0,2*np.pi,numOfSlices+1)
    #print (bounds)
    for i in range(numOfSlices):
        if np.round(bounds[0],8)>=np.round(borders[i],8) and np.round(bounds[1],8)<=np.round(borders[i+1],8):
            return albedo[i] 

#Integral for a single slice.
#Take out i in the (DONE)
#Implement the albedo in there with sliceFinder (DONE, Dont need to)
#Finish commenting code
def integral(time,w_Earth,phi_obs_0,longitudes):
    #start = tyme.time()
    """
    Function that calculates the integral of cosine squared, which is calculated analytically to be 
    the variable "integral" below. This is considered to be the contribution of each slice to the 
    apparent albedo multiplied by some constant. 

    Input(s): 
        phi: longitude array
        phi_obs: the observer longitude, which is dependent on time 

    Output(s): 
        The integral result of the forward model multiplied by the 4/3*pi constant for each slice. 
    """
    C = (4/(3*np.pi))
    phi_obs = phi_obs_0 - 1.0*w_Earth*time # SOP longitude at each time
    starts3 = tyme.time()
    limits = list(map(lambda l : bounds2(l),longitudes))
    sliceNum = [list(limits[x].keys()) for x in range(len(limits))]
    bounds = [list(limits[x].values()) for x in range(len(limits))]
    end = tyme.time()

    print ("to calculate the limits in the integral, it take about ", end-starts3)

    upper = np.asarray([np.asarray([bounds[x][y][0] for y in range(len(bounds[x]))]) for x in range(len(bounds))])
    lower = np.asarray([np.asarray([bounds[x][y][1] for y in range(len(bounds[x]))]) for x in range(len(bounds))])
    x = np.asarray([phi_obs,]*(1)).transpose()
    integral = [C*((1/2)*(up-low)+(1/4)*np.sin(2*up-2*t)-(1/4)*np.sin(2*low-2*t)) for up,low,t in zip(upper,lower,x) ]
    return integral,sliceNum

#UPDATE THIS TO MAIN CODE
#FIX SECOND TO HAVE EXACTLY 60*60*24 for now (Done) 
def timeToLongitude(time):
    """
    Function that converts time since an arbitrary start of the simulation to 
    longitude. Important that the input is in SECONDS. 
    """
    if not isinstance(time,list):
        longitude = np.rad2deg((2*np.pi-(time%86400.0)*(2*np.pi/86400.0)))
        return longitude
    longitude = [(2*np.pi-(t%86400.0)*(2*np.pi/86400.0)) for t in time]
    longitude = np.rad2deg(longitude)
    return longitude

def lightcurve(albedos,longitudes,n,time_days=1.0,phi_obs_0=0.0):
    
    longitudes.reverse()
    longitudes = np.asarray(longitudes)
    trueLon = longitudes

    time = np.linspace(0, 24*time_days , n , False)
    
    w_Earth = 2.0*np.pi/24 # Earth's angular velocity 
    phi_obs = phi_obs_0 - 1.0*w_Earth*time # SOP longitude at each time
    #f = open("test.txt","w+") 

    #integralResults = []

    diff = 24*time_days/len(albedos)
    times = [diff*i for i in range(len(albedos)+1)]
    j = 0
    final = np.zeros([len(albedos),n])
    AllLimits = np.zeros([len(albedos),2])
    
    for i in range(len(time)):
        #Check this line
        if (time[i] <= times[j+1]): 
            longitudes = trueLon
            limit = visibleAngle(time[i],longitudes,time_days)

            if (limit[1]>limit[0]):
                longitudes = [x for x in longitudes if not (limit[0] < x < limit[1])]
                longitudes = [limit[0]] + longitudes + [limit[1]]
                longitudes = sorted(longitudes)
            else:
                longitudes = [x for x in longitudes if (limit[0] > x > limit[1])]
                longitudes = [limit[0]] + longitudes + [limit[1]]
                longitudes = sorted(longitudes)
        else: 
            j +=1
            if (j==len(albedos)+1):
                break
        for k in range(len(albedos)):
            if (len(longitudes)==len(albedos)+2):
                print ("k is ", k)
                AllLimits[k][0] = longitudes[2*k]
                AllLimits[k][1] = longitudes[(2*k)+1]
                print (AllLimits)
            else: 
                AllLimits[k][0] = longitudes[k]
                AllLimits[k][1] = longitudes[k+1]

        #f.write(''.join(str(time[i]))+"\n")
        print (time[i])  
        for l in range(len(albedos)):
            final[l][i] = integral([AllLimits[l][0],AllLimits[l][1]],phi_obs[i],j,albedos)

        #integralResults.append(integral(longitudes,phi_obs[i],j,albedos))
    #final = []

    #for i in range(len(albedos)):
    #    final.append(albedos[i]*np.asarray(integralResults))
    #f.close()
    lightcurve = sum(final)*(3/2)
    lightcurve = np.flip(lightcurve)

    return time, lightcurve

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def bounds(t,bounds,longitudes):
    #start = tyme.time()
    "returns the longitudal bound slice for a given time and limit "
    Two = bounds[0]
    One = bounds[1]
    slices = len(longitudes)-1
    global cond
    #If the region is including the middle
    if (One>Two):
        longitudes = list(filter(lambda x: not Two<=x<=One,longitudes))
        longitudes.extend([One,Two])
        longitudes = np.asarray(sorted(longitudes))
        rightSide = list(pairwise(iter(longitudes[(longitudes >= 0) & (longitudes<=np.pi)])))
        
        leftSide = list(pairwise(iter(longitudes[(longitudes > np.pi) & (longitudes<=2*np.pi)])))[::-1]
        lenLimits = len(rightSide)+len(leftSide)
        finalList1 = leftSide+[(0,0)]*(slices-lenLimits)+rightSide
        finalList2 = rightSide+[(0,0)]*(slices-lenLimits)+leftSide
        if cond == 0:
            return finalList1
        else: 
            return finalList2[::-1]
    #When the region is not including the middle
    elif (Two>One):
        cond +=1
        longitudes = list(filter(lambda x: Two>x>One, longitudes))
        longitudes.extend([One,Two])
        longitudes = sorted(longitudes)
        finalLon = list(pairwise(iter(longitudes)))[::-1]
        finalList = [(0,0)]*(int((slices*(2*np.pi-Two))/(2*np.pi))) + finalLon  + [(0,0)]*(int(One*slices/(2*np.pi))) 
        return finalList

def bounds2(lon):
    "returns the longitudal bound slice for a given time and limit "
    Two = lon[-2]
    One = lon[-1]
    slices = len(lon)-3
    lon = list(dict.fromkeys(lon))
    lon.sort(reverse=True)
    limits = list(pairwise(iter(lon)))
    ind  = [i for i,lim in enumerate(limits) if lim[0]==Two or lim[1]==One]
    if (Two>One): 
        sliceNum = list(range(ind[0],ind[1]+1))
        limits = limits[ind[0]:ind[1]+1]
        if sliceNum[-1] == slices:
            sliceNum = [i-1 for i in sliceNum] 
        #print ("yes")
        #print (sliceNum)
        return dict(zip(sliceNum,limits))
    else:
        if len(limits) == slices:
            limits = limits[0:ind[0]+1] + limits[(ind[1]-ind[0]):slices+1]
            sliceNum = list(range(0,ind[0]+1))+list(range(ind[1]-ind[0],slices))
            #print (sliceNum,"tamere")
        elif len(limits) == slices+1:
            limits = limits[0:ind[0]+1] + limits[(ind[1]-ind[0])+1:slices+1]
            sliceNum = list(range(0,ind[0]+1))+list(range((ind[1]-ind[0])-1,slices))
            #print (sliceNum,"en")
        elif len(limits) == slices+2: 
            limits = limits[0:ind[0]+1] + limits[(ind[1]-ind[0]):slices+2]
            sliceNum = list(range(0,ind[0]+1))+list(range(ind[1]-ind[0]-2,slices))
            #print (sliceNum,"shorts")
        #print ("no")
        return dict(zip(sliceNum,limits))
    """
    #longitudes.extend([One,Two])

    global cond
    #If the region is including the middle
    if (One>Two):
        longitudes = list(filter(lambda x: not Two<=x<=One,longitudes))
        #longitudes = list(longitudes[(longitudes<Two) | (longitudes>One)])

        longitudes.extend([One,Two])
        longitudes = np.asarray(sorted(longitudes))
        rightSide = list(pairwise(iter(longitudes[(longitudes >= 0) & (longitudes<=np.pi)])))
        
        leftSide = list(pairwise(iter(longitudes[(longitudes > np.pi) & (longitudes<=2*np.pi)])))[::-1]
        lenLimits = len(rightSide)+len(leftSide)
        finalList1 = leftSide+[(0,0)]*(slices-lenLimits)+rightSide
        finalList2 = rightSide+[(0,0)]*(slices-lenLimits)+leftSide
        if cond == 0:
            return finalList1
        else: 
            return finalList2[::-1]
    #When the region is not including the middle
    elif (Two>One):
        cond = 1
        longitudes = list(filter(lambda x: Two>x>One, longitudes))
        longitudes.extend([One,Two])
        longitudes = sorted(longitudes)
        finalLon = list(pairwise(iter(longitudes)))[::-1]
        finalList = [(0,0)]*(int((slices*(2*np.pi-Two))/(2*np.pi))) + finalLon  + [(0,0)]*(int(One*slices/(2*np.pi))) 
        return finalList
    """

def longitudeToTime(longitude):
    time = [(2*np.pi-l)/(2*np.pi/86400.0) for l in longitude]
    return time

def lightcurve2(albedos,longitudes,n,time_days=1.0,phi_obs_0=0.0):
    """
    Version 2 of the forward model. 
    """    
    longitudes.reverse()
    time = np.linspace(0, 24*time_days , n , False)
    
    w_Earth = 2.0*np.pi/24 # Earth's angular velocity 
    trueLon = longitudes
    
    longitudes = list(map(lambda t : trueLon+visibleAngle(t,time_days),time))
    
    kern, sliceNum = integral(time,w_Earth,phi_obs_0,longitudes)
    #print (kern)
    albedos= [[albedos[x] for x in timeSlice]for timeSlice in sliceNum ]
    kern = [[albedos[x][y]*kern[x][y] for y in range(len(kern[x]))]for x in range(len(time))]
    #print (kern)
    kern= [sum(kern[i]) for i in range(len(time))]

    lightcurveR = np.asarray(kern)*(3/2)
    return time, lightcurveR

def drawAlbedo(albedo,w,numdata):
    """
    Function that outputs the rate of change of albedo as a function of time
    """
    slices = len(albedo)
    hPerDay = int((w/(2*np.pi))**(-1))
    time = hPerDay/slices 
    x = np.linspace(0,24,numdata)
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
    
    final = np.asarray(albhour)

    return x,final

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
    cs = ax.contourf(longitude,lattitude,A,transform=ccrs.PlateCarree(),cmap='gist_gray',alpha=0.3)
    if data: 
        return cs
    #SET MAX OF COLORBAR TO 1
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r'Apparent Albedo $A^*$')
    ax.coastlines()
    ax.gridlines(color='grey',linewidth=1.5,xlocs = gridlines,ylocs=None) 
    ax.set_global()
    return longitude,lattitude,A,cs,fig,ax,gridlines

def lambert(alpha):
    return (2/(3*np.pi))*(np.sin(alpha)+(np.pi-alpha)*np.cos(alpha)) 


x = np.linspace(0,2*np.pi,1000)
l = lambert(x)
plt.plot(x,l)
plt.show()

albedos = [1,1,1,0]
A1,A2,A3,A4 = np.loadtxt("MCMC_15_04_08__4.csv",delimiter=',',unpack=True)
A1 = sorted(A1)
A2 = sorted(A2)
A3 = sorted(A3)
A4 = sorted(A4)
print (np.mean(A1)-A1[15],A1[83]-np.mean(A1))
print (np.mean(A2)-A2[15],A2[83]-np.mean(A2))
print (np.mean(A3)-A3[15],A3[83]-np.mean(A3))
print (np.mean(A4)-A4[15],A4[83]-np.mean(A4))
#Eckert(albedos,len(albedos))
#plt.show()
    
#numOfSlices = len(albedos)
#

#albedos1 = [0,0,0,1,1,1]
#cond = 0 
"""
start = tyme.time()
#longitudes = np.ndarray.tolist(np.linspace(2*np.pi,0,numOfSlices+1))

time, light = lightcurve2(albedos,longitudes,1000)
print (tyme.time()-start)

hour, alb = drawAlbedo(albedos,2*np.pi/24,5000)

timeComp, lightComp = apparentAlbedo(albedos,alb=True)
#print (light)
#time1,light1 = lightcurve2(albedos1,longitudes,10)

fig,ax = plt.subplots(1,1,gridspec_kw={'height_ratios':[1]},figsize=(10,8))
x = np.linspace(0,24,1000,False)

ax.plot(time,light,'.',label="New Lightcurve")
ax.plot(timeComp,np.round(lightComp,6),'.',label="Old Lightcurve")
ax.set_title('Result for {}'.format(np.round(albedos,3)),fontsize=22)
ax.plot(hour,alb,'--',color='purple',linewidth=3,alpha=0.5)
#ax.plot(time1,light1,'.',label="Simulation result for [1,0]")

ax.set_xlabel("Time (h)",fontsize=22)
ax.set_ylabel("Apparent Albedo",fontsize=22)
ax.legend(fontsize=15)
ax.tick_params(labelsize=22)
plt.show()
"""




"""
One,Two = np.loadtxt("animation.csv",delimiter=',',unpack=True)
plt.ion()
fig = plt.figure(1,figsize=(12,6))
for i in range(len(One)):
    fig.clear()
    plt.xlim(0,2*np.pi)
    plt.annotate(str(time[i]),xy=(2,0.1))

    #plt.axvspan(One[i], Two[i], alpha=0.5, color='red')
    plt.axvline(One[i],color='red',label='One')
    plt.axvline(Two[i],color='blue',label='Two')
    plt.pause(0.5)
#plt.axvline(18)
plt.ioff()
plt.legend()
plt.show()
"""

def roll(Dict,shift):
    slices = np.fromiter(Dict.keys(), dtype=int)
    albedo = np.fromiter(Dict.values(),dtype=float)
    albedo = np.roll(albedo,shift)
    slices = np.roll(slices,shift)
    Dict.clear()
    for i in slices:
        Dict[i] = albedo[i]
    return Dict


"""
hour = dt.datetime.now().time()
fileName = hour.strftime("%H:%M:%S").replace(":","_",2)
f= open("MCMC_{}__{}.csv".format(fileName,numOfSlices),"w+") 
f.write(''.join(str([0.1,0.2,0.3]).replace("[","",1).replace("]","",1))+"\n")
f.write(''.join(str([0.1,0.2,0.3]).replace("[","",1).replace("]","",1))+"\n")
f.close()
"""
