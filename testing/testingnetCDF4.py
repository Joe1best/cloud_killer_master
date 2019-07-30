from netCDF4 import Dataset
import sys
sys.path.insert(0, 'C:\\Users\\joebe\\Desktop\\Research Project 2019\\cloud_killer-master\\cloud_killer_master')
import cloud_killer_lib as ck
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import random as ran
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


#Need to build a function that calculates what angles are visible at each hour. 
def visibleAngle(hour,longitudes,Days):
    """
    Function that calculates the angle visible for the satellite depending on the hour 
    Inputs
        hour: hour elapsed in the simulation. 
        longitudes: longitudes slices that are pre-defined beforehand. 
    """
    hour = [hour*60*60]
    
    currentLon = np.deg2rad(timeToLongitude(hour)) + 2*np.pi*(Days-1)
    print (currentLon, "current longitude is")

    if np.pi*(Days)< currentLon < 2*np.pi*Days:
        TwoBound = (currentLon - np.pi/2) + 2*np.pi*(Days-1)
        diff = (2*np.pi-currentLon) + 2*np.pi*(Days-1)
        OneBound = (np.pi/2 - diff) 
        #print ("here")
    elif np.pi*(Days-1) < currentLon < np.pi*(Days):
        OneBound = currentLon + np.pi/2
        diff = np.pi/2 - currentLon
        TwoBound = 2*np.pi - diff
        #print ("there")
    elif currentLon == 0 or currentLon == 2*np.pi:
        TwoBound = 2*np.pi - np.pi/2
        OneBound = np.pi/2
        #print ("everywhere")
    else: 
        print ("why you here nibba")

    if TwoBound < 0:
        OneBound = 2*np.pi +TwoBound
    elif OneBound < 0 : 
        OneBound = 2*np.pi + OneBound
    
    if TwoBound>2*np.pi*(Days-1):
        TwoBound = TwoBound%(2*np.pi)

    if OneBound>2*np.pi*(Days-1):
        OneBound = OneBound%(2*np.pi)
    
    return OneBound,TwoBound


def sliceFinder(numOfSlices,bounds,albedo):
    """
    Given the bounds, finds which slices exists within those bounds.
    Input: 
        numOfSlices: number of slices in the model
        bounds: the TWO bounds of the integral 
        albedo: albedo map
    """
    borders = np.linspace(0,2*np.pi,numOfSlices+1)
    for i in range(numOfSlices):
        print (i,"slice")
        print (borders[i],borders[i+1],"the borders of slices")
        print (bounds[0], bounds[1], "the bounds")
        if np.round(bounds[0],8)>=np.round(borders[i],8) and np.round(bounds[1],8)<=np.round(borders[i+1],8):
            print ("found your mom")
            print ("\n")
            return albedo[i] 

#Integral for a single slice.
def integral(phi,phi_obs,i,albedos):
    """
    Function that calculates the integral of cosine squared, which is calculated analytically to be 
    the variable "integral" below. This is considered to be the contribution of each slice to the 
    apparent albedo multiplied by some constant. 

    Inputs: 
        i: the ith slice  
        phi: longitude array
        phi_obs: the observer longitude, which is dependent on time 

    Output: 
        The integral result of the forward model multiplied by the 4/3*pi constant for each slice. 
    """
    #print ("input bounds are ", phi)
    #f.write(''.join(str(phi))+"\n")

    C = (4/(3*np.pi))
    #if len(phi)%2 == 0 and i%2 !=0: 
    #    i = i + 1
    #    j = i + 1
    #else: 
    #    j = i+1
    #if j == len(phi):
    #    return 0
    #f.write(''.join(str(j))+"\n")
    #f.write(''.join(str(i))+"\n")
    #f.write("\n")
    a = sliceFinder(len(albedos),phi,albedos)
    integral = a*((1/2)*(phi[1]-phi[0])+(1/4)*np.sin(2*phi[1]-2*phi_obs)-(1/4)*np.sin(2*phi[0]-2*phi_obs))    
    return C*integral

def timeToLongitude(time):
    """
    Function that converts time since an arbitrary start of the simulation to 
    longitude. Important that the input is in SECONDS. 
    """
    #print (time)
    longitude = [(2*np.pi-(t%86148.0)*(2*np.pi/86148.0)) for t in time]
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


def lightcurve2(albedos,longitudes,n,time_days=1.0,phi_obs_0=0.0):
    """
    Version 2 of the forward model. 
    """
    longitudes.reverse()
    longitudes = np.asarray(longitudes)
    trueLon = longitudes

    time = np.linspace(0, 24*time_days , n , False)
    
    w_Earth = 2.0*np.pi/24 # Earth's angular velocity 
    phi_obs = phi_obs_0 - 1.0*w_Earth*time # SOP longitude at each time
    #f = open("test.txt","w+") 

    #integralResults = []

    diff = 24*time_days/len(albedos)
    j = 0
    final = np.zeros([len(albedos),n])
    
    for i in range(len(time)):
        AllLimits = np.zeros([len(albedos),2])
        #Check this line
        longitudes = trueLon
        Two, One = visibleAngle(time[i],longitudes,time_days)
        print (Two, One, "limit is ")
        #When true it is in the first case
        cond = True
        #for k in range(len(albedo)):
        #    AllLimits[k][0] = 
        #    AllLimits[k][1] = 

        #Slice is crossing the middle. Need to split it in 2
        if (One>Two):
            longitudes = [x for x in longitudes if not (Two <= x <= One)]
            print (longitudes,"ostie")
            longitudes =np.asarray([Two] + longitudes + [One])
            rightSide = sorted([x for x in longitudes if (0 <= x <= np.pi)])
            leftSide = sorted([x for x in longitudes if (np.pi < x <= 2*np.pi)])
            print (rightSide)
            print (leftSide)
            cond = True
        #When the region is not including the middle
        elif (Two>One):
            longitudes = [x for x in longitudes if (Two >= x >= One)]
            longitudes = sorted(np.asarray([One] + longitudes + [Two]))
            cond = False
            print (longitudes,"tabarnak")

        #elif(limit[0]>limit[1]):
        #    longitudes = [x for x in longitudes if (limit[0] >= x >= limit[1])]
        #    print (longitudes, "tabarnak")
        #    longitudes = [limit[0]] + longitudes + [limit[1]]
        #    #longitudes = sorted(longitudes)
        #    print (longitudes, "tabarnak")

        count = 0 
        for k in range(len(albedos)):
            print ("k is ", k)
            print (longitudes,"update")
            if cond==True:
                if k+1-count>=len(leftSide):
                    break
                elif k+1>=len(rightSide):
                    AllLimits[k][0] = leftSide[k-count]
                    AllLimits[k][1] = leftSide[k-count+1]
                else:
                    AllLimits[k][0] = rightSide[k]
                    AllLimits[k][1] = rightSide[k+1]
                    count += 1
                print (AllLimits)
            else:
                if k+1 <len(longitudes):
                    AllLimits[k][0] = longitudes[k]
                    AllLimits[k][1] = longitudes[k+1]
                else:
                    break

        #f.write(''.join(str(time[i]))+"\n")
        print (time[i])  
        for l in range(len(albedos)):
            final[l][i] = integral([AllLimits[l][0],AllLimits[l][1]],phi_obs[i],j,albedos)

        #integralResults.append(integral(longitudes,phi_obs[i],j,albedos))
    #final = []

    #for i in range(len(albedos)):
    #    final.append(albedos[i]*np.asarray(integralResults))
    #f.close()
    lightcurveR = sum(final)*(3/2)
    lightcurveR = np.flip(lightcurveR)

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
#lon = []
numOfSlices = 4
albedos = [1,1,1,1]
albedos1 = [0,0,0,1,1,1]
longitudes = np.ndarray.tolist(np.linspace(2*np.pi,0,numOfSlices+1))

hour, alb = drawAlbedo(albedos,2*np.pi/24,100)
time, light = lightcurve2(albedos,longitudes,1000)
print (light)
#time1,light1 = lightcurve2(albedos1,longitudes,10)

fig,ax = plt.subplots(1,1,gridspec_kw={'height_ratios':[1]},figsize=(10,8))
x = np.linspace(0,24,1000,False)

ax.plot(time,np.round(light,4),'.',label="Simulation result for [0,1]")
#ax.plot(x,-0.5*np.sin((2*np.pi/24)*x)+0.5,'--',linewidth=7,label="Rough Prediction for [0,1]")
ax.plot(hour,alb,'--',color='purple',linewidth=6)
#ax.plot(time1,light1,'.',label="Simulation result for [1,0]")
#ax.plot(x,0.5*np.sin((2*np.pi/24)*x)+0.5,'--',linewidth=7,label="Prediction for [1,0]")

ax.set_xlabel("Time (h)",fontsize=22)
ax.set_ylabel("Apparent Albedo",fontsize=22)
ax.legend(fontsize=16)
ax.tick_params(labelsize=22)
plt.show()



def roll(Dict,shift):
    slices = np.fromiter(Dict.keys(), dtype=int)
    albedo = np.fromiter(Dict.values(),dtype=float)
    albedo = np.roll(albedo,shift)
    slices = np.roll(slices,shift)
    Dict.clear()
    for i in slices:
        Dict[i] = albedo[i]
    return Dict


#surf = initialPlanet(numOfSlices,plot=False)
#print (roll(surf,-1)) 

test = [1,2,3,4]
test.reverse()
#print (test)

"""
hour = dt.datetime.now().time()
fileName = hour.strftime("%H:%M:%S").replace(":","_",2)
f= open("MCMC_{}__{}.csv".format(fileName,numOfSlices),"w+") 
f.write(''.join(str([0.1,0.2,0.3]).replace("[","",1).replace("]","",1))+"\n")
f.write(''.join(str([0.1,0.2,0.3]).replace("[","",1).replace("]","",1))+"\n")
f.close()
"""