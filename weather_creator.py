#In[]:
#Importing packages
import numpy as np 
import matplotlib.pyplot as plt 
import random as ran
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib import style
import collections
import datetime as dt
import time as tyme
from netCDF4 import Dataset
import math
from astropy.time import Time
import cloud_killer_lib as ck_lib
import init as var
import Model_Init as M_init 
import MCMC as m
import Utilities as util

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
        title = r"EPIC data [$d$ = {}, $\phi_0$ = {}] ".format(util.date_after(day),np.round(np.deg2rad(longitude[0]),3))
        plt.title(title)
        plt.rcParams.update({'font.size':14})
        plt.show()

    return t, longitude, reflectance, reflectance_err, contains_nan

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
    effAlb=M_init.effectiveAlbedo(numOfSlices,Acloud,False,calClouds=cloudIn,calsurf=surfAlb)
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
        dum_lon, dum_lat, dum_A, dum_grid = M_init.Eckert(dum_alb,numOfSlices,fig=fig,plot=False)
        css = plt.contourf(dum_lon,dum_lat,dum_A,cmap='gist_gray',alpha=0.3)
        condition = True

    hPerDay = int((w/(2*np.pi))**(-1))

    #Don't know if the bottom two lines are necessary. Ill keep them for now.
    dum_alb = [np.random.randint(0,2) for i in range(numOfSlices)]
    dum_lon, dum_lat, dum_A, dum_grid = M_init.Eckert(dum_alb,numOfSlices,plot=False)
    eff = M_init.effectiveAlbedo(numOfSlices,Acloud,plot=False,calClouds=clouds,calsurf=surf)
    daysPassed = 0

    #Loop that loops through the 24 hour cycle (each i is 1 hour)

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
        
        plt.clf()

        #If Animation is TRUE, it will draw an M_init.EckertProjection animation. The plot
        #settings can be found starting from 482 to 488.
        if Animation:
            M_init.Eckert(eff,numOfSlices,fig=fig,bar=css,plot=condition)
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
    surf = M_init.initialPlanet(numOfSlices,False)
    clouds = M_init.cloudCoverage(numOfSlices)
    finalTime = []
    apparentTime = []
    l,d=dataAlbedoDynamic(numOfSlices,Days,w,Acloud,surf,clouds,rateDiss,speedCloud,Animation=False)
        
    print ("Got sum fake data. YOHO, SCIENCE BITCH!")

    for i in range(1,Days+1):
        #Seperates the effective albedo and longitude per day.
        effective = d[(i-1)*numOfSlices:(i)*(numOfSlices)] 
        lon = l[(i-1)*numOfSlices:(i)*(numOfSlices)]
        #Calculates the apparent albedo with the forward model. 
        time, apparent = M_Init.apparentAlbedo(effective,time_days=timespan,
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
            m.MCMC(nwalkers,nsteps,numOfSlices,time,app,lon,timespan,phispan,burning,hPerDay,chainArray,i,ax,plot)
            print ("done MCMC for day {}".format(i))
    for chain in chainArray:
        alb.append(m.mcmc_results(chain,burning))
    return alb

def runShatelite(numOfSlices,Acloud,rateDiss,speedCloud,w,ndata,fastFoward,Days,
        nwalkers,nsamples,nsteps,timespan,phispan,burning,plot=True,mcmc=True):
    #Need to make this a bit more efficient)
    if (numOfSlices>24):
        print ("Cannot have more than 24 number of slices for now")
        return 0,0
    surf = M_init.initialPlanet(numOfSlices,False)
    clouds = M_init.cloudCoverage(numOfSlices)
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
        surfDict =M_init.initialPlanet(numOfSlices,plot=True)
        surf = np.fromiter(surfDict.values(),dtype=float)
        surf = [0.458,0.327,0.332,0.263]
        print ("The planet's surface albedo is theoritically", surf)
        clouds = M_init.cloudCoverage(numOfSlices)
        clouds = [0,0,0,0]
        print ("The effective albedo is ", M_init.effectiveAlbedo(numOfSlices,Acloud,plot=False,calClouds=clouds,calsurf=surf))
        finalTime = []
        apparentTime = []
        print (clouds, "Cloud coverage is ")
        d = dataAlbedoDynamic(numOfSlices,Days,w,Acloud,surf,clouds,rateDiss,speedCloud,Animation=False,hourlyChange=False,repeat=repeat,forming=forming)
        for i in range(1,Days+1):
            effective = d[(i-1)*numOfSlices:(i)*(numOfSlices)] 
            print("The albedo map for day {} is ".format(i-1), effective)
            #start = tyme.time()
            time, apparent = M_init.apparentAlbedo(effective,time_days=timespan,
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
        noise = [np.random.normal(loc=0,scale=0.02*a[i]) for i in range(len(a))]
        a = np.array(a)+np.array(noise)
        if plot: 
            fig,ax = plt.subplots(1,1,gridspec_kw={'height_ratios':[1]},figsize=(10,3))
            tim, albedo = drawAlbedo(d,Days,5000)
            ax.errorbar(t,a,fmt='.',yerr=0.02*np.array(a),color='black',markersize=8,label="Simulated lightcurve",solid_capstyle='projecting', capsize=4)
            #ax.plot(tim_comp,app_comp,'.',color='orange',linewidth=4,label="Early Version of F.M.")
            #ax.plot(tim,albedo,color='purple',linewidth=6,alpha=0.6,label=r'$A_{T.O.P}(\phi)$')
            #ax.plot(t,a,'.',color='purple',label='Albedo Generated for {} slices'.format(numOfSlices),alpha=0.3)
            ax.set_xlabel("Time (h)",fontsize=22)
            ax.set_ylabel("Apparent Albedo ($A^*$)",fontsize = 22)

            #ax.legend(fontsize=17)
            ax.tick_params(labelsize=22)
    else:
        t = Epic[0]
        a = Epic[1]
        t = (t - t[0])*24 
        ax.errorbar(t,a,fmt='.',yerr=Epic[3],color='black',markersize=8,label="EPIC data",solid_capstyle='projecting', capsize=4)

    print ("Done extracting {}".format(numOfSlices))
    chainArray=[]
    alb=[]
    if plot:
        #ax.errorbar(t,a,fmt='.',color='blue',yerr = np.asarray(a)*0.02,markersize=10,solid_capstyle='projecting', capsize=4,
        #            label= "Raw Data from EPIC")
        ax.set_xlabel("Time (h)",fontsize=22)
        ax.set_ylabel("Apparent Albedo ($A^*$)",fontsize = 22)
        if type(Epic)!=type(None):
            title = r"EPIC data [$d$ = {}] ".format(util.date_after(Epic[2]))
            #title = r"Forward model for {} slice albedo map".format(numOfSlices)
            ax.set_title(title,fontsize=22)
        ax.legend(fontsize=20)
        ax.tick_params(labelsize=25)    
    if (mcmc):
        #Implement the MCMC running stuff in a seperate function
        for i in range(1,Days+1):
            time = t[(i-1)*ndata:i*ndata]
            app = a[(i-1)*ndata:i*ndata]
            chain  = m.make_chain(nwalkers,nsteps,ndim,time,app,timespan,phispan,alb=True)
            chainArray.append(chain)
            print ("Well call me a slave, because I just made some chains for day {}...".format(i))
            mean_mcmc_params = m.mcmc_results(chain,burning)
            mean_mcmc_time, mean_mcmc_ref = M_init.apparentAlbedo(mean_mcmc_params,time_days=timespan,
                long_frac=phispan,n=10000,plot=False,alb=True)
            print ("Got the mean MCMC results for day {}. Ya YEET!".format(i))

            flat = m.flatten_chain(chain,burning)
            sample_params = flat[np.random.randint(len(flat),size=nsamples)]
            for s in sample_params:
                sample_time,sample_ref = M_init.apparentAlbedo(s,time_days=timespan,long_frac=phispan,n=10000,plot=False,alb=True)
                sample_time = np.asarray(sample_time)
                mean_mcmc_params = np.asarray(mean_mcmc_params)
                plotting_x = np.asarray(sample_time)+(i-1)*hPerDay
                if (plot):
                    ax.plot(plotting_x,sample_ref,color='k',alpha=0.1)
            if (plot):     
                ax.plot(plotting_x,sample_ref,color='k',alpha=0.1, label='50 samples from MCMC')
                ax.plot(mean_mcmc_time+(i-1)*hPerDay,mean_mcmc_ref,color='red',label="Mean MCMC")
                ax.legend(fontsize=20)

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
        results = m.mcmc_results(chainy,burning)
        if (walkers):
            m.plot_walkers_all(chainy,expAlb=None)
            m.cornerplot(chainy,burning)
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
        ax.axhline(np.mean(M_init.effectiveAlbedo(numOfSlices,Acloud,plot=False,calClouds=clouds,calsurf=surf)),linewidth=8,color='orange',label='Expected Mean')
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



#In[]:
#Main method
if __name__ == "__main__":
    daySim = 790
    t, longitude, reflectance, reflectance_err, contains_nan = EPIC_data(daySim)
    data = [t,reflectance,daySim,reflectance_err]
    
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

    for steps in var.NSTEPS:
        aics = []
        bics = []
        for j in range(var.NTRIALS):
            dumAlb,dumMin,eff= runSatellowan(var.NUMOFSLICES,var.ACLOUD,var.NPARA,var.RATEDISS,var.SPEEDCLOUD,var.WW,var.NDATA,var.FASTFORWARD,var.DAYS,
                    var.NWALKERS,var.NSAMPLES,steps,var.TIMESPAN,var.PHISPAN,var.BURNIN,plot=var.PLOT,mcmc=var.MCMC,repeat=var.REPEAT,walkers=var.WALKERS,forming=var.CLOUDFORMING,Epic=data)
    print ("The surface albedo from MCMC is " ,dumMin," compared to the inputted map of ", eff)
