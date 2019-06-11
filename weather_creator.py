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

#Various functions for a dynamic kernel
def timeToLongitude(time):
    """
    Function that converts time since an arbitrary start of the simulation to 
    longitude. Important that the input is in SECONDS. 
    """
    longitude = [(2*np.pi-(t%86148.0)*(2*np.pi/86148.0)) for t in time]
    longitude = np.rad2deg(longitude)
    return longitude

def longitudeToTime(longitude):
    time = [(2*np.pi-l)/(2*np.pi/86148.0) for l in longitude]
    return time

def findSlice(theta,gridlines):
    for i in range(len(gridlines)-1):
        if theta < gridlines[i] and theta > gridlines[i+1]:
            return i 

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
        plt.show()
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
        #time = np.asarray(longitudeToTime(phi))/(60*60)
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

def Eckert(albedo,numOfSlices,nlats=400,nlons=400,fig=None,bar=None,plot=True,data=False):
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
    cbar = fig.colorbar(bar)
    cbar.ax.set_ylabel(r'Apparent Albedo $A^*$')
    ax.coastlines()
    ax.gridlines(color='grey',linewidth=1.5,xlocs = gridlines,ylocs=None) 
    ax.set_global()
    #plt.show()
    return longitude,lattitude,A,cs,fig,ax,gridlines

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
            plt.show()
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

def move(cloudIn,time,speed):
    """
    Function that moves the clouds from East to West. Important to note, that with the
    definition of longitude of the eckert maps, the first array value starts at the 
    left middle slice and goes westerwards, wraps around the other side and back to 
    the right middle slice. However, clouds move the other way, hence depending on the 
    speed, the index will shift upwards. 
    """
    disSlice = 40070/len(cloudIn)        #distance of each slice
    timeSlice = int(disSlice/speed)    #time of each slice 
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

def rotateEarth(w,albedo,numberOfSlices,t,Days):
    """
    Function that rotates the albedo array based on the angular velocity. 
    Inputs:
        w: angular velocity 
        albedo: albedo array 
        numberOfSlices: number of slices lol 
        t: time (in HOURS)
    Outputs: a shifted array of albedos 
    """
    #shift index by -1 since the Earth rotates from West to East 
    diff = 2*np.pi/numberOfSlices
    cond = diff/w
    #
    time_array = [cond*i for i in range(1,(numberOfSlices+1)*Days)] 

    if any(s == t  for s in time_array):
        albedo = np.roll(albedo,-1)
    return albedo

#MCMC fitting (imported from cloud_killer_lib.py for more efficient runtimes)

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

def lnlike(alpha, time,ref, ref_err,lon,timespan,phispan):
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
                                       0, plot=False, alb=True,lon=lon) 
    
    # compute ln(likelihood)
    chisq_num = np.power(np.subtract(ref,model_ref), 2) # (data-model)**2
    chisq_denom = np.power(ref_err, 2) # (error)**2
    res = -0.5*sum(chisq_num/chisq_denom + np.log(2*np.pi) + np.log(np.power(
            ref_err,2))) #lnlike
    
    return res

def opt_lnlike(alpha, time, ref, ref_err,lon,timespan,phispan):
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
    result = op.minimize(nll, alpha, args=(time,ref,ref_err,lon,timespan,phispan), bounds=bound_alb)
    
    return result['x'] # the optimized parameters

def init_walkers(alpha, time, ref, ref_err, ndim, nwalkers,lon,timespan,phispan):
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
    opt_albs = opt_lnlike(alpha, time, ref, ref_err,lon,timespan,phispan) # mazimize likelihood
    # generate walkers in Gaussian ball
    pos = [opt_albs + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    return pos

def lnprior(alpha):
    """
    Input: guesses for the fit parameters (alpha, an array of albedos,
    representing A(phi))
    Output: The ln(prior) for a given set of albedos 
    """
    if np.all(alpha>0.0) and np.all(alpha<1.0): # if valid albedos
        return 0.0
    return -np.inf # if not, probability goes to 0 

def lnpost(alpha, time, ref, ref_err,lon,timespan,phispan):
    """
    Input: guesses for the fit parameters (alpha, an array of albedos,
    representing A(phi)) and the time, lightcurve, and error on the lightcurve 
    of the data being fit 
    Output: ln(posterior)
    """
    lp = lnprior(alpha)
    if not np.isfinite(lp): # if ln(prior) is -inf (prior->0) 
        return -np.inf      # then ln(post) is -inf too
    return lp + lnlike(alpha, time, ref, ref_err,lon,timespan,phispan)

def make_chain(nwalkers, nsteps, ndim, t,r,lon,timespan,phispan,alb=True):
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
    opt_params  = opt_lnlike(init_guess, t, r, r_err,lon,timespan,phispan) 
    
    # initialize nwalkers in a gaussian ball centered on the opt_params
    print ("Intializing Walkers...")
    init_pos = init_walkers(opt_params, t, r, r_err, ndim, nwalkers,lon,timespan,phispan)
    print ("Walkers initialized, ready for destruction!")

    # set up the sampler object and run MCMC 
    print ("Setting up chain")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(t, r, r_err,lon,timespan,phispan))
    sampler.run_mcmc(init_pos, nsteps)
    print ("chaing completed")
    return sampler.chain

def flatten_chain(chain, burnin):
    """
    Input: an emcee sampler chain and the steps taken during the burnin
    Output: a flattened chain, ignoring all steps pre-burnin
    """
    ndim = len(chain[0][0]) # number of params being fit 
    return chain[:,burnin:,:].reshape(-1, ndim)

#THE function that does dem all
def dataAlbedoDynamic(numOfSlices,Days,w,Acloud,surf,clouds,rateDiss,speedCloud,fastforward=1.0,Animation=True):
    cond = False 
    if (Animation):
        plt.ion()
        fig = plt.figure(1,figsize=(12,6))
        dum_alb = [np.random.randint(0,2) for i in range(numOfSlices)]
        dum_lon, dum_lat,dum_A, dum_grid= Eckert(dum_alb,numOfSlices,fig=fig,plot=False)
        css = plt.contourf(dum_lon,dum_lat,dum_A,cmap='gist_gray',alpha=0.3)
        cond = True
    hPerDay = int((w/(2*np.pi))**(-1))

    hours = np.arange(0,hPerDay)
    longitude = timeToLongitude(hours*60*60)
    longitudeF = []
    ndata = []
    #Dont know if the bottom two lines are necessary. Ill keep them for now
    dum_alb = [np.random.randint(0,2) for i in range(numOfSlices)]
    dum_lon, dum_lat,dum_A, dum_grid= Eckert(dum_alb,numOfSlices,plot=False)

    daysPassed = 0

    for i in range(hPerDay*Days):
        if (i%hPerDay == 0 and i!=0):
            daysPassed += 1
        eff,clouds = dynamicMap(i,numOfSlices,Acloud,surf,clouds,rateDiss,speedCloud*fastforward)
        surf = rotateEarth(w,surf,numOfSlices,i,Days)
        diff = int(22/(numOfSlices-2)) 
        indices = [i*diff for i in range(1,numOfSlices)]
        counter = 0
        while (len(indices) != numOfSlices-2):
            if (counter%2 ==0):
                del indices[-1]
            else:
                del indices[0]            
        if (i%hPerDay == 0 or i%hPerDay == hPerDay-1 or any(c == i%(hPerDay) for c in indices)):
            longitudeF.append(360*(Days-daysPassed-1)+longitude[i%len(longitude)])
            ndata.append(eff[0])
        plt.clf()
        if Animation:
            Eckert(eff,numOfSlices,fig=fig,bar=css,plot=cond)
            plt.pause(0.01)
    plt.ioff() 
    return longitudeF,ndata

def extractN(time,apparent,n):
    """
    Function that extracts N evenly spaced data from the apparent albedo array given. 
    """
    limit = len(apparent)
    diff = int(limit/n)
    indices = [i*diff for i in range(n)]
    t = [time[i] for i in indices]
    a = [apparent[i] for i in indices]
    
    """
    if (len(time)%2==0):
        median = len(time)/2
    else:
        median = (len(time)+1)/2
    minI = median-(int(n/2))
    maxI = median +(int(n/2))
    if (maxI > len(time) or minI <0):
        indices = np.arange(0,len(time),n)
        print (len(time))
        print (indices)
        t = [time[i] for i in indices]
        a = [apparent[i] for i in indices]
    else: 
        t = [time[i] for i in range(int(minI),int(maxI))]
        a = [apparent[i] for i in range(int(minI),int(maxI))]
    """
    return t,a

def runShatelite(numOfSlices,Acloud,rateDiss,speedCloud,w,ndata,fastFoward,Days,nwalkers,nsamples,nsteps,timespan,phispan,burning,plot=True,mcmc=True):
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

    l,d=dataAlbedoDynamic(numOfSlices,Days,w,Acloud,surf,clouds,rateDiss,speedCloud,Animation=False)
    print ("Got sum fake data. YOHO, SCIENCE BITCH!")

    for i in range(1,Days+1):
        effective = d[(i-1)*numOfSlices:(i)*(numOfSlices)] 
        lon = l[(i-1)*numOfSlices:(i)*(numOfSlices)]
        #longitude = np.linspace(2*np.pi*(i)*deg,(i-1)*2*np.pi*deg,len(effective),False)
        time, apparent = apparentAlbedo(effective,time_days=timespan,
                long_frac=phispan,n=10000,plot=False,alb=True,lon=lon)
        finalTime.append(time+(hPerDay*(i-1)))
        apparentTime.append(apparent)
        
    finalTime= np.asarray(finalTime).flatten()
    apparentTime = np.asarray(apparentTime).flatten()
    t,a = extractN(finalTime,apparentTime,ndata*Days)
    print ("Done extracting {}".format(numOfSlices))
    
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

    if (mcmc):
        #Implement the MCMC running stuff in a seperate function
        for i in range(1,Days+1):
            chain  = make_chain(nwalkers,nsteps,ndim,t,a,lon,timespan,phispan,alb=True)
            print ("Well call me a slave, because I just made some chains for day {}...".format(i))
            mean_mcmc_params = mcmc_results(chain,burning)
            mean_mcmc_time, mean_mcmc_ref = apparentAlbedo(mean_mcmc_params,time_days=timespan,
                long_frac=phispan,n=10000,plot=False,alb=True,lon=lon)
            print ("Got the mean MCMC results for day {}. Ya YEET!".format(i))

            flat = flatten_chain(chain,burning)
            sample_params = flat[np.random.randint(len(flat),size=nsamples)]
            for s in sample_params:
                sample_time,sample_ref = apparentAlbedo(s,time_days=timespan,long_frac=phispan,n=10000,plot=False,alb=True,lon=lon)
                sample_time = np.asarray(sample_time)
                mean_mcmc_params = np.asarray(mean_mcmc_params)
                plotting_x = np.asarray(sample_time)+(i-1)*hPerDay
                ax.plot(plotting_x,sample_ref,color='k',alpha=0.1)
            #Need to fix the mean     
            ax.plot(mean_mcmc_time+(i-1)*hPerDay,mean_mcmc_ref,color='red',label="Mean MCMC")
            print ("done MCMC for day {}".format(i))
        plt.show()

    return t,a

if __name__ == "__main__":

    numOfSlices = 8                         #Number of slices 
    Acloud = 0.8                            #The average albedo of clouds
    surf = initialPlanet(numOfSlices,False) #Surface albedo of Planet
    clouds = cloudCoverage(numOfSlices)     #Cloud coverage for each slice 
    rateDiss = 1/300                        #Average Rate of dissipation of clouds in hours 
    speedCloud = 126                        #Speed at which clouds move. I put it real high for effects for now (km/h)
    w = 2*np.pi/(24)                        #Angular frequency of the Earth (rad/h^(-1))
    ndim = numOfSlices                      #Number of dimensions for MCMC
    nsamples = 50                           #Number of samples to graph 
    nwalkers = 1100                         #Number of walkers for MCMC
    nsteps = 2200                           #Number of steps for MCMC
    fastForward = 1                         #Speed of how much the dynamic cloud
    Days = 1                                #Number of days that the model spans 
    timespan = 1                            #Time span of data (default of 1.0)
    phispan = 1                             #Fraction of 2pi for the longitude (1 for a full rotation)
    burnin = 1000                           #Burning period
    McMc = True                             #Run MCMC on the simulation or not
    ndata = 22                              #Nnumber of data taken by the satellite


    #Running the simulation
    t,a = runShatelite(numOfSlices,Acloud,rateDiss,speedCloud,w,ndata,fastForward,Days,nwalkers,nsamples,nsteps,timespan,phispan,burnin,plot=True,mcmc=McMc) 
    