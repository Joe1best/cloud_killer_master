import numpy as np 
import matplotlib.pyplot as plt 
import random as ran
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.animation as animation
from matplotlib import style
import collections
import emcee
import scipy.optimize as op


def timeToLongitude(time):
    """
    Function that converts time since an arbitrary start of the simulation to 
    longitude. Important that the input is in SECONDS. 
    """
    longitude = [(2*np.pi-(t%86148.0)*(2*np.pi/86148.0))%(2*np.pi) for t in time]
    longitude = np.rad2deg(longitude)
    return longitude

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
    time = np.linspace(0.0, time_days*24, n, False)
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
        ax1.plot(time, reflectance,'.', color='red')
        if alb: # if we applied the 3/2 factor
            ax1.set_ylabel("Apparent Albedo "+r"$A^*$")
        else: 
            ax1.set_ylabel("Reflectance")
        ax1.set_xlabel("Time [h]")
        plt.show()
    
    return time, reflectance
#Delete this function
def A(albedo,time,numberOfslices):
    """
    Function that calculates the apparent albedo for a longitudinal slice. It assumes that 
    the Earth rotates in 23.93 hours as per the EPIC data. 
    """
    gridlines = Eckert(albedo,numberOfslices,plot=False)[3]
    A = np.zeros(len(albedo))
    w = 2*np.pi/(23.93)
    #time = [i*23.93/numberOfslices for i in range(numberOfslices)]
    for i in range(len(A)):
        A[i] = 4/(3*np.pi)*albedo[i]*((1/2)*(gridlines[i]-gridlines[i+1])+(1/4)*np.sin(2*(gridlines[i]-gridlines[i+1]+w*time)))
    return A

def Eckert(albedo,numOfSlices,nlats=400,nlons=400,fig=None,plot=True,data=False):
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
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r'Apparent Albedo $A^*$')
    ax.coastlines()
    ax.gridlines(color='grey',linewidth=1.5,xlocs = gridlines,ylocs=None) 
    ax.set_global()
    #plt.show()
    #ax.clear()
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

#MCMC
def lnlike(alpha, time, ref, ref_err):
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
    timespan = (time[-1]-time[0]) # time spanned, in days
    phispan = timespan # longitude spanned, as a fraction of 2pi
    
    # obtain model prediction, in units of apparent albedo
    model_time, model_ref = apparentAlbedo(alpha, timespan, phispan, timepts, 
                                       0, plot=False, alb=True) 
    
    # compute ln(likelihood)
    chisq_num = np.power(np.subtract(ref,model_ref), 2) # (data-model)**2
    chisq_denom = np.power(ref_err, 2) # (error)**2
    res = -0.5*sum(chisq_num/chisq_denom + np.log(2*np.pi) + np.log(np.power(
            ref_err,2))) #lnlike
    
    return res

def opt_lnlike(alpha, time, ref, ref_err):
    """
    Input: guesses for the fit parameters (alpha, an array of albedos, 
    representing A(phi)) and the time, lightcurve, and error on the lightcurve 
    of the data being fit 
    
    Maximizes the ln(likelihood).
    
    Output: The values of albedos with maximum likelihood
    """
    nll = lambda alpha,time,ref,ref_err: -lnlike(alpha,time,ref,ref_err) # return -lnlike of args
    # boundaries on the possible albedos:
    bound_alb = tuple((0.000001,0.999999) for i in range(len(alpha))) 
    # minimize (-ln(like)) to maximimize the likelihood 
    result = op.minimize(nll, alpha, args=(time,ref,ref_err), bounds=bound_alb)
    
    return result['x'] # the optimized parameters

def init_walkers(alpha, time, ref, ref_err, ndim, nwalkers):
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
    opt_albs = opt_lnlike(alpha, time, ref, ref_err) # mazimize likelihood
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

def lnpost(alpha, time, ref, ref_err):
    """
    Input: guesses for the fit parameters (alpha, an array of albedos,
    representing A(phi)) and the time, lightcurve, and error on the lightcurve 
    of the data being fit 
    Output: ln(posterior)
    """
    lp = lnprior(alpha)
    if not np.isfinite(lp): # if ln(prior) is -inf (prior->0) 
        return -np.inf      # then ln(post) is -inf too
    return lp + lnlike(alpha, time, ref, ref_err)

def make_chain(nwalkers, nsteps, ndim, alpha,timespan,phispan,nsamples,phi_init,plot=False,alb=True):
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
    if type(alpha) != type(None): 
        t, r = apparentAlbedo(alpha,timespan,phispan,nsamples,phi_init,plot,alb)
        r_err = 0.02*r # assuming 2% error     
        # add gaussian noise to the data with a variance of up to 2% mean app alb
        gaussian_noise = np.random.normal(0, 0.02*np.mean(r), len(r))
        r += gaussian_noise
    # if neither a day nor an articial albedo map is supplied
    else:
        print("Error: please supply either a day of interest in the EPIC data \
              or a synthetic array of albedo values.")
        return
    
    # guess: alb is 0.25 everywhere
    init_guess = np.asarray([0.25 for n in range(ndim)])
    # better guess: maximize the likelihood
    opt_params  = opt_lnlike(init_guess, t, r, r_err) 
    
    # initialize nwalkers in a gaussian ball centered on the opt_params
    init_pos = init_walkers(opt_params, t, r, r_err, ndim, nwalkers)
    
    # set up the sampler object and run MCMC 
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(t, r, r_err))
    sampler.run_mcmc(init_pos, nsteps)
    return sampler.chain

#dynamic 
def weather_creater(numOfSlices,Days,Acloud,surf,clouds,rateDiss,speedCloud,fastforward=1.0,Animation=True):
    sliceInst = 0
    cond = False 
    if (Animation):
        plt.ion()
        fig = plt.figure(1,figsize=(12,6))
        cond = True
    longitude = np.linspace(2*np.pi*Days,0,24*Days)
    for i in range(24*Days):
        sliceInst = sliceInst+1
        eff,clouds = dynamicMap(i,numOfSlices,Acloud,surf,clouds,rateDiss,speedCloud*fastforward)
        if (sliceInst==24*Days+1):
            break
        plt.clf()
        Eckert(eff,numOfSlices,fig=fig,plot=cond)
        plt.pause(0.01)
        

    
    return 0 

if __name__ == "__main__":

    numOfSlices = 20
    Acloud = 0.8
    surf = initialPlanet(numOfSlices,False)
    clouds = cloudCoverage(numOfSlices)
    rateDiss = 1/180   #Rate of dissipation of clouds 
    speedCloud=80000    #Speed at which clouds move. I put it real high for effects for now
    #plt.ion()          #Responsible for the animations
    #fig = plt.figure(1,figsize=(12,6))

    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fastForward=1
    #sliceInst  =0

    Days = 3
    weather_creater(numOfSlices,Days,Acloud,surf,clouds,rateDiss,speedCloud,Animation=True)

    #Implement apparent albedo (check with Cowan. but for now it is done)
    #Implement this as a function (check)
    #Try and find a way to save this as a gif 
    #Take data with the satellite of this model
    #PARAMTRIZE EFFAPP AND EFF
    """
    ims = []
    fakeData = []
    longitude = np.linspace(2*np.pi*Days,0,24*Days)
    for i in range(24*Days):
        sliceInst = sliceInst+1
        eff,clouds=dynamicMap(i,numOfSlices,Acloud,surf,clouds,rateDiss,speedCloud*fastForward) 
        effApp = A(eff,i,numOfSlices)
        if (sliceInst==24*Days+1):
            break
        fakeData.append(effApp[sliceInst%numOfSlices])
        plt.clf()
        Eckert(effApp,numOfSlices,fig=fig)
        im = Eckert(effApp,numOfSlices,plot=False,data=True)
        ims.append(im)
        print("t =" , i)
        plt.pause(0.01)
        
    g= plt.figure(2)
    for i in range(3):
        plt.axvline((i)*2*np.pi,color='red')
    plt.plot(longitude,fakeData,'.')
    plt.xlabel("Latitude (rad)")
    plt.ylabel("Albedo")
    g.show()

    input()


    #ani = animation.FuncAnimation(fig,ims, interval=50,repeat_delay=3000,blit=True)
    #print (ani)
    #ani.save("test.mp4")
    """