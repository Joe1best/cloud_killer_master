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

#In[]:
#Intializing the model
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
    albedo = [1,1,0,0]

    if plot:
        Eckert(albedo,len(albedo))
        plt.show()
    for i in sliceNum:
        planet[i] = albedo[i]
    
    return planet

def cloudCoverage(numOfSlices):
    """
    Function that generates a cloud coverage, and calculates its effect on the albedo generated
    using initialPlanet.

    Outputs: random values for each slice that ranges from 0 to 1. 1 being total cloud cover 
    and 0 having no clouds. 
    """
    booleanSlice = [ran.randint(0,1) for i in range(numOfSlices)]
    clouds = np.zeros(numOfSlices)
    clouds = [0,0,0,0]
    for i in range(numOfSlices):
        if booleanSlice[i]==1:
            clouds[i] = ran.random()
        else:
            clouds[i] = 0
    return clouds

#In[]:
#Forward model
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

def integral(phi,phi_obs,albedos):
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
    C = (4/(3*np.pi))
    a = sliceFinder(len(albedos),phi,albedos)
    integral = a*((1/2)*(phi[1]-phi[0])+(1/4)*np.sin(2*phi[1]-2*phi_obs)-(1/4)*np.sin(2*phi[0]-2*phi_obs))    
    return C*integral

def visibleAngle(hour,longitudes,Days):
    """
    Function that calculates the angle visible for the satellite depending on the hour 
    Inputs
        hour: hour elapsed in the simulation. 
        longitudes: longitudes slices that are pre-defined beforehand. 
    """
    hour = [hour*60*60]
    
    currentLon = np.deg2rad(timeToLongitude(hour)) + 2*np.pi*(Days-1)

    if np.pi*(Days)< currentLon < 2*np.pi*Days:
        TwoBound = (currentLon - np.pi/2) + 2*np.pi*(Days-1)
        diff = (2*np.pi-currentLon) + 2*np.pi*(Days-1)
        OneBound = (np.pi/2 - diff) 
    elif np.pi*(Days-1) < currentLon < np.pi*(Days):
        OneBound = currentLon + np.pi/2
        diff = np.pi/2 - currentLon
        TwoBound = 2*np.pi - diff
    elif currentLon == 0 or currentLon == 2*np.pi:
        TwoBound = 2*np.pi - np.pi/2
        OneBound = np.pi/2
    else: 
        print ("Something went wrong, code stopping")

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
        if np.round(bounds[0],6)>=np.round(borders[i],6) and np.round(bounds[1],6)<=np.round(borders[i+1],6):
            return albedo[i] 
    raise Exception('Could not find the albedo between {} and {}. The borders were {} and {}. Something fishy going on here...'.format(bounds[0],bounds[1],borders[i],borders[i+1]))

def apparentAlbedo(albedos, time_days=1.0, long_frac=1.0, n=10000, phi_obs_0=0.0, 
               plot=False, alb=False, ndata=None): 
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
    numOfSlices = len(albedos)
    longitudes = np.ndarray.tolist(np.linspace(2*np.pi,0,numOfSlices+1))

    longitudes.reverse()
    longitudes = np.asarray(longitudes)
    trueLon = longitudes

    time = np.linspace(0, 24*time_days , n , False)
    
    w_Earth = 2.0*np.pi/24 # Earth's angular velocity 
    phi_obs = phi_obs_0 - 1.0*w_Earth*time # SOP longitude at each time

    final = np.zeros([len(albedos),n])
    
    for i in range(len(time)):
        AllLimits = np.zeros([len(albedos),2])
        
        #Check this line
        longitudes = trueLon
        Two, One = visibleAngle(time[i],longitudes,time_days)
        cond = True
    
        #Slice is crossing the middle. Need to split it in 2
        if (One>Two):
            longitudes = [x for x in longitudes if not (Two <= x <= One)]
            longitudes = np.asarray([Two] + longitudes + [One])
            rightSide = sorted([x for x in longitudes if (0 <= x <= np.pi)])
            leftSide = sorted([x for x in longitudes if (np.pi < x <= 2*np.pi)])
            cond = True
        #When the region is not including the middle
        elif (Two>One):
            longitudes = [x for x in longitudes if (Two >= x >= One)]
            longitudes = sorted(np.asarray([One] + longitudes + [Two]))
            cond = False

        count = 0 
        for k in range(len(albedos)):
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
            else:
                if k+1 <len(longitudes):
                    AllLimits[k][0] = longitudes[k]
                    AllLimits[k][1] = longitudes[k+1]
                else:
                    break
            final[k][i] = integral([AllLimits[k][0],AllLimits[k][1]],phi_obs[i],albedos)

    lightcurveR = sum(final)
    if alb: 
        lightcurveR = lightcurveR*(3/2)
    #lightcurveR = np.flip(lightcurveR)

    return time, lightcurveR

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
    #SET MAX OF COLORBAR TO 1
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r'Apparent Albedo $A^*$')
    ax.coastlines()
    ax.gridlines(color='grey',linewidth=1.5,xlocs = gridlines,ylocs=None) 
    ax.set_global()
    #plt.show()
    return longitude,lattitude,A,cs,fig,ax,gridlines

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

# In[]:
#Cloud model & earth rotation
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

def dynamicMap(time,numOfSlices,Acloud,surfAlb,cloudIn,rateDiss,speedCloud,forming=True):
    """
    Function that changes the cloud dynamically per iteration. Changes every hour
    """
    dissipate(cloudIn,time,rateDiss)
    if (forming):
        form(cloudIn,time)
    cloudIn = move(cloudIn,time,speedCloud)
    effAlb=effectiveAlbedo(numOfSlices,Acloud,False,calClouds=cloudIn,calsurf=surfAlb)
    return effAlb,cloudIn

def rotate(albedo):
    albedo = np.roll(albedo,-1)
    return albedo 

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
        ax.plot(plotting_x,sample_ref,color='k',alpha=0.1)
        ax.plot(mean_mcmc_time+(i-1)*hPerDay,mean_mcmc_ref,color='red',label="Mean MCMC")
        plt.show()
    return mean_mcmc_ref   

def plot_walkers_all(chain,expAlb):
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
        plt.xlabel("Steps")
        for p in paths:
            if n is not ndim-1:
                plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
            plt.plot(step_number, p,color='k',alpha=0.3) # all walker paths
            plt.axhline(expAlb[n],color='red',linewidth=1) #Draw the expected value
            plt.ylabel(r"$A$"+"[%d]"%(n)) # label parameter
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

#PROBLEM WITH 10 SLICE ALBEDO for dataAlbedoDynamic
#In[]:
#Versions of simulations: runShatellite --> VERSION 1: fits normally
#                         runShatelite  --> VERSION 2: "dynamic fit" (NOT DONE)
#                         runSatellowan --> VERSION 3: static map each day (NEEDS TWEAKING) 
def dataAlbedoDynamic(numOfSlices,Days,w,Acloud,surf,clouds,rateDiss,speedCloud,fastforward=1.0,
        Animation=True,hourlyChange=True,repeat=False,forming=True):
    """
    Function that simulates the satellite taking ndata amount of data points per day, depending on the surface 
    albedo and cloud coverage. It also simulates the Earth rotating and the clouds moving. It was assumed that 
    the clouds moved from East to West (so the cloud coverage array was getting shifted accordingly) and the
    planet itself is rotating West to East (the surface albedo is the one getting shifted).
    Inputs: 
        numOfSlices: Number of slices. 
        Days: How many days the satellites takes data for. 
        w: rotational angular velocity of the planet.
        Acloud: The average albedo of clouds (default 0.8).
        surf: An array of surface albedo for each slice.
        clouds: An array of cloud coverage for each slice. 
        rateDiss: The rate of dissipation of clouds (units of hours, default 1/3)
        speedCloud: The average velocity of clouds in the simulation (in km/h, default 126)
        fastforward: How sped up the model is (default 1)
        Animation: If True, will display an animation of how the slices change.
    """
    
    cond = False 
    if (Animation):
        plt.ion()
        fig = plt.figure(1,figsize=(12,6))
        dum_alb = [np.random.randint(0,2) for i in range(numOfSlices)]
        dum_lon, dum_lat,dum_A, dum_grid= Eckert(dum_alb,numOfSlices,fig=fig,plot=False)
        css = plt.contourf(dum_lon,dum_lat,dum_A,cmap='gist_gray',alpha=0.3)
        cond = True
    hPerDay = int((w/(2*np.pi))**(-1))

    longitudeF = np.linspace(2*np.pi*Days,0,(numOfSlices+1)*Days)
    ndata = []

    #Dont know if the bottom two lines are necessary. Ill keep them for now
    dum_alb = [np.random.randint(0,2) for i in range(numOfSlices)]
    dum_lon, dum_lat,dum_A, dum_grid = Eckert(dum_alb,numOfSlices,plot=False)
    eff = effectiveAlbedo(numOfSlices,Acloud,plot=True,calClouds=clouds,calsurf=surf)
    daysPassed = 0
    #eff = effectiveAlbedo(numOfSlices,Acloud,plot=False,calClouds=clouds,calsurf=surf)
    #Loop that loops through the 24 hour cycle (each i is 1 hour)

    #Change 22 for ndata
    if (numOfSlices == 2):
        diff = 12
    else:
        diff = int(hPerDay/(numOfSlices)) 
    indices = [i*diff for i in range(0,numOfSlices*Days)]
    rotations = 0
    for i in range(hPerDay*Days): 
        #Calculates the days passed depending on how many hours per day.
        if (i%hPerDay == 0 and i!=0):
            daysPassed += 1        
        
        #Changes the map  
        if (hourlyChange):
            eff,clouds = dynamicMap(i,numOfSlices,Acloud,surf,clouds,rateDiss,speedCloud*fastforward,forming=forming)
        elif(i%hPerDay == 0 and hourlyChange == False and repeat==False):
            eff,clouds = dynamicMap(i,numOfSlices,Acloud,surf,clouds,rateDiss,speedCloud*fastforward,forming=forming)

        #Rotates the earth 
        if (hourlyChange and repeat==False):
            surf = rotateEarth(w,surf,numOfSlices,i,Days)
            rotations += 1
        elif(hourlyChange==False and i%hPerDay==0):
            print ("we out here")
            print (surf)
            surf = rotate(surf)
            rotations += 1
        else: 
            eff = rotateEarth(w,eff,numOfSlices,i,Days)
            rotations += 1
        #counter = 0
        
        #Removes the first and last index of the list (this for data extraction)
        """
        while (len(indices) != numOfSlices-2):
            if (counter%2 == 0):
                del indices[-1]
            else:
                del indices[0]            
        """
        #"Data taking", takes it for the 0th and 23th hour and evenly spaced out points 
        #in between depending on ndata.
        
        #Maybe something wrong here??
        if (i%hPerDay == 0 or i%hPerDay == hPerDay or any(c == i%(hPerDay) for c in indices)):
            #longitudeF.append(360*(Days-daysPassed-1)+longitude[i%len(longitude)])
            ndata.append(eff[0])
        plt.clf()
        
        #If Animation is TRUE, it will draw an Eckert Projection animation. The plot
        #settings can be found starting from 482 to 488.
        if Animation:
            Eckert(eff,numOfSlices,fig=fig,bar=css,plot=cond)
            plt.pause(0.01)
    plt.ioff() 

    return longitudeF,ndata,rotations

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
                long_frac=phispan,n=10000,plot=False,alb=True)
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
        nwalkers,nsamples,nsteps,timespan,phispan,burning,plot=True,mcmc=True,repeat=False,walkers=False,forming=True):
    #Need to make this a bit more efficient)
    if (numOfSlices>24):
        print ("Cannot have more than 24 number of slices for now")
        return 0,0
    print (numOfSlices)
    surfDict = initialPlanet(numOfSlices,plot=True)
    surf = np.fromiter(surfDict.values(),dtype=float)
    print ("The planet's surface albedo is theoritically", surf)
    clouds = cloudCoverage(numOfSlices)
    print ("The effective albedo is ", effectiveAlbedo(numOfSlices,Acloud,plot=False,calClouds=clouds,calsurf=surf))
 
    hPerDay = int((w/(2*np.pi))**(-1))
    finalTime = []
    apparentTime = []
    ndim = numOfSlices
    print (clouds, "Cloud coverage is ")
    l,d,rotations = dataAlbedoDynamic(numOfSlices,Days,w,Acloud,surf,clouds,rateDiss,speedCloud,Animation=False,hourlyChange=False,repeat=repeat,forming=forming)

    for i in range(1,Days+1):
        effective = d[(i-1)*numOfSlices:(i)*(numOfSlices)] 
        print (effective,"herehehrehrherheh")
        print("The albedo map for day {} is ".format(i-1), effective)
        #lon = l[(i-1)*numOfSlices:(i)*(numOfSlices)]
        #print (effective)
        #print ("Longitude is ", lon)
        time, apparent = apparentAlbedo(effective,time_days=timespan,
                long_frac=phispan,n=10000,plot=False,alb=True)
        finalTime.append(time+(hPerDay*(i-1)))
        apparentTime.append(apparent)
        
    finalTime= np.asarray(finalTime).flatten()
    apparentTime = np.asarray(apparentTime).flatten()

    t,a = extractN(finalTime,apparentTime,ndata,Days)
    
    print ("Done extracting {}".format(numOfSlices))
    #[]
    if plot:
        fig,ax = plt.subplots(1,1,gridspec_kw={'height_ratios':[1]},figsize=(10,8))
        tim, albedo = drawAlbedo(effectiveAlbedo(numOfSlices,Acloud,plot=False,calClouds=clouds,calsurf=surf),w,50)
        #for i in range(Days+1):
        #    ax.axvline((i)*hPerDay,color='orange',alpha=1,zorder=10)
        ax.plot(finalTime,apparentTime,'--',color='black',linewidth=4,label="Simulated curve")
        ax.plot(tim,albedo,'--',color='purple',label='Albedo Generated for {} slices'.format(numOfSlices),alpha=0.3)
        ax.errorbar(t,a,fmt='.',color='orange',yerr = np.asarray(a)*0.02,markersize=10,solid_capstyle='projecting', capsize=4,
                    label= "Selected {} data".format(ndata))
        ax.set_xlabel("Time (h)",fontsize=22)
        ax.set_ylabel("Apparent Albedo ($A^*$)",fontsize = 22)
        ax.tick_params(labelsize=22)
    
    chainArray=[]
    alb=[]
    if (mcmc):
        #Implement the MCMC running stuff in a seperate function
        for i in range(1,Days+1):
            time = t[(i-1)*ndata:i*ndata]
            app = a[(i-1)*ndata:i*ndata]
            #lon = l[(i-1)*numOfSlices:(i)*(numOfSlices)]
            #lon = [longitude%(2*np.pi) for longitude in lon]
            #for j in range(len(lon)):
            #    if (lon[j]>0 and lon[j]<1) or lon[j]==0:
            #        lon[j] = 2*np.pi
            
            #print (lon,"Longitudes again")
            chain  = make_chain(nwalkers,nsteps,ndim,time,app,timespan,phispan,alb=True)
            chainArray.append(chain)
            print ("Well call me a slave, because I just made some chains for day {}...".format(i))
            mean_mcmc_params = mcmc_results(chain,burning)
            mean_mcmc_time, mean_mcmc_ref = apparentAlbedo(mean_mcmc_params,time_days=timespan,
                long_frac=phispan,n=1000,plot=False,alb=True)
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
    hour = dt.datetime.now().time()
    fileName = hour.strftime("%H:%M:%S").replace(":","_",2)
    f= open("MCMC_{}__{}.csv".format(fileName,numOfSlices),"w+") 
    f.write(''.join(str(effectiveAlbedo(numOfSlices,Acloud,plot=False,calClouds=clouds,calsurf=surf)).replace("[","",1).replace("]","",1))+"\n")
    for chainy in chainArray:
        results = mcmc_results(chainy,burning)
        if (walkers):
            plot_walkers_all(chainy,effectiveAlbedo(numOfSlices,Acloud,plot=False,calClouds=clouds,calsurf=surf))
        alb.append(results)
        print ("The result for day {} is".format(counter),results)
        f.write(''.join(str(results).replace("[","",1).replace("]","",1))+"\n")
        effmean.append(np.mean(results))
        counter += 1
    f.close()
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
    #print (np.transpose(alb),"before")
    #alb = fixAlbedo(alb,rotations)
    #print (np.transpose(alb),"after")
    minAlb = minimumAlb(alb)
    return alb,minAlb
    #,aic,bic

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
    
    final = np.asarray(albhour)*4/(slices)

    return np.flip(x),final

def minimumAlb(albedos):
    """
    Function that returns the minimum albedo of each slice 
    """
    nslice = len(albedos[0])
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

#In[]:
#Main method
if __name__ == "__main__":
    #VARY CLOUDS VERY SLOWLY such that the time map is not varying that much.
    #SEE IF I CAN EXTRACT THE TIME CHANGE dissipation. HOW I CAN FIT FOR THAT
    #Add a verbose option
    #A lot of parameters eh 
    numOfSlices = 4                          #Number of slices 
    Acloud = 0.8                             #The average albedo of clouds
    #surf = initialPlanet(numOfSlices,False) #Surface albedo of Planet
    
    #clouds = cloudCoverage(numOfSlices)    #Cloud coverage for each slice 
    #NEED TO FIT FOR THIS
    rateDiss = 0                            #Average Rate of dissipation of clouds in hours 
    speedCloud = 126                        #Speed at which clouds move. I put it real high for effects for now (km/h)
    w = 2*np.pi/(24)                        #Angular frequency of the Earth (rad/h^(-1))
    ndim = numOfSlices                      #Number of dimensions for MCMC
    nsamples = 50                           #Number of samples to graph 
    nwalkers = 100                          #Number of walkers for MCMC
    #nwalkers = [100,200,300,400,500,600,700,800,900,1000]
    #nwalkers = [1500]
    ntrials = 1
    nsteps = [300]                          #Number of steps for MCMC
    fastForward = 1                         #Speed of how much the dynamic cloud
    Days = 1                                #Number of days that the model spans 
    timespan = 1                            #Time span of data (default of 1.0)
    phispan = 1                             #Fraction of 2pi for the longitude (1 for a full rotation)
    burnin = 150                            #Burning period
    McMc = True                             #Run MCMC on the simulation or not
    PlOt = True                             #Whether to plot things or not
    ndata = 22                              #Nnumber of data taken by the satellite
    npara = numOfSlices
    repeat = False
    cloudForming = False
    #surf = [0.587,0.273,0.219,0.745,0.888,0.156,0.914,0.677]
    #clouds = [0.821,0.234,0.301,0.785,0.122,0.004,0.063,0.925]

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
    
    for steps in nsteps:
        aics = []
        bics = []
        for j in range(ntrials):
            dumAlb,dumMin= runSatellowan(numOfSlices,Acloud,npara,rateDiss,speedCloud,w,ndata,fastForward,Days,
                    nwalkers,nsamples,steps,timespan,phispan,burnin,plot=PlOt,mcmc=McMc,repeat=repeat,walkers=False,forming=cloudForming)
    print (dumMin)
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