import numpy as np 
import matplotlib.pyplot as plt
import Model_Init as M_Init
import corner
import scipy.optimize as op
import emcee
import init as var


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
    model_time, model_ref = M_Init.apparentAlbedo(alpha, timespan, phispan, timepts, 
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


#In[]
#MCMC model 
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
    mean_mcmc_time, mean_mcmc_ref = M_Init.apparentAlbedo(mean_mcmc_params,time_days=timespan,
        long_frac=phispan,n=5000,plot=False,alb=True)
    print ("Got the mean MCMC results for day {}. Ya YEET!".format(i))

    flat = flatten_chain(chain,burning)
    sample_params = flat[np.random.randint(len(flat),size=var.NSAMPLES)]
    for s in sample_params:
        sample_time,sample_ref = M_Init.apparentAlbedo(s,time_days=timespan,long_frac=phispan,n=1000,plot=False,alb=True)
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
