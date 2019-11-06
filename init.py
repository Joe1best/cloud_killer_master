import numpy as np 

#Number of slices
NUMOFSLICES = 4
#The average albedo of clouds 
ACLOUD = 0.8 
#Average rate of dissipation of clouds
RATEDISS = 1/3
#Speed at which clouds move 
SPEEDCLOUD = 126
#Angular frequency of the Earth 
WW = 2*np.pi/24
#Number of dimensions for MCMC
NDIM = NUMOFSLICES
#Number of samples to graph 
NSAMPLES = 25
#Number of walkers to graph
NWALKERS = 100 
#How many times to run the simulation 
NTRIALS = 1
#Number of steps for MCMC. Should be an array 
NSTEPS = [500]
#Speed of how much the clouds move 
FASTFORWARD = 1
#Number of days the model runs for 
DAYS = 1
#Time span of data (Default of 1.0)
TIMESPAN = 1
#Fraction of 2pi for the longitude (2pi for a full rotation)
PHISPAN = 1
#Burning period of the MCMC (Default at 150)
BURNIN = 150
#Run the MCMC for the simulation or not 
MCMC = False
#To plot the things or not
PLOT = True
#Number of data points sampled from the simulated satellite
NDATA = 22
#Plays the same role as NDIM
NPARA = NUMOFSLICES
#
REPEAT = False
#To generate clouds or not 
CLOUDFORMING = True
#To plot the walkers or not
WALKERS = False
