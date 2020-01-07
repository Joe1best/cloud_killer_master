import numpy as np 
import itertools
from astropy.time import Time



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

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def date_after(d):
    """
    Input: an integer d
    
    Quickly find out the actual calendar date of some day in the EPIC dataset. 
    
    Output: the date, d days after 2015-06-13 00:00:00.000
    """
    
    t_i = Time("2015-06-13", format='iso', scale='utc')  # make a time object
    t_new_MJD = t_i.mjd + d # compute the Modified Julian Day (MJD) of the new date
    t_new = Time(t_new_MJD, format='mjd') # make a new time object
    t_new_iso = t_new.iso # extract the ISO (YY:MM:DD HH:MM:SS) of the new date
    t_new_iso = t_new_iso.replace(" 00:00:00.000", "") # truncate after DD
    return t_new_iso
