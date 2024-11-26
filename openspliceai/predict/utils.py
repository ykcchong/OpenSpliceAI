###############################################################################
'''This code has functions which process the information in the .h5 files
datafile_{}_{}.h5 and convert them into a format usable by Keras.'''
###############################################################################

from math import ceil
from openspliceai.constants import *

def ceil_div(x, y):
    """
    Calculate the ceiling of a division between two numbers.

    Parameters:
    - x (int): Numerator
    - y (int): Denominator

    Returns:
    - int: The ceiling of the division result.
    """
    return int(ceil(float(x)/y))