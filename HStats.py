__doc__ = """
A tiny package to calculate 4 distinctive statistics of harmony for
chroma vectors (preferably 12 chroma).

Functions:  ----------------------------------------
End-user functions: --------------------------------


    sortToCif(l):
        iterable(a) -> ndarray(a)
        Hypotheses: len(l) == 12
        >>  Sorts an iterable as if indexed on the circle of fifths and
            turns it into an array.


    toCircD(cv):
        ndarray(real) -> ndarray(complex)
        Hypotheses: len(cv) == 12
        >>  Turns a real array of length 12 into a complex based circular
            distribution. cv[k] maps to cv[k]*exp(k/12*np.pi*j)


    resVect(V):
        ndarray(complex) -> complex
        Hypotheses: V is preferably non-zero, else returns NaN.
        >>  Calculates the resultant vector for a complex based
            circular distribution.


    hCenter(rv):
        complex -> float
        Hypotheses: /
        >>  Calcuates the harmonic center corresponding to a complex based
            resultant vector, that is the location on the circle of fifths
            the resultant vector point to.


    variance(rv):
        complex -> float([0,1])
        Hypotheses: 0 <= abs(rv) <= 1
        >>  returns the circular variance associated with the resultant vector


    harmoniousness(V, rv):
        ndarray(complex) -> complex -> float
        Hypothese: rv == resVect(V) && len(V) == 12
        >>  calculates harmoniousness for a circular distribution V of
            resultant vector rv.


    coharmoniousness(V, rv):
        ndarray(complex) -> complex -> float
        Hypothese: rv == resVect(V) && len(V) == 12
        >>  calculates coharmoniousness for a circular distribution V of
            resultant vector rv.


    computeHStats(cvCif):
        sortToCif(ndarray(Number)) -> ndarray(float)
        Hypotheses: len(cvCif) == 12
        >>  Computes the statistics and returns them as an ndarray:
            [<harmonic center>, <variance>, <harmoniousness>, <coharmoniousness>]


    printHStats(cvCif):
        sortToCif(ndarray(Number)) -> IO
        Hypotheses: len(cvCif) == 12
        >>  Quick and dirty way of printing a chord's stats.


Functions used for miscellanious computations:------


    vabs(V):
        iterable(a) -> ndarray(a)
        Hypotheses: abs() must be defined on 'a' and 'a' must be immutable.
        >>  Maps the absolute value over an iterable and puts it into an array.


    readAsCif(l, i):
        iterable(a) -> nat -> a
        Hypotheses: len(l) must be at least and preferably at most 12.
        >>  Returns l[7*i % 12], so that 'i' acts as an index on the circle
            of fifths.


    hContrib(v, rv) & cohContrib(v, rv):
        complex**2 -> float
        Hypotheses: /
        >>  Formulas for calculating the contribution of each vector 'v'
            given the resultant vector 'rv' into harmoniousness and
            coharmoniousness respectively.


Constants:  ----------------------------------------

    Cst0_11:    Used in the construction of unitCplx.
    unitCplx:   Complex array of the 12th roots of unity,used to vectorize
                the function toCircD.

"""


import numpy as np
import math as m
import numba as nb


# Constants -------------------------------------------------------------------

Cst0_11 = np.array(range(12))
unitCplx = np.array((np.cos(2*np.pi*Cst0_11/12)+ 1j*np.sin(2*np.pi*Cst0_11/12)))

# Functions -------------------------------------------------------------------

# vectorized absolute value
vabs = lambda V: np.fromiter((abs(v) for v in V), float)
# readAsCif is used to read an iterable as if indexed on the circle of fifths.
readAsCif = lambda l, i: l[7*i % 12]

# sortCif sorts an entire array as if indexed on the circle of fifths.
sortToCif = lambda l: np.array([readAsCif(l, i) for i in range(12)])
# toCircD converts a chroma vector to a circular distribution.
toCircD = lambda cv: cv*unitCplx
# calculates the resultant vector of order n
resVect = lambda V: sum(V)/sum(vabs(V))

# harmonic and coharmonic contributions of some vector given the resultant vector of its distribution
hContrib = lambda v, rv: abs(v)*np.cos(np.angle(v)-np.angle(rv))
cohContrib = lambda v, rv: abs(v)*np.sin(np.angle(v)-np.angle(rv))

# harmonic center
hcenter = lambda rv: (12*np.angle(rv)/(2*np.pi)) % 12
# variance of the circular distribution
variance = lambda rv: (1-abs(rv))
# harmoniousness and coharmoniousness for a distribution and resultant vector
harmoniousness = lambda V, rv: sum(np.fromiter((x*abs(x) for x in np.fromiter((hContrib(v, rv) for v in V), float)), float))
coharmoniousness = lambda V, rv: sum(np.fromiter((x*abs(x) for x in np.fromiter((cohContrib(v, rv) for v in V), float)), float))

# Computes the statistics and returns them as a numpy array: [<harmonic center>, <variance>, <harmoniousness>, <coharmoniousness>]
def computeHStats(cvCif):
    V = toCircD(cvCif)
    rv = resVect(V)

    hc = hcenter(rv)
    hv = variance(rv)
    hi = harmoniousness(V, rv)
    cohi = coharmoniousness(V, rv)

    return np.array([hc, hv, hi, cohi])

# Quick and dirty way of showing a chord's stats.
def printHStats(cvCif):
    hc, hv, hi, cohi = computeHStats(cvCif)
    print('Harmonic center =   ', hc)
    print('Variance = ', hv)
    print('Harmoniousness =         ', hi)
    print('Coharmoniousness =       ', cohi)
