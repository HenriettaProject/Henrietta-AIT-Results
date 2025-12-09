import numpy as np
import pylab as pl

import photutils
from photutils.detection import DAOStarFinder
import astropy.io.fits as fits
from scipy import ndimage as ndi
from scipy.signal import convolve, convolve2d
from scipy.spatial.kdtree import KDTree as KDT


"""

On Henrietta the [X, Y] X axis is the spatial direction and Y is spectral.


"""


def hartman_focus_by_peak_finding(datatop, databottom, fwhm_pix=5.0, threshold=1500):
    """Find focus of a pair of hartmann frames using DAOStarFinder

    Args:
        datatop (float array): Top open
        databottom (float array): Bottom open
        fwhm_pix (float, optional): Estimated FWHM. Defaults to 5.0.
        threshold (int, optional): Minimum Threshold, below to ignore.

    Returns:
        tuple (4):
            top KD tree
            bottom KD tree
            focus offsets in X
            focus offsets in Y
    """
    
    print("Threshold is: %i, fwhm is %f" % (threshold, fwhm_pix)) ; 
    daofind = DAOStarFinder(fwhm=fwhm_pix, threshold=threshold, peakmax=60000.)
    xc = "xcentroid"
    yc = "ycentroid"

    s_t = daofind(datatop)
    s_b = daofind(databottom)

    pos_t = np.transpose((s_t[xc], s_t[yc]))
    pos_b = np.transpose((s_b[xc], s_b[yc]))

    tt = KDT(pos_t)
    tb = KDT(pos_b)

    offsets_x = [] ; offsets_y = []
    for point in tt.data:
        _, ix = tb.query(point)
        dx = point[0] - tb.data[ix][0]
        dy = point[1] - tb.data[ix][1]

        offsets_x.append(dx)
        offsets_y.append(dy)


    return tt,tb,np.array(offsets_x),np.array(offsets_y)

    
    
if __name__ == """__main__""":
    tname = "data/hen6860.fits"
    bname = "data/hen6861.fits"
    
    
    dt = fits.open(tname)[0].data
    db = fits.open(bname)[0].data

    
    tt, tb, dx, dy = hartman_focus_by_peak_finding(dt, db)

    x,y = tt.data.T[0], tt.data.T[1]

    ok = (np.abs(dx-2) < 1) & (np.abs(dy - 5) < 4)

    pl.clf()
    pl.figure(1)
    pl.subplot(2,1,1)
    pl.scatter(x[ok], y[ok], c=dx[ok])
    pl.xlabel("X")
    pl.xlabel("y")
    pl.title("dX")
    pl.colorbar()

    pl.subplot(2,1,2)
    pl.scatter(x[ok], y[ok], c=dy[ok])
    pl.title("dY")
    pl.colorbar()

    pl.figure(2)
    pl.clf()
    pl.subplot(2,1,1)
    pl.plot(x[ok], dy[ok], 'o')
    pl.xlabel("X") ; pl.ylabel("dY")

    pl.subplot(2,1,2)
    pl.plot(y[ok], dy[ok], 'o')
    pl.xlabel("Y") ; pl.ylabel("dY")

    
    pl.figure(3)
    pl.clf()
    pl.imshow(dt, vmin=-10, vmax=2000)
    pl.plot(x, y, 'b.')
    pl.plot(x[ok]+dx[ok], y[ok]+dy[ok], 'r.')

    