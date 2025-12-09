import numpy as np
import pylab as pl

import astropy.io.fits as fits


def get_rois(start=50, end=50, skip=2, W=20):
    """ Generate ROIs"""
        
    rois = []
    
    for x in range(start, 2048-end, skip*W):
        for y in range(start, 2048-end, skip*W):
            roi = (slice(x-W, x+W), slice(y-W,y+W))
            rois.append(roi)

    return rois
    

def handle_pair(A, B):
    rois = get_rois()
    N = int(np.ceil(np.sqrt(len(rois))))

    sigs = np.zeros(len(rois))
    vars = np.zeros_like(sigs)
    mmm = np.zeros_like(sigs)

    AmB = A-B
    for ix, roi in enumerate(rois):
        mn, md, mx = np.quantile(A[roi], [.05, .5, .95])
        sigs[ix]= md
        vars[ix] = np.var(AmB[roi])/2
        mmm[ix] = (mx-mn)/md
        

    return sigs, vars, mmm


if __name__ == "__main__":

    dats = []
    all_sigs = []
    all_vars = []
    all_mmms = []
    for i in [6893, 6914, 6953, 6973]:
        fnA = "data/hen%0.4i.fits" % i
        fnB = "data/hen%0.4i.fits" % (i+1)
        print(fnA,fnB)
        A = fits.open(fnA)[0].data
        B = fits.open(fnB)[0].data
        sigs, vars, mmms = handle_pair(A, B)

        all_sigs.append(sigs)
        all_vars.append(vars)
        all_mmms.append(mmms)
        
    sigs = np.ndarray.flatten(np.array(all_sigs))
    vars = np.ndarray.flatten(np.array(all_vars))
    mmms = np.ndarray.flatten(np.array(all_mmms))
        
    ok = (mmms < 0.5) 
    sigs = sigs[ok]
    vars = vars[ok]

    pl.figure(2)
    pl.clf()
    pl.subplot(2,1,1)
    pl.plot(sigs, vars, 'o')
    pl.ylim(0,10_000)
    pl.xlabel("Signal [DN]")
    pl.ylabel("Variance [DN^2]")
    pl.grid(True)
    pl.subplot(2,1,2)
    pl.loglog(sigs, vars, 'o')
    pl.xlabel("Signal [DN]")
    pl.ylabel("Variance [DN^2]")
    pl.grid(True)

    pl.savefig("sig-var-8-dec-b.pdf")