import numpy as np
import pylab as pl

import astropy.io.fits as fits



if __name__ == "__main__":
    

    dats = []
    for i in range(6893, 6893+2):
        fn = "data/hen%0.4i.fits" % i
        print(fn)
        dat = fits.open(fn)[0].data

        dats.append(dat)

    dats = np.array(dats)

    
    rois = []
    
    W = 15
    for x in range(50, 2048-50, 6*W):
        for y in range(50, 2048-50, 3*W):
            roi = (slice(x-W, x+W), slice(y-W,y+W))
            rois.append(roi)

    
    N = int(np.ceil(np.sqrt(len(rois))))
    pl.figure(1)
    pl.clf()
    dat = dats[0]
    for ix, roi in enumerate(rois):
        #pl.subplot(N,N,ix+1)
        #pl.imshow(dat[roi])
        pass

        
            

    sigs = np.zeros(len(rois))
    vars = np.zeros_like(sigs)
    
    A = dats[0]
    B = dats[1]
    AmB = A-B
    for ix, roi in enumerate(rois):
        sigs[ix]= np.nanmedian(A[roi])
        vars[ix] = np.var(AmB[roi])/2
        

    pl.figure(2)
    pl.clf()
    pl.subplot(2,1,1)
    pl.plot(sigs, vars, 'o')
    pl.xlabel("Signal [DN]")
    pl.ylabel("Variance [DN^2]")
    pl.grid(True)
    pl.subplot(2,1,2)
    pl.loglog(sigs, vars, 'o')
    pl.xlabel("Signal [DN]")
    pl.ylabel("Variance [DN^2]")
    pl.grid(True)

    pl.savefig("sig-var.pdf")