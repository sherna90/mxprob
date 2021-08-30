import h5py
import dask.array as da
from dask.array.fft import *
#from scipy.fftpack import next_fast_len

def normalize(x):
    xp=(x-x.mean(axis=0))/x.std(axis=0)
    return xp

def potential_scale_reduction(sample):
    #https://mc-stan.org/docs/2_18/reference-manual/notation-for-samples-chains-and-draws.html
    m=sample.shape[0]
    n=sample.shape[1]
    theta_m=[da.mean(sample[i,:],axis=0) for i in range(m)]
    theta_t=da.stack(theta_m).mean(axis=0)
    B=[da.square(theta-theta_t) for theta in theta_m]
    B=(n/(m-1))*da.sum(da.stack(B),axis=0)
    s2=[(1./(n-1))*da.sum(da.square(sample[i,:]-theta_m[i]),axis=0) for i in range(m)]
    W=da.mean(da.stack(s2),axis=0)
    var=(n-1)/n*W+(1/n)*B
    rhat=da.sqrt(var/W)
    return rhat.compute()

def autocorr(x):
    '''fft, pad 0s, non partial'''
    n=x.shape[0]
    # pad 0s to 2n-1
    ext_size=2*n-1
    # nearest power of 2
    fsize=2**np.ceil(np.log2(ext_size)).astype('int')
    xp=(x-x.mean(axis=0))
    # do fft and ifft
    cf=rfft(xp,fsize,axis=0)
    sf=da.conj(cf)*cf
    cov=irfft(sf,fsize,axis=0)
    cov=cov/n
    return cov[:n-1,:]

def effective_sample_size(sample):
    #https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html
    m=sample.shape[0]
    n=sample.shape[1]
    ro_t_m=[autocorr(sample[i,:]) for i in range(m)]
    theta_m=[da.mean(sample[i,:],axis=0) for i in range(m)]
    s2=[(1./(n-1))*da.sum(da.square(sample[i,:]-theta_m[i]),axis=0) for i in range(m)]
    W=da.mean(da.stack(s2),axis=0)
    theta_t=da.stack(theta_m).mean(axis=0)
    B=[da.square(theta-theta_t) for theta in theta_m]
    B=(n/(m-1))*da.sum(da.stack(B),axis=0)
    var=((n-1)/n)*W+(1/n)*B
    #chain_variance=[s2[i]*ro_t_m[i] for i in range(m)]
    ro_t=(W-da.stack(ro_t_m).mean(axis=0))/var
    n_eff=n*m/(1+2*da.sum(ro_t,axis=0))
    return n_eff.compute()

#df=h5py.File('posterior_samples_lenet.h5','r')
#samples={var:da.from_array(df[var]) for var in df.keys()}
#rhat={var:potential_scale_reduction(samples[var]) for var in samples.keys()}
#neff={var:effective_sample_size(samples[var]) for var in samples.keys()}
