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
    # nearest power of 2
    ext_size=_fft_next_fast_len(n)
    fsize=2*ext_size
    xp=(x-x.mean(axis=0))
    var=x.var(axis=0, keepdims=True)
    # do fft and ifft
    cf=rfft(xp,fsize,axis=0)
    sf=da.conj(cf)*cf
    corr=irfft(sf,fsize,axis=0)
    corr=corr[:n,:]
    norm=np.repeat(np.arange(n,0.0,-1),np.prod(x.shape[1:])).reshape(x.shape)
    corr=corr/norm
    return corr*var

def _fft_next_fast_len(target):
    # find the smallest number >= N such that the only divisors are 2, 3, 5
    # works just like scipy.fftpack.next_fast_len
    if target <= 2:
        return target
    while True:
        m = target
        while m % 2 == 0:
            m //= 2
        while m % 3 == 0:
            m //= 3
        while m % 5 == 0:
            m //= 5
        if m == 1:
            return target
        target += 1

def effective_sample_size(sample):
    #https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html
    #http://num.pyro.ai/en/latest/_modules/numpyro/diagnostics.html#effective_sample_size
    m=sample.shape[0]
    n=sample.shape[1]
    rho_t_m=[autocorr(sample[i,:]) for i in range(m)]
    theta_m=[da.mean(sample[i,:],axis=0) for i in range(m)]
    s2=[(1./(n-1))*da.sum(da.square(sample[i,:]-theta_m[i]),axis=0) for i in range(m)]
    W=da.mean(da.stack(s2),axis=0)
    theta_t=da.stack(theta_m).mean(axis=0)
    B=[da.square(theta-theta_t) for theta in theta_m]
    B=(n/(m-1))*da.sum(da.stack(B),axis=0)
    var=((n-1)/n)*W+(1/n)*B
    #chain_variance=[s2[i]*rho_t_m[i] for i in range(m)]
    rho_t=1-(W-da.stack(rho_t_m).mean(axis=0))/var
    rho_t[0,:]=1
    # initial positive sequence (formula 1.18 in [1]) applied for autocorrelation
    Rho_t = rho_t[:-1:2, ...] + rho_t[1::2, ...]
    Rho_t=da.clip(Rho_t, 0,None)
    init_sequence=[Rho_t[0]]
    for i in range(1,Rho_t.shape[0]-1):
        init_sequence.append(da.minimum(Rho_t[i,:],Rho_t[i+1,:]))
    Rho_t=da.stack(init_sequence,axis=0)
    tau = -1.0 + 2.0 * Rho_t.sum(axis=0)
    n_eff=(n*m)/tau
    return n_eff.compute()

#df=h5py.File('posterior_samples_lenet.h5','r')
#samples={var:da.from_array(df[var]) for var in df.keys()}
#rhat={var:potential_scale_reduction(samples[var]) for var in samples.keys()}
#neff={var:effective_sample_size(samples[var]) for var in samples.keys()}
