"""Memory-bounded, exact two-pass Barnes delete-one-cluster calculations.

The full-sample estimator follows the notebook, including its treatment of
out-of-grid interpolation residuals: their contribution to the correction is
zero, while their distance weights remain in the denominator. Input QC is held
fixed. This module does not treat individual 1-Hz measurements as replicates.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
from scipy.stats import t as student_t


def within_sigma(values, k=2.5):
    values = np.asarray(values, float)
    if values.size < 5:
        return np.ones(values.size, bool)
    mean, sd = np.nanmean(values), np.nanstd(values, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.ones(values.size, bool)
    return abs(values-mean) <= k*sd


def barnes_jackknife(x, y, z, cluster, X, Y, n_clusters, kappa=6.7,
                     gamma=.3, min_neighbors=1, chunk=64):
    """Full field and every leave-one-day-out field, retaining pooled weighting.

    At most chunk*n_observations kernel weights are allocated at once. First
    pass sums are accumulated by cluster. The entire second pass, including
    interpolation of the first pass, is recalculated for every deletion.
    """
    x,y,z = [np.asarray(a,float) for a in (x,y,z)]
    cluster = np.asarray(cluster,int)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x,y,z,cluster = [a[good] for a in (x,y,z,cluster)]
    if not len(z) or min_neighbors < 1 or kappa <= 0 or gamma <= 0:
        raise ValueError("Empty observations or invalid Barnes parameters")
    order=np.argsort(cluster,kind='stable')
    x,y,z,cluster=[a[order] for a in (x,y,z,cluster)]
    groups, starts=np.unique(cluster,return_index=True)
    if groups.min()<0 or groups.max()>=n_clusters:
        raise ValueError("Invalid cluster code")
    obs=np.column_stack([x,y]); grid=np.column_stack([X.ravel(),Y.ravel()])
    n_grid=len(grid); n_col=n_clusters+1
    first=np.full((n_grid,n_col),np.nan)
    support=np.zeros(n_grid,int); effective=np.zeros(n_grid)

    def cluster_sum(a):
        result=np.zeros((len(a),n_clusters))
        result[:,groups]=np.add.reduceat(a,starts,axis=1)
        return result

    def kernel(points,scale):
        d2=cdist(points,obs,metric='sqeuclidean')
        outside=d2>9*scale
        np.multiply(d2,-1/scale,out=d2);np.exp(d2,out=d2)
        d2[outside]=0.
        return d2

    def full_and_deleted(a):
        total=a.sum(axis=1,keepdims=True)
        return np.concatenate([total,total-a],axis=1)

    for lo in range(0,n_grid,chunk):
        hi=min(lo+chunk,n_grid);w=kernel(grid[lo:hi],kappa)
        by_w=cluster_sum(w);den=full_and_deleted(by_w)
        num=full_and_deleted(cluster_sum(w*z))
        counts=full_and_deleted(cluster_sum(w>0))
        np.divide(num,den,out=first[lo:hi],where=(den>0)&(counts>=min_neighbors))
        support[lo:hi]=(by_w>0).sum(axis=1)
        effective[lo:hi]=by_w.sum(axis=1)**2/np.maximum((by_w**2).sum(axis=1),1e-300)
    f=RegularGridInterpolator((Y[:,0],X[0,:]),first.reshape(*X.shape,n_col),
                              bounds_error=False,fill_value=np.nan)
    residual=z[:,None]-f(np.column_stack([y,x]))
    residual[np.arange(len(z)),cluster+1]=0.
    residual=np.nan_to_num(residual,nan=0.)
    result=np.full_like(first,np.nan)
    for lo in range(0,n_grid,chunk):
        hi=min(lo+chunk,n_grid);w=kernel(grid[lo:hi],gamma*kappa)
        den=full_and_deleted(cluster_sum(w))
        counts=full_and_deleted(cluster_sum(w>0))
        correction=np.full_like(den,np.nan)
        np.divide(w@residual,den,out=correction,where=(den>0)&(counts>=min_neighbors))
        result[lo:hi]=first[lo:hi]+correction
    return dict(mean=result[:,0].reshape(X.shape),
                leave_one_out=result[:,1:].T.reshape(n_clusters,*X.shape),
                n_days=support.reshape(X.shape),effective_days=effective.reshape(X.shape))


def jackknife_interval(mean, deleted, df):
    """Normal/t jackknife interval, conditional on QC and fixed analysis settings."""
    n=len(deleted)
    center=np.mean(deleted,axis=0)
    se=np.sqrt((n-1)/n*np.sum((deleted-center)**2,axis=0))
    half=student_t.ppf(.975,df)*se
    return mean-half,mean+half,se


def fdr_adjust(p, arbitrary_dependence=False):
    """BH, or BY for arbitrary dependence, over the supplied complete family."""
    p=np.asarray(p,float);out=np.full(p.shape,np.nan);valid=np.isfinite(p)
    a=p[valid];n=len(a)
    if n==0:return out
    order=np.argsort(a);rank=np.arange(1,n+1)
    factor=np.sum(1/rank) if arbitrary_dependence else 1.
    adjusted=np.minimum.accumulate((a[order]*n*factor/rank)[::-1])[::-1]
    restored=np.empty(n);restored[order]=np.minimum(adjusted,1.)
    out[valid]=restored
    return out
