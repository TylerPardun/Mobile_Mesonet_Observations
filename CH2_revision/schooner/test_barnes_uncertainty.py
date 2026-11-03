import unittest
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from barnes_uncertainty import barnes_jackknife, jackknife_interval, fdr_adjust


def reference(x,y,z,X,Y,kappa,gamma,min_neighbors):
    # Deliberately slow, independent observation/grid loops for a small fixture.
    def weighted(values,scale):
        result=np.full(X.shape,np.nan)
        for ij in np.ndindex(X.shape):
            d2=(x-X[ij])**2+(y-Y[ij])**2;ids=d2<=9*scale
            if ids.sum()>=min_neighbors:
                w=np.exp(-d2[ids]/scale)
                result[ij]=np.nansum(w*values[ids])/w.sum()
        return result
    a=weighted(z,kappa)
    interp=RegularGridInterpolator((Y[:,0],X[0,:]),a,bounds_error=False,fill_value=np.nan)
    return a+weighted(z-interp(np.column_stack([y,x])),kappa*gamma)


class BarnesTests(unittest.TestCase):
    def test_full_and_all_deleted_fields_match_independent_recomputations(self):
        rng=np.random.default_rng(812)
        x,y=rng.uniform(-3,3,(2,90));z=np.sin(x)+.4*y+rng.normal(0,.2,90)
        days=np.arange(90)%6
        X,Y=np.meshgrid(np.linspace(-2,2,9),np.linspace(-2,2,9))
        for minimum in (1,3):
            a=barnes_jackknife(x,y,z,days,X,Y,7,kappa=1.7,gamma=.3,min_neighbors=minimum,chunk=11)
            np.testing.assert_allclose(a['mean'],reference(x,y,z,X,Y,1.7,.3,minimum),atol=1e-12,equal_nan=True)
            for day in range(7): # includes a day absent from this group
                use=days!=day
                np.testing.assert_allclose(a['leave_one_out'][day],reference(x[use],y[use],z[use],X,Y,1.7,.3,minimum),atol=1e-12,equal_nan=True)

    def test_jackknife_se_reduces_to_mean_standard_error(self):
        x=np.array([1.,2.,4.,8.,10.]);deleted=np.array([np.delete(x,i).mean() for i in range(len(x))])
        low,high,se=jackknife_interval(x.mean(),deleted,len(x)-1)
        self.assertAlmostEqual(se,x.std(ddof=1)/np.sqrt(len(x)))
        self.assertLess(low,x.mean());self.assertGreater(high,x.mean())

    def test_fdr_known_values_and_missing_cells(self):
        p=np.array([.01,.04,.03,.002,np.nan])
        np.testing.assert_allclose(fdr_adjust(p),[.02,.04,.04,.008,np.nan],equal_nan=True)
        np.testing.assert_allclose(fdr_adjust(p,True),np.array([.02,.04,.04,.008,np.nan])*(1+1/2+1/3+1/4),equal_nan=True)

    def test_constant_field_and_no_spurious_cluster_precision(self):
        x,y=np.meshgrid(np.linspace(-1,1,7),np.linspace(-1,1,7));z=np.full(x.size,4.)
        X,Y=np.meshgrid(np.linspace(-1,1,5),np.linspace(-1,1,5))
        days=np.arange(x.size)%4
        a=barnes_jackknife(x.ravel(),y.ravel(),z,days,X,Y,4,kappa=1.,gamma=.3)
        np.testing.assert_allclose(a['mean'],4.,atol=1e-12)
        np.testing.assert_allclose(a['leave_one_out'],4.,atol=1e-12)
        self.assertTrue(np.all(a['effective_days']<=4+1e-12))


if __name__=='__main__':unittest.main()
