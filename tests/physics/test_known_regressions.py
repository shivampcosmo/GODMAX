"""Known bugs, recorded as strict expected failures.

Each test asserts the CORRECT behaviour. While the bug is present the test fails and pytest
reports it as XFAIL. When the bug is fixed the test passes, `strict=True` turns that into a
FAILURE, and the xfail marker must be removed so the test guards the fix from then on.
"""
import pytest

from godmax.get_Pkzs import get_Pkz


@pytest.mark.xfail(strict=True, reason="14ae4e4 replaced the squeezed CMB-kernel stack with a plain vmap; "
                                       "chi_CMB has shape (1,), so Wk_mat gains an axis and nbins > 1 fails")
def test_cmb_lensing_with_two_source_bins(model):
    import numpy as np
    zs = np.linspace(0.01, 1.5, 128)
    nz = {"nbins": 2, "z_array_source": zs, "nz0": ((zs > 0.2) & (zs < 0.6)).astype(float),
          "nz1": ((zs > 0.6) & (zs < 1.2)).astype(float)}
    o = model("on", ana_over={"is_cmb_lensing": True, "nz_source_info_dict": nz})
    assert np.asarray(o.Cl_kappa_kappa_tot_mat).shape[1:] == (2, 2)


@pytest.mark.xfail(strict=True, reason="14ae4e4 removed the HMF name aliases; the code default 'tinker08' "
                                       "is no longer recognised")
def test_hmf_model_code_default(model):
    model("on", cls=get_Pkz, halo_over={"hmf_model": "tinker08"})


@pytest.mark.xfail(strict=True, reason="14ae4e4 changed jnp.ones((self.nk, ...)) to len(self.nk); nk is an int")
def test_do_corr_2h_mm_false(model):
    model("on", cls=get_Pkz, halo_over={"do_corr_2h_mm": False})
