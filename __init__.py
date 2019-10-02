"""This lensit package contains some convenience functions in its __init__.py for quick startup.

"""
from __future__ import print_function

import numpy as np
import os

from lensit.ffs_covs import ffs_cov, ell_mat
from lensit.sims import ffs_phas, ffs_maps, ffs_cmbs
from lensit.pbs import pbs
from lensit.misc.misc_utils import enumerate_progress, camb_clfile, gauss_beam

"""
import ffs_covs
import ffs_iterators
import ffs_deflect
import curvedskylensing
import ffs_qlms
import misc
import pbs
import qcinv
import sims
import pseudocls
import os
"""

def _get_lensitdir():
    assert 'LENSIT' in os.environ.keys(), 'Set LENSIT env. variable to somewhere safe to write'
    LENSITDIR = os.environ.get('LENSIT')
    CLSPATH = os.path.join(os.path.dirname(__file__), 'data', 'cls')
    return LENSITDIR, CLSPATH


def get_fidcls(ellmax_sky=6000):
    r"""Returns *lensit* fiducial CMB spectra (Planck 2015 cosmology)

    Args:
        ellmax_sky: optionally reduces outputs spectra :math:`\ell_{\rm max}`

    Returns:
        unlensed and lensed CMB spectra (dicts)


    """
    cls_unl = {}
    cls_unlr = camb_clfile(os.path.join(_get_lensitdir()[1], 'fiducial_flatsky_lenspotentialCls.dat'))
    for key in cls_unlr.keys():
        cls_unl[key] = cls_unlr[key][0:ellmax_sky + 1]
        if key == 'pp': cls_unl[key] = cls_unlr[key][:]  # might need this one to higher lmax
    cls_len = {}
    cls_lenr = camb_clfile(os.path.join(_get_lensitdir()[1], 'fiducial_flatsky_lensedCls.dat'))
    for key in cls_lenr.keys():
        cls_len[key] = cls_lenr[key][0:ellmax_sky + 1]
    return cls_unl, cls_len


def get_fidtenscls(ellmax_sky=6000):
    cls = {}
    cls_tens = camb_clfile(os.path.join(_get_lensitdir()[1], 'fiducial_tensCls.dat'))
    for key in cls_tens.keys():
        cls[key] = cls_tens[key][0:ellmax_sky + 1]
    return cls

def get_ellmat(LD_res, HD_res):
    r"""Default ellmat instances.


    Returns:
        *ell_mat* instance describing a flat-sky square patch of physical size :math:`\sim 0.74 *2^{\rm HDres}` arcmin,
        sampled with :math:`2^{\rm LDres}` points on a side.

    The patch area is :math:`4\pi` if *HD_res* = 14

    """
    assert HD_res <= 14 and LD_res <= 14, (LD_res, HD_res)
    lcell_rad = (np.sqrt(4. * np.pi) / 2 ** 14) * (2 ** (HD_res - LD_res))
    shape = (2 ** LD_res, 2 ** LD_res)
    lsides = (lcell_rad * 2 ** LD_res, lcell_rad * 2 ** LD_res)
    lib_dir = os.path.join(_get_lensitdir()[0], 'temp', 'ellmats', 'ellmat_%s_%s' % (HD_res, LD_res))
    return ell_mat.ell_mat(lib_dir, shape, lsides)


def get_lencmbs_lib(res=14, cache_sims=True, nsims=120, num_threads=int(os.environ.get('OMP_NUM_THREADS', 1))):
    r"""Default lensed CMB simulation library

    Lensing is always performed at resolution of :math:`0.75` arcmin

    Args:
        res: lensed CMBs are generated on a square box with of physical size  :math:`\sim 0.74 \cdot 2^{\rm res}` arcmin
        cache_sims: saves the lensed CMBs when produced for the first time
        nsims: number of simulations in the library
        num_threads: number of threads used by the pyFFTW fft-engine.

    Note:
        All simulations random phases will be generated at the very first call if not performed previously; this might take some time

    """
    HD_ellmat = get_ellmat(res, HD_res=res)
    ellmax_sky = 6000
    fsky = int(np.round(np.prod(HD_ellmat.lsides) / 4. / np.pi * 1000.))
    lib_skyalm = ell_mat.ffs_alm_pyFFTW(HD_ellmat, num_threads=num_threads,
                                                 filt_func=lambda ell: ell <= ellmax_sky)
    skypha_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d' % fsky, 'len_alms', 'skypha')
    skypha = ffs_phas.ffs_lib_phas(skypha_libdir, 4, lib_skyalm, nsims_max=nsims)
    if not skypha.is_full() and pbs.rank == 0:
        for i, idx in enumerate_progress(np.arange(nsims, dtype=int), label='Generating CMB phases'):
            skypha.get_sim(int(idx))
    pbs.barrier()
    cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky)
    sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d' % fsky, 'len_alms')
    return ffs_cmbs.sims_cmb_len(sims_libdir, lib_skyalm, cls_unl, lib_pha=skypha, cache_lens=cache_sims)


def get_maps_lib(exp, LDres, HDres=14, cache_lenalms=True, cache_maps=False,
                 nsims=120, num_threads=int(os.environ.get('OMP_NUM_THREADS', 1))):
    r"""Default CMB data maps simulation library

    Args:
        exp: experimental configuration (see *get_config*)
        LDres: the data is generated on a square patch with :math:` 2^{\rm LDres}` pixels on a side
        HDres: The physical size of the path is :math:`\sim 0.74 \cdot 2^{\rm HDres}` arcmin
        cache_lenalms: saves the lensed CMBs when produced for the first time (defaults to True)
        cache_maps: saves the data maps when produced for the first time (defaults to False)
        nsims: number of simulations in the library
        num_threads: number of threads used by the pyFFTW fft-engine.

    Note:
        All simulations random phases (CMB sky and noise) will be generated at the very first call if not performed previously; this might take some time

    """
    sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    len_cmbs = get_lencmbs_lib(res=HDres, cache_sims=cache_lenalms, nsims=nsims)
    lmax_sky = len_cmbs.lib_skyalm.ellmax
    cl_transf = gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=lmax_sky)
    lib_datalm = ffs_covs.ell_mat.ffs_alm_pyFFTW(get_ellmat(LDres, HDres), filt_func=lambda ell: ell <= lmax_sky,
                                                 num_threads=num_threads)
    fsky = int(np.round(np.prod(len_cmbs.lib_skyalm.ell_mat.lsides) / 4. / np.pi * 1000.))
    vcell_amin2 = np.prod(lib_datalm.ell_mat.lsides) / np.prod(lib_datalm.ell_mat.shape) * (180 * 60. / np.pi) ** 2
    nTpix = sN_uKamin / np.sqrt(vcell_amin2)
    nPpix = sN_uKaminP / np.sqrt(vcell_amin2)

    pixpha_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims' % nsims, 'fsky%04d' % fsky, 'res%s' % LDres, 'pixpha')
    pixpha = ffs_phas.pix_lib_phas(pixpha_libdir, 3, lib_datalm.ell_mat.shape, nsims_max=nsims)

    if not pixpha.is_full() and pbs.rank == 0:
        for _i, idx in enumerate_progress(np.arange(nsims), label='Generating Noise phases'):
            pixpha.get_sim(idx)
    pbs.barrier()
    sims_libdir = os.path.join(_get_lensitdir()[0], 'temp', '%s_sims'%nsims,'fsky%04d'%fsky, 'res%s'%LDres,'%s'%exp, 'maps')
    return ffs_maps.lib_noisemap(sims_libdir, lib_datalm, len_cmbs, cl_transf, nTpix, nPpix, nPpix,
                                      pix_pha=pixpha, cache_sims=cache_maps)


def get_isocov(exp, LD_res, HD_res=14, pyFFTWthreads=int(os.environ.get('OMP_NUM_THREADS', 1))):
    r"""Default *ffs_cov.ffs_diagcov_alm* instances.


    Returns:
        *ffs_cov.ffs_diagcov_alm* instance on a flat-sky square patch of physical size :math:`\sim 0.74 \cdot 2^{\rm HDres}` arcmin,
        sampled with :math:`2^{\rm LDres}` points on a side.


    """
    ellmax_sky = 6000
    sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky)

    cls_noise = {'t': (sN_uKamin * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1),
                 'q':(sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1),
                 'u':(sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1)}  # simple flat noise Cls
    cl_transf = gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=ellmax_sky)
    lib_alm = ell_mat.ffs_alm_pyFFTW(get_ellmat(LD_res, HD_res=HD_res),
                        filt_func=lambda ell: (ell >= ellmin) & (ell <= ellmax), num_threads=pyFFTWthreads)
    lib_skyalm = ell_mat.ffs_alm_pyFFTW(get_ellmat(LD_res, HD_res=HD_res),
                        filt_func=lambda ell: (ell <= ellmax_sky), num_threads=pyFFTWthreads)

    lib_dir = os.path.join(_get_lensitdir()[0], 'temp', 'Covs', '%s' % exp, 'LD%sHD%s' % (LD_res, HD_res))
    return ffs_cov.ffs_diagcov_alm(lib_dir, lib_alm, cls_unl, cls_len, cl_transf, cls_noise, lib_skyalm=lib_skyalm)


"""
def get_config(exp):
    #Returns noise levels, beam size and multipole cuts for some configurations

    
    sN_uKaminP = None
    if exp == 'Planck':
        sN_uKamin = 35.
        Beam_FWHM_amin = 7.
        ellmin = 10
        ellmax = 2048
    elif exp == 'Planck_65':
        sN_uKamin = 35.
        Beam_FWHM_amin = 6.5
        ellmin = 100
        ellmax = 2048
    elif exp == 'S4':
        sN_uKamin = 1.5
        Beam_FWHM_amin = 3.
        ellmin = 10
        ellmax = 3000
    elif exp == 'S5':
        sN_uKamin = 1.5 / 4.
        Beam_FWHM_amin = 3.
        ellmin = 10
        ellmax = 3000
    elif exp == 'S6':
        sN_uKamin = 1.5 / 4. / 4.
        Beam_FWHM_amin = 3.
        ellmin = 10
        ellmax = 3000
    elif exp == 'SO':
        sN_uKamin = 3.
        Beam_FWHM_amin = 3.
        ellmin = 10
        ellmax = 3000
    elif exp == 'SOb1':
        sN_uKamin = 3.
        Beam_FWHM_amin = 1.
        ellmin = 10
        ellmax = 3000
    elif exp == 'PB85':
        sN_uKamin = 8.5 /np.sqrt(2.)
        Beam_FWHM_amin = 3.5
        ellmin = 10
        ellmax = 3000
    elif exp == 'PB5':
        sN_uKamin = 5. / np.sqrt(2.)
        Beam_FWHM_amin = 3.5
        ellmin = 10
        ellmax = 3000
    elif exp == 'fcy_mark':
        sN_uKamin = 5.
        Beam_FWHM_amin = 1.4
        ellmin=10
        ellmax=3000
    elif exp == 'SO_1p7':
        sN_uKamin = 3.
        Beam_FWHM_amin = 3.
        ellmin = 10
        ellmax = 3000
    elif exp == 'SO_1p7H':
        sN_uKamin = 3.
        Beam_FWHM_amin = 3.
        ellmin = 10
        ellmax = 3000
    elif exp == 'SOJP':
        sN_uKamin = 4.015 #5/np.sqrt(2.)
        Beam_FWHM_amin = 1.4 #1.8
        ellmin = 10
        ellmax = 3000
        
    elif exp == 'SO_1p7_new':
        sN_uKamin = 4.015 #5/np.sqrt(2.)
        Beam_FWHM_amin = 1.4 #1.8
        ellmin = 10
        ellmax = 3000
    elif exp == 'SO_1p7_LAT_patch':
        sN_uKamin = 4.015 #5/np.sqrt(2.)
        Beam_FWHM_amin = 1.4 #1.8
        ellmin = 10
        ellmax = 3000
    elif exp == 'SO_1p7_v3':
        sN_uKamin = 4.015 #5/np.sqrt(2.)
        Beam_FWHM_amin = 1.4 #1.8
        ellmin = 10
        ellmax = 3000 #3000
    elif exp == 'SO_1p7_v3_SAT_fullsky':
        sN_uKamin = 4.015 #5/np.sqrt(2.)
        Beam_FWHM_amin = 1.4 #1.8
        ellmin = 10
        ellmax = 3000
    elif exp == 'SO_1p7_LAT_patch_v3':
        sN_uKamin = 4.015 #5/np.sqrt(2.)
        Beam_FWHM_amin = 1.4 #1.8
        ellmin = 10
        ellmax = 3000
    
    elif exp == 'SOJP_small':
        sN_uKamin = 4.015 #5/np.sqrt(2.)
        Beam_FWHM_amin = 1.4 #1.8
        ellmin = 10
        ellmax = 3000
    elif exp == 'SOJP0745':
        sN_uKamin = 5/np.sqrt(2.)
        Beam_FWHM_amin = 1.8
        ellmin = 10
        ellmax = 3000
    elif exp == 'null':
        sN_uKamin = 0.00001
        Beam_FWHM_amin = 0.00001
        ellmin = 10
        ellmax = 3000
    else:
        sN_uKamin = 0
        Beam_FWHM_amin = 0
        ellmin = 0
        ellmax = 0
        assert 0, '%s not implemented' % exp
    sN_uKaminP = sN_uKaminP or np.sqrt(2.) * sN_uKamin
    return sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax
"""

def get_fidcls(ellmax_sky=ellmax_sky,path_to_inputs='.'):
    cls_unl = {}
    for key, cl in misc.jc_camb.spectra_fromcambfile(path_to_inputs+'/inputs/cls/fiducial_lenspotentialCls.dat').iteritems():
        cls_unl[key] = cl[0:ellmax_sky + 1]
        if key == 'pp': cls_unl[key] = cl[:]  # might need this one
    cls_len = {}
    for key, cl in misc.jc_camb.spectra_fromcambfile(path_to_inputs+'/inputs/cls/fiducial_lensedCls.dat').iteritems():
        cls_len[key] = cl[0:ellmax_sky + 1]
    return cls_unl, cls_len

def get_partiallylenfidcls(w,ellmax_sky=ellmax_sky,path_to_inputs='.'):
    # Produces spectra lensed with w_L * cpp_L
    params = misc.jc_camb.read_params(path_to_inputs + '/inputs/cls/fiducial_flatsky_params.ini')
    params['lensing_method'] = 4
    #FIXME : this would anyway not work in MPI mode beacause lensing method 4 does not.
    params['output_root'] = os.path.abspath(path_to_inputs + '/temp/camb_rank%s' % pbs.rank)
    ell = np.arange(len(w),dtype = int)
    np.savetxt(misc.jc_camb.PathToCamb + '/cpp_weights.txt', np.array([ell, w]).transpose(), fmt=['%i', '%10.5f'])
    misc.jc_camb.run_camb_fromparams(params)
    cllen = misc.jc_camb.spectra_fromcambfile(params['output_root'] + '_' + params['lensed_output_file'])
    ret = {}
    for key, cl in cllen.iteritems():
        ret[key] = cl[0:ellmax_sky + 1]
    return ret


def get_fidtenscls(ellmax_sky=ellmax_sky,path_to_inputs = '.'):
    cls = {}
    for key, cl in misc.jc_camb.spectra_fromcambfile(path_to_inputs + '/inputs/cls/fiducial_tensCls.dat').iteritems():
        cls[key] = cl[0:ellmax_sky + 1]
    return cls

def get_ellmat(LD_res, HD_res=14,res_formula=False):
    """
    Standardized ellmat instances.
    Returns ellmat with 2 ** LD_res squared points with
    lcell = 0.745 * (2 ** (HD_res - LD_res)) and lsides lcell * 2 ** LD_res.
    Set HD_res to 14 for full sky ell_mat.
    :param LD_res:
    :param HD_res:
    :return:
    """
    #assert HD_res <= 14 and LD_res <= 14, (LD_res, HD_res)
    if res_formula:
        lcell_rad = (np.sqrt(4. * np.pi) / 2 ** 14) * (2 ** (HD_res - LD_res))
    else:
        lcell_rad = 1.7*np.pi/180/60
    if LD_res<100:
        shape = (2 ** LD_res, 2 ** LD_res)
    else:
        shape = (LD_res, LD_res)
    lsides = (lcell_rad * shape[0], lcell_rad * shape[1])
    if os.path.exists('/global'):
        if res_formula:
            return ffs_covs.ell_mat.ell_mat('/global/cscratch1/sd/markm/lensit/temp/ellmats/ellmat_%s_%s' % (HD_res, LD_res), shape, lsides)
        else:
            return ffs_covs.ell_mat.ell_mat('/global/cscratch1/sd/markm/lensit/temp/ellmats/ellmat_%s_%s_1p7' % (HD_res, LD_res), shape, lsides)
    else:
        return ffs_covs.ell_mat.ell_mat('/media/sf_C_DRIVE/Users/DarkMatter42/OneDrive - University of Sussex/LensIt/temp/ellmats/ellmat_%s_%s' % (HD_res, LD_res), shape, lsides)


def get_lencmbs_lib(res=14, cache_sims=True, nsims=120, num_threads=4,path_to_inputs='.', Beam_FWHM_amin = 0.0):
    """
    Default simulation library of 120 lensed CMB sims.
    Lensing is always performed at lcell 0.745 amin or so, and lensed CMB are generated on a square with sides lcell 2 ** res
    Will build all phases at the very first call if not already present.
    """
    HD_ellmat = get_ellmat(res, HD_res=res)
    ellmax_sky = 6000
    fsky = int(np.round(np.prod(HD_ellmat.lsides) / 4. / np.pi * 1000.))
    lib_skyalm = ffs_covs.ell_mat.ffs_alm_pyFFTW(HD_ellmat, num_threads=num_threads,
                                                 filt_func=lambda ell: ell <= ellmax_sky)
    if os.path.exists('/global'):
        skypha = sims.ffs_phas.ffs_lib_phas('/global/cscratch1/sd/markm/lensit/temp/%s_sims/fsky%04d/beam%d/len_alms/skypha' % (nsims, fsky, Beam_FWHM_amin*10), 4, lib_skyalm, nsims_max=nsims)
    else:
        skypha = sims.ffs_phas.ffs_lib_phas('/media/sf_C_DRIVE/Users/DarkMatter42/OneDrive - University of Sussex/LensIt/temp/%s_sims/fsky%04d/beam%d/len_alms/skypha' % (nsims, fsky, Beam_FWHM_amin*10), 4, lib_skyalm, nsims_max=nsims)
			
    if not skypha.is_full() and pbs.rank == 0:
        for _i, idx in misc.misc_utils.enumerate_progress(np.arange(nsims), label='Generating CMB phases'):
            skypha.get_sim(idx)
    pbs.barrier()
    cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky,path_to_inputs=path_to_inputs)
    if os.path.exists('/global'):
        return sims.ffs_cmbs.sims_cmb_len('/global/cscratch1/sd/markm/lensit/temp/%s_sims/fsky%04d/beam%d/len_alms' % (nsims, fsky, Beam_FWHM_amin*10), lib_skyalm, cls_unl, lib_pha=skypha, cache_lens=cache_sims)
    else:
        return sims.ffs_cmbs.sims_cmb_len('/media/sf_C_DRIVE/Users/DarkMatter42/OneDrive - University of Sussex/LensIt/temp/%s_sims/fsky%04d/beam%d/len_alms' % (nsims, fsky, Beam_FWHM_amin*10), lib_skyalm, cls_unl, lib_pha=skypha, cache_lens=cache_sims)


def get_maps_lib(params, exp, LDres, HDres=14, cache_lenalms=True, cache_maps=False, nsims=120, num_threads=4,path_to_inputs='.'):
    """
    Default simulation library of 120 full flat sky sims for exp 'exp' at resolution LDres.
    Different exp at same resolution share the same random phases both in CMB and noise
        Will build all phases at the very first call if not already present.
    :param exp: 'Planck', 'S4' ... See get_config
    :param LDres: 14 : cell length is 0.745 amin, 13 : 1.49 etc.
    :return: sim library instance
    """
    #sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    sN_uKamin = params.sN_uKamin
    sN_uKaminP = params.sN_uKaminP
    Beam_FWHM_amin = params.Beam_FWHM_amin
    ellmin = params.lmin
    ellmax = params.lmax
    len_cmbs = get_lencmbs_lib(res=HDres, cache_sims=cache_lenalms, nsims=nsims,path_to_inputs=path_to_inputs, Beam_FWHM_amin = Beam_FWHM_amin)
    lmax_sky = len_cmbs.lib_skyalm.ellmax
    cl_transf = hp.gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=lmax_sky)
    lib_datalm = ffs_covs.ell_mat.ffs_alm_pyFFTW(get_ellmat(LDres, HDres), filt_func=lambda ell: ell <= lmax_sky,
                                                 num_threads=num_threads)
    fsky = int(np.round(np.prod(len_cmbs.lib_skyalm.ell_mat.lsides) / 4. / np.pi * 1000.))
    vcell_amin2 = np.prod(lib_datalm.ell_mat.lsides) / np.prod(lib_datalm.ell_mat.shape) * (180 * 60. / np.pi) ** 2
    nTpix = sN_uKamin / np.sqrt(vcell_amin2)
    nPpix = sN_uKaminP / np.sqrt(vcell_amin2)

    if os.path.exists('/global'):
        pixpha = sims.ffs_phas.pix_lib_phas('/global/cscratch1/sd/markm/lensit/temp/%s_sims/fsky%04d/beam%d/res%s/pixpha' % (nsims, fsky, Beam_FWHM_amin*10, LDres), 3, lib_datalm.ell_mat.shape, nsims_max=nsims)
    else:
        pixpha = sims.ffs_phas.pix_lib_phas('./temp/%s_sims/fsky%04d/beam%d/res%s/pixpha' % (nsims, fsky, Beam_FWHM_amin*10, LDres), 3, lib_datalm.ell_mat.shape, nsims_max=nsims)
		
    if not pixpha.is_full() and pbs.rank == 0:
        for _i, idx in misc.misc_utils.enumerate_progress(np.arange(nsims), label='Generating Noise phases'):
            pixpha.get_sim(idx)
    pbs.barrier()
    if os.path.exists('/global'):
        lib_dir = '/global/cscratch1/sd/markm/lensit/temp/%s_sims/fsky%04d/beam%d/res%s/%s/maps' % (nsims, fsky, Beam_FWHM_amin*10, LDres, exp)
    else:
        lib_dir = '/media/sf_C_DRIVE/Users/DarkMatter42/OneDrive - University of Sussex/LensIt/temp/%s_sims/fsky%04d/beam%d/res%s/%s/maps' % (nsims, fsky, Beam_FWHM_amin*10, LDres, exp)
    return sims.ffs_maps.lib_noisemap(lib_dir, lib_datalm, len_cmbs, cl_transf, nTpix, nPpix, nPpix,
                                      pix_pha=pixpha, cache_sims=cache_maps)


def get_noisefree_maps_lib(params, exp, LDres, HDres=14, cache_lenalms=True, cache_maps=False, nsims=120, num_threads=4,path_to_inputs='.'):
    """
    Default simulation library of 120 full flat sky sims with no instrument noise at resolution LDres.
    Different exp at same resolution share the same random phases both in CMB and noise
        Will build all phases at the very first call if not already present.
    :param exp: 'Planck', 'S4' ... See get_config
    :param LDres: 14 : cell length is 0.745 amin, 13 : 1.49 etc.
    :return: sim library instance
    """
    #_, _, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    Beam_FWHM_amin = params.Beam_FWHM_amin
    ellmin = params.lmin
    ellmax = params.lmax
    len_cmbs = get_lencmbs_lib(res=HDres, cache_sims=cache_lenalms, nsims=nsims,path_to_inputs=path_to_inputs, Beam_FWHM_amin=Beam_FWHM_amin)
    lmax_sky = len_cmbs.lib_skyalm.ellmax
    cl_transf = hp.gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=lmax_sky)
    lib_datalm = ffs_covs.ell_mat.ffs_alm_pyFFTW(get_ellmat(LDres, HDres), filt_func=lambda ell: ell <= lmax_sky,
                                                 num_threads=num_threads)
    fsky = int(np.round(np.prod(len_cmbs.lib_skyalm.ell_mat.lsides) / 4. / np.pi * 1000.))
    vcell_amin2 = np.prod(lib_datalm.ell_mat.lsides) / np.prod(lib_datalm.ell_mat.shape) * (180 * 60. / np.pi) ** 2

    if os.path.exists('/global'):
        pixpha = sims.ffs_phas.pix_lib_phas('/global/cscratch1/sd/markm/lensit/temp/%s_sims/fsky%04d/beam%d/res%s/pixpha' % (nsims, fsky, Beam_FWHM_amin*10, LDres), 3, lib_datalm.ell_mat.shape, nsims_max=nsims)
    else:
        pixpha = sims.ffs_phas.pix_lib_phas('/media/sf_C_DRIVE/Users/DarkMatter42/OneDrive - University of Sussex/LensIt/temp/%s_sims/fsky%04d/beam%d/res%s/pixpha' % (nsims, fsky, Beam_FWHM_amin*10, LDres), 3, lib_datalm.ell_mat.shape, nsims_max=nsims)
    		
    if not pixpha.is_full() and pbs.rank == 0:
        for _i, idx in misc.misc_utils.enumerate_progress(np.arange(nsims), label='Generating Noise phases'):
            pixpha.get_sim(idx)
    pbs.barrier()
    if os.path.exists('/global'):
        lib_dir = '/global/cscratch1/sd/markm/lensit/temp/AN_sims/%s_sims/fsky%04d/beam%d/res%s/%s/maps' % (nsims, fsky, Beam_FWHM_amin*10, LDres, exp)
    else:
        lib_dir = '/media/sf_C_DRIVE/Users/DarkMatter42/OneDrive - University of Sussex/LensIt/temp/AN_sims/%s_sims/fsky%04d/beam%d/res%s/%s/maps' % (nsims, fsky, Beam_FWHM_amin*10, LDres, exp)
    return sims.ffs_maps.lib_noisefree(lib_dir, lib_datalm, len_cmbs, cl_transf, cache_sims=cache_maps)

def get_noisefree_maps_lib_for_MCN1(params, exp, LDres, HDres=14, cache_lenalms=True, cache_maps=False, nsims=120, num_threads=4,path_to_inputs='.'):
    """
    Default simulation library of 120 full flat sky sims with no instrument noise at resolution LDres.
    Different exp at same resolution share the same random phases both in CMB and noise
        Will build all phases at the very first call if not already present.
    :param exp: 'Planck', 'S4' ... See get_config
    :param LDres: 14 : cell length is 0.745 amin, 13 : 1.49 etc.
    :return: sim library instance
    """
    #_, _, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    Beam_FWHM_amin = params.Beam_FWHM_amin
    ellmin = params.lmin
    ellmax = params.lmax
    len_cmbs = get_lencmbs_lib_for_MCN1(res=HDres, cache_sims=cache_lenalms, nsims=nsims,path_to_inputs=path_to_inputs, Beam_FWHM_amin=Beam_FWHM_amin)
    lmax_sky = len_cmbs.lib_skyalm.ellmax
    cl_transf = hp.gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=lmax_sky)
    lib_datalm = ffs_covs.ell_mat.ffs_alm_pyFFTW(get_ellmat(LDres, HDres), filt_func=lambda ell: ell <= lmax_sky,
                                                 num_threads=num_threads)
    fsky = int(np.round(np.prod(len_cmbs.lib_skyalm.ell_mat.lsides) / 4. / np.pi * 1000.))
    vcell_amin2 = np.prod(lib_datalm.ell_mat.lsides) / np.prod(lib_datalm.ell_mat.shape) * (180 * 60. / np.pi) ** 2
    
    if os.path.exists('/global'):
        pixpha = sims.ffs_phas.pix_lib_phas('/global/cscratch1/sd/markm/lensit/temp/%s_sims/fsky%04d/beam%d/res%s/pixpha' % (nsims, fsky, Beam_FWHM_amin*10, LDres), 3, lib_datalm.ell_mat.shape, nsims_max=nsims)
    else:
        pixpha = sims.ffs_phas.pix_lib_phas('/media/sf_C_DRIVE/Users/DarkMatter42/OneDrive - University of Sussex/LensIt/temp/%s_sims/fsky%04d/beam%d/res%s/pixpha' % (nsims, fsky, Beam_FWHM_amin*10, LDres), 3, lib_datalm.ell_mat.shape, nsims_max=nsims)
    		
    if not pixpha.is_full() and pbs.rank == 0:
        for _i, idx in misc.misc_utils.enumerate_progress(np.arange(nsims), label='Generating Noise phases'):
            pixpha.get_sim(idx)
    pbs.barrier()
    if os.path.exists('/global'):
        lib_dir = '/global/cscratch1/sd/markm/lensit/temp/AN_sims_for_MCN1/%s_sims/fsky%04d/beam%d/res%s/%s/maps' % (nsims, fsky, Beam_FWHM_amin*10, LDres, exp)
    else:
        lib_dir = '/media/sf_C_DRIVE/Users/DarkMatter42/OneDrive - University of Sussex/LensIt/temp/AN_sims_for_MCN1/%s_sims/fsky%04d/beam%d/res%s/%s/maps' % (nsims, fsky, Beam_FWHM_amin*10, LDres, exp)
    return sims.ffs_maps.lib_noisefree(lib_dir, lib_datalm, len_cmbs, cl_transf, cache_sims=cache_maps)

def get_lencmbs_lib_for_MCN1(res=14, cache_sims=True, nsims=120, num_threads=4,path_to_inputs='.', Beam_FWHM_amin=0.0):
    """
    Default simulation library of 120 lensed CMB sims.
    Lensing is always performed at lcell 0.745 amin or so, and lensed CMB are generated on a square with sides lcell 2 ** res
    Will build all phases at the very first call if not already present.
    """
    HD_ellmat = get_ellmat(res, HD_res=res)
    ellmax_sky = 6000
    fsky = int(np.round(np.prod(HD_ellmat.lsides) / 4. / np.pi * 1000.))
    lib_skyalm = ffs_covs.ell_mat.ffs_alm_pyFFTW(HD_ellmat, num_threads=num_threads,
                                                 filt_func=lambda ell: ell <= ellmax_sky)
    if os.path.exists('/global'):
        skypha = sims.ffs_phas.ffs_lib_phas('/global/cscratch1/sd/markm/lensit/temp/%s_sims/fsky%04d/beam%d/len_alms/skypha' % (nsims, fsky, Beam_FWHM_amin*10), 4, lib_skyalm, nsims_max=nsims)
    else:
        skypha = sims.ffs_phas.ffs_lib_phas('/media/sf_C_DRIVE/Users/DarkMatter42/OneDrive - University of Sussex/LensIt/temp/%s_sims/fsky%04d/beam%d/len_alms/skypha' % (nsims, fsky, Beam_FWHM_amin*10), 4, lib_skyalm, nsims_max=nsims)
			
    if not skypha.is_full() and pbs.rank == 0:
        for _i, idx in misc.misc_utils.enumerate_progress(np.arange(nsims), label='Generating CMB phases'):
            skypha.get_sim(idx)
    pbs.barrier()
    cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky,path_to_inputs=path_to_inputs)
    if os.path.exists('/global'):
        return sims.ffs_cmbs.sims_cmb_len_for_MCN1('/global/cscratch1/sd/markm/lensit/temp/MCN1_sims/%s_sims/fsky%04d/beam%d/len_alms' % (nsims, fsky, Beam_FWHM_amin*10), lib_skyalm, cls_unl, lib_pha=skypha, cache_lens=cache_sims)
    else:
        return sims.ffs_cmbs.sims_cmb_len_for_MCN1('/media/sf_C_DRIVE/Users/DarkMatter42/OneDrive - University of Sussex/LensIt/temp/MCN1_sims/%s_sims/fsky%04d/beam%d/len_alms' % (nsims, fsky, Beam_FWHM_amin*10), lib_skyalm, cls_unl, lib_pha=skypha, cache_lens=cache_sims)



def get_isocov(params, exp, LD_res, HD_res=14, pyFFTWthreads=4):
    """
    Set HD_res to 14 for full sky sampled at res LD.
    """
    #sN_uKamin, sN_uKaminP, Beam_FWHM_amin, ellmin, ellmax = get_config(exp)
    sN_uKamin = params.sN_uKamin
    sN_uKaminP = params.sN_uKaminP
    Beam_FWHM_amin = params.Beam_FWHM_amin
    ellmin = params.lmin
    ellmax = params.lmax
    cls_unl, cls_len = get_fidcls(ellmax_sky=ellmax_sky,path_to_inputs=path_to_inputs)

    cls_noise = {}
    cls_noise['t'] = (sN_uKamin * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1)  # simple flat noise Cls
    cls_noise['q'] = (sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1)  # simple flat noise Cls
    cls_noise['u'] = (sN_uKaminP * np.pi / 180. / 60.) ** 2 * np.ones(ellmax_sky + 1)  # simple flat noise Cls
    cl_transf = hp.gauss_beam(Beam_FWHM_amin / 60. * np.pi / 180., lmax=ellmax_sky)
    lib_alm = ffs_covs.ell_mat.ffs_alm_pyFFTW(get_ellmat(LD_res, HD_res=HD_res),
                                              filt_func=lambda ell: (ell >= ellmin) & (ell <= ellmax),
                                              num_threads=pyFFTWthreads)
    lib_skyalm = ffs_covs.ell_mat.ffs_alm_pyFFTW(get_ellmat(LD_res, HD_res=HD_res),
                                                 filt_func=lambda ell: (ell <= ellmax_sky), num_threads=pyFFTWthreads)
    if os.path.exists('/global'):
        lib_dir = '/global/cscratch1/sd/markm/lensit/temp/Covs/%s/LD%sHD%s' % (exp, LD_res, HD_res)
    else:
        lib_dir = '/media/sf_C_DRIVE/Users/DarkMatter42/OneDrive - University of Sussex/LensIt/temp/Covs/%s/LD%sHD%s' % (exp, LD_res, HD_res)
    return ffs_covs.ffs_cov.ffs_diagcov_alm(lib_dir, lib_alm, cls_unl, cls_len, cl_transf, cls_noise,
                                            lib_skyalm=lib_skyalm)

											
