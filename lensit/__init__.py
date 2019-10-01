"""
Contains some pure convenience functions for quick startup.
"""
import numpy as np
import healpy as hp

try:
    import gpu
except:
    print "NB : import of GPU module unsuccessful"
import ffs_covs
import ffs_iterators
import ffs_deflect
import ffs_qlms
import misc
import pbs
import qcinv
import sims
import pseudocls
import os

ellmax_sky = 6000

"""
def get_config(exp):
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
