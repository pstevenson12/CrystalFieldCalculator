import numpy as np
from sympy.physics.wigner import wigner_3j


def find_ind(target_freq,freq_ax):
    return np.argmin(np.abs(target_freq-freq_ax))


def calc_spectrum(gs_dict,es_dict,**kwargs):
    # parse kwargs and do some error handling
    # kwargs are Spectrum, PopVecs, Temperature, FreqParams, Polarization

    # building frequency axis
    freq_params = kwargs.get('FreqParams',[6000.,7000.,1.])
    freq_npts = np.round((freq_params[1]-freq_params[0])/freq_params[2])
    freq_axis = np.linspace(freq_params[0],freq_params[1],int(freq_npts))

    # defining spectrum type to calculate
    spec_type = kwargs.get('Spectrum','Excitation')
    if spec_type not in ['Excitation','Emission','2D']:
        spec_type = 'Excitation'
        print(r"I didn't understand that type of spectrum. I'm defaulting to Excitation.")

    # get z1y1 transition energy
    z1y1_freq = kwargs.get('z1y1',6500.)

    # define populations of states. Either directly with PopVecs, or thermal. Popvecs overrides thermal
    if 'PopVecs' in kwargs:
        pop_list = kwargs.get('PopVecs')
        z_pops = pop_list[0]
        y_pops = pop_list[1]
    else:
        temperature = kwargs.get('Temperature',4.0)
        z_pops = np.exp(-gs_dict['eigenvalues']/(0.695*temperature))
        y_pops = np.exp(-es_dict['eigenvalues'] / (0.695 * temperature))

    z_pops = z_pops/np.sum(z_pops)
    z_pops[z_pops<1e-6] = 0.0
    z_pops = z_pops / np.sum(z_pops)

    y_pops = y_pops / np.sum(y_pops)
    y_pops[y_pops < 1e-6] = 0.0
    y_pops = y_pops / np.sum(y_pops)

    # now parse the polarization
    pol_string = kwargs.get('Polarization','Linear')

    if pol_string=='Linear':
        q = 0
    elif pol_string=='RHC':
        q = 1
    elif pol_string=='LHC':
        q = -1
    else:
        q = 0
        print(r"I couldn't recognize that polarization. Defaulting to Linear")

    # now calculate the magnetic dipole operator matrix
    # future note: if we need to do this a lot (e.g. fitting) we can precalculate this
    j_g = gs_dict['Jval']
    j_e = es_dict['Jval']

    mjg_list = np.arange(-j_g, j_g + 0.001)
    mje_list = np.arange(-j_e, j_e + 0.001)

    m_mat = np.zeros((mjg_list.size,mje_list.size))

    for mjg in mjg_list:
        im_jg = int(mjg + j_g)
        for mje in mje_list:
            im_je = int(mje + j_e)
            m_mat[im_jg, im_je] = (-1) ** (j_g - 1 + mje) * np.sqrt(2 * j_e + 1) * wigner_3j(j_g, 1, j_e,
                                                                                                 mjg, q, -mje)

    # calculate the line strengths
    spec_mat = np.zeros_like(m_mat)

    for m, gs_vec in enumerate(gs_dict['eigenvecs'].T):
        for n, es_vec in enumerate(es_dict['eigenvecs'].T):
            gs_vec[np.abs(gs_vec) < 1e-15] = 0.0
            es_vec[np.abs(es_vec) < 1e-15] = 0.0
            ref_mat = np.outer(gs_vec, es_vec)
            inten_mat = (m_mat[:, :] * ref_mat) ** 2
            spec_mat[m, n] = np.sum(np.sum(inten_mat))

    # calculate the spectrum

    if spec_type == 'Excitation':
        spec_out = np.zeros_like(freq_axis)
        for m, zval in enumerate(gs_dict['eigenvalues']):
            for n, yval in enumerate(es_dict['eigenvalues']):
                temp_spec = np.zeros_like(freq_axis)
                temp_ind = find_ind(z1y1_freq+yval-zval,freq_axis)
                temp_spec[temp_ind] = z_pops[m]*spec_mat[m, n]
                spec_out+=temp_spec
    elif spec_type == 'Emission':
        spec_out = np.zeros_like(freq_axis)
        for m, zval in enumerate(gs_dict['eigenvalues']):
            for n, yval in enumerate(es_dict['eigenvalues']):
                temp_spec = np.zeros_like(freq_axis)
                temp_ind = find_ind(z1y1_freq+yval-zval,freq_axis)
                temp_spec[temp_ind] = y_pops[n]*spec_mat[m, n]
                spec_out+=temp_spec
    else:
        spec_out = np.zeros_like(np.outer(freq_axis,freq_axis))
        spec_exc = np.zeros_like(freq_axis)
        spec_em = np.zeros_like(freq_axis)
        for m, zval in enumerate(gs_dict['eigenvalues']):
            for n, yval in enumerate(es_dict['eigenvalues']):
                temp_spec_exc = np.zeros_like(freq_axis)
                temp_spec_em = np.zeros_like(freq_axis)
                temp_ind = find_ind(z1y1_freq+yval-zval,freq_axis)
                temp_spec_exc[temp_ind] = z_pops[m] * spec_mat[m, n]
                temp_spec_em[temp_ind] = y_pops[n] * spec_mat[m, n]
                spec_exc+=temp_spec_exc
                spec_em += temp_spec_exc
        spec_out = np.outer(spec_exc,spec_em)

    return freq_axis, spec_out
