# CrystalFieldCalculator
 CrystalFieldCalculator is a series of functions designed to facilitate *rough* calculations of the crystal field states of rare earth ions
 
 The aim of this package is not to provide a method for quantitative calculation, but instead to offer a straightforward pipeline from crystal structure to optical and ESR spectra. The only unit supported currently is cm-1.
 
 As of March 25th, it is composed of three different elements:
 
 **PointChargeFunctions.py**<br/>
 A class with functions designed to take in an input structure (in cartesian coordinates) and output the crystal field (B) parameters. Future versions will support automated parsing of .cif and .xyz files, but for now an array of coordinates with the REI at the center must be input.
 
 **HamiltonianFunctions.py**<br/>
 This class contains the necessary functions to build the crystal field Hamiltonian, generating the underlying angular momentum operators (up to sixth order), and builds and diagonalizes the Hamiltonian for each *J* manifold using input B parameters. The Zeeman term is **not** yet incorporated.
 
 **SpecFunctions.py**<br/>
 This class takes the input eigenvectors and eigenvalues from the HamiltonianFunctions.py class and calculates the spectrum for transitions between $J$ manifolds. Thermal effects are incorporated by an input temperature, and the user specifies either linear or left- or right-handed circular polarization.

## PointChargeFunctions.py
The usage of PointChargeFunctions.py is

    from PointChargeFunctions import *
    pc = PointChargeFunctions()
    cf_params = pc.calc_cf_params(ion_coords)
 
 where *cf_params* returns a dictionary with elements *B2*, *B4* and *B6* which are arrays of size 3,5 and 7, respectively. These elements are the m elements of the B^l_m terms.

The input *ion_coords* is an array of size *(n,4)* for *n* ions. The order of elements is *x,y,z,q* where *q* is the charge in units of elementary charge *e.g.* 1, not 1.6E-19. An example function for generating a distored octahedral coordination is given below as an example

    def gen_dist_oh(x0,z0):
         ion_pos = np.asarray([[x0,0,0,-2],
                          [-x0,0,0,-2],
                          [0,x0,0,-2],
                          [0,-x0,0,-2],
                          [0,0,z0,-2],
                          [0,0,-z0,-2]
                          ])
         return ion_pos
         
These parameters are strictly only valid for Er3+. However, the only changes required in the code are the expectation values of the radial part of the wavefunction, and the Steven's multiplicative factors. As a temporary work-around, these values can be altered after the class is initialized and values from the Mantid program (expectation values) and the multiplicative factors (Hutchings, 1964) can be rewritten.

## HamiltonianFunctions.py
A long-term goal is to implement different approaches to describing the crystal-field Hamiltonian, compatible with the point-charge estimator and the spectrum calculator. The current version only supports the Steven's operator description of the Hamiltonian, and currently only for Er3+. 

### StevensOperators()

The usage is

    from HamiltonianFunctions import *
    so = StevensOperators(Jval)
    ham_j = so.build_ham(cf_params)
    ev_dict = so.proc_ham(ham_j)

where the output *ev_dict* is a dictionary with elements *eigenvalues*, *eigenvecs* and *Jval*. The latter is stored for calculation of the spectrum. All states are stored - there is no checking from Kramer's doublets or other degeneracies. A degeneracy calculcator is a simple add-on that could be added.

## SpecFunctions.py

This class takes in the outputs from the Hamiltonian calculator and generates an output spectrum. The basic usage is

    from SpecFunctions import *
    sc = SpectrumCalculator()
    freq_ax, spec_out = sc.calc_spectrum(gs_ev_dict,es_ev_dict,**kwargs)
    
where *gs_ev_dict* and *es_ev_dict* are the outputs of the *build_ham()* function for the ground and excited *J* manifolds. \*\*kwargs (default values) are:
* **Spectrum** ('Excitation') - accepts 'Excitation', 'Emission' and '2D'. This defines the type of spectrum calculated. In all cases, thermalization in the excited state is assumed to be infinitely fast. Calculates population weighted spectra automatically using either specified state vector or Boltzmann populations for a temperature.
* **PopVecs** - accepts a list of size 2. First element is a vector with the ground state populations, second element is a vector with excited state populations. If a conflict arises with the temperature kwarg, PopVecs has priority
* **Temperature** (4.0) - accepts a single float with the temperature in K. Only read if no PopVecs is present. Generates Boltzmann-weighted populations at this temperature for the population vectors used in the spectrum weighting.
* **Polarization** ('Linear') - accepts 'Linear', 'RHS' and 'LHS'. This defines the polarization used for selection rules. All selection rules assume magnetic dipole transitions.
* **FreqParams** ([6000.,7000.,1.])- accepts a list of three elements. Start frequency in (cm-1), end frequency and resolution. Used in the output spectrum
* **z1y1** (6500.) - the Z1-Y1 transition energy in wavenumbers

Also included is a function *convolve_spectrum* that accepts the *freq_ax, spec_out* generated as well as a linewidth, and returns a uniformly broadened spectrum. Future versions may include the option for state-dependent linewidths.
