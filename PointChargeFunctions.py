from __future__ import division
import numpy as np
import warnings
from scipy.special import sph_harm


class PointCharge:

    def __init__(self):
        # these are the radial averages <r^l> of the wavefunctions
        # I think I took these numbers from B M Angelov 1984 J. Phys. C: Solid State Phys. 17 1709
        self.rad_avg = np.zeros((7,1))
        self.rad_avg[2] = 0.44
        self.rad_avg[4] = 1.11
        self.rad_avg[6] = 4.816

        # these are the Steven's multiplicitive factors. Taken from Hutchins 1964
        self.stevens = np.zeros((7,1))
        self.stevens[2] = 2 ** 2 / (3 ** 2 * 5 ** 2 * 7)
        self.stevens[4] = 2 / (3 ** 2 * 5 * 7 * 11 * 13)
        self.stevens[6] = 2 * 2 * 2 / (3 * 3 * 3 * 7 * 11 ** 2 * 13 ** 2)

        # these scalings are strictly taken from the SPECTRE program
        # https://xray.physics.ox.ac.uk/UserGuide.pdf
        # but I think it can be originally derived from the Hutchins paper
        # it seems to be a geometric factor coming from definitions of spherical harmonics
        # versus the geometric meaning of the Steven's operators
        self.spectre_scaling = np.zeros((7,7))
        self.spectre_scaling[2, 0] = 0.5
        self.spectre_scaling[2, 1] = np.sqrt(6)
        self.spectre_scaling[2, 2] = 0.6*np.sqrt(6)
        self.spectre_scaling[4, 0] = 0.125
        self.spectre_scaling[4, 1] = 0.5*np.sqrt(5)
        self.spectre_scaling[4, 2] = 0.25*np.sqrt(10)
        self.spectre_scaling[4, 3] = 0.5 * np.sqrt(35)
        self.spectre_scaling[4, 4] = 0.125 * np.sqrt(70)
        self.spectre_scaling[6, 0] = 1 / 16
        self.spectre_scaling[6, 1] = 1 / 8 * np.sqrt(42)
        self.spectre_scaling[6, 2] = 1 / 16 * np.sqrt(105)
        self.spectre_scaling[6, 3] = 1 / 8 * np.sqrt(105)
        self.spectre_scaling[6, 4] = 3 / 16 * np.sqrt(14)
        self.spectre_scaling[6, 5] = 3 / 8 * np.sqrt(77)
        self.spectre_scaling[6, 6] = 1 / 16 * np.sqrt(231)

        self.echarge = -1.6e-19  # in C
        self.eps0 = 8.854e-12  # in F/m
        self.a0 = 5.3e-11  # in m
        return

    @staticmethod
    def cart_to_polar(ions):
        # ions array should be in the form x,y,z,q, with Er at (0,0,0)
        r_vec = np.sqrt(ions[:, 0] ** 2 + ions[:, 1] ** 2 + ions[:, 2] ** 2)
        phi_vec = np.arctan(ions[:, 1] / ions[:, 0])
        phi_vec[np.isnan(phi_vec)] = 0.0
        theta_vec = np.arccos(ions[:, 2] / r_vec)
        return r_vec, phi_vec, theta_vec

    def calc_b_param(self,l, m, ions):
        prefactor = 4 * np.pi / (2 * l + 1) * self.echarge ** 2 / (np.pi * 4 * self.eps0)

        # ions array should be in the form x,y,z,q, with Er at (0,0,0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_vec, phi_vec, theta_vec = self.cart_to_polar(ions)

        # scipy uses an inconvenient definition for spherical harmonic angles
        # checked that this gives the correct results for Oh and Td symmetries
        # another formulation of this uses Tesserel harmonics isntead of just
        # taking the real part of the spherical harmonic. The difference is a
        # normalization factor
        sum_vec = ions[:, 3] / np.power(r_vec, l + 1) * np.power(self.a0, l) * self.rad_avg[l] * np.real(
            sph_harm(m, l, phi_vec, theta_vec))
        temp = prefactor * np.sum(sum_vec) * self.stevens[l] * self.spectre_scaling[l,m]
        return -1 * temp * 6.24e21 * 8.06554

    def calc_cf_params(self,ions):
        lvec = [2, 4, 6]
        Bparams = []
        for l in lvec:
            mvec = range(0, l + 1)
            for m in mvec:
                Btemp = self.calc_b_param(l, m, ions)
                Bparams.append(Btemp)
        # now format into a dictionary
        Bparams = np.asarray(Bparams)
        b_dict = {'B2':Bparams[0:3],'B4':Bparams[3:8],'B6':Bparams[8:]}
        return b_dict
