import numpy as np


# noinspection PyCompatibility
class ExtendedStevensOperators:
    # this class contains the functions for the extended Stevens operators.
    # These are lifted verbatim from Barnes et al, 1964 Phys Rev 136. Most of
    # the structure and basis functions are taken from HamiltonianFunctions.py,
    # which should also be in this repository

    def __init__(self, jval):
        self.jval = jval
        self.bfieldvec = [0, 0, 0]  # applied Bfield (x,y,z) in Gauss
        self.ub_cm = 4.66e-5
        self.Sval = 3 / 2
        self.Lval = self.jval - self.Sval
        self.gj = 1.5 + (self.Sval * (self.Sval + 1) - self.Lval * (self.Lval + 1)) / (2 * self.jval * (self.jval + 1))
        if self.jval == 15/2:
            self.prefactors = [2.6947e-3,4.5187e-5,1.9993e-6]
        elif self.jval == 13/2:
            self.prefactors = [3.1409e-3,5.654e-5,1.7801e-6]
        else:
            self.prefactors = [1e-3,1e-5,1e-6]
            print('Uknown scalings for this J value. Set to order-of-magnitude guesses')

    def set_bfield(self, bfield_in):
        self.bfieldvec = bfield_in
        return

    # define some of the basic angular momentum operators.
    # this should keep things (relatively) compact later

    def j2(self):
        return self.jval * (self.jval + 1) * np.eye(int(2 * self.jval + 1))

    def jz(self):
        return np.diag(np.arange(-1 * self.jval, self.jval + 1e-3))

    def jp(self):
        return np.diag(np.sqrt(
            self.jval * (self.jval + 1) - np.arange(-1 * self.jval, self.jval + 1e-3)[0:-1] * np.arange(-1 * self.jval,
                                                                                                        self.jval + 1e-3)[
                                                                                              1:]), k=1)

    def jm(self):
        return np.diag(np.sqrt(
            self.jval * (self.jval + 1) - np.arange(-1 * self.jval, self.jval + 1e-3)[0:-1] * np.arange(-1 * self.jval,
                                                                                                        self.jval + 1e-3)[
                                                                                              1:]), k=-1)

    # define the Stevens operator equivalents - NOTE uses the negative superscripts from Barnes 1964

    # m = 0 terms
    def o_2_0(self):
        o20 = 3 * self.jz() @ self.jz() - self.j2()
        return o20

    def o_4_0(self):
        o40 = (35 * np.linalg.matrix_power(self.jz(), 4) - 30 * self.j2() @ np.linalg.matrix_power(self.jz(), 2)
               + 25 * self.jz() @ self.jz() - 6 * self.j2() + 3 * self.j2() @ self.j2())
        return o40

    def o_6_0(self):
        o60 = (231 * np.linalg.matrix_power(self.jz(), 6) - 315 * self.j2() @ np.linalg.matrix_power(self.jz(), 4)
               + 735 * np.linalg.matrix_power(self.jz(), 4) + 105 * np.linalg.matrix_power(self.j2(),
                                                                                           2) @ np.linalg.matrix_power(
                    self.jz(), 2)
               - 525 * self.j2() @ np.linalg.matrix_power(self.jz(), 2) + 294 * np.linalg.matrix_power(self.jz(), 2)
               - 5 * np.linalg.matrix_power(self.j2(), 3) + 40 * np.linalg.matrix_power(self.j2(), 2)
               - 60 * self.j2())
        return o60

    # m = 2 terms
    def o_2_2(self):
        o22 = 1/2 * (self.jp() @ self.jp() + self.jm() @ self.jm())
        return o22

    def o_4_2(self):
        term1 = 7 * self.jz() @ self.jz() - self.j2() - 5 * np.eye(int(2 * self.jval + 1))
        term2 = self.jp() @ self.jp() + self.jm() @ self.jm()
        o42 = 1/4 * (term1 @ term2 + term2 @ term1)
        return o42

    def o_6_2(self):
        term1 = (33 * np.linalg.matrix_power(self.jz(), 4) - 18 * self.jz() @ self.jz() @ self.j2()
                 - 123 * self.jz() @ self.jz() + self.j2() @ self.j2() + 10 * self.j2()
                 + 102 * np.eye(int(2 * self.jval + 1)))
        term2 = self.jp() @ self.jp() + self.jm() @ self.jm()
        o62 = 0.25 * (term1 @ term2 + term2 @ term1)
        return o62

    # m=-2 terms
    def o_2_m2(self):
        o2m2 = 1 / (2 * 1j) * (self.jp() @ self.jp() - self.jm() @ self.jm())
        return o2m2

    def o_4_m2(self):
        term1 = 7 * self.jz() @ self.jz() - self.j2() - 5 * np.eye(int(2 * self.jval + 1))
        term2 = self.jp() @ self.jp() - self.jm() @ self.jm()
        o4m2 = 1 / (4 * 1j) * (term1 @ term2 + term2 @ term1)
        return o4m2

    def o_6_m2(self):
        term1 = (33 * np.linalg.matrix_power(self.jz(), 4) - 18 * self.jz() @ self.jz() @ self.j2()
                 - 123 * self.jz() @ self.jz() + self.j2() @ self.j2() + 10 * self.j2()
                 + 102 * np.eye(int(2 * self.jval + 1)))
        term2 = self.jp() @ self.jp() - self.jm() @ self.jm()
        o6m2 = 1 / (4 * 1j) * (term1 @ term2 + term2 @ term1)
        return o6m2

    # m=4 terms
    def o_4_4(self):
        o44 = 0.5 * (np.linalg.matrix_power(self.jp(), 4) + np.linalg.matrix_power(self.jm(), 4))
        return o44

    def o_6_4(self):
        term1 = 11 * self.jz() @ self.jz() - self.j2() - 38 * np.eye(int(2 * self.jval + 1))
        term2 = np.linalg.matrix_power(self.jp(), 4) + np.linalg.matrix_power(self.jm(), 4)
        o64 = 0.25 * (term1 @ term2 + term2 @ term1)
        return o64

    # m=-4 terms
    def o_4_m4(self):
        o4m4 = 1 / (2 * 1j) * (np.linalg.matrix_power(self.jp(), 4) - np.linalg.matrix_power(self.jm(), 4))
        return o4m4

    def o_6_m4(self):
        term1 = 11 * self.jz() @ self.jz() - self.j2() - 38 * np.eye(int(2 * self.jval + 1))
        term2 = np.linalg.matrix_power(self.jp(), 4) - np.linalg.matrix_power(self.jm(), 4)
        o6m4 = 1 / (4 * 1j) * (term1 @ term2 + term2 @ term1)
        return o6m4

    # m=6 terms
    def o_6_6(self):
        o66 = 1/2 * (np.linalg.matrix_power(self.jp(), 6) + np.linalg.matrix_power(self.jm(), 6))
        return o66

    # m=-6 terms
    def o_6_m6(self):
        o6m6 = 1/(2*1j) * (np.linalg.matrix_power(self.jp(), 6) - np.linalg.matrix_power(self.jm(), 6))
        return o6m6


    def build_ham(self, coefflist):
        ham = (coefflist[0] * self.prefactors[0] * self.o_2_0() +
               coefflist[1] * self.prefactors[1] * self.o_4_0() +
               coefflist[2] * self.prefactors[2] * self.o_6_0() +
               coefflist[3] * self.prefactors[1] * self.o_4_4() +
               coefflist[4] * self.prefactors[1] * self.o_4_m4() +
               coefflist[5] * self.prefactors[2] * self.o_6_4() +
               coefflist[6] * self.prefactors[2] * self.o_6_m4())
        ham += (self.ub_cm * self.gj * self.bfieldvec[0] * 0.5 * (self.jp() + self.jm())
                + self.ub_cm * self.gj * self.bfieldvec[1] * 0.5 * (self.jp() + self.jm())
                + self.ub_cm * self.gj * self.bfieldvec[2] * self.jz())
        return ham

    def calc_g(self,eigvectors):
        jx = 1 / 2*(self.jp() + self.jm())
        gper = np.abs(self.gj*2*np.transpose(np.conj(eigvectors[1]))@jx@eigvectors[0])
        gpar = np.abs(self.gj * 2 * np.transpose(np.conj(eigvectors[0])) @ self.jz() @ eigvectors[0])
        return [gpar,gper]
