import numpy as np


class StevensOperators:
    # this class contains the functions for building a Hamiltonian using
    # the Steven's operator formalism. This gives us some nice physical
    # intuition for the operators because we can construct them using only
    # Jz, J+, J- and J^2. However, this comes at the cost of assuming that only
    # states within the same angular momentum manifold couple.
    # The structure of CrystalFieldREI should allow you to replace this
    # set of functions with your own if you don't want to make this assumption.

    def __init__(self,jval):
        self.jval = jval

    # define some of the basic angular momentum operators.
    # this should keep things (relatively) compact later

    def j2(self):
        return self.jval*(self.jval+1)*np.eye(int(2*self.jval+1))

    def jz(self):
        return np.diag(np.arange(-1*self.jval,self.jval+1e-3))

    def jp(self):
        return np.diag(np.sqrt(self.jval*(self.jval+1)-np.arange(-1*self.jval,self.jval+1e-3)[0:-1]*np.arange(-1*self.jval,self.jval+1e-3)[1:]),k=1)

    def jm(self):
        return np.diag(np.sqrt(self.jval*(self.jval+1)-np.arange(-1*self.jval,self.jval+1e-3)[0:-1]*np.arange(-1*self.jval,self.jval+1e-3)[1:]),k=-1)

    def build_ham(self,bdict):
        ham = np.zeros_like(self.jz())
        for m,bval in zip(range(3),bdict['B2']):
            ham+=bval*self.o_mat(2,m)
        for m,bval in zip(range(5),bdict['B4']):
            ham+=bval*self.o_mat(4,m)
        for m,bval in zip(range(7),bdict['B6']):
            ham+=bval*self.o_mat(6,m)
        return ham

    def proc_ham(self,ham):
        eigvals, eigvecs = self.calc_eigvals(ham)
        out_dict = {'eigenvalues':eigvals,'eigenvecs':eigvecs,'Jval':self.jval}
        return out_dict

    @staticmethod
    def calc_eigvals(ham):
        return np.linalg.eigh(ham)

    def o_mat(self,l,m):
        if l == 2:
            if m == 0:
                omat = self.o_2_0()
            elif m == 1:
                omat = self.o_2_1()
            elif m == 2:
                omat = self.o_2_2()
            else:
                print('Invalid value of m. m cannot be greater than l')
                omat = 0.0
        elif l == 4:
            if m == 0:
                omat = self.o_4_0()
            elif m == 1:
                omat = self.o_4_1()
            elif m == 2:
                omat = self.o_4_2()
            elif m == 3:
                omat = self.o_4_3()
            elif m == 4:
                omat = self.o_4_4()
            else:
                print('Invalid value of m. m cannot be greater than l')
                omat = 0.0
        elif l == 6:
            if m == 0:
                omat = self.o_6_0()
            elif m == 1:
                omat = self.o_6_1()
            elif m == 2:
                omat = self.o_6_2()
            elif m == 3:
                omat = self.o_6_3()
            elif m == 4:
                omat = self.o_6_4()
            elif m == 5:
                omat = self.o_6_5()
            elif m == 6:
                omat = self.o_6_6()
            else:
                print('Invalid value of m. m cannot be greater than l')
                omat = 0.0
        else:
            print('Invalid value of l. l must be 2,4,6')
            omat = 0.0
        return omat

    # now define the Stevens Operators. This is based on Hutching 1964
    # functions are checked against the tabulated values in this paper

    # m = 0 terms
    def o_2_0(self):
        o20 = 3*self.jz()@self.jz() - self.j2()
        return o20

    def o_4_0(self):
        o40 = (35*np.linalg.matrix_power(self.jz(), 4) - 30*self.j2()@np.linalg.matrix_power(self.jz(), 2)
               + 25*self.jz()@self.jz() - 6*self.j2() + 3*self.j2()@self.j2())
        return o40

    def o_6_0(self):
        o60 = (231*np.linalg.matrix_power(self.jz(), 6) - 315*self.j2()@np.linalg.matrix_power(self.jz(), 4)
               + 735*np.linalg.matrix_power(self.jz(), 4) + 105*np.linalg.matrix_power(self.j2(), 2)@np.linalg.matrix_power(self.jz(), 2)
               - 525*self.j2()@np.linalg.matrix_power(self.jz(), 2) + 294*np.linalg.matrix_power(self.jz(), 2)
               - 5*np.linalg.matrix_power(self.j2(), 3) + 40*np.linalg.matrix_power(self.j2(), 2)
               -60*self.j2())
        return o60

    # m =1 terms
    def o_2_1(self):
        term1 = self.jz()
        term2 = self.jp() + self.jm()
        o21 = 0.25*(term1@term2 + term2@term1)
        return o21

    def o_4_1(self):
        term1 = 7*np.linalg.matrix_power(self.jz(),3) - 3*self.j2()@self.jz() + self.jz()
        term2 = self.jp() + self.jm()
        o41 = 0.25 * (term1 @ term2 + term2 @ term1)
        return o41

    def o_6_1(self):
        term1 = (33 * np.linalg.matrix_power(self.jz(), 5)
                 - 30*self.j2()@np.linalg.matrix_power(self.jz(),3)
                 - 15*np.linalg.matrix_power(self.jz(),3)
                 + 5*self.j2()@self.j2()@self.jz() - 10*self.j2()@self.jz()
                 + 12*self.jz())
        term2 = self.jp() + self.jm()
        o61 = 0.25 * (term1 @ term2 + term2 @ term1)
        return o61

    # m = 2 terms
    def o_2_2(self):
        o22 = 0.5*(self.jp()@self.jp() + self.jm()@self.jm())
        return o22

    def o_4_2(self):
        term1 = 7*self.jz()@self.jz() - self.j2() - 5*np.eye(int(2*self.jval+1))
        term2 = self.jp()@self.jp() + self.jm()@self.jm()
        o42 = 0.25*(term1@term2 + term2@term1)
        return o42

    def o_6_2(self):
        term1 = (33*np.linalg.matrix_power(self.jz(),4) - 18*self.jz()@self.jz()@self.j2()
                 - 123*self.jz()@self.jz() + self.j2()@self.j2() + 10*self.j2()
                 + 102*np.eye(int(2*self.jval+1)))
        term2 = self.jp()@self.jp() + self.jm()@self.jm()
        o62 = 0.25 * (term1 @ term2 + term2 @ term1)
        return o62

    # m = 3 terms
    def o_4_3(self):
        term1 = self.jz()
        term2 = np.linalg.matrix_power(self.jp(),3) + np.linalg.matrix_power(self.jm(),3)
        o43 = 0.25*(term1@term2 + term2@term1)
        return o43

    def o_6_3(self):
        term1 = 11*np.linalg.matrix_power(self.jz(),3) - 3*self.jz()@self.j2() - 59*self.jz()
        term2 = np.linalg.matrix_power(self.jp(),3) + np.linalg.matrix_power(self.jm(),3)
        o63 = 0.25*(term1@term2 + term2@term1)
        return o63

    # m=4 terms
    def o_4_4(self):
        o44 = 0.5*(np.linalg.matrix_power(self.jp(),4) + np.linalg.matrix_power(self.jm(),4))
        return o44

    def o_6_4(self):
        term1 = 11*self.jz()@self.jz() - self.j2() - 38*np.eye(int(2*self.jval+1))
        term2 = np.linalg.matrix_power(self.jp(),4) + np.linalg.matrix_power(self.jm(),4)
        o64 = 0.25*(term1@term2 + term2@term1)
        return o64

    # m=5 terms.
    def o_6_5(self):
        term1 = self.jz()
        term2 = np.linalg.matrix_power(self.jp(), 5) + np.linalg.matrix_power(self.jm(), 5)
        o65 = 0.25 * (term1 @ term2 + term2 @ term1)
        return o65

    # m=6 terms
    def o_6_6(self):
        o66 = 0.5*(np.linalg.matrix_power(self.jp(),6) + np.linalg.matrix_power(self.jm(),6))
        return o66
