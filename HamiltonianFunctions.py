import numpy as np


class StevensOperators:

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
    # these are always 0 I think, but let's put some dummy terms to help with indexing
    def o_2_1(self):
        o21 = 0*self.jz()
        return o21

    def o_4_1(self):
        o41 = 0*self.jz()
        return o41

    def o_6_1(self):
        o61 = 0*self.jz()
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