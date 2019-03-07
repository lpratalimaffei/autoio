""" common elstruct parameters
"""


class MODULE():
    """ elstruct module names """
    WRITER = 'writer'
    READER = 'reader'


MODULES = (MODULE.WRITER, MODULE.READER)


class BASIS():
    """ electronic structure basis sets """
    STO3G = 'sto-3g'
    P321G = '3-21g'
    P631G = '6-31g'
    PVDZ = 'cc-pvdz'
    PVTZ = 'cc-pvtz'


class METHOD():
    """ electronic structure methods """

    class SCF():
        """ scf method names """
        RHF = 'rhf'
        UHF = 'uhf'
        ROHF = 'rohf'
        LIST = (RHF, UHF, ROHF)

    class CORR():
        """ correlation method names """
        MP2 = 'mp2'

    # RHF Methods
    RHF = SCF.RHF
    RHF_MP2 = '-'.join([SCF.RHF, CORR.MP2])
    # UHF Methods
    UHF = SCF.UHF
    UHF_MP2 = '-'.join([SCF.UHF, CORR.MP2])
    # ROHF Methods
    ROHF = SCF.ROHF
    ROHF_MP2 = '-'.join([SCF.ROHF, CORR.MP2])

    @classmethod
    def split_name(cls, method_name):
        """ split a method name into an scf method and a correlated method
        """
        split = method_name.split('-')
        assert len(split) <= 2
        scf_method = split[0]
        assert scf_method in cls.SCF.LIST
        corr_method = split[1] if len(split) == 2 else None
        return scf_method, corr_method


class PROGRAM():
    """ Programs to be called """
    PSI4 = 'psi4'


class JOB():
    """ The type of job
    """
    ENERGY = 'energy'
    GRADIENT = 'gradient'
    HESSIAN = 'hessian'
    OPTIMIZATION = 'optimization'


class ERROR():
    """ Job errors
    """
    SCF_NOCONV = 'scf_noconv'
    OPT_NOCONV = 'opt_noconv'


class OPTION():
    """ Writer options
    """
    class SCF():
        """ SCF options """
        KEY = 'scf_options'

    class OPT():
        """ optimization options """
        KEY = 'opt_options'
