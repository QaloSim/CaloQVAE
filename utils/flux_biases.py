"""
    This was taken from https://github.com/dwavesystems/dwave-system/pull/513/files#diff-9a532ba673abaa03a87ac2a1c71e0f57f84e6eff24c6e7081a2fe23f4f5e6f43R443
    A simple use case: Consider three spins
    
    h = {0: 0.0, 1: 0., 2: -3.5}
    J = {(0, 1): 1, (1, 2): 1}
    embedding = {0: [30], 1: [45], 2: [2985]}
    fb = [0]*qpu.properties['num_qubits']    
    sampleset = FixedEmbeddingComposite(qpu, embedding).sample_ising(h, J,
                   num_reads=100, flux_drift_compensation=False, flux_biases=fb)       
    sampleset.record["sample"]
    >> [[-1,1,1],
        [1,-1,1]]
        
    Since spin 2985 has a self-field -3.5 which minimizes energy when it's 1
    
    Now, we do:
    
    fb[2985] = h_to_fluxbias(20)
    sampleset = FixedEmbeddingComposite(qpu, embedding).sample_ising(h, J,
                   num_reads=100, flux_drift_compensation=False, flux_biases=fb)       
    sampleset.record["sample"]
    >> [[-1,1,-1]]
    
    For our purposes, suppose spin 60 needs to be up and spin 61 down. then we do:
    
    fb[60] = h_to_fluxbias(-20)  
    fb[61] = h_to_fluxbias(20)    # I was able to go up to 50, but ~20 should be enough
    response = self._qpu_sampler.sample_ising(h, J, num_reads=num_samples, answer_mode='raw', auto_scale=False, flux_drift_compensation=False, flux_biases=fb)
"""

from typing import Optional, Literal, Union
import numpy as np


def h_to_fluxbias(h: Union[float, np.ndarray]=1,
                  Ip: Optional[float]=None,
                  B: float=1.391, MAFM: Optional[float]=1.647,
                  units_Ip: Optional[str]='uA',
                  units_B : str='GHz',
                  units_MAFM : Optional[str]='pH') -> Union[float, np.ndarray]:
    """Convert problem Hamiltonian bias ``h`` to equivalent flux bias.
    Unitless bias ``h`` is converted to the equivalent flux bias in units 
    :math:`\Phi_0`, the magnetic flux quantum.
    The dynamics of ``h`` and flux bias differ, as described in the
    :func:`Ip_in_units_of_B` function.
    Equivalence at a specific point in the anneal is valid under a 
    freeze-out (quasi-static) hypothesis.
    Defaults are based on the published physical properties of 
    `Leap <https://cloud.dwavesys.com/leap/>`_\ 's  
    ``Advantage_system4.1`` solver at single-qubit freezeout (:math:`s=0.612`).
    Args:
        Ip:
            Persistent current, :math:`I_p(s)`, in units of amps or 
            microamps. When not provided, inferred from :math:`M_{AFM}` 
            and and :math:`B(s)` based on the relation 
            :math:`B(s) = 2 M_{AFM} I_p(s)^2`. 
    
        B:
            Annealing schedule field, :math:`B(s)`, associated with the 
            problem Hamiltonian. Schedules are provided for each quantum 
            computer in the 
            :ref:`system documentation <sysdocs_gettingstarted:doc_qpu_characteristics>`. 
            This parameter is ignored when ``Ip`` is specified.
        
        MAFM:
            Mutual inductance, :math:`M_{AFM}`, specified for each quantum 
            computer in the 
            :ref:`system documentation <sysdocs_gettingstarted:doc_qpu_characteristics>`. 
            ``MAFM`` is ignored when ``Ip`` is specified.
        units_Ip:
            Units in which the persistent current, ``Ip``, is specified. 
            Allowed values are ``'uA'`` (microamps) and ``'A'`` (amps)
        units_B:
            Units in which the schedule ``B`` is specified. Allowed values
            are ``'GHz'`` (gigahertz) and ``'J'`` (Joules).
        units_MAFM:
            Units in which the mutual inductance, ``MAFM``, is specified. Allowed 
            values are ``'pH'`` (picohenry) and ``'H'`` (Henry).
    
    Returns:
        Flux-bias values producing equivalent longitudinal fields to the given 
        ``h`` values.
    """
    Ip = Ip_in_units_of_B(Ip, B, MAFM,
                          units_Ip, units_B, units_MAFM)  # Convert/Create Ip in units of B, scalar
    # B(s)/2 h_i = Ip(s) phi_i 
    # print(B, Ip, h)
    return -B/2/Ip*h



def Ip_in_units_of_B(Ip: Union[None, float, np.ndarray]=None,
                     B: Union[None, float, np.ndarray]=1.391,
                     MAFM: Optional[float]=1.647,
                     units_Ip: Optional[str]='uA',
                     units_B: Literal['GHz', 'J'] = 'GHz',
                     units_MAFM : Optional[str]='pH') -> Union[float, np.ndarray]:
    """Estimate qubit persistent current :math:`I_p(s)` in schedule units. 
    Under a simple, noiseless freeze-out model, you can substitute flux biases 
    for programmed linear biases, ``h``, in the standard transverse-field Ising 
    model as implemented on D-Wave quantum computers. Perturbations in ``h`` are 
    not, however, equivalent to flux perturbations with respect to dynamics 
    because of differences in the dependence on the anneal fraction, :math:`s`: 
    :math:`I_p(s) \propto \sqrt(B(s))`. The physical origin of each term is different, 
    and so precision and noise models also differ.
    
    Assume a Hamiltonian in the :ref:`documented form <sysdocs_gettingstarted:doc_qpu>` 
    with an additional flux-bias-dependent component 
    :math:`H(s) \Rightarrow H(s) - H_F(s) \sum_i \Phi_i \sigma^z_i`,
    where :math:`\Phi_i` are flux biases (in units of :math:`\Phi_0`), 
    :math:`\sigma^z_i` is the Pauli-z operator, and 
    :math:`H_F(s) = Ip(s) \Phi_0`. Schedules for D-Wave quantum computers 
    specify energy in units of Joule or GHz. 
    Args:
        Ip:
            Persistent current, :math:`I_p(s)`, in units of amps or 
            microamps. When not provided, inferred from :math:`M_{AFM}` 
            and and :math:`B(s)` based on the relation 
            :math:`B(s) = 2 M_{AFM} I_p(s)^2`. 
    
        B:
            Annealing schedule field, :math:`B(s)`, associated with the 
            problem Hamiltonian. Schedules are provided for each quantum 
            computer in the 
            :ref:`system documentation <sysdocs_gettingstarted:doc_qpu_characteristics>`. 
            This parameter is ignored when ``Ip`` is specified.
        
        MAFM:
            Mutual inductance, :math:`M_{AFM}`, specified for each quantum 
            computer in the 
            :ref:`system documentation <sysdocs_gettingstarted:doc_qpu_characteristics>`. 
            ``MAFM`` is ignored when ``Ip`` is specified.
        units_Ip:
            Units in which the persistent current, ``Ip``, is specified. 
            Allowed values are ``'uA'`` (microamps) and ``'A'`` (amps)
        units_B:
            Units in which the schedule ``B`` is specified. Allowed values
            are ``'GHz'`` (gigahertz) and ``'J'`` (Joules).
        units_MAFM:
            Units in which the mutual inductance, ``MAFM``, is specified. Allowed 
            values are ``'pH'`` (picohenry) and ``'H'`` (Henry).
    
    Returns:
        :math:`I_p(s)` with units matching the Hamiltonian :math:`B(s)`.
    """
    h = 6.62607e-34  # Plank's constant for converting energy in Hertz to Joules 
    Phi0 = 2.0678e-15  # superconducting magnetic flux quantum (h/2e); units: Weber=J/A

    if units_B == 'GHz':
        B_multiplier = 1e9*h  # D-Wave schedules use GHz by convention
    elif units_B == 'J':
        B_multiplier = 1
    else:
        raise ValueError('Schedule B must be in units GHz or J, ' 
                         f'but given {units_B}')
    if Ip is None:
        B = B*B_multiplier # To Joules
        if units_MAFM == 'pH':
            MAFM = MAFM*1e-12  # conversion from picohenry to Henry
        elif units_MAFM != 'H':
            raise ValueError('MAFM must be in units pH or H, ' 
                             f'but given {units_MAFM}')
        Ip = np.sqrt(B/(2*MAFM))  # Units of A = C/s, O(1e-6) 
    else:
        if units_Ip == 'uA':
            Ip = Ip*1e-6  # Conversion from microamps to amp
        elif units_Ip != 'A':
            raise ValueError('Ip must be in units uA or A, ' 
                             f'but given {units_Ip}')

    return Ip*Phi0/B_multiplier

