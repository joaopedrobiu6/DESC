"""Objectives for optimizing the equilibrium from tracing particles."""

import warnings

import numpy as np

from desc.backend import jnp
from desc.compute import compute as compute_fun
from desc.compute import get_params, get_profiles, get_transforms
from desc.grid import Grid, LinearGrid
from desc.utils import Timer
from desc.vmec_utils import ptolemy_linear_transform
from jax.experimental.ode import odeint as jax_odeint
from functools import partial
from jax import jit

from .normalization import compute_scaling_factors
from .objective_funs import _Objective

class ParticleTracer(_Objective):
    """Particle Tracer using Guiding Center equations of motion.

    Parameters
    ----------
    eq : Equilibrium, optional
        Equilibrium that will be optimized to satisfy the Objective.
    target : float, ndarray, optional
        Target value(s) of the objective. Only used if bounds is None.
        len(target) must be equal to Objective.dim_f
    bounds : tuple, optional
        Lower and upper bounds on the objective. Overrides target.
        len(bounds[0]) and len(bounds[1]) must be equal to Objective.dim_f
    weight : float, ndarray, optional
        Weighting to apply to the Objective, relative to other Objectives.
        len(weight) must be equal to Objective.dim_f
    normalize : bool
        Whether to compute the error in physical units or non-dimensionalize.
    normalize_target : bool
        Whether target and bounds should be normalized before comparing to computed
        values. If `normalize` is `True` and the target is in physical units,
        this should also be set to True.
    output_time : ndarray
        Values of time where the system is evaluated.
    initial_conditions : tuple, array
        Initial conditions (psi, theta, zeta, vpar) to solve the system of equations.
        Starting state of the system.
    initial_parameters : tuple, array
        Parameters needed in the system, such as the magnetic momentum, mu, and the mass-charge ratio, m_q.
    compute_option: str
        Select the compute() output. Can be "optimization" for the optimization metric; "tracer" for the full 
        solution of the system; "average psi/theta/zeta/vpar" for the mean value of psi/theta/zeta/vpar in the 
        computed time.
    name : str
        Name of the objective function.
    """
    
    _scalar = False
    _linear = False
    _units = ""
    _print_value_fmt = "System solution: {:10.3e}"

    def __init__(
        self,
        eq=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        output_time=None,
        initial_conditions=None,
        initial_parameters=None,
        compute_option=None,
        tolerance =1.4e-8,
        name="Particle Tracer"
    ):
        self.output_time = output_time
        self.initial_conditions=jnp.asarray(initial_conditions) 
        self.initial_parameters=jnp.asarray(initial_parameters)
        self.compute_option=compute_option
        self.tolerance = tolerance
        
        if target is None and bounds is None:
            target = 0
        
        super().__init__(
            eq=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            name=name,
        )

        self._print_value_fmt = (
            "System solution for initial conditions"
        )

    def build(self, eq=None, use_jit=True, verbose=1):
        
        self._data_keys = ["psidot", "thetadot", "zetadot", "vpardot"]
        self._args = get_params(
            self._data_keys,
            obj="desc.equilibrium.equilibrium.Equilibrium",
            has_axis=False,
        )
        
        self.charge = 1.6e-19
        self.mass = 1.673e-27 # CHECK VALUES
        self.Energy = 3.52e6*self.charge 
        eq = eq or self._eq

        if self.compute_option == "optimization":
            self._dim_f = len(self.output_time)
        elif self.compute_option == "tracer":
            self._dim_f = [len(self.output_time), 4]
        elif self.compute_option == "average psi":
            self._dim_f = len(self.output_time)
        elif self.compute_option == "average theta":
            self._dim_f = len(self.output_time)
        elif self.compute_option == "average zeta":
            self._dim_f = len(self.output_time)
        elif self.compute_option == "average vpar":
           self._dim_f = len(self.output_time)

        super().build(eq=eq, use_jit=use_jit, verbose=verbose)

    def compute(self, *args, **kwargs):

        params, constants = self._parse_args(*args, **kwargs)
        if constants is None:
            constants = self.constants
        
        
        def system(initial_conditions = self.initial_conditions, t = self.output_time, initial_parameters = self.initial_parameters):
            #initial conditions
            psi = initial_conditions[0]
            theta = initial_conditions[1]
            zeta = initial_conditions[2]
            vpar = initial_conditions[3]
            
            grid = Grid(jnp.array([jnp.sqrt(psi), theta, zeta]).T, jitable=True, sort=False)
            transforms = get_transforms(self._data_keys, self._eq, grid, jitable=True)
            profiles = get_profiles(self._data_keys, self._eq, grid, jitable=True)
            
            data = compute_fun("desc.equilibrium.equilibrium.Equilibrium", self._data_keys, params, transforms, profiles, mu=initial_parameters[0], m_q=initial_parameters[1], vpar=vpar)
            
            psidot = data["psidot"]
            thetadot = data["thetadot"]
            zetadot = data["zetadot"]
            vpardot = data["vpardot"]

            return jnp.array([psidot, thetadot, zetadot, vpardot])
        
        initial_conditions_jax = jnp.array(self.initial_conditions, dtype=jnp.float64)
        t_jax = self.output_time
        system_jit = jit(system)
        solution = jax_odeint(partial(system_jit, initial_parameters=self.initial_parameters), initial_conditions_jax, t_jax, rtol = self.tolerance)

        if self.compute_option == "optimization":
            return jnp.sum((solution[:, 0] - solution[0, 0]) * (solution[:, 0] - solution[0, 0]), axis=-1)
        elif self.compute_option == "tracer":
            return solution
        elif self.compute_option == "average psi":
            return jnp.mean(solution[:, 0])
        elif self.compute_option == "average theta":
            return jnp.mean(solution[:, 1])
        elif self.compute_option == "average zeta":
            return jnp.mean(solution[:, 2])
        elif self.compute_option == "average vpar":
            return jnp.mean(solution[:, 3])