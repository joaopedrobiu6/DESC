from desc import set_device
set_device("gpu")
import desc.io
from desc.backend import jnp
import scipy.constants
from time import time as timet
import desc.equilibrium
from desc.objectives import ParticleTracer

from jb_utils import set_mu, output_to_file

time1 = timet()

# Load Equilibrium
print("\nStarting Equilibrium")
eq_file = "eq_2411_M1_N1.h5"
eq = desc.io.load(eq_file)
eq._iota = eq.get_profile("iota").to_powerseries(order=eq.L, sym=True)
eq._current = None
print(f"Loaded Equilibrium: {eq_file}\n")

time2 = timet()
print(f"Time to load equilibrium: {time2 - time1}s")

# Energy and Mass info
Energy_eV = 10 #1 # eV (3.52e6 eV proton energy)
Proton_Mass = scipy.constants.proton_mass
Proton_Charge = scipy.constants.elementary_charge
Energy_SI = Energy_eV*Proton_Charge

# Particle Info
Mass = 4*Proton_Mass
Charge = 2*Proton_Charge

# Initial State
psi_i = 0.8
zeta_i = 0.5
theta_i = jnp.pi/2
vpar_i = -0.1*jnp.sqrt(2*Energy_SI/Mass)
ini_cond = jnp.array([float(psi_i), theta_i, zeta_i, float(vpar_i)])

# Time
tmin = 0
tmax = 5e-2
nt = 7500
time = jnp.linspace(tmin, tmax, nt)

# Initial State
initial_conditions = ini_cond
Mass_Charge_Ratio = Mass/Charge

# Mu 
# grid = Grid(jnp.array([jnp.sqrt(psi_i), theta_i, zeta_i]).T, jitable=True, sort=False)
# data = eq.compute("|B|", grid=grid)
# mu = Energy_SI/(Mass*data["|B|"]) - (vpar_i**2)/(2*data["|B|"])

mu = set_mu(psi_i, theta_i, zeta_i, vpar_i, eq, Energy_SI, Mass)

# Initial Parameters
ini_param = jnp.array([mu[0], Mass_Charge_Ratio])       # this works

time3 = timet()

# TRACER DIFFRAX
tracer_diffrax = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="tracer", tolerance=1.4e-8, integration_lib="diffrax")
tracer_diffrax.build()
solution_diffrax = tracer_diffrax.compute(*tracer_diffrax.xs(eq))

print(solution_diffrax.shape)
# transform shape from [7500, 4, 1] to [7500, 4]
solution_diffrax = solution_diffrax[:, :, 0]
print(solution_diffrax.shape)

time4 = timet()
timediff_diffrax = time4 - time3
print(f"Time to compute tracer - DIFFRAX: {time4 - time3}s")

time5 = timet()

# TRACER JAX
tracer_jax = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="tracer", tolerance=1.4e-8, integration_lib="jaxint")
tracer_jax.build()
solution_jax = tracer_jax.compute(*tracer_jax.xs(eq))

time6 = timet()
timediff_jax = time6 - time5
print(f"Time to compute tracer - JAX: {time6 - time5}s")

# Compare Results
diff=solution_diffrax-solution_jax
output_to_file(sol=diff, name="solution_difference_dopri5")

# PLOT BAR PLOT OF TIME TAKEN
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar(["DIFFRAX", "JAX"], [timediff_diffrax, timediff_jax])
ax.set_ylabel("Time (s)")
ax.set_xlabel("Integration Library")

# Save the figure and show
plt.savefig('time_comparison_dopri5.png')
plt.show()
