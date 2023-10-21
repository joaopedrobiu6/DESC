from desc import set_device
set_device("gpu")
from desc.grid import Grid
import desc.io
from desc.backend import jnp
import scipy.constants
import matplotlib.pyplot as plt
import numpy as np
from time import time as timet
import desc.equilibrium
from desc.objectives import ParticleTracer, ObjectiveFunction, ForceBalance, FixBoundaryR, FixBoundaryZ, FixPressure, FixIota, FixPsi

initial_time = timet()

# Functions

def output_to_file(solution, name):
    list1 = solution[:, 0]
    list2 = solution[:, 1]
    list3 = solution[:, 2]
    list4 = solution[:, 3]

    combined_lists = zip(list1, list2, list3, list4)
    
    file_name = f'{name}.txt'

    with open(file_name, 'w') as file:        
        for row in combined_lists:
            row_str = '\t'.join(map(str, row))
            file.write(row_str + '\n')

def Trajectory_Plot(solution, save_name="Trajectory_Plot.png"):
    fig, ax = plt.subplots()
    ax.plot(np.sqrt(solution[:, 0]) * np.cos(solution[:, 1]), np.sqrt(solution[:, 0]) * np.sin(solution[:, 1]))
    ax.set_aspect("equal", adjustable='box')
    plt.xlabel(r'$\sqrt{\psi}cos(\theta)$')
    plt.ylabel(r'$\sqrt{\psi}sin(\theta)$')
    fig.savefig(save_name, bbox_inches="tight", dpi=300)
    print(f"Trajectory Plot Saved: {save_name}")

def Quantity_Plot(solution, save_name="Quantity_Plot.png"):
    fig, axs = plt.subplots(2, 2)
    axs[0, 1].plot(time, solution[:, 0], 'tab:orange')
    axs[0, 1].set_title(r'$\psi$ (t)')
    axs[1, 0].plot(time, solution[:, 1], 'tab:green')
    axs[1, 0].set_title(r'$\theta$ (t)')
    axs[1, 1].plot(time, solution[:, 2], 'tab:red')
    axs[1, 1].set_title(r'$\zeta$ (t)')
    axs[0, 0].plot(time, solution[:, 3], 'tab:blue')
    axs[0, 0].set_title(r"$v_{\parallel}$ (t)")
    fig = plt.gcf()
    fig.set_size_inches(10.5, 10.5)
    fig.savefig(save_name, bbox_inches="tight", dpi=300)
    print(f"Quantity Plot Saved: {save_name}")

def Energy_Plot(solution, save_name="Energy_Plot.png"):
    plt.figure()
    grid = Grid(np.vstack((np.sqrt(solution[:, 0]), solution[:, 1], solution[:, 2])).T,sort=False)
    B_field = eq.compute("|B|", grid=grid)
    Energy = 0.5*(solution[:, 3]**2 + 2*B_field["|B|"]*mu)*Mass

    plt.plot(time, (Energy-Energy_SI)/Energy_SI)
    plt.title(r"(E - E$_0$)/E$_0$")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig(save_name, bbox_inches="tight", dpi=300)
    print(f"Energy Plot Saved: {save_name}")

print("*************** START ***************")

# Load Equilibrium

print("Starting Equilibrium")
eq_file = "input.LandremanPaul2021_QA_scaled_output.h5"
opt_file = "optimized_" + eq_file
print(f"Loaded Equilibrium: {eq_file}")

eq = desc.io.load(eq_file)[-1]
eq._iota = eq.get_profile("iota").to_powerseries(order=eq.L, sym=True)
eq._current = None

# Energy and Mass info
Energy_eV = 3.52e6
Proton_Mass = scipy.constants.proton_mass
Proton_Charge = scipy.constants.elementary_charge
Energy_SI = Energy_eV*Proton_Charge
# Energy and Mass info
Energy_eV = 3.52e6
Proton_Mass = scipy.constants.proton_mass
Proton_Charge = scipy.constants.elementary_charge

Energy_SI = Energy_eV*Proton_Charge

# Particle Info
Mass = 4*Proton_Mass
Charge = 2*Proton_Charge

# Initial State
psi_i = 0.2
zeta_i = 0
theta_i = 0
vpar_i = 0.7*jnp.sqrt(2*Energy_SI/Mass)
ini_cond = [float(psi_i), theta_i, zeta_i, float(vpar_i)]

# Time
tmin = 0
tmax = 1e-4
nt = 250
time = jnp.linspace(tmin, tmax, nt)

# Initial State
initial_conditions = ini_cond
Mass_Charge_Ratio = Mass/Charge

# Mu 
grid = Grid(jnp.array([jnp.sqrt(psi_i), theta_i, zeta_i]).T, jitable=True, sort=False)
data = eq.compute("|B|", grid=grid)
mu = Energy_SI/(Mass*data["|B|"]) - (vpar_i**2)/(2*data["|B|"])

# Initial Parameters
ini_param = [float(mu), Mass_Charge_Ratio]

intermediate_time = timet()
print(f"Time from beginning until here: {intermediate_time - initial_time}s")

# Objective Function
objective = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="optimization", tolerance=1.4e-8)
objective.build()

# Compute optimization
solution = objective.compute(*objective.xs(eq))

intermediate_time_2 = timet()
print(f"Time to build and compute solution: {intermediate_time_2 - intermediate_time}s")

# Objective Object
ObjFunction = ObjectiveFunction([objective])
ObjFunction.build()

intermediate_time_3 = timet()
print(f"Time to build and compile ObjFunction: {intermediate_time_3 - intermediate_time_2}s")

# Optimization
R_modes = np.array([[0, 0, 0]])
constraints = (ForceBalance(eq), FixBoundaryR(eq, modes=R_modes), FixBoundaryZ(eq, modes=False), FixPressure(eq), FixIota(eq), FixPsi(eq))
eq.optimize(objective=ObjFunction, optimizer = "fmin-auglag-bfgs", constraints=constraints, verbose=3, maxiter=100) # Mudar o número de iterações para 3, 10, 100
eq.save(opt_file)

intermediate_time_4 = timet()
print(f"Time to optimize: {intermediate_time_4 - intermediate_time_3}s")

print("Optimization Completed")
print(f"Optimized Filename: {opt_file}")
optimization_final_time = timet()
print(f"Total time: {optimization_final_time - initial_time}s")
print("*********************** OPTIMIZATION END ***********************")

print("*************** TRACING ***************")

tracing_original = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="tracer", tolerance=1.4e-8)

# Compute tracing original equilibrium

intermediate_time_5 = timet()
objective.build()
tracer_solution_original = objective.compute(*objective.xs(eq))
intermediate_time_6 = time()
print(f"Time to build and trace (original): {intermediate_time_6 - intermediate_time_5}s")

output_to_file(solution=tracer_solution_original, name="tracing_original")

# Compute tracing optimized equilibrium
opt_eq = desc.io.load(opt_file)[-1]
opt_eq._iota = opt_eq.get_profile("iota").to_powerseries(order=opt_eq.L, sym=True)
opt_eq._current = None

tracing_optimized = ParticleTracer(eq=opt_eq, output_time=time, initial_conditions=ini_cond, initial_parameters=ini_param, compute_option="tracer", tolerance=1.4e-8)

intermediate_time_7 = timet()
objective.build()
tracer_solution_optimized = objective.compute(*objective.xs(opt_eq))
intermediate_time_8 = timet()
print(f"Time to build and trace (optimized): {intermediate_time_8 - intermediate_time_7}s")

output_to_file(solution=tracer_solution_optimized, name="tracing_optimized")

# Comparison
diffence = tracer_solution_original - tracer_solution_optimized

output_to_file(solution=diffence, name="tracing_difference")

print("*********************** TRACING END ***********************")

print("*************** PLOTTING ***************")

print("Original Equilibrium")
Trajectory_Plot(solution=tracer_solution_original, save_name="Trajectory_Plot_original.png")
Quantity_Plot(solution=tracer_solution_original, save_name="Quantity_Plot_original.png")
Energy_Plot(solution=tracer_solution_original, save_name="Energy_Plot_original.png")

print("Optimized Equilibrium")
Trajectory_Plot(solution=tracer_solution_optimized, save_name="Trajectory_Plot_optimized.png")
Quantity_Plot(solution=tracer_solution_optimized, save_name="Quantity_Plot_optimized.png")
Energy_Plot(solution=tracer_solution_optimized, save_name="Energy_Plot_optimized.png")

print("*************** PLOTTING END ***************")

final_time = timet()
print(f"Total time: {final_time - initial_time}s")
print("*************** END ***************")