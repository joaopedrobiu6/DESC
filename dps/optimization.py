from desc.objectives import ParticleTracer, ObjectiveFunction
from desc.grid import Grid, LinearGrid
import desc.io
from desc.backend import jnp
import matplotlib.pyplot as plt
import numpy as np
import jax.random

eq = desc.io.load("test_run.h5")
eq._iota = eq.get_profile("iota").to_powerseries(order=eq.L, sym=True)
eq._current = None
# eq.solve()


def output_to_file(solution, filename):
    list1 = solution[:, 0]
    list2 = solution[:, 1]
    list3 = solution[:, 2]
    list4 = solution[:, 3]

    combined_lists = zip(list1, list2, list3, list4)

    file_name = filename

    with open(file_name, 'w') as file:
        for row in combined_lists:
            row_str = '\t'.join(map(str, row))
            file.write(row_str + '\n')


mass = 1.673e-27
Energy = 3.52e6*1.6e-19
key = jax.random.PRNGKey(int(4120))

# v_init = jax.random.maxwell(key, (size,))*jnp.sqrt(2*Energy/mass)

psi_init = jax.random.uniform(key, (1,), minval=1e-4, maxval=1-1e-4)
zeta_init = 0.2
theta_init = 0.2
v_init = (0.3 + jax.random.uniform(key, (1,), minval=-
          1e-2, maxval=1e-2)) * jnp.sqrt(2*Energy/mass)

ini_cond = [[float(psi_init[i]), theta_init, zeta_init, float(v_init[i])]
                   for i in range(0, 1)]


tmin = 0
tmax = 0.0007
nt = 500
time = jnp.linspace(tmin, tmax, nt)


mass = 1.673e-27
Energy = 3.52e6*1.6e-19
initial_conditions = ini_cond

mass_charge = mass/1.6e-19

grid = Grid(jnp.array([jnp.sqrt(ini_cond[0]), ini_cond[1],
            ini_cond[2]]).T, jitable=True, sort=False)
data = eq.compute("|B|", grid=grid)

mu = Energy/(mass*data["|B|"]) - (ini_cond[3]**2)/(2*data["|B|"])

ini_param = [float(mu), mass_charge]

objective = ParticleTracer(eq=eq, output_time=time, initial_conditions=ini_cond,
                           initial_parameters=ini_param, compute_option="optimization")

objective.build()
solution = objective.compute(*objective.xs(eq))

print("*************** SOLUTION .compute() ***************")
print(solution)
print("***************************************************")

ObjFunction = ObjectiveFunction([objective])
ObjFunction.build()

print("*************** ObjFunction.compile() ***************")
ObjFunction.compile(mode="bfgs")
print("*****************************************************")

gradient = ObjFunction.grad(ObjFunction.x(eq))
print("*************** GRADIENT ***************")
print(gradient)
print("****************************************")
output_to_file(gradient, "gradient.txt")

# print(ObjFunction.x(eq))
# xs = objective.xs(eq)
# print("*************** xs **************")
# print(xs)
# print("*********************************")