

# from pymoo.algorithms.moo.moead import MOEAD, ParallelMOEAD
# from pymoo.factory import get_problem, get_visualization, get_reference_directions
# from pymoo.optimize import minimize

# problem = get_problem("WFG7",n_var=3,n_obj=2)

# ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

# algorithm = MOEAD(
#     ref_dirs,
#     n_neighbors=15,
#     prob_neighbor_mating=0.7,
# )

# res = minimize(problem,
#                algorithm,
#                ('n_gen', 200),
#                seed=1,
#                verbose=True)

# get_visualization("scatter").add(res.F).show()


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

problem = get_problem("OSY")

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()