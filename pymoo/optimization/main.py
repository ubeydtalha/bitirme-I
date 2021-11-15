import asyncio
import os
from hspicpy import HSpicePy
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.algorithms.moo.moead import MOEAD, ParallelMOEAD
from pymoo.factory import get_problem, get_visualization, get_reference_directions , get_decomposition
from pymoo.optimize import minimize
from pymoo.optimize import minimize
from pymoo.factory import get_termination
from pymoo.visualization.scatter import Scatter
class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=8,
                         n_obj=2,
                         n_constr=0,
                         xl=np.array([1.0092e-06,1.2842e-06,1.5865e-06,2.4734e-05,1.9268e-05,1.2646e-05,10000,0.1]),
                         xu=np.array([9.0092e-06,9.2842e-06,9.5865e-06,9.4734e-05,9.9268e-05,9.2646e-05,25000,1]))

    def _evaluate(self, x, out, *args, **kwargs):

        meAbsPath = os.path.dirname(os.path.realpath(__file__))
        h = HSpicePy(file_name="amp",design_file_name="designparam",path=meAbsPath,timeout="")
        h.change_parameters_to_cir(LM1 = x[0], LM2 = x[1], LM3 = x[2],
                        WM1 = x[3], WM2 = x[4], WM3 = x[5],
                        Rb = x[6],Vb = x[7])
        asyncio.run(h.run_async())
        # h.run()
        f1_ = h.result["bw"]
        f2_ = h.result["gain"]
        
        if "failed" in [f1_,f2_]:
            return
            out["F"] = [0, 0]
            
        else:
            f1 = float(h.result["bw"])
            f2 = -float(h.result["gain"])

            out["F"] = [f1, f2]
            


problem = MyProblem()


ref_dirs =  get_reference_directions(
    "multi-layer",
    get_reference_directions("das-dennis", 2, n_partitions=100, scaling=1.5),
    get_reference_directions("das-dennis", 2, n_partitions=100, scaling=0.5)
)
algorithm = MOEAD(
    ref_dirs,
    n_neighbors=7, 
    decomposition=get_decomposition("pbi", theta=5, eps=0),
    prob_neighbor_mating=0.7,
    sampling=get_sampling("real_random"), 
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20), 
    
)

res = minimize(problem,
               algorithm,
               ('n_gen', 50),
               seed=3878595,
               verbose=False,
               return_least_infeasible=False,
               save_history=True
               )


plot = Scatter()
plot.add(res.F, color="red")
plot.show()

import numpy as np
import matplotlib.pyplot as plt

n_evals = np.array([e.evaluator.n_eval for e in res.history])
opt = np.array([e.opt[0].F for e in res.history])

plt.title("Convergence")
plt.plot(n_evals, opt, "--")
plt.yscale("log")
plt.show()

# algorithm = NSGA2(
#     pop_size=100,
#     n_offsprings=10,
#     sampling=get_sampling("real_random"),
#     crossover=get_crossover("real_sbx", prob=0.9, eta=15),
#     mutation=get_mutation("real_pm", eta=20),
#     eliminate_duplicates=True
# )


# termination = get_termination("n_gen", 100)


# res = minimize(problem,
#                algorithm,
#                termination,
#                seed=1,
#                save_history=True,
#                verbose=True)

# X = res.X
# F = res.F

# import matplotlib.pyplot as plt
# xl, xu = problem.bounds()
# plt.figure(figsize=(7, 5))
# plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
# plt.xlim(xl[0], xu[0])
# plt.ylim(xl[1], xu[1])
# plt.title("Design Space")
# plt.show()

# plt.figure(figsize=(7, 5))
# plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
# plt.title("Objective Space")
# plt.show()