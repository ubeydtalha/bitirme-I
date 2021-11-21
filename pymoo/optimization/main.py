import asyncio
from asyncio.tasks import sleep
import json
import os
from hspicpy import HSpicePy
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.algorithms.moo.moead import MOEAD, ParallelMOEAD
from pymoo.factory import get_problem, get_visualization, get_reference_directions , get_decomposition
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.factory import get_termination
from pymoo.visualization.scatter import Scatter
from pymoo.core.repair import Repair



class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=8,
                         n_obj=2,
                         n_constr=3,
                         xl=np.array([.12e-06,.12e-06,.12e-06,0.16e-06,0.16e-06,0.16e-06,5000,0.4]),
                         xu=np.array([4.0e-06,4.0e-06,4.0e-06,200.0e-06,200.0e-06,200.0e-06,25000,0.8]))
        meAbsPath = os.path.dirname(os.path.realpath(__file__))
        self.h = HSpicePy(file_name="amp",design_file_name="designparam",path=meAbsPath,timeout="")
        self.my_history = {}

    def _evaluate(self, x, out, *args, **kwargs):

        
        self.h.change_parameters_to_cir( 
                                    LM1 = x[0], LM2 = x[1], LM3 = x[2],
                                    WM1 = x[3], WM2 = x[4], WM3 = x[5],
                                    Rb = x[6],Vb = x[7])
        # asyncio.run(self.h.run_async())
        # asyncio.run(asyncio.sleep(.08))
        self.h.runner = True
        self.h.run()
        

        
        
        try : 
            f1_ = self.h.result["bw"]
            f2_ = self.h.result["gain"]

            himg = self.h.result["himg"]
            hreal = self.h.result["hreal"]

            zpower = self.h.mt0_result["zpower"]
            zsarea = self.h.mt0_result["zsarea"]

            f1_ = float(f1_)
            f2_ = float(f2_)
            zpower = float(zpower)
            zsarea = float(zsarea)
            himg = float(himg)
            hreal = float(hreal)
        except Exception as e:

            out["F"] = [99,-9999999]
            out["G"] = [1, 1, 1]
            
        else:
            f1 = float(self.h.result["bw"])
            f2 = float(self.h.result["gain"])
            zpower = float(zpower)
            zsarea = float(zsarea)
            himg = float(himg)
            hreal = float(hreal)

            pm = None
            if himg > 0.0 and hreal > 0.0:
                pm = np.arctan(himg/hreal)*180/np.pi
            elif himg > 0.0 and hreal < 0.0:
                pm = 0.1
            elif himg < 0.0 and hreal < 0.0:
                pm = np.arctan(himg/hreal)*180/np.pi
            else:
                pm = 10    
            
            out["F"] = [f1, -f2]
            G_pm = -pm+45
            G_zpower = zpower-0.5e-3
            G_zsarea = zsarea-1e-3
            out["G"] = [G_pm, G_zpower, G_zsarea]
class Repair(Repair):

    def _do(self, problem : ElementwiseProblem , pop, **kwargs):
        

        values = pop.get("X")
        new_values = []
        i = 0
        for individual in values:
            new_individual = []
            # control = list(map(lambda x: True if isinstance(x,bool) else False,individual))
            if str(individual.dtype) == "bool":
                for lower,upper in zip(problem.xl,problem.xu):
                    new_individual.append(np.random.uniform(lower,upper,1)[0])
            else:
                new_individual = individual
            new_values.append(new_individual)
            i+=0

        pop.set("X", new_values)
        return pop



problem = MyProblem()


ref_dirs =  get_reference_directions(
    "multi-layer",
    get_reference_directions("das-dennis", 2, n_partitions=150),
    get_reference_directions("das-dennis", 2, n_partitions=150, scaling=0.5)
)
# ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=50)
algorithm = MOEAD(
    ref_dirs,
    n_neighbors=7, 
    
    # decomposition=get_decomposition("pbi", theta=5, eps=0.5),
    # prob_neighbor_mating=0.7,
    sampling=get_sampling("real_random"), 
    crossover=get_crossover("bin_ux", prob=0.9),
    mutation=get_mutation("bin_bitflip", prob=0.1), 
    repair = Repair(),
    
)

res = minimize(problem,
               algorithm,
               
               ('n_gen', 10),
               seed=6178,
               verbose=True,
            #    return_least_infeasible=True,
               save_history=True,
               
               )

# np.save("checkpoint", algorithm)

# plot = Scatter()
# plot.add(res.F, color="red")
# # plot.add(res.F, plot_type="line", color="black", linewidth=2)
# plot.show()
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

# print(res.F)
# import numpy as np
# import matplotlib.pyplot as plt

# n_evals = np.array([e.evaluator.n_eval for e in res.history])
# opt = np.array([e.opt[0].F for e in res.history])

# plt.title("Convergence")
# plt.plot(n_evals, opt, "--")
# plt.yscale("log")
# plt.show()

# algorithm = NSGA2(
#     pop_size=100,
#     n_offsprings=100,
#     sampling=get_sampling("real_random"),
#     crossover=get_crossover("real_sbx", prob=0.9, eta=15),
#     mutation=get_mutation("real_pm", eta=20),
#     eliminate_duplicates=True,
#     # repair = Repair(),
# )


# termination = get_termination("n_gen", 10)


# res = minimize(problem,
#                algorithm,
#                termination,
#                seed=1,
#                save_history=True,
#                verbose=True)

X = res.X
F = res.F

plot = Scatter()
plot.add(res.F, color="red")

plot.show()

# print(res.F)


import matplotlib.pyplot as plt
# xl, xu = problem.bounds()
# plt.figure(figsize=(7, 5))
# plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
# plt.xlim(xl[0], xu[0])
# plt.ylim(xl[1], xu[1])
# plt.title("Design Space")
# plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')

plt.title("Objective Space")
plt.show()

history = {}
def changer(x):
    return x if  isinstance(x,(float,np.float64)) else f"{x}" 
i = 0
for gen in res.history:
    
    history[i] = {
        "generetion": i,
        "population":{} 
    }
    j = 0
    for individual in gen.pop.copy():
        history[i]["population"][j] = {
            "LM1" :individual.X[0], 
            "LM2" :individual.X[1], 
            "LM3" :individual.X[2],
            "WM1" :individual.X[3], 
            "WM2" :individual.X[4], 
            "WM3" :individual.X[5],
            "Rb" :individual.X[6],
            "Vb" :individual.X[7],
            "bw": changer(individual.F[0]) ,
            "gain": changer(individual.F[1]),
            "pm": changer(individual.G[0]),
            "zpower": changer(individual.G[1]),
            "zarea": changer(individual.G[2]),
        }
        j += 1
    i+=1



meAbsPath = os.path.dirname(os.path.realpath(__file__))
with open(f"{meAbsPath}\\out\\history.json","w") as outfile:
    json.dump(history, outfile)   

n_evals = []             # corresponding number of function evaluations\
hist_F = []              # the objective space values in each generation
hist_cv = []             # constraint violation in each generation
hist_cv_avg = []         # average constraint violation in the whole population

for algo in res.history:

    # store the number of function evaluations
    n_evals.append(algo.evaluator.n_eval)

    # retrieve the optimum from the algorithm
    opt = algo.opt

    # store the least contraint violation and the average in each population
    hist_cv.append(opt.get("CV").min())
    hist_cv_avg.append(algo.pop.get("CV").mean())

    # filter out only the feasible and append and objective space values
    feas = np.where(opt.get("feasible"))[0]
    hist_F.append(opt.get("F")[feas])


k = np.where(np.array(hist_cv) <= 0.0)[0].min()
print(f"At least one feasible solution in Generation {k} after {n_evals[k]} evaluations.")

# replace this line by `hist_cv` if you like to analyze the least feasible optimal solution and not the population
vals = hist_cv_avg

k = np.where(np.array(vals) <= 0.0)[0].min()
print(f"Whole population feasible in Generation {k} after {n_evals[k]} evaluations.")

plt.figure(figsize=(7, 5))
plt.plot(n_evals, vals,  color='black', lw=0.7, label="Avg. CV of Pop")
plt.scatter(n_evals, vals,  facecolor="none", edgecolor='black', marker="p")
plt.axvline(n_evals[k], color="red", label="All Feasible", linestyle="--")
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Constraint Violation")
plt.legend()
plt.show()

# replace this line by `hist_cv` if you like to analyze the least feasible optimal solution and not the population
vals = hist_cv

k = np.where(np.array(vals) <= 0.0)[0].min()
print(f"Whole population feasible in Generation {k} after {n_evals[k]} evaluations.")

plt.figure(figsize=(7, 5))
plt.plot(n_evals, vals,  color='black', lw=0.7, label="Avg. CV of Pop")
plt.scatter(n_evals, vals,  facecolor="none", edgecolor='black', marker="p")
plt.axvline(n_evals[k], color="red", label="All Feasible", linestyle="--")
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Constraint Violation")
plt.legend()
plt.show()

approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)

from pymoo.indicators.hv import Hypervolume

metric = Hypervolume(ref_point= np.array([1.1, 1.1]),
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=approx_ideal,
                     nadir=approx_nadir)

hv = [metric.do(_F) for _F in hist_F]

plt.figure(figsize=(7, 5))
plt.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
plt.scatter(n_evals, hv,  facecolor="none", edgecolor='black', marker="p")
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.show()