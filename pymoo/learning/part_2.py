import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation,get_termination
from pymoo.optimize import minimize


#%% Problem 
class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(
            n_var=2,
            n_constr=2, 
            n_obj=2, 
            xl=np.array([-2,-2]),
            xu=np.array([2,2])
        )

    
    def _evaluate(self, x, out, *args, **kwargs):
        """_________\n 
        X
        -
        Değerlendirilecek tek bir çözümü temsil eden, uzunluk n_var olan tek boyutlu bir NumPy dizisidir.
        \n_________

        out
        -
        Objektif değerler, n_obj uzunluğundaki NumPy dizisinin bir listesi olarak 
        out["F"]'ye ve out["G"] için kısıtlamalar n_constr uzunluğunda 
        (eğer problemin karşılanacak kısıtlamaları varsa) yazılmalıdır.

        """
        
        f1 = 100 *(x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2

        g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        g2 = - 20 * (x[0]-0.4) * (x[0]-0.6)/ 4.8
        # g fonksiyonları minimize etmek için kullanılır
        # pymoo için her fonksiyon minimize etmeye dayalıdır.
        # maksimize  edilecek problem -1 ile çarpılarak minimize edilmelidir.

        out["F"] = [f1,f2]
        out["G"] = [g1,g2]

problem = MyProblem()
        
#%% algorithm
algorithm = NSGA2(
    pop_size=40, 
    n_offsprings=10, 
    sampling= get_sampling("real_random"),
    crossover= get_crossover("real_sbx",prob=0.9,eta=15),
    mutation= get_mutation("real_pm",eta=20),
    eliminate_duplicates=True
)

#%% Termination Criteria
termination = get_termination("n_gen",40)

#%% Run Optimization - Result 
res = minimize(
    problem,
    algorithm,
    seed=1,
    save_history=True, 
    verbose=True
)

X = res.X
F = res.F


"""

Kısıtlı problemler için sonraki iki sütun,
 mevcut popülasyondaki minimum kısıtlama ihlalini (cv (min)) 
 ve ortalama kısıtlama ihlalini (cv (ort)) gösterir. 
 Bunu, baskın olmayan çözümlerin sayısı (n_nds) ve 
 nesnel uzaydaki hareketi temsil eden iki metrik daha takip eder.
"""

#%% Visualization

import matplotlib.pyplot as plt

xl, xu = problem.bounds()
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
plt.xlim(xl[0], xu[0])
plt.ylim(xl[1], xu[1])
plt.title("Design Space")
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
plt.show()

#%% Multi-Criteria Decision Making 
"""

Artık bir dizi baskın olmayan çözüm elde ettikten sonra, 
bir karar vericinin seti nasıl sadece birkaç hatta 
tek bir çözüme indirebileceği sorulabilir. 
Çok amaçlı problemler için bu karar verme süreci,
Çok Kriterli Karar Verme (ÇKKV) olarak da bilinir.
Multi-Criteria Decision Making (MCDM)



Herhangi bir tekniği kullanmaya başlamadan önce, 
amaçların farklı bir ölçeği olduğunu belirtmeliyiz.

"""

fl = F.min(axis=0)
fu = F.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")

#f1 ve f2 ye ait upper ve lower değerler birbirinden çok farklıdır 
# bu yüzden normalizasyon yapılmalıdır.

approx_ideal = F.min(axis=0) # en mükemmel nokta 
approx_nadir = F.max(axis=0) # en düşük nokta 

"""
Minimize ettiğimiz için 0 a en yakın nokta ideal noktadır.
en uzak nokta ise en düşük noktadır.

"""
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(approx_ideal[0], approx_ideal[1], facecolors='none', edgecolors='red', marker="*", s=100, label="Ideal Point (Approx)")
plt.scatter(approx_nadir[0], approx_nadir[1], facecolors='none', edgecolors='black', marker="p", s=100, label="Nadir Point (Approx)")
plt.title("Objective Space")
plt.legend()
plt.show()


# Sınır noktalarıyla ilgili olarak elde edilen nesnel değerlerin normalleştirilmesi şu şekilde

nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

fl = nF.min(axis=0)
fu = nF.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")

plt.figure(figsize=(7, 5))
plt.scatter(nF[:, 0], nF[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
plt.show()

#%% Compromise Programming

"""
Ayrıştırma işlemi için amaçlara ağırlık verme
"""
weights = np.array([0.2, 0.8]) # Sırasıyla fonksiyonlara ağırlık veriliyor



from pymoo.decomposition.asf import ASF
# Decomposition method : Augmented Scalarization Function (ASF)
decomp = ASF()

i = decomp.do(nF, 1/weights).argmin()

print("Best regarding ASF: Point \ni = %s\nF = %s" % (i, F[i]))

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
plt.title("Objective Space")
plt.show()

#%% Pseudo-Weights

"""

Bu denklem, her bir hedefle ilgili olarak en kötü çözüme olan normalleştirilmiş i. mesafeyi hesaplar.

Lütfen dışbükey olmayan Pareto cepheleri için sözde ağırlığın, 
ağırlıklı toplamı kullanan bir optimizasyonun sonucuna karşılık gelmediğini unutmayın. 
Bununla birlikte, dışbükey Pareto cepheleri için, sözde ağırlıklar, amaç uzayındaki konumu gösterir.
"""
from pymoo.mcdm.pseudo_weights import PseudoWeights

i = PseudoWeights(weights).do(nF)

print("Best regarding Pseudo Weights: Point \ni = %s\nF = %s" % (i, F[i]))

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
plt.title("Objective Space")
plt.show()

#%% Analysis of Convergence - Result

from pymoo.util.misc import stack

class MyTestProblem(MyProblem):

    def _calc_pareto_front(self,flatten=True, *args, **kwargs):

        f2 = lambda f1:((f1/100)** 0.5 - 1 )**2

        F1_a , F1_b = np.linspace(1,16,300) , np.linspace(36,81,300)
        F2_a , F2_b = f2(F1_a), f2(F1_b)

        pf_a = np.column_stack([F1_a,F2_a])
        pf_b = np.column_stack([F1_b,F2_b])

        return stack(pf_a, pf_b, flatten=flatten)

    def _calc_pareto_set(self, flatten=True, *args, **kwargs):

        x1_a = np.linspace(0.1,0.4,50)
        x1_b = np.linspace(0.6,0.9,50)
        x2 = np.zeros(50)

        a,b = np.column_stack([x1_a,x2]), np.column_stack([x1_b,x2])
        return stack(a,b,flatten=flatten)

problem = MyTestProblem()
pf_a, pf_b = problem.pareto_front(use_cache=False, flatten=False)
pf = problem.pareto_front(use_cache=False, flatten=True)
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='b', label="Solutions")
plt.plot(pf_a[:, 0], pf_a[:, 1], alpha=0.5, linewidth=2.0, color="red", label="Pareto-front")
plt.plot(pf_b[:, 0], pf_b[:, 1], alpha=0.5, linewidth=2.0, color="red")
plt.title("Objective Space")
plt.legend()
plt.show()

"""
for analyzing the convergence, historical data need to be stored. 
One way of accomplishing that is enabling the save_history flag, 
which will store a deep copy of the algorithm object in each 
iteration and save it in the Result object. 
"""
"""
strongly recommend not only analyzing the final result but also the algorithm’s behavior. This gives more insights into the convergence of the algorithm.
"""

hist = res.history
print(len(hist))

n_evals = []             # corresponding number of function evaluations\
hist_F = []              # the objective space values in each generation
hist_cv = []             # constraint violation in each generation
hist_cv_avg = []         # average constraint violation in the whole population

for algo in hist:

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
