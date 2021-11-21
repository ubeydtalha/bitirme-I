import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.moead import MOEAD, ParallelMOEAD
from pymoo.factory import get_reference_directions, get_sampling, get_crossover, get_mutation,get_termination, get_visualization
from pymoo.optimize import minimize


#%% Problem 
class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(
            n_var=2,
            
            n_obj=3, 
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
        # f3 = (3*x[1]+(x[0]**3))
        g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        g2 = - 20 * (x[0]-0.4) * (x[0]-0.6)/ 4.8
        # g fonksiyonları minimize etmek için kullanılır
        # pymoo için her fonksiyon minimize etmeye dayalıdır.
        # maksimize  edilecek problem -1 ile çarpılarak minimize edilmelidir.

        out["F"] = [f1,f2]
        # out["G"] = [g1,g2]

problem = MyProblem()
        
#%% algorithm
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=50)

algorithm = MOEAD(
    ref_dirs,
    n_neighbors=15,
    prob_neighbor_mating=0.7,
)
#%% Termination Criteria
termination = get_termination("n_gen",40)

#%% Run Optimization - Result 
res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

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



get_visualization("scatter").add(res.F).show()