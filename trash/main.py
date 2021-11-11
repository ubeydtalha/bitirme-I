import time
import random


class Genetic:

    def __init__(self,target,population_number):
        self.gene_pool =  '''abcçdefgğhıijklmnoöpqrsştuüvwxyzABCÇDEFGĞHİIJKLMNOÖPQRŞSTUÜVWXYZ 1234567890,.-;:_!"#%&/()=?@${[]}'''
        self.target = target
        self.population_number = population_number
        self.target_text_length = len(self.target)
        self.population = [] 
        self.next_generation = []
        self.found = False
        self.generation_timer = 0
        self.found_text = ""

    class Member:

        def __init__(self,chromosome):
            self.chromosome = chromosome
            self.fitness = 0

    
    def random_gene(self):
        gene = random.choice(self.gene_pool)
        return gene

    def create_chromosome(self):
        
        chromosome = [self.random_gene() for _ in range(self.target_text_length)]
        return chromosome

    def calculate_fitness(self):
        for Member in self.population:
            Member.fitness = 0

            for i in range(self.target_text_length):
                if Member.chromosome[i] == self.target[i]:
                    Member.fitness +=1
            
            if Member.fitness == self.target_text_length:
                self.found_text = Member.chromosome
                self.found = True


    def crossover(self):
        last_best = int((90 * self.population_number) / 100)

        self.next_generation = []

        self.next_generation.extend((self.population[last_best:]))

        while 1:

            if len(self.next_generation) < self.population_number:
                member_1 = random.choice(self.population[last_best:]).chromosome
                member_2 = random.choice(self.population[last_best:]).chromosome

                new_member_chromosome = []

                for gene1,gene2 in zip(member_1,member_2):
                    prob = random.random()

                    if prob < 0.47:
                        new_member_chromosome.append(gene1)
                    elif prob < 0.94:
                        new_member_chromosome.append(gene2)
                    else:
                        new_member_chromosome.append(self.random_gene())
                
                self.next_generation.append(self.Member(new_member_chromosome))

            else:
                break

        self.population = self.next_generation




    def main(self):
        

        for _ in range(self.population_number):
            self.population.append(self.Member(self.create_chromosome()))

        while not self.found:
            self.calculate_fitness()
            self.population = sorted(self.population, key= lambda member:member.fitness)
            self.crossover()
            self.generation_timer +=1
        
        print(f"succeess, I found {self.found_text} , in {self.generation_timer} ")

target = "Mustafa Kemal Atatürk hja"
population_number = 1000

ga = Genetic(target,population_number)

ga.main()
