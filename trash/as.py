import asyncio,random



class aile:

    def __init__(self, anne , baba):
        self.anne = anne
        self.baba = baba
        self.fitness = 0
    
    

class birey:

    def __init__(self,kromozom):
        self.kromozom = kromozom
        self.fitness = 0


class population:

    def __init__(self,target,nesil_sayisi):
        self.target = target
        self.nesil_sayisi =nesil_sayisi
        self.current = []
        self.next = []
        self.found = False
        self.max = 100
        self.min = 0
        self.max_pop = 100

    def create_population(self):
        self.current = [birey((bin(random.randint(self.max)),bin(random.randint(self.max)))) for _ in range(self.max_pop)]

    def puanla(self,function,hata):

        for birey in range(self.current):

            result = function(int(birey.kromozom[0],2),int(birey.kromozom[1],2))
