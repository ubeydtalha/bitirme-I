

# class araba():

#     def __init__(self,teker,koltuk,motor):
#         self.teker_sayısı = teker
#         self.kolt = koltuk
#         self.motor_hacmi = motor

#     def calistır(self):
#         print("Araba Çalıştırıldı")
    
#     def dur(self):
#         print("Araba Durduruldu")

# opel = araba(3,5,1.6)
# opel1 = araba(4,3,1.3)

# opel.calistır()

# print()



# def a(b):
#     print("a",b)

# a(10)
# a(11)

class hayvan():
    def __init__(self,bacak_sayisi):
        self.bacak_sayisi = bacak_sayisi
        self.göz_sayisi = 2
        self.__dna = "ATC"
    def konus(self):
        print("Ben bir hayvanım")


class kanguru(hayvan):
    def __init__(self,kol,bacak_sayisi):
        super().__init__(bacak_sayisi=2)
        self.kol = kol
    def konus(self):
        print("Ben iki bacaklıyım")

    
class dort_bacakli():
    def __init__(self,bacak_sayisi):
        self.bacak_sayisi = bacak_sayisi
    def konus(self):
        print("Ben iki bacaklıyım")

    

# a = dort_bacakli(2)

# b = kanguru(2,4)
# g = kanguru(2,5)
c = hayvan(6)
print(c.__dna)



# print(g.__dict__)
# print(c.__dict__)


# def hesapla(b,a=10):
#     return a+b


# print(hesapla(30,a=20))

# import random

# random.seed(11)
# print(random.random())

# array = [0,1,2]

# array[0] = 10

# sayi = 10
# sayi = float(sayi)
# text = str(sayi)

# print(sayi,text)

