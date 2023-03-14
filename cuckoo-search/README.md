# Cuckoo Search
Bu klasör, Cuckoo Search algoritması için Python kodunu içerir. Cuckoo Search, metaheuristic bir optimizasyon algoritmasıdır. Asıl amacı, bir işlevin minimum veya maksimum değerini bulmaktır.

# Nasıl Kullanılır?
cuckoo_search() fonksiyonu, Cuckoo Search algoritmasını çalıştırmak için kullanılır. Bu fonksiyon, aşağıdaki parametreleri alır:

fitness_func: En aza indirgenmesi gereken işlev.
dim: Arama uzayının boyutu.
lower_bound: Arama uzayının alt sınırı.
upper_bound: Arama uzayının üst sınırı.
num_nests: Yumurta yuvası sayısı (default: 25).
pa: Yumurta yuvalarının yüzdesi (default: 0.25).
alpha: Yumurtaların yeniden konumlandırılması için kullanılan parametre (default: 1).
beta: Yumurtaların yeniden konumlandırılması için kullanılan parametre (default: 0.5).
max_iter: Maksimum iterasyon sayısı (default: 1000).
Fonksiyon, en iyi çözümü ve en iyi uygunluğu döndürür.

python
Copy code
from cuckoo_search import cuckoo_search

# Define the fitness function to minimize
def fitness(x):
    return x**2

# Set the bounds of the search space
lb = -5
ub = 5

# Set the dimension of the search space
dim = 1

# Run the Cuckoo Search algorithm
best_nest, best_fitness = cuckoo_search(fitness, dim, lb, ub)

# Print the results
print("Best solution found:")
print(best_nest)
print("Best fitness found:")
print(best_fitness)
Referanslar
Xin-She Yang and Suash Deb. Cuckoo search via Lévy flights. In World Congress on Nature & Biologically Inspired Computing (NaBIC 2009), December 2009.
E. Talbi, A. A. Jerraya, J. -P. Merlet, L. Lemaitre and A. Hafid. Metaheuristics: from design to implementation. John Wiley & Sons, 2009.
# Lisans
Bu klasör, MIT lisansı altında lisanslanmıştır. Daha fazla bilgi için LICENSE dosyasını okuyun.
