---
layout: post
title:  "Algoritmos Genéticos com Python"
subtitle: "Exemplo de implementação do problema da mochila com biblioteca DEAP"
date:   2019-06-04
background: '/img/posts/01/cover.jpg'
---

Olá!

Hoje vou compartilhar o código do exemplo da apresentação sobre Algoritmos Genéticos com Python que eu fiz recentemente. 

Quando eu postar aqui no site, eu vou colocar um link para a apresentação [aqui]


```python
import random
import numpy
from deap import creator, base, tools, algorithms
import matplotlib.pyplot as plt
```

    D:\Programas\Anaconda3\lib\site-packages\deap\tools\_hypervolume\pyhv.py:33: ImportWarning: Falling back to the python version of hypervolume module. Expect this to be very slow.
      "module. Expect this to be very slow.", ImportWarning)
    

# Definindo a classe Produto


```python
class Produto():
    def __init__(self, nome, espaco, valor):
        self.nome = nome
        self.espaco = espaco
        self.valor = valor
```

# Criando lista de produtos


```python
lista_produtos = []
lista_produtos.append(Produto("Geladeira Dako", 0.751, 999.90))
lista_produtos.append(Produto("Iphone 6", 0.0000899, 2911.12))
lista_produtos.append(Produto("TV 55' ", 0.400, 4346.99))
lista_produtos.append(Produto("TV 50' ", 0.290, 3999.90))
lista_produtos.append(Produto("TV 42' ", 0.200, 2999.00))
lista_produtos.append(Produto("Notebook Dell", 0.00350, 2499.90))
lista_produtos.append(Produto("Ventilador Panasonic", 0.496, 199.90))
lista_produtos.append(Produto("Microondas Electrolux", 0.0424, 308.66))
lista_produtos.append(Produto("Microondas LG", 0.0544, 429.90))
lista_produtos.append(Produto("Microondas Panasonic", 0.0319, 299.29))
lista_produtos.append(Produto("Geladeira Brastemp", 0.635, 849.00))
lista_produtos.append(Produto("Geladeira Consul", 0.870, 1199.89))
lista_produtos.append(Produto("Notebook Lenovo", 0.498, 1999.90))
lista_produtos.append(Produto("Notebook Asus", 0.527, 3999.00))

espacos = []
valores = []
nomes = []
for produto in lista_produtos:
    espacos.append(produto.espaco)
    valores.append(produto.valor)
    nomes.append(produto.nome)
    
limite = 3
```

# Definindo a função objetivo

Se couber no caminhão, o valor é a soma dos valores dos itens selecionados
Se não couber, o valor é zero


```python
def avaliacao(individual):
    nota = 0
    soma_espacos = 0
    for i in range(len(individual)):
       if individual[i] == 1:
           nota += valores[i]
           soma_espacos += espacos[i]
    if soma_espacos > limite:
        nota = 0
    return nota,
```

# Definindo os parâmetros do algoritmo


```python
toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, n=len(espacos))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", avaliacao)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb = 0.01)
toolbox.register("select", tools.selRoulette)
```

# Definição dos hiper-parâmetros


```python
populacao = toolbox.population(n = 500)
probabilidade_crossover = 0.95
probabilidade_mutacao = 0.15
numero_geracoes = 100
```

# Definição das estatísticas a serem retornadas pelo algoritmo


```python
estatisticas = tools.Statistics(key=lambda individuo: individuo.fitness.values)
estatisticas.register("max", numpy.max)
estatisticas.register("min", numpy.min)
estatisticas.register("med", numpy.mean)
estatisticas.register("std", numpy.std)
```


```python
populacao, info = algorithms.eaSimple(populacao, toolbox,
									  probabilidade_crossover,
									  probabilidade_mutacao,
									  numero_geracoes, estatisticas)
```

    gen	nevals	max    	min	med    	std    
    0  	500   	22385.7	0  	9043.55	6388.65
    1  	493   	24684.9	0  	11979.9	5493.94
    2  	481   	24385.6	0  	13080.7	5476.4 
    3  	477   	24993.6	0  	13768.5	5441.56
    4  	490   	24993.6	0  	14725.3	5338.08
    5  	479   	24993.6	0  	15375.6	5476.3 
    6  	463   	24363.7	0  	15960.2	5195.54
    7  	482   	24494.3	0  	16866.8	5061.43
    8  	481   	24694.3	0  	17556.2	5187.06
    9  	483   	24993.6	0  	18107.2	4806.62
    10 	476   	24993.6	0  	18661.6	4644.9 
    11 	470   	24993.6	0  	19532.1	3940.61
    12 	482   	24993.6	0  	19965.9	3797.84
    13 	477   	24993.6	0  	20112.7	3902.99
    14 	489   	24993.6	0  	20253.8	4372.03
    15 	480   	24993.6	0  	20992.9	3383.36
    16 	487   	24793.6	0  	21066.5	3463.21
    17 	476   	24993.6	0  	21003.4	4048.29
    18 	484   	24993.6	0  	21546.5	3343.34
    19 	473   	24993.6	0  	21866.1	2671.21
    20 	485   	24993.6	0  	21273.1	4477.49
    21 	469   	24993.6	0  	21910.5	3238.49
    22 	482   	24993.6	0  	22167.3	2553.28
    23 	483   	24993.6	0  	22104.6	3310.65
    24 	479   	24993.6	0  	22159  	3571.21
    25 	473   	24993.6	0  	22423.8	3093.63
    26 	483   	24993.6	0  	21957.4	4307.6 
    27 	481   	24993.6	0  	22483.1	3099.9 
    28 	485   	24993.6	0  	22600.2	3026.55
    29 	486   	24993.6	0  	22709.3	2221.23
    30 	490   	24993.6	0  	22407.5	3343.49
    31 	471   	24993.6	0  	22698.6	2637.46
    32 	490   	24993.6	0  	22699.8	2622.51
    33 	481   	24993.6	0  	22748.5	2404.95
    34 	474   	24993.6	0  	22903.1	1827.46
    35 	479   	24993.6	0  	22816  	2541.41
    36 	488   	24993.6	0  	22976.8	1512.44
    37 	469   	24993.6	0  	22864.9	1866.13
    38 	484   	24993.6	0  	22916.4	1580.95
    39 	478   	24993.6	0  	22935.3	1887.5 
    40 	486   	24993.6	0  	23026  	1566.69
    41 	477   	24993.6	0  	23044.9	2050.75
    42 	476   	24993.6	0  	23126.7	1462.74
    43 	485   	24993.6	0  	23010.2	2076.16
    44 	477   	24993.6	16583.3	23146  	1061.79
    45 	486   	24993.6	0      	23107.5	1819.65
    46 	482   	24993.6	0      	23066.2	2108.02
    47 	482   	24993.6	0      	23151.5	1832.53
    48 	482   	24993.6	0      	23193.3	1838.77
    49 	478   	24993.6	0      	23074.4	2349.45
    50 	475   	24993.6	0      	23121.6	2334.37
    51 	481   	24993.6	0      	23154.4	2114.46
    52 	475   	24993.6	0      	22981.1	2951.05
    53 	488   	24993.6	0      	23145.3	2098.95
    54 	478   	24993.6	17732.5	23402.5	940.326
    55 	475   	24993.6	0      	23151.2	2545.25
    56 	476   	24993.6	0      	23291.1	2045.04
    57 	471   	24993.6	0      	23312.7	1797.99
    58 	484   	24993.6	0      	23279.2	2321.66
    59 	465   	24993.6	0      	23312.2	2123.67
    60 	475   	24993.6	0      	23421.1	1841.1 
    61 	474   	24993.6	0      	23473.5	1545.78
    62 	478   	24993.6	0      	23334.4	2383.97
    63 	472   	24993.6	0      	23338.1	2625.69
    64 	477   	24993.6	0      	23482.2	2373.06
    65 	470   	24993.6	0      	23461.4	2769.33
    66 	473   	24993.6	0      	23604  	2064.1 
    67 	475   	24993.6	0      	23519.6	2525.37
    68 	486   	24993.6	0      	23593  	2067.94
    69 	479   	24993.6	0      	23422.9	3126.51
    70 	480   	24993.6	0      	23606.6	2533.55
    71 	478   	24993.6	0      	23512.1	2766.75
    72 	465   	24993.6	0      	23464.5	2970.43
    73 	474   	24993.6	0      	23679.9	1877.24
    74 	482   	24993.6	18142.9	23747.8	1101.87
    75 	478   	24993.6	0      	23416.6	2993.23
    76 	476   	24993.6	0      	23707.7	1845.07
    77 	480   	24993.6	0      	23625.9	1645.29
    78 	487   	24993.6	0      	23371.2	3053.04
    79 	487   	24993.6	0      	23580.2	1995.23
    80 	485   	24993.6	0      	23298.1	3384.35
    81 	485   	24993.6	0      	23388.5	3081.09
    82 	484   	24993.6	0      	23530.3	2045.56
    83 	476   	24993.6	0      	23415.7	2698.44
    84 	478   	24993.6	0      	23435.9	2510.74
    85 	477   	24993.6	0      	23090.3	3729.28
    86 	477   	24993.6	0      	23387.3	2500.44
    87 	483   	24993.6	0      	23376.6	2902.26
    88 	485   	24993.6	0      	23395.9	3093.58
    89 	482   	24993.6	0      	23370.4	3236.2 
    90 	485   	24993.6	0      	23623  	2181.1 
    91 	483   	24993.6	0      	23515.9	2638.5 
    92 	467   	24993.6	0      	23617.1	2422.3 
    93 	478   	24993.6	0      	23634.5	2191.44
    94 	485   	24993.6	0      	23590.8	2158.54
    95 	483   	24993.6	17794.8	23755.7	1167.82
    96 	485   	24993.6	0      	23732.3	1787.9 
    97 	470   	24993.6	0      	23755.3	1486.48
    98 	467   	24993.6	0      	23618.9	2402.14
    99 	479   	24993.6	0      	23688.3	2159.56
    100	479   	24993.6	0      	23395.2	3359.47
    


```python
melhores = tools.selBest(populacao, 1)
```


```python
for individuo in melhores:
	print(individuo)
	print(individuo.fitness)
	soma = 0
	for i in range(len(lista_produtos)):
		if individuo[i] == 1:
			soma += valores[i]
			print("Nome: %s R$ %s " % (lista_produtos[i].nome,
									   lista_produtos[i].valor))
	print("Melhor solução: %s" % soma)
```

    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    (24993.550000000003,)
    Nome: Iphone 6 R$ 2911.12 
    Nome: TV 55'  R$ 4346.99 
    Nome: TV 50'  R$ 3999.9 
    Nome: TV 42'  R$ 2999.0 
    Nome: Notebook Dell R$ 2499.9 
    Nome: Microondas Electrolux R$ 308.66 
    Nome: Microondas LG R$ 429.9 
    Nome: Microondas Panasonic R$ 299.29 
    Nome: Geladeira Consul R$ 1199.89 
    Nome: Notebook Lenovo R$ 1999.9 
    Nome: Notebook Asus R$ 3999.0 
    Melhor solução: 24993.550000000003
    


```python
valores_grafico = info.select("max")
plt.plot(valores_grafico)
plt.title("Acompanhamento dos valores")
plt.show()
```


![png](/img/posts/01/output.png)

