# coding: utf-8
import random
import math
import numpy as np
import sys
from numba import jit
# from numba import autojit, prange
# from numba import njit, prange
# import numba
# import cython
import humanfriendly
import time
# import multiprocessing
# from multiprocessing import Pool
# from itertools import product
# from line_profiler import LineProfiler


def Load_Match_inputs(quantidade=None):

    if quantidade is None:
        quantidade = 50

    with open('hero_winrate_python_obj.pkl', 'rb') as fp:
        hero_winrate = np.load(fp)

    with open('hero_syn_vntg_python_obj.pkl', 'rb') as fp:
        hero_syn_vntg = np.load(fp)

    with open('partidas_python_obj.pkl', 'rb') as fp:
        partidas = np.load(fp)


    x = np.zeros((quantidade,250))
    y = np.zeros((quantidade,1))


    for i in range(quantidade):
        count_index_x = 0

        y[i] = partidas[i][10]

        for hero in partidas[i][0:5]:
            x[i][count_index_x : count_index_x+16] = hero_winrate[hero]
            count_index_x += 16
            for hero_amigo in partidas[i][0:5]:
                if hero != hero_amigo:
                    x[i][count_index_x] = hero_syn_vntg[hero][hero_amigo][0]
                    count_index_x += 1
            for hero_inimigo in partidas[i][5:10]:
                x[i][count_index_x] = hero_syn_vntg[hero][hero_inimigo][1]
                count_index_x += 1

        for hero in partidas[i][5:10]:
            x[i][count_index_x : count_index_x+16] = hero_winrate[hero]
            count_index_x += 16
            for hero_amigo in partidas[i][5:10]:
                if hero != hero_amigo:
                    x[i][count_index_x] = hero_syn_vntg[hero][hero_amigo][0]
                    count_index_x += 1
            for hero_inimigo in partidas[i][0:5]:
                x[i][count_index_x] = hero_syn_vntg[hero][hero_inimigo][1]
                count_index_x += 1
    return (x,y)


def sigmoide(layer_output):
    for i in range(len(layer_output)):
        layer_output[i] = 1 / (1 + math.exp(-1*layer_output[i]))
    return layer_output

# @numba.jit(nopython=True, parallel=True)
@jit
def atualiza_pesos_camada_inicial(qtd_inputs, qtd_nr_layer, back_prp_erro, ni, x):
    tmp_pesos = np.zeros((251,125))
    for i in range(qtd_inputs+1):                                      # quantidade de entradas na camada inicial. (4 + 1 com bias = 5)
       for j in range(qtd_nr_layer[0]):                                     # quantidade de neuronios na camada inicial:  4
           tmp_pesos[i,j] = ni * back_prp_erro[j+1] * x[i] 
    return tmp_pesos


def MLP_single_pass(pesos, x, y, ni, qtd_nr_layer, qtd_inputs, qtd_de_camadas):


    output_layer = []

    x = np.concatenate( ( [1], x ) )                   # Colocando o bias.
    output_layer.append(x @ pesos[0])                  # Multiplicação de matriz (1x250) por (250x125), resultando numa matriz (1x125)    np.matmul com perf output é mais fast.
    output_layer[0] = sigmoide(output_layer[0])        # Função de ativação sigmoide. 
    

    for num_camada in range(qtd_de_camadas-1):         # calculando a saída de cada camada seguinte.
        output_layer[num_camada] = np.concatenate( ( [1], output_layer[num_camada] ) )   # Colocando o bias.
        output_layer.append(output_layer[num_camada] @ pesos[num_camada+1])              # Multiplicando a saída da camada atual pelos pesos da próxima camada, e colocando o resultado no output da camada [próximaa]
        output_layer[num_camada+1] = sigmoide(output_layer[num_camada+1])


    erro = y - output_layer[qtd_de_camadas-1] 

    

    #%  ===============================================================================================================================================    
    back_prp_erro = np.zeros(qtd_nr_layer[0]+1)              #  Quantidade de neuronios na primeira camada +1. [1 x 126] ////// é o tamanho da ultima camada.(?????)
    

    for i in range (qtd_nr_layer[0]+1):                 # quantidade de pesos na camada anterior a última (125 + bias = 126)
        temp = 0                                        
        for j in range(qtd_nr_layer[1]):                # quantidade de neuronios de saída (3)
            temp += erro[j] * pesos[1][i,j]              # multiplica o erro da saída de cada neuronio, com cada peso que contribuiu para a saída.           
        back_prp_erro[i] = output_layer[0][i] * (1-output_layer[0][i]) * temp

    

    #   Atulizando os pesos da camada final    
    for i in range(qtd_nr_layer[0]+1):                                      # quantidade de entradas na camada intermediária, provenientes da anterior. (4 + 1 com bias = 5)
        for j in range (qtd_nr_layer[1]):                                   # quantidade de neuronios na camada intermediária:  3
            pesos[1][i,j] = pesos[1][i,j] + ni * erro[j] * output_layer[0][i]                       

    #   Atulizando os pesos da primeira camada
    
    diff_pesos = atualiza_pesos_camada_inicial(qtd_inputs, qtd_nr_layer, back_prp_erro, ni, x)
    pesos[0] = pesos[0] + diff_pesos
       

    erro = abs( y - output_layer[1] )

    return (erro, pesos, output_layer[1])



# x = np.array([[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],[4.6,3.1,1.5,0.2],[5.0,3.6,1.4,0.2],[5.4,3.9,1.7,0.4],[4.6,3.4,1.4,0.3],[5.0,3.4,1.5,0.2],[4.4,2.9,1.4,0.2],[4.9,3.1,1.5,0.1],[5.4,3.7,1.5,0.2],[4.8,3.4,1.6,0.2],[4.8,3.0,1.4,0.1],[4.3,3.0,1.1,0.1],[5.8,4.0,1.2,0.2],[5.7,4.4,1.5,0.4],[5.4,3.9,1.3,0.4],[5.1,3.5,1.4,0.3],[5.7,3.8,1.7,0.3],[5.1,3.8,1.5,0.3],[5.4,3.4,1.7,0.2],[5.1,3.7,1.5,0.4],[4.6,3.6,1.0,0.2],[5.1,3.3,1.7,0.5],[4.8,3.4,1.9,0.2],[5.0,3.0,1.6,0.2],[5.0,3.4,1.6,0.4],[5.2,3.5,1.5,0.2],[5.2,3.4,1.4,0.2],[4.7,3.2,1.6,0.2],[4.8,3.1,1.6,0.2],[5.4,3.4,1.5,0.4],[5.2,4.1,1.5,0.1],[5.5,4.2,1.4,0.2],[4.9,3.1,1.5,0.2],[5.0,3.2,1.2,0.2],[5.5,3.5,1.3,0.2],[4.9,3.6,1.4,0.1],[4.4,3.0,1.3,0.2],[5.1,3.4,1.5,0.2],[5.0,3.5,1.3,0.3],[4.5,2.3,1.3,0.3],[4.4,3.2,1.3,0.2],[5.0,3.5,1.6,0.6],[5.1,3.8,1.9,0.4],[4.8,3.0,1.4,0.3],[5.1,3.8,1.6,0.2],[4.6,3.2,1.4,0.2],[5.3,3.7,1.5,0.2],[5.0,3.3,1.4,0.2],[7.0,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.9,3.1,4.9,1.5],[5.5,2.3,4.0,1.3],[6.5,2.8,4.6,1.5],[5.7,2.8,4.5,1.3],[6.3,3.3,4.7,1.6],[4.9,2.4,3.3,1.0],[6.6,2.9,4.6,1.3],[5.2,2.7,3.9,1.4],[5.0,2.0,3.5,1.0],[5.9,3.0,4.2,1.5],[6.0,2.2,4.0,1.0],[6.1,2.9,4.7,1.4],[5.6,2.9,3.6,1.3],[6.7,3.1,4.4,1.4],[5.6,3.0,4.5,1.5],[5.8,2.7,4.1,1.0],[6.2,2.2,4.5,1.5],[5.6,2.5,3.9,1.1],[5.9,3.2,4.8,1.8],[6.1,2.8,4.0,1.3],[6.3,2.5,4.9,1.5],[6.1,2.8,4.7,1.2],[6.4,2.9,4.3,1.3],[6.6,3.0,4.4,1.4],[6.8,2.8,4.8,1.4],[6.7,3.0,5.0,1.7],[6.0,2.9,4.5,1.5],[5.7,2.6,3.5,1.0],[5.5,2.4,3.8,1.1],[5.5,2.4,3.7,1.0],[5.8,2.7,3.9,1.2],[6.0,2.7,5.1,1.6],[5.4,3.0,4.5,1.5],[6.0,3.4,4.5,1.6],[6.7,3.1,4.7,1.5],[6.3,2.3,4.4,1.3],[5.6,3.0,4.1,1.3],[5.5,2.5,4.0,1.3],[5.5,2.6,4.4,1.2],[6.1,3.0,4.6,1.4],[5.8,2.6,4.0,1.2],[5.0,2.3,3.3,1.0],[5.6,2.7,4.2,1.3],[5.7,3.0,4.2,1.2],[5.7,2.9,4.2,1.3],[6.2,2.9,4.3,1.3],[5.1,2.5,3.0,1.1],[5.7,2.8,4.1,1.3],[6.3,3.3,6.0,2.5],[5.8,2.7,5.1,1.9],[7.1,3.0,5.9,2.1],[6.3,2.9,5.6,1.8],[6.5,3.0,5.8,2.2],[7.6,3.0,6.6,2.1],[4.9,2.5,4.5,1.7],[7.3,2.9,6.3,1.8],[6.7,2.5,5.8,1.8],[7.2,3.6,6.1,2.5],[6.5,3.2,5.1,2.0],[6.4,2.7,5.3,1.9],[6.8,3.0,5.5,2.1],[5.7,2.5,5.0,2.0],[5.8,2.8,5.1,2.4],[6.4,3.2,5.3,2.3],[6.5,3.0,5.5,1.8],[7.7,3.8,6.7,2.2],[7.7,2.6,6.9,2.3],[6.0,2.2,5.0,1.5],[6.9,3.2,5.7,2.3],[5.6,2.8,4.9,2.0],[7.7,2.8,6.7,2.0],[6.3,2.7,4.9,1.8],[6.7,3.3,5.7,2.1],[7.2,3.2,6.0,1.8],[6.2,2.8,4.8,1.8],[6.1,3.0,4.9,1.8],[6.4,2.8,5.6,2.1],[7.2,3.0,5.8,1.6],[7.4,2.8,6.1,1.9],[7.9,3.8,6.4,2.0],[6.4,2.8,5.6,2.2],[6.3,2.8,5.1,1.5],[6.1,2.6,5.6,1.4],[7.7,3.0,6.1,2.3],[6.3,3.4,5.6,2.4],[6.4,3.1,5.5,1.8],[6.0,3.0,4.8,1.8],[6.9,3.1,5.4,2.1],[6.7,3.1,5.6,2.4],[6.9,3.1,5.1,2.3],[5.8,2.7,5.1,1.9],[6.8,3.2,5.9,2.3],[6.7,3.3,5.7,2.5],[6.7,3.0,5.2,2.3],[6.3,2.5,5.0,1.9],[6.5,3.0,5.2,2.0],[6.2,3.4,5.4,2.3],[5.9,3.0,5.1,1.8]])
# y = np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])


def main():

    # pool = multiprocessing.Pool(processes=2)
    # pool = Pool()



    qtd_treinos = 100
    if len(sys.argv) == 2:
        qtd_treinos = int(sys.argv[1])

    learning_rate = 0.05

    x,y = Load_Match_inputs(25000)
    samplesize = len(x)
    qtd_inputs = len(x[0])


    qtd_nr_layer = []
    qtd_nr_layer.append(125)
    qtd_nr_layer.append(1)

    qtd_de_camadas = len(qtd_nr_layer)


    pesos = None

    if pesos is None:
        pesos = []
        pesos.append(np.random.rand(qtd_inputs+1,qtd_nr_layer[0]) * 0.4 - 0.2)
        for i in range (1,qtd_de_camadas):
            pesos.append(np.random.rand(qtd_nr_layer[i-1]+1,qtd_nr_layer[i]) * 0.4 - 0.2)


    
    print ("Começando")
    sum_error = np.zeros( qtd_treinos ) 
    for i in range (qtd_treinos):
        sum_erro = 0
        num_acertos = 0
        t1 = time.time()
        for j in range (samplesize):
            # lp = LineProfiler()
            # lp_wrapper = lp(MLP_single_pass)
            # lp_wrapper(pesos, x[j,:], y[j,:], learning_rate, qtd_nr_layer, qtd_inputs, qtd_de_camadas)
            # lp.print_stats()            
            (erro, pesos, output) = MLP_single_pass(pesos, x[j,:], y[j,:], learning_rate, qtd_nr_layer, qtd_inputs, qtd_de_camadas)        
            
            sum_erro += sum(erro)
            
            if y[j][0] == round(output[0]) :
                num_acertos = num_acertos + 1
        t2 = time.time()
        print ('Treino %d - Acertos: %d/%d Sum: %.5f Tempo gasto: %s\n' % (i+1, num_acertos, samplesize, sum_erro, humanfriendly.format_timespan(t2 - t1, detailed=True, max_units=5) ) )
        

# import profile
# profile.run('main()')
if __name__ == "__main__":
    main()