# coding: utf-8
import random
import math
import numpy as np
import sys
from numba import autojit, prange, jit
import cython
import humanfriendly
import time
from line_profiler import LineProfiler
import profile


def Load_Match_inputs(quantidade=None):

    if quantidade is None:
        quantidade = 50

    with open('hero_winrate_python_obj.pkl', 'rb') as fp:
        hero_winrate = np.load(fp)

    with open('hero_syn_vntg_python_obj.pkl', 'rb') as fp:
        hero_syn_vntg = np.load(fp)

    with open('partidas_python_obj.pkl', 'rb') as fp:
        partidas = np.load(fp)

    x = np.zeros((quantidade, 250))
    y = np.zeros((quantidade, 1))

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
            x[i][count_index_x: count_index_x+16] = hero_winrate[hero]
            count_index_x += 16
            for hero_amigo in partidas[i][5:10]:
                if hero != hero_amigo:
                    x[i][count_index_x] = hero_syn_vntg[hero][hero_amigo][0]
                    count_index_x += 1
            for hero_inimigo in partidas[i][0:5]:
                x[i][count_index_x] = hero_syn_vntg[hero][hero_inimigo][1]
                count_index_x += 1
    return (x,y)


@autojit(nogil=True, nopython=True, cache=True)
def sigmoide(layer_output):
    for i in range(len(layer_output)):
        layer_output[i] = 1 / (1 + math.exp(-1*layer_output[i]))
    return layer_output


# @jit(nopython=True, parallel=True)
@autojit(nogil=True, nopython=True, cache=True)
def atualiza_pesos_camada_inicial_pro(back_prp_erro, x, ni):   
    return (x * back_prp_erro * ni)


#@autojit(nogil=True, cache=True)
def MLP_single_pass(pesos, x, y, ni, qtd_nr_layer, qtd_inputs, qtd_de_camadas):

    output_layer = []

    x = np.concatenate( ( [1], x ) )                   
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
    
    
    diff_pesos = atualiza_pesos_camada_inicial_pro( back_prp_erro[1:,],   x.reshape((-1, 1)),  ni )
    pesos[0] = pesos[0] + diff_pesos
       

    erro = abs( y - output_layer[1] )

    return (erro, pesos, output_layer[1])


# @autojit()
def main():

    qtd_treinos = 100
    if len(sys.argv) == 2:
        qtd_treinos = int(sys.argv[1])

    learning_rate_begin = 0.05
    learning_rate_final = 0.005
    learning_rate = 0.05

    x,y = Load_Match_inputs(5000)
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
    min_error = 9999999
    min_error_t = 0
    max_acertos = 0
    max_acertos_t = 0

    for i in range (qtd_treinos):
        sum_erro = 0
        num_acertos = 0

        learning_rate = learning_rate_final + (1 - i/qtd_treinos) * (learning_rate_begin - learning_rate_final)
        t1 = time.time()        
        for j in range (samplesize):
            # lp = LineProfiler()            
            # lp_wrapper = lp(MLP_single_pass)
            # (erro, pesos, output) = lp_wrapper(pesos, x[j,:], y[j,:], learning_rate, qtd_nr_layer, qtd_inputs, qtd_de_camadas)
            # lp.print_stats()     
            # time.sleep(999)       
            (erro, pesos, output) = MLP_single_pass(pesos, x[j,:], y[j,:], learning_rate, qtd_nr_layer, qtd_inputs, qtd_de_camadas)        
            


            sum_erro += sum(erro)
            
            if y[j][0] == round(output[0]) :
                num_acertos = num_acertos + 1
        t2 = time.time()
        if sum_erro < min_error:
            min_error = sum_erro
            min_error_t = (i+1)
        if num_acertos > max_acertos:
            max_acertos = num_acertos
            max_acertos_t = (i+1)

        print ('Época %d/%d\n - AvgErro: %.4f - Acertos: %.4f LR: %.4f Tempo gasto: %s\n' % (i+1, qtd_treinos, sum_erro/samplesize, num_acertos/samplesize, learning_rate, humanfriendly.format_timespan(t2 - t1, detailed=True, max_units=5) ) )
    print ("Menor soma de erros registrada: %.4f no treino número: %d \nMaior qtd de acertos registrada: %d/%d no treino número: %d" % (min_error,min_error_t,max_acertos,samplesize,max_acertos_t) )
    lp = LineProfiler()            
    lp_wrapper = lp(MLP_single_pass)
    (erro, pesos, output) = lp_wrapper(pesos, x[j,:], y[j,:], learning_rate, qtd_nr_layer, qtd_inputs, qtd_de_camadas)
    lp.print_stats()         

if __name__ == "__main__":
    main()
# profile.run('main()')
