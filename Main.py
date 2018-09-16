import random
import math
import numpy as np
import sys
# import time


def sigmoide(layer_output):
    for i in range(len(layer_output)):
        layer_output[i] = 1 / (1 + math.exp(-1*layer_output[i]))
    return layer_output


 
def MLP_single_pass(pesos_first_layer, pesos_last_layer, x, y, ni, qtd_nr_c0, qtd_nr_c1, qtd_nr_c2, qtd_inputs):

    if pesos_first_layer is None and pesos_last_layer is None:            
        pesos_first_layer  = np.random.rand(qtd_inputs+1,qtd_nr_c0) * 0.4 - 0.2
        # pesos_middle_layer = np.random.rand(qtd_nr_c0+1,qtd_nr_c1) * 0.4 - 0.2
        pesos_last_layer   = np.random.rand(qtd_nr_c0+1,qtd_nr_c1) * 0.4 - 0.2
        # pesos_last_layer   = np.random.rand(qtd_nr_c1+1,qtd_nr_c2) * 0.4 - 0.2

    qtd_nr_layer = [qtd_nr_c0, qtd_nr_c1]

    #%  Primeira camada de neurônios
    x        = np.concatenate( ( [1], x ) )     # Colocando o bias.
    output_0 = x @ pesos_first_layer  # Multiplicação de matriz (1x5) por (5x4), resultando numa matriz (1x4)    np.matmul com perf output é mais fast.
    output_0 = sigmoide(output_0)
    
    #%  Segunda camada de neurônios    
    output_0 = np.concatenate( ( [1], output_0 ) )       
    output_1 = output_0 @ pesos_last_layer   
    output_1 = sigmoide(output_1)

    # output_1 = np.concatenate( ( [1], output_1 ) )       # Colocando o bias.
    # output_2 = output_1 @ pesos_last_layer   # Multiplicação de matriz: (1x5) por (5x3), resultando numa matriz (1x3)    
    # output_2 = sigmoide(output_2)


    erro = y - output_1 


    #%  ============================================================================    
    icognita = np.zeros(qtd_nr_layer[0]+1)              # é o tamanho da pesos_last_layer
    
    for i in range (qtd_nr_layer[0]+1):                 # quantidade de pesos na primeira camada (4 + bias = 5)
        temp = 0                                        # np.zeros(1) 
        for j in range(qtd_nr_layer[1]):                # quantidade de neuronios de saída (3)
            temp += erro[j] * pesos_last_layer[i,j]              # multiplica o erro da saída de cada neuronio, com cada peso que contribuiu para a saída.           
        icognita[i] = output_0[i] * (1-output_0[i]) * temp

    

    #   Atulizando os pesos da camada final    
    for i in range(qtd_nr_layer[0]+1):                                      # quantidade de entradas na camada intermediária, provenientes da anterior. (4 + 1 com bias = 5)
        for j in range (qtd_nr_layer[1]):                                   # quantidade de neuronios na camada intermediária:  3
            pesos_last_layer[i,j] = pesos_last_layer[i,j] + ni * erro[j] * output_0[i]                       


    #   Atualizando os pesos da camada inicial    
    for i in range(qtd_inputs+1):                                      # quantidade de entradas na camada inicial. (4 + 1 com bias = 5)
       for j in range(qtd_nr_layer[0]):                                     # quantidade de neuronios na camada inicial:  4
           pesos_first_layer[i,j] = pesos_first_layer[i,j] + ni * icognita[j+1] * x[i] 
       

    erro = abs( y - output_1 )

    return (erro, pesos_first_layer, pesos_last_layer, output_1)



x = np.array([[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],[4.6,3.1,1.5,0.2],[5.0,3.6,1.4,0.2],[5.4,3.9,1.7,0.4],[4.6,3.4,1.4,0.3],[5.0,3.4,1.5,0.2],[4.4,2.9,1.4,0.2],[4.9,3.1,1.5,0.1],[5.4,3.7,1.5,0.2],[4.8,3.4,1.6,0.2],[4.8,3.0,1.4,0.1],[4.3,3.0,1.1,0.1],[5.8,4.0,1.2,0.2],[5.7,4.4,1.5,0.4],[5.4,3.9,1.3,0.4],[5.1,3.5,1.4,0.3],[5.7,3.8,1.7,0.3],[5.1,3.8,1.5,0.3],[5.4,3.4,1.7,0.2],[5.1,3.7,1.5,0.4],[4.6,3.6,1.0,0.2],[5.1,3.3,1.7,0.5],[4.8,3.4,1.9,0.2],[5.0,3.0,1.6,0.2],[5.0,3.4,1.6,0.4],[5.2,3.5,1.5,0.2],[5.2,3.4,1.4,0.2],[4.7,3.2,1.6,0.2],[4.8,3.1,1.6,0.2],[5.4,3.4,1.5,0.4],[5.2,4.1,1.5,0.1],[5.5,4.2,1.4,0.2],[4.9,3.1,1.5,0.2],[5.0,3.2,1.2,0.2],[5.5,3.5,1.3,0.2],[4.9,3.6,1.4,0.1],[4.4,3.0,1.3,0.2],[5.1,3.4,1.5,0.2],[5.0,3.5,1.3,0.3],[4.5,2.3,1.3,0.3],[4.4,3.2,1.3,0.2],[5.0,3.5,1.6,0.6],[5.1,3.8,1.9,0.4],[4.8,3.0,1.4,0.3],[5.1,3.8,1.6,0.2],[4.6,3.2,1.4,0.2],[5.3,3.7,1.5,0.2],[5.0,3.3,1.4,0.2],[7.0,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.9,3.1,4.9,1.5],[5.5,2.3,4.0,1.3],[6.5,2.8,4.6,1.5],[5.7,2.8,4.5,1.3],[6.3,3.3,4.7,1.6],[4.9,2.4,3.3,1.0],[6.6,2.9,4.6,1.3],[5.2,2.7,3.9,1.4],[5.0,2.0,3.5,1.0],[5.9,3.0,4.2,1.5],[6.0,2.2,4.0,1.0],[6.1,2.9,4.7,1.4],[5.6,2.9,3.6,1.3],[6.7,3.1,4.4,1.4],[5.6,3.0,4.5,1.5],[5.8,2.7,4.1,1.0],[6.2,2.2,4.5,1.5],[5.6,2.5,3.9,1.1],[5.9,3.2,4.8,1.8],[6.1,2.8,4.0,1.3],[6.3,2.5,4.9,1.5],[6.1,2.8,4.7,1.2],[6.4,2.9,4.3,1.3],[6.6,3.0,4.4,1.4],[6.8,2.8,4.8,1.4],[6.7,3.0,5.0,1.7],[6.0,2.9,4.5,1.5],[5.7,2.6,3.5,1.0],[5.5,2.4,3.8,1.1],[5.5,2.4,3.7,1.0],[5.8,2.7,3.9,1.2],[6.0,2.7,5.1,1.6],[5.4,3.0,4.5,1.5],[6.0,3.4,4.5,1.6],[6.7,3.1,4.7,1.5],[6.3,2.3,4.4,1.3],[5.6,3.0,4.1,1.3],[5.5,2.5,4.0,1.3],[5.5,2.6,4.4,1.2],[6.1,3.0,4.6,1.4],[5.8,2.6,4.0,1.2],[5.0,2.3,3.3,1.0],[5.6,2.7,4.2,1.3],[5.7,3.0,4.2,1.2],[5.7,2.9,4.2,1.3],[6.2,2.9,4.3,1.3],[5.1,2.5,3.0,1.1],[5.7,2.8,4.1,1.3],[6.3,3.3,6.0,2.5],[5.8,2.7,5.1,1.9],[7.1,3.0,5.9,2.1],[6.3,2.9,5.6,1.8],[6.5,3.0,5.8,2.2],[7.6,3.0,6.6,2.1],[4.9,2.5,4.5,1.7],[7.3,2.9,6.3,1.8],[6.7,2.5,5.8,1.8],[7.2,3.6,6.1,2.5],[6.5,3.2,5.1,2.0],[6.4,2.7,5.3,1.9],[6.8,3.0,5.5,2.1],[5.7,2.5,5.0,2.0],[5.8,2.8,5.1,2.4],[6.4,3.2,5.3,2.3],[6.5,3.0,5.5,1.8],[7.7,3.8,6.7,2.2],[7.7,2.6,6.9,2.3],[6.0,2.2,5.0,1.5],[6.9,3.2,5.7,2.3],[5.6,2.8,4.9,2.0],[7.7,2.8,6.7,2.0],[6.3,2.7,4.9,1.8],[6.7,3.3,5.7,2.1],[7.2,3.2,6.0,1.8],[6.2,2.8,4.8,1.8],[6.1,3.0,4.9,1.8],[6.4,2.8,5.6,2.1],[7.2,3.0,5.8,1.6],[7.4,2.8,6.1,1.9],[7.9,3.8,6.4,2.0],[6.4,2.8,5.6,2.2],[6.3,2.8,5.1,1.5],[6.1,2.6,5.6,1.4],[7.7,3.0,6.1,2.3],[6.3,3.4,5.6,2.4],[6.4,3.1,5.5,1.8],[6.0,3.0,4.8,1.8],[6.9,3.1,5.4,2.1],[6.7,3.1,5.6,2.4],[6.9,3.1,5.1,2.3],[5.8,2.7,5.1,1.9],[6.8,3.2,5.9,2.3],[6.7,3.3,5.7,2.5],[6.7,3.0,5.2,2.3],[6.3,2.5,5.0,1.9],[6.5,3.0,5.2,2.0],[6.2,3.4,5.4,2.3],[5.9,3.0,5.1,1.8]])
y = np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])

pesos_first_layer = None
pesos_last_layer = None

learning_rate = 0.02

qtd_treinos = 100

if len(sys.argv) == 2:
    qtd_treinos = int(sys.argv[1])

qtd_inputs = 4
qtd_nr_c0 = 12
qtd_nr_c1 = 3
qtd_nr_c2 = 3

sum_error = np.zeros( qtd_treinos ) 

samplesize = len(x)

print ("Começando")

for i in range (qtd_treinos):
    sum_erro = 0
    num_acertos = 0
    for j in range (samplesize):

        (erro, pesos_first_layer, pesos_last_layer, o) = MLP_single_pass(pesos_first_layer, pesos_last_layer, x[j,:], y[j,:], learning_rate, qtd_nr_c0, qtd_nr_c1, qtd_nr_c2, qtd_inputs)        

        outputs = np.array([o[0], o[1], o[2]])
        sum_erro += sum(erro)
        index = np.unravel_index(np.argmax(outputs), outputs.shape)[0]        
        if (y[j,index] == 1):
            num_acertos = num_acertos + 1
    print ('Treino %d - Acertos: %d/%d Sum: %.5f \n' % (i+1, num_acertos, samplesize, sum_erro ) )
    