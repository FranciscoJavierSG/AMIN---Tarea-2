import sys
import pandas as pd
import numpy as np

def obtenerMatrizDistancias(numVar,matriz):
    matrizDistancias = np.full((numVar,numVar), fill_value=-1.0, dtype=float)

    for i in range(numVar-1):
        for j in range(i+1, numVar):
            distancia = np.sqrt(np.sum(np.square(matriz[i]-matriz[j])))
            matrizDistancias[i][j] = distancia
            matrizDistancias[j][i] = distancia

    return matrizDistancias

def obtenerMatrizHeuristicas(matriz):
    matrizHeuristicas = np.full_like(matriz, fill_value=(1/matriz), dtype=float)
    np.fill_diagonal(matrizHeuristicas, 0.0)

    return matrizHeuristicas

def solucionCalcularCosto(cantNodos,matrizSolucion,matrizBase):
    costo = matrizBase[matrizSolucion[cantNodos-1]][matrizSolucion[0]]
    for i in range(cantNodos-1):
        costo = costo + (matrizBase[matrizSolucion[i]][matrizSolucion[i+1]])
    return costo

if len(sys.argv) == 8:
    semilla = int(sys.argv[1])
    matrizBerlin32 = str(sys.argv[2])
    tamañoPob = int(sys.argv[3])
    numIt = int(sys.argv[4])
    factEvapFeromona = float(sys.argv[5])
    valHeuristica = float(sys.argv[6])
    probLimite = float(sys.argv[7])
    print("Semilla: ", semilla)
    print("Matriz de Distancias: ", matrizBerlin32)
    print("Tamaño de Población: ", tamañoPob)
    print("Número de Iteraciones: ", numIt)
    print("Factor de Evaporación de la Feromona: ", factEvapFeromona)
    print("Peso del Valor de la Heuristica: ", valHeuristica)
    print("Valor de Probabilidad Límite: ", probLimite)
else:
    print('Error en la entrada de los parametros')
    sys.exit(0)

np.random.seed(semilla)

berlin52 = pd.read_csv("berlin32.txt", header=None, skiprows=6, skipfooter=1, engine='python', delim_whitespace=True, usecols=(1,2), dtype=int)
matrizBerlin52 = berlin52.to_numpy() 
numVariables = matrizBerlin52.shape[0]
matrizDistancias = obtenerMatrizDistancias(numVariables,matrizBerlin52)
matrizHeuristicas = obtenerMatrizHeuristicas(matrizDistancias)
mejorSol = np.arange(0,numVariables)
np.random.shuffle(mejorSol)
solucionMejorCosto = solucionCalcularCosto(numVariables,mejorSol,matrizHeuristicas)
