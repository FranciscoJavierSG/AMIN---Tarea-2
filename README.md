# Algoritmos Metaheurísticos Inspirados en la Naturaleza
*Autores: José Avilán (javilan@ing.ucsc.cl) y Francisco Salazar (fsalazarg@ing.ucsc.cl)*

El programa tiene como objetivo desarrollar una aplicación que implemente el Problema del Vendedor Viajero a través del método de Sistema de Colonia de Hormigas.

## Instalación
Antes de ejecutar el programa, es necesario:
- Tener instalado Python 3.8.10 en su sistema.
- Para descargar el programa, presione en el siguiente [enlace](https://codeload.github.com/FranciscoJavierSG/AMIN---Tarea-2/zip/refs/heads/main).

## Ejecución 
Para ejecutar el programa, escriba lo siguiente en la consola o terminal de su sistema operativo:

```       
py hormigas.py <Semilla> <Matriz_Coords> <Tamaño_Pob> <Iteraciones> <Factor_Evap> <Valor_Heur> <Prob_Limite>
```

Donde:
- **Semilla**: Número entero positivo que representa el valor de la semilla.
- **Matriz_Coords**: Archivo de entrada, documento con las coordenadas de los nodos.
- **Tamaño_Pob**: Tamaño de la colonia o número de hormigas, valor entero.
- **Iteraciones**: Condición de término o número de iteraciones, valor entero.
- **Factor_Evap**: Factor de evaporación de la feromona, valor entero entre 0 y 1.
- **Valor_Heur**: El peso del valor de la heurística, valor entero entre 0 y 1.
- **Prob_Limite**: Valor de probabilidad límite, valor entero entre 0 y 1.

## Ejemplo de Ejecución

```
py hormigas.py 4 berlin32.txt 40 200 0.1 2.5 0.9
```