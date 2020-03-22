# Problema del agrupamiento de restricciones
El problema del agrupamiento con restricciones, abreviado PAR, es una generalización del agrupamiento clásico, el llamado clustering, típico problema en el ámbito de Machine Learning.

Se ha implementado utilizando dos algoritmos diferentes:

- Greedy  
- Búsqueda Local

## Ficheros

- Ejecución `/bin`
    
    - `clustering.py` contiene los métodos necesarios para ejecutar los dos algoritmos:
        - `greedy(data, r_matrix, k, seed)`
        - `local_search(data, r_list, k, seed)`
    - `clustering_test.py` el fichero para ejecutar las pruebas del algoritmo
        - `results()` 5 ejecuciones de cada dataset y cada % de restricciones del algoritmo especificado. Por defecto se ejecuta `local_search` , para ejecutar `greedy` hay que comentar la línea de local_search y descomentar la de greedy
        - `convergence()` Una ejecución del dataset especificado para el método especificado `greedy_convergence()` o `local_seach_converge()`
- Ficheros de datos `/bin`
    - `ecoli_set.dat` Conjunto de datos ecoli
    - `iris_set.dat` Conjunto de datos iris
    - `rand_set.dat` Conjunto de datos rand
    - `ecoli_set_const_10.const` Restricciones ecoli 10%
    - `ecoli_set_const_20.const` Restricciones ecoli 20%
    - `iris_set_const_10.const` Restricciones iris 10%
    - `iris_set_const_20.const` Restricciones iris 20%
    - `rand_set_const_10.const` Restricciones rand 10%
    - `rand_set_const_20.const` Restricciones rand 20%
- Ficheros de resultados `/results`
    - `results_greedy.txt`  Resultados globales greedy
    - `results_local_search.txt` Resultados globales búsqueda local
    - `convergence_greedy_ecoli.txt` Convergencia greedy con ecoli
    - `convergence_greedy_iris.txt` Convergencia greedy con iris
    - `convergence_local_search_ecoli.txt` Convergencia búsqueda local con ecoli
    - `convergence_local_search_iris.txt` Convergencia búsqueda local con iris
    
    
         
