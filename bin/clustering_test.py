from bin import clustering
import time

ecoli = clustering.read_file("bin/ecoli_set.dat")
ecoli_matrix_20 = clustering.read_file("bin/ecoli_set_const_20.const")
ecoli_list_20 = clustering.build_restrictions_list(ecoli_matrix_20)
ecoli_matrix_10 = clustering.read_file("bin/ecoli_set_const_10.const")
ecoli_list_10 = clustering.build_restrictions_list(ecoli_matrix_10)

iris = clustering.read_file("bin/iris_set.dat")
iris_matrix_20 = clustering.read_file("bin/iris_set_const_20.const")
iris_list_20 = clustering.build_restrictions_list(iris_matrix_20)
iris_matrix_10 = clustering.read_file("bin/iris_set_const_10.const")
iris_list_10 = clustering.build_restrictions_list(iris_matrix_10)

rand = clustering.read_file("bin/rand_set.dat")
rand_matrix_20 = clustering.read_file("bin/rand_set_const_20.const")
rand_list_20 = clustering.build_restrictions_list(rand_matrix_20)
rand_matrix_10 = clustering.read_file("bin/rand_set_const_10.const")
rand_list_10 = clustering.build_restrictions_list(rand_matrix_10)

# Executions

# Ecoly

start_time = time.time()
greedy_ecoli_10 = clustering.greedy(ecoli, ecoli_matrix_10, 8)
time_greedy_ecoli_10 = time.time() - start_time

start_time = time.time()
greedy_ecoli_20 = clustering.greedy(ecoli, ecoli_matrix_20, 8)
time_greedy_ecoli_20 = time.time() - start_time

start_time = time.time()
ls_ecoli_10 = clustering.local_search(ecoli, ecoli_list_10, 8)
time_ls_ecoli_10 = time.time() - start_time

start_time = time.time()
ls_ecoli_20 = clustering.local_search(ecoli, ecoli_list_20, 8)
time_ls_ecoli_20 = time.time() - start_time

# Iris

start_time = time.time()
greedy_iris_10 = clustering.greedy(iris, iris_matrix_10, 8)
time_greedy_iris_10 = time.time() - start_time

start_time = time.time()
greedy_iris_20 = clustering.greedy(iris, iris_matrix_20, 8)
time_greedy_iris_20 = time.time() - start_time

start_time = time.time()
ls_iris_10 = clustering.local_search(iris, iris_list_10, 8)
time_ls_iris_10 = time.time() - start_time

start_time = time.time()
ls_iris_20 = clustering.local_search(iris, iris_list_20, 8)
time_ls_iris_20 = time.time() - start_time


# Rand

start_time = time.time()
greedy_rand_10 = clustering.greedy(rand, rand_matrix_10, 8)
time_greedy_rand_10 = time.time() - start_time

start_time = time.time()
greedy_rand_20 = clustering.greedy(rand, rand_matrix_20, 8)
time_greedy_rand_20 = time.time() - start_time

start_time = time.time()
ls_rand_10 = clustering.local_search(rand, rand_list_10, 8)
time_ls_rand_10 = time.time() - start_time

start_time = time.time()
ls_rand_20 = clustering.local_search(rand, rand_list_20, 8)
time_ls_rand_20 = time.time() - start_time