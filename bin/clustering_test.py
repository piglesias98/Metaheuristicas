from bin import clustering
import time
#
# ecoli = clustering.read_file("bin/ecoli_set.dat")
# ecoli_matrix_20 = clustering.read_file("bin/ecoli_set_const_20.const")
# ecoli_list_20 = clustering.build_restrictions_list(ecoli_matrix_20)
# ecoli_matrix_10 = clustering.read_file("bin/ecoli_set_const_10.const")
# ecoli_list_10 = clustering.build_restrictions_list(ecoli_matrix_10)
#
# iris = clustering.read_file("bin/iris_set.dat")
# iris_matrix_20 = clustering.read_file("bin/iris_set_const_20.const")
# iris_list_20 = clustering.build_restrictions_list(iris_matrix_20)
# iris_matrix_10 = clustering.read_file("bin/iris_set_const_10.const")
# iris_list_10 = clustering.build_restrictions_list(iris_matrix_10)
#
# rand = clustering.read_file("bin/rand_set.dat")
# rand_matrix_20 = clustering.read_file("bin/rand_set_const_20.const")
# rand_list_20 = clustering.build_restrictions_list(rand_matrix_20)
# rand_matrix_10 = clustering.read_file("bin/rand_set_const_10.const")
# rand_list_10 = clustering.build_restrictions_list(rand_matrix_10)
#
# # Executions
#
# seeds = [94, 31, 42, 22, 104]
#
# # Arrays of results
# greedy_ecoli_10 = []
# time_greedy_ecoli_10 = []
# greedy_ecoli_20 = []
# time_greedy_ecoli_20 = []
# ls_ecoli_10 = []
# time_ls_ecoli_10 = []
# ls_ecoli_20 = []
# time_ls_ecoli_20 = []
#
# greedy_iris_10 = []
# time_greedy_iris_10 = []
# greedy_iris_20 = []
# time_greedy_iris_20 = []
# ls_iris_10 = []
# time_ls_iris_10 = []
# ls_iris_20 = []
# time_ls_iris_20 = []
#
# greedy_rand_10 = []
# time_greedy_rand_10 = []
# greedy_rand_20 = []
# time_greedy_rand_20 = []
# ls_rand_10 = []
# time_ls_rand_10 = []
# ls_rand_20 = []
# time_ls_rand_20 = []
#
#
#
# for i in range(5):
#
#     # Ecoly
#
#     start_time = time.time()
#     greedy_ecoli_10.append(clustering.greedy(ecoli, ecoli_matrix_10, 8))
#     time_greedy_ecoli_10.append(time.time() - start_time)
#
#     start_time = time.time()
#     greedy_ecoli_20.append(clustering.greedy(ecoli, ecoli_matrix_20, 8))
#     time_greedy_ecoli_20.append(time.time() - start_time)
#
#     start_time = time.time()
#     ls_ecoli_10.append(clustering.local_search(ecoli, ecoli_list_10, 8))
#     time_ls_ecoli_10.append(time.time() - start_time)
#
#     start_time = time.time()
#     ls_ecoli_20.append(clustering.local_search(ecoli, ecoli_list_20, 8))
#     time_ls_ecoli_20.append(time.time() - start_time)
#
#     # Iris
#
#     start_time = time.time()
#     greedy_iris_10.append(clustering.greedy(iris, iris_matrix_10, 8))
#     time_greedy_iris_10.append(time.time() - start_time)
#
#     start_time = time.time()
#     greedy_iris_20.append(clustering.greedy(iris, iris_matrix_20, 8))
#     time_greedy_iris_20.append(time.time() - start_time)
#
#     start_time = time.time()
#     ls_iris_10.append(clustering.local_search(iris, iris_list_10, 8))
#     time_ls_iris_10.append(time.time() - start_time)
#
#     start_time = time.time()
#     ls_iris_20.append(clustering.local_search(iris, iris_list_20, 8))
#     time_ls_iris_20.append(time.time() - start_time)
#
#
#     # Rand
#
#     start_time = time.time()
#     greedy_rand_10.append(clustering.greedy(rand, rand_matrix_10, 8))
#     time_greedy_rand_10.append(time.time() - start_time)
#
#     start_time = time.time()
#     greedy_rand_20.append(clustering.greedy(rand, rand_matrix_20, 8))
#     time_greedy_rand_20.append(time.time() - start_time)
#
#     start_time = time.time()
#     ls_rand_10.append(clustering.local_search(rand, rand_list_10, 8))
#     time_ls_rand_10.append(time.time() - start_time)
#
#     start_time = time.time()
#     ls_rand_20.append(clustering.local_search(rand, rand_list_20, 8))
#     time_ls_rand_20.append(time.time() - start_time)
#
#
# # Create a file
# f = open("bin/results.txt", "x")
#
# for i in range(5):
#     f.write("greedy_ecoli_10", greedy_ecoli_10[i])
#     f.write("time_greedy_ecoli_10",time_greedy_ecoli_10[i])
#     f.write("greedy_ecoli_20",greedy_ecoli_20[i])
#     f.write("time_greedy_ecoli_20",time_greedy_ecoli_20[i])
#     f.write("ls_ecoli_10",ls_ecoli_10[i])
#     f.write("time_ls_ecoli_10",time_ls_ecoli_10[i])
#     f.write("ls_ecoli_20",ls_ecoli_20[i])
#     f.write("time_ls_ecoli_20", time_ls_ecoli_20[i])
#
#     # f.writegreedy_iris_10[i]
#     # f.writetime_greedy_iris_10[i]
#     # f.writegreedy_iris_20[i]
#     # f.writetime_greedy_iris_20[i]
#     # f.writels_iris_10[i]
#     # f.writetime_ls_iris_10[i]
#     # f.writels_iris_20[i]
#     # f.writetime_ls_iris_20[i]
#     #
#     # f.writegreedy_rand_10[i]
#     # f.writetime_greedy_rand_10[i]
#     # f.writegreedy_rand_20[i]
#     # f.writetime_greedy_rand_20[i]
#     # f.writels_rand_10[i]
#     # f.write.("time_ls_rand_10",time_ls_rand_10[i])
#     # f.write("ls_rand_20",ls_rand_20[i])
#     # f.write("time_ls_rand_20", time_ls_rand_20[i])

executions = 5
datasets = ["iris", "ecoli", "rand"]
clusters = [3, 8, 3]
restrictions = ["10", "20"]

# Results file
f = open("bin/results.txt", "x")

for d in range(len(datasets)):
    data = clustering.read_file("bin/"+datasets[d]+"_set.dat")
    f.write("------------------  "+datasets[d]+"  ------------------\n")
    for r in restrictions:
        r_matrix = clustering.read_file("bin/"+datasets[d]+"_set_const_"+r+".const")
        r_list = clustering.build_restrictions_list(r_matrix)
        f.write("--------> Restriction: " + r + "\n")
        for i in range(executions):
            f.write("--EXECUTION: " + str(i) + "\n")
            start_time = time.time()
            sol = clustering.local_search(data, r_list, clusters[d])
            time_sol = time.time() - start_time
            c_rate = clustering.c(sol, data, clusters[d])
            inf_rate = clustering.infeasibility_total(sol, r_list)
            obj_rate = clustering.objective(sol, data, clusters[d], r_list, clustering.compute_lambda(data, r_list))
            f.write("SOL: " + str(sol) + "\n")
            f.write("TIME: " + str(time_sol) + "\n")
            f.write("C_RATE" + str(c_rate) + "\n")
            f.write("INF_RATE" + str(inf_rate) + "\n")
            f.write("OBJ_RATE" + str(obj_rate) + "\n")