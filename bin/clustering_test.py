from bin import clustering
import time

executions = 5
datasets = ["iris", "ecoli", "rand"]
clusters = [3, 8, 3]
restrictions = ["10", "20"]
seeds = [94, 33, 42, 22, 110]
# Results file
f = open("bin/greedy_v4.txt", "w")

for d in range(len(datasets)):
    data = clustering.read_file("bin/"+datasets[d]+"_set.dat")
    f.write("\n\n------------------------------------  "+datasets[d]+"  ------------------------------------\n")
    for r in restrictions:
        r_matrix = clustering.read_file("bin/"+datasets[d]+"_set_const_"+r+".const")
        r_list = clustering.build_restrictions_list(r_matrix)
        f.write("\n\n--------> Restriction: " + r + "\n")
        for i in range(executions):
            f.write("--EXECUTION: " + str(i) + "\n")
            start_time = time.time()
            sol = clustering.local_search(data, r_list, clusters[d], seeds[i])
            # sol = clustering.greedy(data, r_matrix, clusters[d], seeds[i])
            time_sol = time.time() - start_time
            c_rate = clustering.c(sol, data, clusters[d])
            inf_rate = clustering.infeasibility_total(sol, r_list)
            obj_rate = clustering.objective(sol, data, clusters[d], r_list, clustering.compute_lambda(data, r_list))
            f.write("SOL: " + str(sol) + "\n")
            f.write("TIME: " + str(time_sol) + "\n")
            f.write("C_RATE: " + str(c_rate) + "\n")
            f.write("INF_RATE: " + str(inf_rate) + "\n")
            f.write("OBJ_RATE: " + str(obj_rate) + "\n")


