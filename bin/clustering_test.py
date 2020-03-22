from bin import clustering
import time

'''
Clustering test
-> results() choose between local_search or greedy
    5 executions with seeds 27, 33, 56, 22, 29
    of 3 datasets "iris", "ecoli", "rand" each with
    restrictions 10% and 20%
-> convergence() choose a between local_search or greedy
    to see the convergence of the algorithm printing in 
    each iteration the objective function value
'''

# Parameters:
executions = 5
datasets = ["iris", "ecoli", "rand"]
clusters = [3, 8, 3]
restrictions = ["10", "20"]
seeds = [27, 33, 56, 22, 29]
# Results file
f = open("bin/results.txt", "w")


def results():
    for d in range(len(datasets)):
        data = clustering.read_file("bin/"+datasets[d]+"_set.dat")
        f.write("\n\n------------------------------------  "+datasets[d]+"  ------------------------------------\n")
        print(datasets[d])
        for r in restrictions:
            r_matrix = clustering.read_file("bin/"+datasets[d]+"_set_const_"+r+".const")
            r_list = clustering.build_restrictions_list(r_matrix)
            f.write("\n\n--------> Restriction: " + r + "\n")
            print("Restriction: ", r)
            for i in range(executions):
                f.write("--EXECUTION: " + str(i) + " Seed: " + str(seeds[i]) + "\n")
                print("Execution: ", i)
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


def convergence():
    f.write("IRIS RESTRICTION 20, SEED 27\n\n")
    data = clustering.read_file("bin/iris_set.dat")
    r_matrix = clustering.read_file("bin/iris_set_const_20.const")
    r_list = clustering.build_restrictions_list(r_matrix)
    start_time = time.time()
    # sol, results = clustering.local_search_convergence(data, r_list,3, 27)
    sol, results = clustering.greedy_convergence(data, r_matrix, 3, 27, r_list)
    time_sol = time.time() - start_time
    c_rate = clustering.c(sol, data, 3)
    inf_rate = clustering.infeasibility_total(sol, r_list)
    obj_rate = clustering.objective(sol, data, 3, r_list, clustering.compute_lambda(data, r_list))
    f.write("SOL: " + str(sol) + "\n")
    f.write("TIME: " + str(time_sol) + "\n")
    f.write("C_RATE: " + str(c_rate) + "\n")
    f.write("INF_RATE: " + str(inf_rate) + "\n")
    f.write("OBJ_RATE: " + str(obj_rate) + "\n")
    f.write("\nRESULTS: \n")
    for i in results:
        f.write(str(i)+"\n")



# Try it out

results()