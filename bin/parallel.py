from multiprocessing import Pool


datasets = ["iris", "ecoli", "rand", "newthyroid"]
argument = [0,1,2,3]

def run_in_parallel(d):
    # Results file
    f = open("prueba_paralelo_" + datasets[d] + ".txt", "w")
    f.write("\n\n------------------------------------  " + datasets[d] + "  ------------------------------------\n")
    print(datasets[d])


if __name__ == '__main__':
    pool = Pool()
    pool.map(run_in_parallel, argument)
    pool.close()
    pool.join()