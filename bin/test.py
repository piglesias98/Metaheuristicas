def read_file(file):
    with open(file, 'r') as f:
        line = [i.strip().split(',') for i in f]
        data = [[float(i) for i in l] for l in line]
        return data


iris = read_file('bin/iris_set.dat')
print(iris)
print(len(iris))
iris_restr = read_file('bin/iris_set_const_10.const')
print(len(iris_restr))