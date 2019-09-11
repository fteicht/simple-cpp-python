import build.__simple_cpp_python as wow
import getopt
import math
import operator
import time
import sys
import matplotlib.pyplot as plt
from pathos import multiprocessing
from functools import reduce
from collections import defaultdict
from pandas import DataFrame

def python_inner_loop(M):
    # computes pi/8 as a series
    return reduce(operator.add, [1 / ((4*k + 1) * (4*k + 3)) for k in range(M)], 0)

def python_outer_loop(xf, N, M):
    # computes the series of cos(n * theta) + sin(n * theta)
    with multiprocessing.Pool() as pool:
        v = pool.map(lambda n: math.cos((n+1) * xf(M)) + math.sin((n+1) * xf(M)), [n for n in range(N)])
        return reduce(operator.add, v, 0)

def full_python(N, M):
    return python_outer_loop(python_inner_loop, N, M)

def mixed_op(pool, lr, n, M):
    lr[n] = pool.apply_async(lambda : python_inner_loop(M))

def mixed_get(lr, n):
    return lr[n].get()


if __name__ == "__main__":

    try:
        options, remainder = getopt.getopt(sys.argv[1:],
                                           "x:n:",
                                           ["xarg=", "narg="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    
    N = 1000

    for opt, arg in options:
        if opt in ('-n', '--narg'):
            n = int(arg)

    start = time.time()
    r = full_python(N, 10*N)
    end = time.time()
    print('full python result: ' + str(r) + ' in ' + str(end-start) + ' seconds')

    start = time.time()
    r = wow.full_cpp(N, 10*N)
    end = time.time()
    print('full c++ result: ' + str(r) + ' in ' + str(end-start) + ' seconds')

    start = time.time()
    r = wow.cpp_outer_loop_gil(lambda M: python_inner_loop(M), N, 10*N)
    end = time.time()
    print('mixed result (GIL): ' + str(r) + ' in ' + str(end-start) + ' seconds')

    start = time.time()
    with multiprocessing.Pool() as pool:
        lr = [None]*N
        r = wow.cpp_outer_loop_process(lambda n, M : mixed_op(pool, lr, n, M), lambda n : mixed_get(lr, n), N, 10*N)
    end = time.time()
    print('mixed result (process): ' + str(r) + ' in ' + str(end-start) + ' seconds')

    x = math.pi / 8
    r = 0.5 * ((1.0 / math.tan(x / 2)) - 1 + (math.sin(x * (N + 0.5)) - math.cos(x * (N + 0.5))) / math.sin(x / 2))
    print('exact result: ' + str(r))

    # Statistical analysis
    print('Statistic analysis...')

    N = 1000
    d = defaultdict(list)
    index = [100, 1000, 10000, 100000, 1000000]
    
    for M in index:
        print("N={0}, M={1}".format(N, M))

        start = time.time()
        full_python(N, M)
        end = time.time()
        ref = end - start
        d['full_python'].append(1.0)
        print('   full python:', str(end-start))

        start = time.time()
        wow.cpp_outer_loop_gil(lambda m: python_inner_loop(m), N, M)
        end = time.time()
        d['mixed_gil'].append(ref / (end - start))
        print('   mixed gil:', str(end-start))

        start = time.time()
        with multiprocessing.Pool() as pool:
            lr = [None]*N
            wow.cpp_outer_loop_process(lambda n, m : mixed_op(pool, lr, n, m), lambda n : mixed_get(lr, n), N, M)
        end = time.time()
        d['mixed_process'].append(ref / (end - start))
        print('   mixed process:', str(end-start))

        start = time.time()
        wow.full_cpp(N, M)
        end = time.time()
        d['full_cpp'].append(ref / (end - start))
        print('   full cpp:', str(end-start))
    
    df = DataFrame(data=d, index=index)
    df.plot.bar(rot=0, logy=True)
    plt.xlabel('M')
    plt.ylabel('(log) speedup compared to full python')
    plt.axhline(y=1, linewidth=3, color='gray', linestyle='dashed')
    plt.savefig('comparison_M.png', transparent=True)
    plt.show()

    M = 10000
    d = defaultdict(list)
    index = [10, 100, 1000, 10000, 100000]
    
    for N in index:
        print("N={0}, M={1}".format(N, M))

        start = time.time()
        full_python(N, M)
        end = time.time()
        ref = end - start
        d['full_python'].append(1.0)
        print('   full python:', str(end-start))

        start = time.time()
        wow.cpp_outer_loop_gil(lambda m: python_inner_loop(m), N, M)
        end = time.time()
        d['mixed_gil'].append(ref / (end - start))
        print('   mixed gil:', str(end-start))

        start = time.time()
        with multiprocessing.Pool() as pool:
            lr = [None]*N
            wow.cpp_outer_loop_process(lambda n, m : mixed_op(pool, lr, n, m), lambda n : mixed_get(lr, n), N, M)
        end = time.time()
        d['mixed_process'].append(ref / (end - start))
        print('   mixed process:', str(end-start))

        start = time.time()
        wow.full_cpp(N, M)
        end = time.time()
        d['full_cpp'].append(ref / (end - start))
        print('   full cpp:', str(end-start))
    
    df = DataFrame(data=d, index=index)
    df.plot.bar(rot=0, logy=True)
    plt.xlabel('N')
    plt.ylabel('(log) speedup compared to full python')
    plt.axhline(y=1, linewidth=3, color='gray', linestyle='dashed')
    plt.savefig('comparison_N.png', transparent=True)
    plt.show()
