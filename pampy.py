import numpy

from kmedoids import pam

import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import csv

import sys

import os


#########################################
# creating the opt parser
parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, nargs=1,
                    help='Specify a data file path to load from')

parser.add_argument('-i', '--iters', type=int, nargs='?',
                    default=100,
                    help='Max number of iterations')


parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/',
                    help='Output dir path')

parser.add_argument('-k', '--n-clusters', type=int, nargs=1,
                    default=10,
                    help='Number of clusters')

parser.add_argument('--cluster-ids', action='store_true',
                    help='Returning cluster labels as medoid ids')

parser.add_argument('--theano', action='store_true',
                    help='Computing closest medoids and the'
                    'cost function with Theano')

# parsing the args
args = parser.parse_args()

#
# random generator initing
#
rand_gen = numpy.random.RandomState(args.seed)

#
# reading the from the data file, assuming a distance matrix in csv form
#


def csv_2_numpy(file, path='', sep=',', type='int8'):
    """
    WRITEME
    """
    file_path = path + file
    reader = csv.reader(open(file_path, "r"), delimiter=sep)
    x = list(reader)
    dataset = numpy.array(x).astype(type)
    return dataset

datapath, = args.data
distances = csv_2_numpy(datapath)

#
# calling the algorithm
#
pam_start_t = perf_counter()
clustering = pam(distances,
                 args.n_clusters,
                 n_iters=args.iters,
                 rand_gen=rand_gen,
                 medoids_2_clusters=args.cluster_ids,
                 theano=args.theano)
pam_end_t = perf_counter()
print('PAM done (elapsed {0} secs)'.format(pam_end_t - pam_start_t))
print('clustering found:', clustering)
