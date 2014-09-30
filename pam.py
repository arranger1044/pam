import numpy

import scipy
import scipy.spatial.distance

import sys


from numba import jit

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time


INF = float('inf')


def closest_medoid(distances, point_id, medoid_ids):
    """
    WRITEME
    """
    return medoid_ids[numpy.argmin(distances[point_id, medoid_ids])]


# @autojit
def closest_medoids_cost(distances, medoid_ids):
    """
    WRITEME
    """
    closest_ids = numpy.argmin(distances[:, medoid_ids], axis=1)
    return closest_ids, numpy.sum(distances[:, closest_ids])


@jit
def closest_medoids_numba(distances, medoid_ids, clustering):
    """
    WRITEME
    """
    n_instances = distances.shape[0]
    tot_cost = 0
    for i in range(n_instances):
        best_medoid = i
        best_cost = INF
        for j in medoid_ids:
            cost = distances[i, j]
            if cost < best_cost:
                best_cost = cost
                best_medoid = j
        tot_cost += cost
        clustering[i] = best_medoid

    return tot_cost


def closest_medoids(distances, points_2_medoids, medoid_ids):
    """
    WRITEME
    """
    medoids_2_instances = [{} for i in range(len(medoid_ids))]
    for i in range(points_2_medoids):
        medoid_id = closest_medoid(distances, i, medoid_ids)
        points_2_medoids[i] = medoid_id
        medoids_2_instances[medoid_id].add(i)
    return medoids_2_instances


def closest_medoids_2(distances, points_2_medoids, medoid_ids):
    """
    WRITEME
    """
    for i in range(points_2_medoids):
        points_2_medoids[i] = closest_medoid(distances, i, medoid_ids)


def total_cost_by_swap(distances, medoid_id):
    """
    WRITEME
    """
    return numpy.sum(distances[medoid_id, :])


# @autojit
def pam(distances,
        k,
        n_iters=100,
        delta_cost=1e-5,
        rand_gen=None):
    """
    WRITEME
    """
    n_instances = distances.shape[0]
    stop = False
    iter = 0

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(1337)

    instance_ids = [i for i in range(n_instances)]
    #
    # using an int array for medoid ids
    # initing with random points
    # medoid_ids = rand_gen.choice(instance_ids, k, replace=False)
    # using an array of bits
    medoid_ids_vec = numpy.zeros(n_instances, dtype=bool)
    # setting k random medoids
    medoid_ids = rand_gen.choice(instance_ids, k, replace=False)
    medoid_ids_vec[medoid_ids] = True

    #
    # using another array for the medoid correspondences
    # assigning to the closest medoid
    clustering, cost = closest_medoids_cost(distances, medoid_ids)

    while iter < n_iters and not stop:

        best_cost = INF
        best_clustering = None
        best_medoid_ids = None
        best_medoid_ids_vec = None

        iter_start_t = perf_counter()
        #
        # For each current medoid
        #
        for i, medoid_id in enumerate(medoid_ids):

            avg_swap_t = 0
            #
            # Trying to swap each other instance
            #
            for j in instance_ids:

                swap_start_t = perf_counter()

                if medoid_id != j:

                    # remove medoid
                    # medoid_ids_cp = set(medoid_ids)
                    # medoid_ids_cp.remove(medoid_id)
                    # add instance
                    # medoid_ids_cp.add(instance_id)
                    medoid_ids_vec[i] = False
                    medoid_ids_vec[j] = True

                    # compute the cost
                    # clustering, cost = closest_medoids_cost(distances,
                    #                                         medoid_ids_vec)
                    cost = closest_medoids_numba(distances,
                                                 medoid_ids,
                                                 clustering)

                    if cost < best_cost:
                        best_cost = cost
                        best_clustering = numpy.copy(clustering)
                        best_medoid_ids_vec = numpy.copy(medoid_ids_vec)
                        best_medoid_ids = numpy.where(medoid_ids_vec)

                    medoid_ids_vec[i] = True
                    medoid_ids_vec[j] = False

                swap_end_t = perf_counter()
                avg_swap_t += (swap_end_t - swap_start_t)
                sys.stdout.write('\rswapped {0}-{1} [{2:.4f} secs avg]'.
                                 format(i, j, avg_swap_t / (j + 1)))
                sys.stdout.flush()

        #
        # checking for the best clustering
        #
        medoid_ids = best_medoid_ids
        medoid_ids_vec = best_medoid_ids_vec

        if numpy.all(clustering != best_clustering):
            clustering = best_clustering
        else:
            stop = False

        #
        # checking for the cost as well?
        #

        iter_end_t = perf_counter()
        print('\n-->Elapsed {0:.4f} secs for iteration {1}/{2}'.
              format(iter_end_t - iter_start_t,
                     iter + 1,
                     n_iters))

    return clustering


def compute_similarity_matrix(data_slice):
    """
    From a matrix m x n creates a kernel matrix
    according to a metric of size m x m
    (it shall be symmetric, and (semidefinite) positive)

    ** USES SCIPY **

    """

    pairwise_dists = \
        scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(data_slice,
                                         'sqeuclidean'))

    return pairwise_dists

if __name__ == '__main__':

    #
    # create fake dataset
    #
    print('Creating synthetic data')
    synth_data = numpy.random.binomial(100, 0.5, (2000, 15))

    #
    # create distance matrix
    #
    print('Computing the distance matrix')
    distances = compute_similarity_matrix(synth_data)

    #
    # clustering
    #
    k = 30
    clustering = pam(distances, k)
