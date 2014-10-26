import numpy

import scipy
import scipy.spatial.distance

import sys


from numba import jit

import theano

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time


INF = float('inf')


# @autojit
def closest_medoids_cost(distances, medoid_ids):
    """
    Computing the closest medoids and the cost of the new configuration
    using numpy
    """
    closest_ids = numpy.argmin(distances[:, medoid_ids], axis=1)
    return closest_ids, numpy.sum(distances[:, closest_ids])


@jit
def closest_medoids_numba(distances, medoid_ids, clustering):
    """
    Computing the closest medoids and the cost of the new configuration
    using a simple loop optimized with Numba
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

        tot_cost += best_cost
        clustering[i] = best_medoid

    return tot_cost


def create_theano_cost_function(distances):
    """
    Creating the function to compute the closest medoids and the cost
    of that configuration using Theano
    """
    #
    # the distance matrix is a shared var
    #
    D = theano.shared(distances)
    M = theano.tensor.bvector()
    #
    # Some tricky stuff here, I am using max_and_argmax to get both
    # things so I shall use an inverted distance matrix
    #
    Inf = theano.shared(float('inf'))
    D_meds = theano.tensor.switch(M, -D, -Inf)
    # D_meds = theano.printing.Print('DM')(D_meds_a)
    Closest, Idx = theano.tensor.max_and_argmax(D_meds, axis=1)
    # Closest = theano.printing.Print('CLS')(Closest_a)
    # Idx = theano.printing.Print('IDX')(Idx_a)
    Cost = theano.tensor.sum(Closest)
    return theano.function([M], [-Cost, Idx])


def medoid_assoc_to_clustering(clustering):
    """
    Converting a clustering by medoid ids into one with sequential ids
    (starting from 0)
    """
    medoids_to_clusters = set(clustering)

    clustering_assoc = {medoid_id: cluster_id for cluster_id, medoid_id
                        in enumerate(medoids_to_clusters)}
    clustering_ids = [clustering_assoc[medoid_id] for medoid_id in clustering]
    return clustering_ids

# @autojit


def pam(distances,
        k,
        n_iters=100,
        delta_cost=1e-5,
        rand_gen=None,
        medoids_2_clusters=False,
        theano=False):
    """
    Partitioning Around Medoids (PAM) implementation of K-Medoids
    """
    n_instances = distances.shape[0]
    stop = False
    iter = 0

    if theano:
        closest_medoids_theano = create_theano_cost_function(distances)

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(1337)

    instance_ids = [i for i in range(n_instances)]
    #
    # using an array of bits
    medoid_ids_vec = numpy.zeros(n_instances, dtype=bool)

    # setting k random medoids
    medoid_ids = rand_gen.choice(instance_ids, k, replace=False)
    # medoid_ids = [1, 7]
    medoid_ids_vec[medoid_ids] = True

    #
    # using another array for the medoid correspondences
    # assigning to the closest medoid
    if not theano:
        clustering = numpy.zeros(n_instances, dtype='int32')
        cost = closest_medoids_numba(distances, medoid_ids, clustering)

    else:
        cost, clustering = closest_medoids_theano(medoid_ids_vec)

    print('Initial cost {0} clustering {1} medoids {2} vec {3}'.
          format(cost, clustering, medoid_ids, medoid_ids_vec))

    best_cost = cost
    best_clustering = clustering
    curr_best_cost = cost
    curr_best_clustering = numpy.copy(clustering)
    curr_best_medoid_ids = medoid_ids
    curr_best_medoid_ids_vec = medoid_ids_vec

    while iter < n_iters and not stop:

        iter_start_t = perf_counter()
        #
        # Maximization step
        #
        for i, medoid_id in enumerate(medoid_ids):

            avg_swap_t = 0
            #
            # Trying to swap each other instance
            #
            for j in instance_ids:

                swap_start_t = perf_counter()

                # j is not a medoid
                if not medoid_ids_vec[j]:

                    # remove medoid
                    # medoid_ids_cp = set(medoid_ids)
                    # medoid_ids_cp.remove(medoid_id)
                    # add instance
                    # medoid_ids_cp.add(j)

                    # print('\n curr vec', medoid_ids_vec)
                    medoid_ids_vec[medoid_id] = False
                    medoid_ids_vec[j] = True

                    medoid_ids_cp, = numpy.where(medoid_ids_vec)

                    # compute the Expected cost
                    # clustering, cost = closest_medoids_cost(distances,
                    #                                         medoid_ids_vec)
                    if not theano:
                        cost = closest_medoids_numba(distances,
                                                     medoid_ids_cp,
                                                     clustering)
                    else:
                        cost, clustering = closest_medoids_theano(
                            medoid_ids_vec)

                    #       format(cost, clustering,
                    #              medoid_ids_cp, medoid_ids_vec))

                    if cost < curr_best_cost:
                        print(
                            '\nnew best cost {0}/{1}'.format(cost,
                                                             curr_best_cost))
                        curr_best_cost = cost
                        curr_best_clustering = numpy.copy(clustering)
                        curr_best_medoid_ids_vec = numpy.copy(medoid_ids_vec)
                        curr_best_medoid_ids = medoid_ids_cp

                    medoid_ids_vec[medoid_id] = True
                    medoid_ids_vec[j] = False

                swap_end_t = perf_counter()
                avg_swap_t += (swap_end_t - swap_start_t)
                sys.stdout.write('\rswapped {0}-{1} [{2:.4f} secs avg]'.
                                 format(i, j, avg_swap_t / (j + 1)))
                sys.stdout.flush()

        #
        # checking for the cost as well?
        #
        delta_cost = best_cost - curr_best_cost

        #
        # checking for changes in the clustering scheme
        #
        if numpy.any(medoid_ids_vec != curr_best_medoid_ids_vec):
            print('\nNew clustering', best_clustering)
            best_clustering = curr_best_clustering
            medoid_ids = curr_best_medoid_ids
            medoid_ids_vec = curr_best_medoid_ids_vec
            best_cost = curr_best_cost
        else:
            stop = True

        iter_end_t = perf_counter()
        print('\n-->Elapsed {0:.4f} secs for iteration {1}/{2} COST:{3}'.
              format(iter_end_t - iter_start_t,
                     iter + 1,
                     n_iters,
                     best_cost))
        iter += 1

    #
    # translating from medoids to clusters?
    #
    print('Best clustering (medoid ids)', best_clustering)
    if medoids_2_clusters:
        best_clustering = medoid_assoc_to_clustering(best_clustering)
        print('Best clustering (cluster ids)', best_clustering)

    return best_clustering


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
    clustering = pam(distances, k, theano=True)

if __name__ == '__main__2':

    distances = numpy.array([[0, 3, 3, 3, 8, 6, 8, 7, 7, 5],
                             [3, 0, 4, 4, 5, 3, 4, 4, 6, 6],
                             [3, 4, 0, 2, 9, 7, 9, 8, 8, 6],
                             [3, 4, 2, 0, 7, 5, 7, 6, 6, 4],
                             [8, 5, 9, 7, 0, 2, 2, 3, 5, 5],
                             [6, 3, 7, 5, 2, 0, 2, 1, 3, 3],
                             [8, 4, 9, 7, 2, 2, 0, 1, 3, 3],
                             [7, 4, 8, 6, 3, 1, 1, 0, 2, 2],
                             [7, 6, 8, 6, 5, 3, 3, 2, 0, 2],
                             [5, 6, 6, 4, 5, 3, 3, 2, 2, 0]])
    k = 2
    clustering = pam(distances, k, n_iters=20, theano=True)
