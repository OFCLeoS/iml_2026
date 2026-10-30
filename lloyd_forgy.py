import numpy as np
import random

### DELETE THIS ###
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
###################


def lloyd_forgy(data, k, max_iterations=500):
    # Max Iterations to avoid infinite loop
    # 1. Initialize Centroids (Pick random ah points)
        # 1.1. Assign each data point to nearest centroid
        # 1.2. Update centroids to mean of the points within it
        # Repeat until no more assignments to a data point is done or max iterations have been reached (to avoid infinite loop)

    # 1.
    # We initialize the centroids, and
    # The centroids dictionary (maps a centroid to an array of data points that belong to said centroid)
    # The data points dictionary (maps a data_point str to a centroid index)
    centroids = []
    centroids_dict = {}
    data_points_dict = {}
    # TODO: Could replace this dic/also do a dictionary that maps a data point to a centroid (better performance?)
    for i in range(k):
        index = random.randint(0, len(data)-1)
        centroids.append(data[index])
        centroids_dict[i] = []

    repeat = True
    iterations = 0
    # We repeat until no more assignments to a data point is done
    while repeat and iterations <= max_iterations:
        repeat = False
        # 1.1
        for data_point in data:
            # We start with a big ass number as a reference (max 32bit integer)
            closest_centroid_distance = 2147483647
            closest_centroid_index = -1
            for i in range(len(centroids)):
                distance = get_euclidean_distance(data_point, centroids[i])
                if distance < closest_centroid_distance:
                    closest_centroid_distance = distance
                    closest_centroid_index = i
            
            # 1.2
            if str(data_point) in data_points_dict and data_points_dict[str(data_point)] != closest_centroid_index:
                key = data_points_dict[str(data_point)]
                repeat = True
                # TODO: Exit on spotting the entry? (this scans the whole list)
                centroids_dict[key] = [entry for entry in centroids_dict[key] if not (entry == data_point).all()]   
                centroids_dict[closest_centroid_index].append(data_point)
                data_points_dict[str(data_point)] = closest_centroid_index
            elif str(data_point) not in data_points_dict:
                repeat = True
                centroids_dict[closest_centroid_index].append(data_point)
                data_points_dict[str(data_point)] = closest_centroid_index
        # If repeat is not True, then no assignment was made, and we do not need to update the centroid mean
        if repeat:
            for i in range(len(centroids)):
                # Centroid will be new average of points that are assigned to it
                centroids[i] = sum(centroids_dict[i])/len(centroids_dict[i])
        iterations += 1

    print(f"FINISHED IN {iterations-1} ITERATIONS")
    return centroids


def get_euclidean_distance(start, end):
    if len(start) != len(end):
        raise ValueError(
            "Euclidian Distance Error: The 2 points need to be of the same length.")
    euclidian_distance = 0
    for i in range(len(start)):
        euclidian_distance += (start[i]-end[i])**2
    return euclidian_distance


######################### TESTING #########################
iris = load_iris().data

kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(iris)
print(kmeans.cluster_centers_)

print(lloyd_forgy(iris, 3))
