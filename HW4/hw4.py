import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


def load_data(filepath):
    # Intialize list which will contain dicts as elements
    pokemon_stat_list = []

    # Open file and create a dict for the values for each line
    with open(filepath, newline="\n") as datafile:
        reader = csv.DictReader(datafile)
        for line in reader:
            dict = {
                "HP": line["HP"],
                "Attack": line["Attack"],
                "Defense": line["Defense"],
                "Sp. Atk": line["Sp. Atk"],
                "Sp. Def": line["Sp. Def"],
                "Speed": line["Speed"],
            }
            # Append dict of stats to list
            pokemon_stat_list.append(dict)

    # Return the list of dictionaries
    return pokemon_stat_list


def calc_features(row):
    # Convert the dict of attributes to a list of integer values
    list_attributes = [
        int(row["Attack"]),
        int(row["Sp. Atk"]),
        int(row["Speed"]),
        int(row["Defense"]),
        int(row["Sp. Def"]),
        int(row["HP"]),
    ]

    # Return the list as an numpy array of shape 6 with int64 datatypes
    return np.array(list_attributes, dtype=np.int64)


def get_indicies(indices_min):
    if len(indices_min[0]) > 1:
        # return indices_min
        possible_indecies = []

        for i in range(len(indices_min[0])):
            possible_indecies.append(indices_min[0][i])

        # check number of occurences for the minimum index for i_1
        min_index = min(possible_indecies)
        occurrences = possible_indecies.count(min_index)

        # if more than one occurence return indecies for min i_2
        if occurrences > 1:
            possible_indecies_2 = []
            for i in range(len(indices_min[0])):
                possible_indecies_2.append(indices_min[1][i])

            min_index_b = min(possible_indecies_2)
            for i in range(len(indices_min[0])):
                if indices_min[1][i] == min_index_b:
                    return [[indices_min[0][i]], [indices_min[1][i]]]
        else:
            # return the row, col of cluster with the min index as first index
            for i in range(len(indices_min[0])):
                if indices_min[0][i] == min_index:
                    return [[indices_min[0][i]], [indices_min[1][i]]]
    else:
        return indices_min


def minimize_distances(
    distances,
    numbered_clusters,
    return_list,
    iteration,
    num_datapoints,
    row_index,
):
    # minimum distance
    min_distance = np.min(distances)
    # row/column of minimum distance
    indices_min = np.where(distances == min_distance)

    indices = get_indicies(indices_min)

    row = indices[0][0]
    col = indices[1][0]

    # calculate repalacement distances
    temp_array = distances.copy()
    for i in range(distances.shape[0]):
        for j in range(0, i):
            temp_array[j][i] = temp_array[i][j]

    values = []
    for i in range(temp_array.shape[0]):
        if i != row and i != col:
            values.append(max(temp_array[row][i], temp_array[col][i]))

    distances = np.delete(distances, row, 0)
    distances = np.delete(distances, col, 0)
    distances = np.delete(distances, row, 1)
    distances = np.delete(distances, col, 1)

    appendColumn = np.full((distances.shape[0], 1), np.inf)
    distances = np.append(distances, appendColumn, 1)

    appendRow = np.full((1, distances.shape[1]), np.inf)
    distances = np.append(distances, appendRow, 0)

    # add values for distance to new cluster
    for i in range(len(values)):
        distances[distances.shape[0] - 1][i] = values[i]

    cntr = 0
    for i in range(len(numbered_clusters)):
        numbered_clusters[i]["distances_index"] = cntr
        cntr += 1

    # Get index of two clusters that are being merged
    to_remove = [None] * 2
    for i in range(len(numbered_clusters)):
        if numbered_clusters[i]["distances_index"] == row:
            to_remove[0] = i
            break
    for i in range(len(numbered_clusters)):
        if numbered_clusters[i]["distances_index"] == col:
            to_remove[1] = i
            break

    # Get the clusters that are being merged
    old_cluster_1 = numbered_clusters[to_remove[0]]
    old_cluster_2 = numbered_clusters[to_remove[1]]

    # Generate the new clusterd
    new_cluster = {
        "index": num_datapoints + row_index,
        "pokemons": old_cluster_1["pokemons"] + old_cluster_2["pokemons"],
        "distances_index": distances.shape[0] - 2,
    }

    numbered_clusters.pop(to_remove[0])
    numbered_clusters.pop(to_remove[1])
    numbered_clusters.append(new_cluster)

    # Update the return list with the new cluster info
    if old_cluster_1["index"] < old_cluster_2["index"]:
        return_list[iteration] = old_cluster_1["index"]
        return_list[iteration + 1] = old_cluster_2["index"]
    else:
        return_list[iteration] = old_cluster_2["index"]
        return_list[iteration + 1] = old_cluster_1["index"]

    return_list[iteration + 2] = min_distance
    return_list[iteration + 3] = new_cluster["pokemons"]

    return distances


def hac(features):
    # linkage function
    # data = linkage(features, "complete")
    # return data

    # number of datapoints
    num_datapoints = len(features)

    numbered_features = []
    for i in range(num_datapoints):
        numbered_features.append(
            {
                "index": i,
                "pokemons": 1,
                "distances_index": i,
            }
        )

    # matrix to hold distances between clusters
    distances = np.full([num_datapoints, num_datapoints], np.inf)

    # result matrix
    clustering = [0] * ((num_datapoints - 1) * 4)

    # fill matrix with inital distances
    for i in range(num_datapoints):
        for j in range(0, i):
            if i != j:
                feature_a = features[i]
                feature_b = features[j]
                distances[i][j] = np.linalg.norm(feature_a - feature_b)

    reference = distances
    minimize_array = reference
    counter = 0
    row_index = 0
    while minimize_array.shape[0] > 2:
        minimize_array = minimize_distances(
            minimize_array,
            numbered_features,
            clustering,
            counter,
            num_datapoints,
            row_index,
        )
        row_index += 1
        counter += 4

    val1 = 0
    val2 = 0

    # for i in range(len(numbered_features)):
    #     if numbered_features[i]["distances_index"] == 0:
    val1 = numbered_features[0]["index"]
    # elif numbered_features[i]["distances_index"] == 1:
    val2 = numbered_features[1]["index"]

    if val1 > val2:
        clustering[len(clustering) - 4] = val2
        clustering[len(clustering) - 3] = val1
    else:
        clustering[len(clustering) - 4] = val1
        clustering[len(clustering) - 3] = val2

    clustering[len(clustering) - 2] = minimize_array[1][0]
    clustering[len(clustering) - 1] = (
        numbered_features[0]["pokemons"] + numbered_features[1]["pokemons"]
    )

    return np.array(clustering).reshape([num_datapoints - 1, 4])


def imshow_hac(Z):
    dendrogram(Z)
    plt.show()


def main():
    imshow_hac(hac([calc_features(row) for row in load_data("Pokemon.csv")][:15]))


if __name__ == "__main__":
    main()
