import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


from matplotlib import pyplot as plt

from progress_bar import *

halos = pd.read_csv("halos_999.csv", index_col=0)

data = halos.drop(["ID", "Rvir", "Mass"], axis=1)

scaler = StandardScaler()
data[["X_T", "Y_T", "Z_T", "VX_T", "VY_T", "VZ_T"]] = scaler.fit_transform(halos[["X", "Y", "Z", "VX", "VY", "VZ"]])


# Optimum number of cluster?

def optimise_k_means(df, max_k):
    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)

        means.append(k)
        inertias.append(kmeans.inertia_)
        progressbar(k, max_k)

    figure = plt.subplots(figsize=(10, 5))
    plt.plot(means, inertias, "o-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.grid()
    plt.show()


optimise_k_means(data[["X_T", "Y_T", "Z_T", "VX_T", "VY_T", "VZ_T"]], 20)

n = int(input("Enter the number of clusters : "))

kmeans = KMeans(n_clusters=n)

# We need the 6D phase space, since some halos may be dynamically bound, but may be more separated in space
kmeans.fit(data[["X_T", "Y_T", "Z_T", "VX_T", "VY_T", "VZ_T"]])

data["kmeans_n"] = kmeans.labels_

# Plot the position of the halos, classified by cluster number
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.cla()
ancho = 1500

sc = ax.scatter(data["X"], data["Y"], data["Z"], c=data["kmeans_n"], s=100 * halos["Rvir"] / np.max(halos["Rvir"]),
                alpha=0.5, cmap="jet")
plt.tick_params(labelsize=8)
cb = plt.colorbar(sc)

ax.set_ylabel('Y (kpc)', fontsize=10)
ax.set_xlabel('X (kpc)', fontsize=10)
ax.set_ylim(-ancho, ancho)
ax.set_xlim(-ancho, ancho)

plt.show()

#TODO: testing. Classification in virial radius for minor sub halos which could be dynamically bound