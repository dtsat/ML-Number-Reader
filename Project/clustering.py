#!/usr/bin/env python3

import os

#Importing numpy
import numpy as np

# Feature extractors
from svm import SVMFeatureExtractor
from three_by_three import ThreeByThreeFeatureExtractor
from gradient import GradientFeatureExtractor

# Image loader and cross-validater
from importer import Importer

# Library to create math plots and graphs. Can be used to show images
from matplotlib import pyplot as plt

#Principal Component Analysis to reduce the number of dimensions for plots and get representation of data
from sklearn.decomposition import PCA

# KMeans cluster
from sklearn.cluster import KMeans

#Affinity Propagation cluster
from sklearn.cluster import AffinityPropagation as AP

#Birch cluster
from sklearn.cluster import Birch


from itertools import cycle

class Clustering:
    """
    This class creates clusters out of the fitted images extracted using the Three by Three feature extractor
    """

    def __init__(self, crossValidate=False):
        """
        Parameters:
            -- crossValidate: randomly select images for learn, validate, and test sets
        """

        self.path = os.getcwd()
        #used to retrieve image file name for cluster center
        self.images = []
    
        # Run tests using 3 methods of feature extraction
        self.load_sets(crossValidate)
        self.__kmeans()
        self.__birch()
        self.__affinityPropagation()

        # self.load_sets('gradient', crossValidate)
        # self.__kmeans()

        # self.load_sets('svm', crossValidate)
        # self.__kmeans()

    def load_sets(self, crossValidate=False):
        """
        Parses the images for features and loads the descriptors
        using the specified method of extraction.

        Parameters:
            -- which: `svm`, `kmeans`, or `three`
            -- crossValidate: Set by the constructor
        """
        
        self.extractor = ThreeByThreeFeatureExtractor()
        self.__crossValidate = crossValidate
        self.__load_one_descriptor()
        # self.__which = which

    def __load_one_descriptor(self):
        self.learn = [[],[],[]]
        # Load images for learn and validate sets
        # TODO: Load and use test set
        importer = Importer('images/4', self.__crossValidate)
        for image in importer.learn:
            self.learn[0].append(self.extractor.run(image).data)
            self.learn[1].append(self.extractor.run(image).data)
            self.learn[2].append(self.extractor.run(image).data)
        for value in importer.grabImagesName:
            for key, val in value.items():
                img = {}
                img[key] = value
                self.images.append(img)

    def __kmeans(self):
        print("KMEANS Clustering on PCA-reduced data")
        reduced_data = PCA(n_components=2).fit_transform(self.learn[0])
        cluster = KMeans(init='k-means++', random_state=22)
        result = cluster.fit(reduced_data)

        self.kmeansData = reduced_data
        self.kmeansResult = result

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max]self.

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = cluster.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

        # Plot the centroids as a white X
        centroids = result.cluster_centers_
        print (centroids)
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        plt.title('K-means clustering on the 9 dataset (PCA-reduced data)\n'
                  'Centroids are marked with white cross')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        path = self.path + '/ClusteringImages/KMeans/Number9.png'
        plt.savefig(path)
        # plt.show()

    def __birch(self):
        print("Birch Clustering on PCA-reduced data")
        reduced_data = PCA(n_components=2).fit_transform(self.learn[1])
        cluster = Birch()
        result = cluster.fit(reduced_data)

        self.birchResult = result
        self.birchData = reduced_data

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max]self.

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = cluster.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

        plt.title('Birch clustering on the 9 dataset (PCA-reduced data)')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        path = self.path + '/ClusteringImages/Birch/Number9.png'
        plt.savefig(path)
        # plt.show()

    def __affinityPropagation(self):
        print("Affinity Propagation Clustering on PCA-reduced data")
        reduced_data = PCA(n_components=2).fit_transform(self.learn[2])
        af = AP().fit(reduced_data)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_

        #print the image file names in the terminal
        for obj in self.images:
            for key,value in obj.items():
                if (value[key] in cluster_centers_indices):
                    print("cp " + key + " ../APClustersImagesFile/")

        self.affPropResult = af
        self.affPropData = reduced_data

        n_clusters_ = len(cluster_centers_indices)

        plt.figure(1)
        plt.clf()

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            class_members = labels == k
            cluster_center = reduced_data[cluster_centers_indices[k]]
            plt.plot(reduced_data[class_members, 0], reduced_data[class_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)
            for x in reduced_data[class_members]:
                plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

        plt.title('Affinity Propagation clustering on the 9 dataset (PCA-reduced data)')
        path = self.path + '/ClusteringImages/AffinityPropagation/Number9.png'
        plt.savefig(path)
        # plt.show()





if __name__ == "__main__":
    c = Clustering()