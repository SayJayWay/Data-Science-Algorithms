# Simple k-means clustering model
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    """ performs k-means clustering"""
    def __init__(self,k):
        self.k = k          # number of clusters
        self.meas = None    # means of clusters

    def classify(self, input):
        """ return the index of the cluster closest to the input"""
        return min(range(self.k),
                   key = lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs):
        # choose k random points as the initial means
        self.means = random.sample(inputs, self.k)
        assignments = None

        while True:
            # Find new assignments
            new_assignments = map(self.classify, inputs)

            # If no assignments have changed, we're done.
            if assignments == new_assignments:
                return

            # Otherwise keep the new assignments,
            assignments = new_assignments

            # And compute new means based on the new assignments
            for i in range(self.k):
                # Find all the points assigned to cluster i
                i_points = [p for p, a in zip(inputs, assignments) if a==i]

                # make sure i_points is not empty so don't divide by 0
                if i_points:
                    self.means[i] = np.mean(i_points)

""" Implementation is simple:
cluster = Kmeans(3) # of any appropriate number of k-means
cluster.train(inputs)
print cluster.means # these are the location/points of the means (i.e. the center of the clusters)
"""

# If don't know what k to choose, we can plot the sum of squared errors between each point and the mean
# of its cluster as a function of k and see where the graph "bends"

def squared_clustering_errors(inputs, k):
    """ finds the total squared error from k-means clustering the inputs"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = map(clusterer.classify, inputs)

    return sum(np.dot(np.subtract(input,means[cluster],np.subtract(input,means[cluster])))
               for input, cluster in zip(inputs, assignments))

    # plot from 1 up to len(inputs) clusters
    ks = range(1,len(inputs) + 1)
    errors = [squared_clustering_errors(inputs,k) for k in ks]

    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel("k")
    plt.ylabel("total squared error")
    plt.title("Total Error VS. # of Clusters")
    plt.show()

""" Example: Reducing number of colours in an image
The idea here is to group similar colours into k-means (here 5 means) to reduce the amount of colours

path_to_img = ________ # Add your image path here
import matplotlib.image as mpimg
img = mpimg.imread(path_to_img)

# Note: each pixel is a list [red,green,blue] indicating the color of that pixel where the
        pixel location is: img[i][j]

pixels = [pixel for row in img for pixel in row]
clusterer = KMeans(5)
clusterer.train(pixels)

# Now we can reconstruct a new image:
def recolor(pixel):
    cluster = clusterer.classify(pixel)      # index of the closets cluster
    return clusterer.means[cluster]          # mean of the clsoets cluster

new_img = [[recolor(pixel) for pixel in row] # recolor this row of pixels
            for row in img]                  # for each row in the image

plt.imshow(new_img)
plt.axis('off')
plt.show()

    

