# *************************************************
# Unsupervised learning PYTHON 
#   finds patterns in data Eg. clustering customers by their purchases
#       or compressning data using puchase patterns (dimmersion reduciton)
# *************************************************

#Iris dataset PetalLentgh, PetalWidh, SepalLength, SepalWidth
#   columns - measurements - features 4Dim
#   rows - iris plants - samples
#   k-means sklearn 
from sklearn.cluster import KMeans

samples = [  [5., 3.3, 1.4, 0.2], 
            [4., 3.3, 1.4, 0.2],]

model = KMeans(n_clusters=3)
model.fit(samples)
labels = model.predict(samples)
print(labels)

# visualize in scatterplot
import matplotlib.pyplot as plt
xs = samples[:,0]
ys = samples[:,1]
plt.scatter(xs,ys, c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1] # Diamon market
plt.scatter(centroids_x,centroids_y, marker='D', s=50)
plt.show()

#Cross tabulations - when cluster mix different species 

# inertia measures 


#standard scaler - normalization 0-1 - 
# others: MaxAbsScaler and Normalizer
# 2 steps - standardscaler and kmeans => combine two steps with a pipeline 
