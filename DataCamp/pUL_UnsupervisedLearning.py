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

sample = [  [5., 3.3, 1.4, 0.2], 
            [4., 3.3, 1.4, 0.2],]
            
model = KMeans(n_clusters=3)
