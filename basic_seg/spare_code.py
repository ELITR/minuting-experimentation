
# distortions, distances = [], [] 
# K = range(3, 8)
# print(X)
# print(type(X))
# # iterate to find optimal number of clusters (3 - 7)
# for n_c in K:
#   model = KMeans(n_clusters=n_c, init='k-means++', max_iter=100, n_init=1)
#   model.fit(X)
#   # distortions.append(model.inertia_)
#   print(model.cluster_centers_)
#   print(type(model.cluster_centers_))
#   distortions.append(sum(np.min(cdist(np.array(X), model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# kn = KneeLocator(list(K), distortions, S=1.0, curve='convex', direction='decreasing')
# optimum_nc = kn.knee

# # elbow method for finding the optimal number of clusters
# for i in range(0, 5):
#   p1 = Point(initx=1,inity=distortions[0])
#   p2 = Point(initx=5,inity=distortions[4])
#   p = Point(initx=i+1,inity=distortions[i])
#   distances.append(p.distance_to_line(p1,p2))
# optimum_nc = distances.index(max(distances)) + 3


