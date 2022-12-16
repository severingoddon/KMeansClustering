from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt

# Folder Path
path = "/Users/sevi/Desktop/Docs/PycharmProjects/kmeansTextClustering/documents"
os.chdir(path)

documents = []

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        documents.append(f.read())


for file in os.listdir():
    if file.endswith(".txt"):
        file_path = f"{path}/{file}"
        read_text_file(file_path)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

clustersCount = 3
model = KMeans(n_clusters=clustersCount, init='random', max_iter=10000, n_init=1)
model.fit(X)

centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(clustersCount):
    print("Cluster %d:" % i),
    for ind in centroids[i, :10]:
        print(' %s' % terms[ind]),
    print()

print("\n")
print("Prediction")


plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
#plt.show()

Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

test_file = open("/Users/sevi/Desktop/Docs/PycharmProjects/kmeansTextClustering/testing/Hoopoe.txt", "r")
testData = test_file.read()
test_file.close()


Y = vectorizer.transform([testData])
prediction = model.predict(Y)
print(prediction)