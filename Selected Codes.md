CORRELATION BETWEEN PREDICTORS
sns.set(style="white")

# Compute the correlation matrix
corr = loan_data.iloc[:,:].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(6, 5))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot= True);
plt.xticks(rotation=45)
plt.title("CROSS CORRELATION BETWEEN PREDICTORS", fontsize=12)
plt.show()
 # Model Building
Using Stratified K-Fold Cross Validation
from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_index, test_index in folds.split(X,Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]z

# Setup arrays to store training and test accuracies
neighbors = np.arange(1,9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Fit the model
    knn.fit(X_train, Y_train)
    
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, Y_train)
    
    # Compute accuracy on test set
    test_accuracy[i] = knn.score(X_test, Y_test)
    
    plt.title('K-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.show()
We can observe we are getting maximum testing accuracy when k=5. So lets create a KNeighorsClassifier with number of neigbors as 5.

# Setting up Knn classifier with K neighbors as 5
knn = KNeighborsClassifier(n_neighbors=5)
# Fit the model
knn.fit(X_train, Y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
# Getting accuracy
knn.score(X_test, Y_test)
0.7540983606557377

# Confusion Matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
Y_pred = knn.predict(X_test)
confusion_matrix(Y_test, Y_pred)
array([[10, 28],
       [ 2, 82]], dtype=int64)
# Classification Report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))
              precision    recall  f1-score   support

           0       0.83      0.26      0.40        38
           1       0.75      0.98      0.85        84

    accuracy                           0.75       122
   macro avg       0.79      0.62      0.62       122
weighted avg       0.77      0.75      0.71       122

# ROC ( Reciever Operating Characteristic) Curve
It is the plot of the true positive rate against the false positive rate for the different possible cutpoints of a diagnostic test.

Y_pred_proba = knn.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve
fpr, tpr, thresholds  = roc_curve(Y_test, Y_pred_proba)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=5) ROC Curve')
plt.show()
# Area under curve
from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test, Y_pred_proba)
score = 0.7196115288220553
# K-Means Clustering
In this method, the user has to define the number of clusters. The clusters are formed based on the closeness to the center value of the clusters. Here, we decide the number of clusters(k) and then group the data points into "k" clusters.

The method which allows us to seperate the data into groups and so to create clusters with similar values. To define "K" the Elbow graph will be used.

from sklearn.cluster import KMeans
value= range(1,10)
kmeans = [KMeans(n_clusters=i) for i in value]

score = [kmeans[i].fit(happiness_data.iloc[:,2:]).score(happiness_data.iloc[:,2:]) for i in range(len(kmeans))]
plt.figure(figsize=(8,5))
plt.plot(value, score)
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.title("Elbow Curve")
plt.show()
We are observing multiple elbows here, i.e K=2 and K=3. However, we can consider either of the elbows. But, we will be exploring another method to get a better understanding in choosing the elbow.

# Silhouette Analysis
Silhouette Analysis is a way to measure how close each point in a cluster is to the points in its neighbouring clusters. It is a neat way to find out the optimum value for k during k-means clustering. Silhouette values lies in the range of [-1,1]. A value of +1 indicates that the sample is far away from its neighboring cluster and very close to the cluster its assigned. Similarly, value of -1 indicates that the point is close to its neighboring cluster than the cluster its assigned. And, a value of 0 means its at the boundary of the distance between the two cluster. Value of +1 is ideal and -1 is least preferred. Hence, higher the value better is the cluster configuration.
from sklearn.metrics import silhouette_score
# Use silhouette score
n_clusters = list(range(2,4))
print("Number of clusters from 2 to 4: \n", n_clusters)

for clusters in range(2,4):
    clusterer = KMeans(n_clusters = clusters, random_state=0)
    preds = clusterer.fit_predict(happiness_data.iloc[:,2:])
    centers = clusterer.cluster_centers_
    
    score = silhouette_score(happiness_data.iloc[:,2:], preds, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {}".format(clusters,score))
Number of clusters from 2 to 4: 
 [2, 3]
For n_clusters = 2, silhouette score is 0.3482961119644963
For n_clusters = 3, silhouette score is 0.30320523814626266
For are observing that cluster 2 has maximum silhouette score. Hence, we will be grouping data into 2 clusters.

# Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
# Model Fitting
model.fit(X_train, y_train)
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
predicted = model.predict(X_test)
# max_depth of tree
Let us consider max_depth from 0 to 100.

from sklearn.metrics import accuracy_score
max_depths = np.linspace(1,100,100, endpoint=True)

train_results= []
test_results = []

for max_depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)
    
    train_pred = dt.predict(X_train)
    
    acc_score_train = accuracy_score(y_train,train_pred)
    
    # Add accuracy score to train results
    train_results.append(acc_score_train)
    y_pred = dt.predict(X_test)
    
    acc_score_test = accuracy_score(y_test, y_pred)
    # Add accuracy score to test results
    test_results.append(acc_score_test)
    
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_depths, train_results, 'b', label='Train data')
line2, = plt.plot(max_depths, test_results, 'r', label='Test data')

plt.legend(handler_map = {line1 : HandlerLine2D(numpoints=2)})

plt.ylabel("Accuracy Score")
plt.xlabel("Size of tree(Number of nodes)")
plt.show()
From the above results , we see that the test accuracy is declining as the depth size is increasing so can say they the model is overfitting as the depth of the tree increases.
    
