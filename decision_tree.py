import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import os

#load the data
path = os.path.abspath(os.path.dirname(__file__)) + "\dummy_data_large_v2.csv"
data = pd.read_csv(path, sep = ';')
#split the data in train and test set
Xs = data.drop(['SuccessfulCall'], axis=1)
y = data['SuccessfulCall']
X_train, X_test, y_train, y_test = train_test_split(Xs.values, y.values, test_size = 0.2, random_state = 0)
#decision tree using gini criterion
dtc = DecisionTreeClassifier(criterion = 'gini')
clf = dtc.fit(X_train,y_train)
#accuracy score on train and test set
clf.score(X_train, y_train)
clf.score(X_test, y_test)

#calculate accuracy for varying depth of the tree and see how it changes
""" dtc = DecisionTreeClassifier()
scores = []

for dep in range(1,30):
    
    dtc = DecisionTreeClassifier(max_depth = dep)
    clf = dtc.fit(X_train,y_train)
    scores.append([dep, clf.score(X_test, y_test)])

print(scores) """

""" plt.figure(figsize=(30,10), facecolor ='k')
a = tree.plot_tree(clf,
                   feature_names = ['type','WeekDay','StartTime','SuccessfulCall'],
                   class_names = ['type','WeekDay','StartTime','SuccessfulCall'],
                   rounded = True,
                   filled = True,
                   fontsize=14)
plt.show() """


tempHeatMapArray = np.zeros((24,7))
print(tempHeatMapArray)
for hour in range(24):
    for weekday in range(7):
        tempHeatMapArray[hour][weekday] = clf.predict([[0,weekday,hour]])

timesLabel = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00','08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']
daysLabel = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
ax = sns.heatmap(tempHeatMapArray, linewidth = 0.5 , cmap = 'coolwarm' )
plt.title( "Chance of successful calls on outbound calls" )
plt.ylabel('time')
plt.xlabel('day of week')
plt.yticks(ticks=range(24), labels=timesLabel)
plt.xticks(ticks=range(7), labels=daysLabel)
plt.show()

#plot confusion matrix on predictions
""" dtc = DecisionTreeClassifier(criterion = 'entropy')
clf = dtc.fit(X_train,y_train)
y_pred = clf.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {:.3f}'.format(clf.score(X_test,y_test)) 
plt.title(all_sample_title, size = 15)
plt.show() """