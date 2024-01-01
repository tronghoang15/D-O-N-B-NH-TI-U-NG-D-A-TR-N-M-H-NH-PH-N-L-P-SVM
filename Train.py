from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from preprocessing import dienkhuyetthieu,chuanhoa
import pandas as pd
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv("data/diabetes.csv")
dienkhuyetthieu(df)
X = chuanhoa(df)
y=df["Outcome"]
# tạo tệp train, test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y, random_state=2)

# tạo mô hình SVM
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,y_train)

# Dự đoán kết quả trên tập train
X_train_prediction = classifier.predict(X_train)
training_data_accuracy_score = accuracy_score(y_train, X_train_prediction)

# Tạo biểu đồ tròn
labels = ['Correct Predictions', 'Incorrect Predictions']
sizes = [training_data_accuracy_score, 1 - training_data_accuracy_score]
colors = ['green', 'red']
explode = (0.1, 0)

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Training Data Accuracy')
plt.show()

# Dự đoán kết quả trên tập test
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy_score = accuracy_score(y_test, X_test_prediction)

# Tạo biểu đồ tròn
labels = ['Correct Predictions', 'Incorrect Predictions']
sizes = [testing_data_accuracy_score, 1 - testing_data_accuracy_score]
colors = ['blue', 'red']
explode = (0.1, 0)

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Training Data Accuracy')
plt.show()

# lưu file train
filename="diabetesmodel.sav"
pickle.dump(classifier,open(filename,'wb'))