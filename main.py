import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from svm import svm_,svmLinear,svmPoly,svmRbf
from knn import knn3,bestKnn
from logisticregression import lr
from cnn import cnn_,cnn2D
from decisiontree import dt

def fill_data():
    # Read all the info from the CSV
    data=pd.read_csv("voice.csv")
    # change 0 for female and 1 for male
    data.label=[0 if each=="female" else 1 for each in data.label]
    return data
    
    
def shuffle_data(data):
    y=data.label.values
    x_data=data.drop(["label"],axis=1)
    
    # normalization
    # (x-max)/(max-min)
    x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.3)

    scaler = StandardScaler()
    x_cnn2d = scaler.fit_transform(x_data)
    
    return x_train,x_test,y_train,y_test,x,x_cnn2d,y




if __name__ == '__main__':
    data = fill_data()
    # 0.svm regular 1.svm Linear 2.svm Poly 3.svm Rbf 4.knn3 5.logistic regression 6.decision tree 
    arr_sum = [0] * 7
    for i in range(100):
        # Split dataset into training set and test set
        x_train,x_test,y_train,y_test,x,x_cnn2d,y = shuffle_data(data) # 80% training and 20% test
        # arr_sum[0] += svm_(x_train,x_test,y_train,y_test)
        # arr_sum[1] += svmLinear(x_train,x_test,y_train,y_test)
        # arr_sum[2] += svmPoly(x_train,x_test,y_train,y_test)
        # arr_sum[3] += svmRbf(x_train,x_test,y_train,y_test)
        # arr_sum[4] += knn3(x_train,x_test,y_train,y_test)
        # arr_sum[5] += lr(x_train,x_test,y_train,y_test)
        # arr_sum[6] += dt(x_train,x_test,y_train,y_test)
      
    
    
    # print("________________________________SVM________________________________\n")
    # print("Svm Accuracy: ",arr_sum[0]/100)
    # print("Linear Svm Accuracy: ",arr_sum[1]/100)
    # print("Poly Svm Accuracy: ",arr_sum[2]/100)
    # print("Rbf Svm Accurancy: ",arr_sum[3]/100)
    # print("___________________________________________________________________\n")
    
    # print("________________________________KNN [k=3]________________________________\n")
    # print("Knn [k=3] Accuracy: ",arr_sum[4]/100)
    # print("_________________________________________________________________________\n")
    # print("________________________________KNN - Find best K ___________________________________\n")
    # acc,k=bestKnn(x_train, x_test, y_train, y_test)
    # print("Knn Accuracy: " , acc , " With k: " , k)
    # print("______________________________________________________________________\n")

    # print("________________________________Logistic Regression________________________________\n")
    # print("Logistic Regression Accuracy: ",arr_sum[5]/100)
    # print("___________________________________________________________________________________\n")

    # print("________________________________Decision Tree________________________________\n")
    # print("Decision Tree Accuracy: ",arr_sum[6]/100)
    # print("_____________________________________________________________________________\n")
    
    print("________________________________CNN________________________________\n")
    cnn_(x_train,x_test,y_train,y_test,x)
    print("___________________________________________________________________\n")
    print("________________________________CNN 2D________________________________\n")
    cnn2D(x_train,x_test,y_train,y_test, x_cnn2d,y)
    print("______________________________________________________________________\n")