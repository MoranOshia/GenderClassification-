# # KNN implemantion

from sklearn.neighbors import KNeighborsClassifier

def knn3(x_train, x_test, y_train, y_test):
#knn for k=3
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(x_train,y_train)
    prediction = knn.predict(x_test)
    return knn.score(x_test,y_test)*100




def bestKnn(x_train, x_test, y_train, y_test):

    score_list=[]
    c=0
    bestK=0 
    bestAcc=0
    for each in range(1,100,2):
        knn2=KNeighborsClassifier(n_neighbors=each)
        knn2.fit(x_train,y_train)
        score_list.append(knn2.score(x_test,y_test))
        print("Round: " + str(c) + "   K is now: " + str(each) + "    accuracy: " + str(score_list[c]*100))
        if score_list[c]*100 > bestAcc:
            bestAcc = score_list[c]*100
            bestK = each
        c=c+1

    return bestAcc,bestK

