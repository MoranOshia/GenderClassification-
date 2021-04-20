from sklearn.tree import DecisionTreeClassifier

def dt(x_train, x_test, y_train, y_test):

    dt=DecisionTreeClassifier()
    dt.fit(x_train,y_train)
    return dt.score(x_test,y_test) * 100