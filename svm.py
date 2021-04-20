from sklearn.svm import SVC


def svm_(x_train, x_test, y_train, y_test):
    svm=SVC(random_state=1) 
    svm.fit(x_train,y_train)
    acc_svm = svm.score(x_test,y_test) * 100
    return acc_svm


def svmLinear(x_train, x_test, y_train, y_test):
    svm1 = SVC(kernel='linear')
    svm1.fit(x_train, y_train)
    acc_linear = svm1.score(x_test, y_test) * 100
    return acc_linear


def svmPoly(x_train, x_test, y_train, y_test):
    svm2 = SVC(kernel='poly')
    svm2.fit(x_train, y_train)
    acc_poly = svm2.score(x_test, y_test) * 100
    return acc_poly


def svmRbf(x_train, x_test, y_train, y_test):
    svm3 = SVC(kernel='rbf')
    svm3.fit(x_train, y_train) 
    acc_rbf = svm3.score(x_test, y_test) * 100
    return acc_rbf
    