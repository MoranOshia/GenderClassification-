from sklearn.linear_model import LogisticRegression


def lr(x_train, x_test, y_train, y_test):
    lr=LogisticRegression()
    lr.fit(x_train,y_train)
    return lr.score(x_test,y_test) * 100
    
