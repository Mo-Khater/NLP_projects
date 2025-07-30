
def train_multinomial_nb(X_train, y_train):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model