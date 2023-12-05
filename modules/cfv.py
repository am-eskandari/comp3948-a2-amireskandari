from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def crossfold_validation(X_scaled, y):
    # Create a logistic regression model
    model = LogisticRegression(max_iter=1000)

    # Perform 5-fold cross-validation with the scaled data
    scores = cross_val_score(model, X_scaled, y, cv=10, scoring='accuracy')

    # Print the accuracy for each fold
    print(scores)

    # Print the mean accuracy of the folds
    print('Mean accuracy:', scores.mean())

    # Print the standard deviation of the accuracy
    print('Accuracy standard deviation:', scores.std())
