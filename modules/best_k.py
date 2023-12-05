from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def best_k(X_scaled, y):
    # Define the pipeline with feature selection and logistic regression
    pipeline = Pipeline([
        ('select_kbest', SelectKBest(score_func=chi2)),
        ('logistic_regression', LogisticRegression(max_iter=1000, solver='liblinear'))
    ])

    # Define the parameter grid to search over
    param_grid = {
        'select_kbest__k': list(range(1, X_scaled.shape[1] + 1))  # Searching over all possible values of k
    }

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy')

    # Fit the grid search to the data
    grid_search.fit(X_scaled, y)

    # Print the best score found by the grid search
    # print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

    # Print the best k value
    k_value = grid_search.best_estimator_.named_steps['select_kbest'].k
    # print(f"Best k value: {best_k}")

    # Extract the best model (if needed for further processing or analysis)
    # best_model = grid_search.best_estimator_
    return int(k_value)

# Ensure you have the correct best k value


# Call the modified chi_squares function
