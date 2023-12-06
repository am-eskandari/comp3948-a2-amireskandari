from sklearn.feature_selection import SelectKBest, chi2


def chi_squares(X, X_scaled, y, k):
    # Show chi-square scores for each feature.
    test = SelectKBest(score_func=chi2, k=k)
    test.fit(X_scaled, y)  # Fit on scaled data

    # Get the mask of the selected features
    mask = test.get_support()  # Boolean mask of selected features

    # Get the names of the selected features from the original DataFrame
    selected_features = X.columns[mask]

    # Print the selected feature names
    print("\nSelected features: " + str(selected_features.tolist()))

    return selected_features
