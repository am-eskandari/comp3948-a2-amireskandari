import numpy as np
from sklearn.feature_selection import SelectKBest, chi2


# def chi_squares(X, y, predictorVariables, k):
#     # Show chi-square scores for each feature.
#     # There is 1 degree freedom since 1 predictor during feature evaluation.
#     # Generally, >=3.8 is good)
#     test = SelectKBest(score_func=chi2, k=k)
#     chiScores = test.fit(X, y)  # Summarize scores
#     np.set_printoptions(precision=3)
#     print("\nPredictor variables: " + str(predictorVariables))
#     print("Predictor Chi-Square Scores: " + str(chiScores.scores_))
#
#     # Another technique for showing the most statistically
#     # significant variables involves the get_support() function.
#     cols = chiScores.get_support(indices=True)
#     print(cols)
#     features = X.columns[cols]
#     # print(np.array(features))
#     selected_feature = np.array(features)
#     return selected_feature

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
