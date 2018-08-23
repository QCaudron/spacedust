import os
from collections import defaultdict
from itertools import product, combinations, combinations_with_replacement, chain

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureInteractions(BaseEstimator, TransformerMixin):
    """
    This class generates polynomial features by multiplying features together. The interaction
    of two ( or more ) features results in a feature indicative of whether those features are
    activated at the same time. A self-interaction is just the same feature raised to some power.

    This is basically the same as sklearn.preprocessing.PolynomialFeatures, except that it
    accounts for pd.DataFrame column names and keeps those in a sensible format.
    """

    def __init__(self, degree=2, interaction_only=False, include_bias=True):
        """
        Fix the parameters to the object.
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def fit(self, *args):
        """
        This method is required by the sklearn Pipeline API, but this transformer has
        no internal parameters, so this does nothing.
        """
        return self

    def transform(self, features):
        """
        Generate combinatorial features with interaction.
        """

        # Are we only crossing features or taking self-powers ?
        combine = combinations if self.interaction_only else combinations_with_replacement

        # Generator of all possible combinations of features
        all_combinations = chain.from_iterable(
            combine(features.columns, degree) for degree in range(1, self.degree + 1)
        )

        # Generate all features by crossing them together
        transformed = pd.DataFrame()
        for combination in all_combinations:
            transformed.loc[:, "__x__".join(combination)] = features[list(combination)].prod(1)

        # Generate the bias column if needed
        if self.include_bias:
            transformed.loc[:, "__bias__"] = 1

        return transformed


class Dust(object):
    """
    Main deduplicator class. Inherit from this to get spacedusting.
    """

    # A list of functions used to construct features from pairs of datapoints.
    featureset = None

    # Root of the filename to save the model to
    filename = "spacedust/parent_deduplicator"

    # Machine learning pipeline
    estimators = [
        ("feature_crossing", FeatureInteractions()),
        ("boosted_forest", xgb.XGBClassifier(silent=True, nthread=-1))
    ]

    # Hyperparameters
    parameters = {
        "feature_crossing__degree": [1],
        "feature_crossing__interaction_only": [False],
        "boosted_forest__n_estimators": [100]
    }

    def __init__(self, **kwargs):
        """
        Instantiate empty sparse matrices for the distances and clustering results.
        """

        # Ensure this class is inherited from and the featureset is defined
        if self.featureset is None:
            raise NotImplementedError("You must inherit from this class and generate features !")

        # Save or overwrite any attributes passed in at init time
        for attr, val in kwargs.items():
            setattr(self, attr, val)
        
        # If the machine learning model exists, load it
        if os.path.exists("{}.pkl".format(self.filename)):
            self.model = joblib.load("{}.pkl".format(self.filename))

    def generate_features(self, x1, x2, cross=True):
        """
        Generates a dataframe of features from pairs of datapoints.

        If cross=True, return one row for each pairwise comparison from elements of x1 and x2.
        If cross=False, x1 and x2 must have the same length, and only pair x1[i] with x2[i].
        Typically, cross=True for predicting, returning a (len(x1), len(x2))-shaped array
        of probabilities, whereas cross=False for training, where we have known duplicate labels.
        """

        # Ensure that both x1 and x2 are iterable lists or arrays
        if not isinstance(x1, (list, tuple, np.ndarray)):
            x1 = [x1]
        if not isinstance(x2, (list, tuple, np.ndarray)):
            x2 = [x2]

        # Features will first be held in a dict, then turned into a dataframe
        features = {}
        
        # Select the mode by which we create pairs from the data
        if cross:
            iterator = product
        else:
            iterator = zip

        # Then, for each feature function, generate an array of feature values
        for feature in self.featureset:

            # Initialise with an empty array with the correct shape
            features[feature.__name__] = np.zeros(len(x1) * len(x2) if cross else len(x1))

            # Iterate over pairs and generate that feature for each pair
            for idx, (i, j) in enumerate(iterator(x1, x2)):
                features[feature.__name__][idx] = feature(i, j)

        # Turn features into a DataFrame
        features = pd.DataFrame(features)

        return features, len(x1), len(x2)

    def predict(self, x1, x2):
        """
        Given two lists x1 and x2, predict the probability that each product pair between x1 and x2
        are duplicates of one another.

        x1 and x2 can be either lists or numpy arrays of objects, or single objects.
        """

        # Generate features from the two lists
        features, l1, l2 = self.generate_features(x1, x2)

        # Generate predictions from these features
        probabilities = self.model.predict_proba(features)[:, 1].reshape((l1, l2))
        return np.maximum(probabilities, 1e-15)

    def fit(self, x1, x2, targets, max_candidates=2000):
        """
        Train a deduplication pipeline on known data.

        x1 and x2 are lists of objects to be compared, and y is a boolean array.
        len(x1) == len(x2) == len(y). 
        Elements of these lists correspond to each other, such that x1[i] and x2[i] are two 
        objects to be compared; if they are duplicates, y[i] is True, and otherwise False.

        max_candidates allows you to limit the search through hyperparameter space to a number
        of candidates. 
        """

        # Generate features
        features, *_ = self.generate_features(x1, x2, cross=False)

        # Initialise pipeline and fit to data
        pipeline = Pipeline(self.estimators)

        # Count the total number of parameter combinations
        num_parameter_combinations = np.product([len(p) for p in self.parameters.values()])

        # Select the search strategy according to this nuber
        if num_parameter_combinations > max_candidates:
            cross_val = RandomizedSearchCV(
                pipeline, 
                self.parameters, 
                n_jobs=-1, 
                verbose=1, 
                n_iter=max_candidates,
                cv=5
            )

        else:
            cross_val = GridSearchCV(pipeline, self.parameters, n_jobs=1, verbose=1, cv=5)

        # Fit the pipeline
        cross_val.fit(features, targets)

        # Save the model to disk and to object
        joblib.dump(cross_val.best_estimator_, "{}.pkl".format(self.filename))
        self.model = cross_val.best_estimator_

        # Score the best model by 5-fold cross-validation
        scores = cross_val_score(self.model, features, targets, cv=5, n_jobs=1)
        print("Model score : {} +/- {}".format(scores.mean(), 2 * scores.std()))

    def __repr__(self):
        return "Dust"
