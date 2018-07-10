import os
from collections import defaultdict
from itertools import product, combinations, combinations_with_replacement, chain

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import dok_matrix
from scipy.io import mmwrite, mmread
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin

from django.db.models import QuerySet, Model
from deduplication import transactions_features


class FeatureInteractions(BaseEstimator, TransformerMixin):
    """
    This class generates polynomial features by multiplying features together. The interaction
    of two ( or more ) features results in a feature indicative of whether those features are
    activated at the same time. A self-interaction is just the same feature raised to some power.
    """

    def __init__(self, degree=3, only_interactions=False):
        """
        Fix the parameters to the object.
        """
        self.degree = degree
        self.only_interactions = only_interactions

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
        combine = combinations if self.only_interactions else combinations_with_replacement

        # Generator of all possible combinations of features
        all_combinations = chain.from_iterable(combine(features.columns, degree)
                                               for degree in range(1, self.degree + 1))

        # Generate all features by crossing them together
        transformed = pd.DataFrame()
        for combination in all_combinations:
            transformed["__x__".join(combination)] = features[list(combination)].prod(1)

        return transformed


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    This class returns features that have been normalised ( to the interval [-1, 1] ) or
    standardised ( to zero mean and unit variance ).
    """

    def __init__(self, standardise=[], normalise=[], drop_below=0):
        """
        Fix the parameters to the object.
        """
        self.standardise = standardise
        self.normalise = normalise
        self.drop_below = drop_below

    def fit(self, features, *args):
        """
        Find the minimum, maximum, mean, and standard deviation of the data.
        """

        # Parameters for standardisation and normalisation
        self.min = features.min()
        self.max = features.max()
        self.mean = features.mean()
        self.std = features.std()

        # If we want all features to be normalised or standardised
        if self.standardise is None:
            self.standardise = features.columns
        if self.normalise is None:
            self.normalise = features.columns

        return self

    def transform(self, features):
        """
        Scale features.
        """

        # Define standardisation and normalisation
        def standardised(feature):
            return (features[feature] - self.mean[feature]) / self.std[feature]

        def normalised(feature):
            return (2 * features[feature] - self.max[feature] - self.min[feature]) / \
                   (self.max[feature] - self.min[feature])

        # Make a copy of the input data
        transformed_features = features.copy()

        # Standardise it to zero-mean and unit variance
        for feature in self.standardise:
            transformed_features[feature] = standardised(feature)

        # Normalise to between -1 and 1
        for feature in self.normalise:
            transformed_features[feature] = normalised(feature)

        # Calculate the fraction of missing data per column
        missing_ratio = transformed_features.isnull().sum() / len(transformed_features)

        return transformed_features[missing_ratio[missing_ratio >= self.drop_below].index]


class Deduplicator(object):

    # List of methods from features.py relevant to this particular deduplicator
    featureset = None

    # Root filenames to save data to
    filename = "deduplication/parent_deduplicator"

    # A realtime model does no saving of feature data - it simply computes things on the fly
    realtime = True

    # If true, resets all features and probability values; only relevant for non-realtime models
    clean = False

    # Define machine learning pipeline
    estimators = [
        ("feature_crossing", FeatureInteractions()),
        ("feature_scaling", FeatureScaler(normalise=None)),
        ("boosted_forest", xgb.XGBClassifier(silent=True, nthread=1))
    ]

    # Hyperparameter tuning
    parameters = {
        "feature_crossing__degree": [1],
        "feature_crossing__only_interactions": [False],
        "boosted_forest__n_estimators": [200]
    }

    def __init__(self, **kwargs):
        """
        Instantiate empty sparse matrices for the distances and clustering results.
        """

        # Overwrite any attributes
        for attr, val in kwargs.items():
            setattr(self, attr, val)

        # Ensure this class is inherited and the featureset is defined
        if self.featureset is None:
            raise NotImplementedError("You need to inherit from this class !")

        # If the machine learning model exists, load it
        if os.path.exists("{}__model.pkl".format(self.filename)) and not self.clean:
            self.model = joblib.load("{}__model.pkl".format(self.filename))

        # If the model has feature and probability data on disk
        if not self.realtime:

            # If we want to use this feature data, load the matrices from file
            if os.path.exists("{}_probabilities.mtx".format(self.filename)) and not self.clean:
                self.load()

            # Otherwise, initialise them as large sparse matrices
            else:

                matrix_size = (2**31, 2**31)

                # The various distance matrices
                self.feature_matrices = {feature.__name__: dok_matrix(matrix_size)
                                         for feature in self.featureset}

                # The matrix of pairwise duplicate probabilities
                self.probability_matrix = dok_matrix(matrix_size, dtype=np.float64)

                # The matrix labelling datapoints as user-verified
                self.verified = dok_matrix(matrix_size, dtype=np.float64)

                self.save()

    def __getitem__(self, querysets):
        """
        Signature : d = Deduper(); d[queryset, other_queryset]
        Returns the pairwise probabilities of rows being duplicates.
        Arguments must be either querysets or Django model instances, like a Transaction object.
        This is the workhorse method of this class.
        """

        # Get indices for this queryset
        index_info = self.indices_from_querysets(querysets, return_all=True)
        indices, pairs, queryset_x, queryset_y, primary_keys_x, primary_keys_y = index_info

        # Dict of unknown features
        features = defaultdict(list)

        # Map out indices of the pairs to the objects from their querysets
        x = {element.pk: element for element in queryset_x}
        y = {element.pk: element for element in queryset_y}

        # If we're running in realtime mode, with no stored information
        if self.realtime:

            # Probabilities that objects are duplicates
            probabilities = np.zeros(pairs.shape[0])

            # Calculate the probability that pairs of objects are duplicates
            for idx, unknown_pair in enumerate(pairs):

                # If the objects are the same, they're duplicates
                if unknown_pair[0] == unknown_pair[1]:
                    probabilities[idx] = 1

                # Otherwise, calculate their features and duplicate probabilities
                else:

                    # Begin by calculating all features
                    for feature in self.featureset:
                        features[feature.__name__].append(
                            feature(x[unknown_pair[0]], y[unknown_pair[1]]))

        # If we're not operating in realtime mode, we need to pull up what's already been stored
        else:

            # Pull probabilities from the probability matrix
            probabilities = np.atleast_1d(self.probability_matrix[indices].toarray()[0].squeeze())

            # Find those that have unknown pairwise distance
            unknown_indices = np.where(probabilities == 0)[0]
            unknown_pairs = pairs[unknown_indices]

            # Calculate the pairwise distance for unknowns
            for idx, unknown_pair in zip(unknown_indices, unknown_pairs):

                # Expand indices
                expanded_idx, _ = self.expand(unknown_pair)

                # If a row is being compared to itself, it's a guaranteed duplicate of itself
                if unknown_pair[0] == unknown_pair[1]:
                    probabilities[idx] = 1

                else:  # Otherwise, calculate each feature, store in the relevant matrix
                    for feature in self.featureset:

                        # Store the features in the feature matrices
                        features[feature.__name__].append(
                            feature(x[unknown_pair[0]], y[unknown_pair[1]]))
                        self.feature_matrices[feature.__name__][expanded_idx] = \
                            features[feature.__name__][-1]
                        self.feature_matrices[feature.__name__][expanded_idx[::-1]] = \
                            features[feature.__name__][-1]  # store the reverse comparison toofeatca

        # Calculate the probability that elements are duplicates for unknown pairs, if any
        if len(features):
            calculated_probabilities = self.calculate_probability(features)

            # Allocate results to the probabilities array
            # Everything that isn't a zero at this point is a one due to being the same object pair
            unknown_indices = np.where(probabilities == 0)[0]
            for i, j in enumerate(unknown_indices):
                probabilities[j] = calculated_probabilities[i]

        # If we're saving features and probabilities, update the probability matrix and save
        if not self.realtime:
            for idx, pair in enumerate(pairs):
                self.probability_matrix[self.expand(pair)[0]] = probabilities[idx]
                self.probability_matrix[self.expand(pair)[0][::-1]] = probabilities[idx]  # reverse
            self.save()

        # Ultimately, return the probabilities as an array the same shape as the original query
        return probabilities.reshape((len(queryset_x), len(queryset_y)))

    def indices_from_querysets(self, querysets, return_all=False):
        """
        Return a set of linear indices and pairs of indices from a query.
        """

        # Cast the query, be it an individual model, a list, or a queryset, to an iterable.
        def cast_queryset(queryset):
            if isinstance(queryset, QuerySet):
                queryset = list(queryset)
            elif isinstance(queryset, Model):
                queryset = [queryset]
            return queryset

        queryset_x = cast_queryset(querysets[0])
        queryset_y = cast_queryset(querysets[1])

        # Pull the primary keys of the rows passed, as lists; fix order of the querysets
        primary_keys_x = [element.pk for element in queryset_x]  # slower than .values_list
        primary_keys_y = [element.pk for element in queryset_y]  # but the ordering is fixed

        # Expand indices for linear access of the sparse matrices
        indices, pairs = self.expand((primary_keys_x, primary_keys_y))

        if return_all:
            return indices, pairs, queryset_x, queryset_y, primary_keys_x, primary_keys_y
        return indices, pairs

    def expand(self, idx):
        """
        Turns Python's standard 2D slice notation, such as [(1, 3, 8), 4:6], into two tuples
        of individual coordinates, ensuring all pairwise combinations are generated.
        Now with more integer, slicing, or tuple indexing action !
        TL; DR : turns [(1, 3, 8), 4:6] into ([1, 1, 3, 3, 8, 8], [4, 5, 4, 5, 4, 5]).
        """

        # Expand out slice and tuple notation to a coordinate system
        square_coords = [np.r_[idx[0]], np.r_[idx[1]]]

        # Take the cross product between square coordinates to obtain individual matrix elements
        cross_product = list(product(*square_coords))

        # Generate linear coordinates to index numpy arrays
        coords = [c[0] for c in cross_product], [c[1] for c in cross_product]

        return coords, np.array(cross_product)

    def calculate_probability(self, features):
        """
        Machine learning ! Predict the probability that given indices are duplicates.
        """

        if len(list(features.values())[0]) == 1:
            features = pd.DataFrame(features, index=[0])
        else:
            features = pd.DataFrame(features)

        # Predict the probability of this entry being a duplicate
        probability = self.model.predict_proba(features)[:, 1]

        # Return the probability, with a minimum of epsilon to avoid recalculating this
        return np.maximum(probability, 1e-15)

    def save(self):
        """
        Save the feature matrices to disk.
        """

        # Do nothing for realtime models
        if self.realtime:
            return

        # Write the probabilities matrix
        mmwrite("{}_probabilities".format(self.filename), self.probability_matrix)

        # Write the verification labels
        mmwrite("{}_verified".format(self.filename), self.verified)

        # Write all the feature matrices
        for name, matrix in self.feature_matrices.items():
            mmwrite("{}_feature__{}".format(self.filename, name), matrix)

    def load(self):
        """
        Load the feature matrices from file.
        """

        # Do nothing for realtime models
        if self.realtime:
            return

        # Read in the probabilities matrix
        self.probability_matrix = dok_matrix(mmread("{}_probabilities.mtx".format(self.filename)))

        # Read in the verification labels
        self.verified = dok_matrix(mmread("{}_verified.mtx".format(self.filename)))

        # Read in all the feature matrices
        self.feature_matrices = {}
        for feature in self.featureset:
            self.feature_matrices[feature.__name__] = dok_matrix(
                mmread("{}_feature__{}.mtx".format(self.filename, feature.__name__)))

        # Read in the machine learning model
        self.model = joblib.load("{}__model.pkl".format(self.filename))

    def train(self, x1, x2, targets, max_candidates=2000, real_data=False):
        """
        Train a machine learning pipeline on some known data.
        x1 and x2 are lists of transactions, and y is a boolean array.
        len(x1) == len(x2) == len(y) must be true. Elements of these lists correspond to each
        other, such that x1[i] and x2[i] are two transactions which are duplicates if and only
        if y[i] is True, and they are not duplicatees if y[i] is False.
        If update_feature_matrix is True, then the data passed into training is considered real
        data, and the objects are treated as real database items : they are added to the feature
        matrices, to the probability matrix, and are listed as verified entries. If False, passed
        data is considered fake data, good for model training but not to be kept afterwards.
        """

        # Initialise empty distance lists
        feature_dict = {feature.__name__: [] for feature in self.featureset}

        # Calculate features for each object pair
        for a, b, is_duplicate in zip(x1, x2, targets):
            for feature in self.featureset:
                feature_value = feature(a, b)
                feature_dict[feature.__name__].append(feature_value)

                # Update data matrices if we're not running in realtime mode and the data is real
                if not self.realtime and real_data:
                    idx = self.expand((a.pk, b.pk))[0]
                    self.distance_matrices[feature.__name__][idx] = feature_value
                    self.probability_matrix[idx] = is_duplicate
                    self.verified[idx] = 1  # entry is verified, not predicted

        # Cast as a dataframe for passing to the pipeline object
        training_set = pd.DataFrame(feature_dict)

        # Initialise pipeline and fit to data
        pipeline = Pipeline(self.estimators)

        # Select the search strategy
        num_parameter_combinations = np.product([len(p) for p in self.parameters.values()])
        if num_parameter_combinations > max_candidates:
            cross_val = RandomizedSearchCV(
                pipeline, self.parameters, n_jobs=-1, verbose=1, n_iter=max_candidates)
        else:
            cross_val = GridSearchCV(pipeline, self.parameters, n_jobs=-1, verbose=1)

        # Fit the pipeline
        cross_val.fit(training_set, targets)

        # Save the model to disk and to object
        joblib.dump(cross_val.best_estimator_, "{}__model.pkl".format(self.filename))
        self.model = cross_val.best_estimator_

        # Score the best model by 5-fold cross-validation
        scores = cross_val_score(self.model, training_set, targets, cv=5, n_jobs=-1)
        print("Model score : {} +/- {}".format(scores.mean(), 2 * scores.std()))

    def __repr__(self):
        return "Deduplicator"


class TransactionsDeduplicator(Deduplicator):

    # Set the filename and set the model
    filename = "deduplication/transactions_deduplicator"

    # Define the set of features
    featureset = [
        transactions_features.transaction_type,
        transactions_features.contact_name_full,
        transactions_features.contact_name_partial,
        transactions_features.client_name_full,
        transactions_features.client_name_partial,
        transactions_features.client_name_tokenset,
        transactions_features.client_address_full,
        transactions_features.client_address_partial,
        transactions_features.client_address_tokenset,
        transactions_features.transaction_managing_office_equal,
        transactions_features.date_distance
    ]

    # Define hyperparameters
    parameters = {
        # Feature interactions
        "feature_crossing__degree": [1, 2],
        "feature_crossing__only_interactions": [True, False],

        # Boost classification forest
        "boosted_forest__n_estimators": [50, 200, 500, 1000],
        "boosted_forest__max_depth": [2, 4, 7],
        "boosted_forest__learning_rate": np.logspace(-5, -1, 4),
        "boosted_forest__subsample": [0.6, 0.8, 1],
        "boosted_forest__reg_alpha": np.logspace(-3, 0, 4),
        "boosted_forest__reg_lambda": np.logspace(-3, 0, 4)
    }               
