spacedust
*********

Sprinkle a little cosmic dust on your data.


Pairwise object deduping
########################


**spacedust** is a convenience interface around ``xgboost`` and ``sklearn``, providing an API for building out pairwise deduplication models. It works like this.

Data
#####

You have two lists of data, ``x1`` and ``x2``. For each element in these, you know whether those two are duplicates of one another. Let's keep this information in ``y``, another list of the same length.

For example, you have a list of known addresses :

.. code-block:: python

    addresses = ["123 Main Street", "420 5th Ave", "123, Main St"]


You know that the first and third elements refer to the same place, whereas the second element is a distinct address. You might build up the lists :

.. code-block:: python

    x1 = [
        "123 Main Street",
        "123 Main Street",
        "123 Main Street",
        "420 5th Ave",
        "420 5th Ave",
        "420 5th Ave",
        "123, Main St",
        "123, Main St",
        "123, Main St"
    ]

    x2 = [
        "123 Main Street",
        "420 5th Ave",
        "123, Main St",
        "123 Main Street",
        "420 5th Ave",
        "123, Main St",
        "123 Main Street",
        "420 5th Ave",
        "123, Main St"
    ]

    y = [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True
    ]


How do you build this list ? Pairwise deduping is firmly a *supervised learning* task, which means we need labelled data. "Labelled" means that we have a number of examples and we know whether they are duplicates or not. Often, that might mean some human time spent labelling stuff, which isn't fun but is unfortunately necessary.


Features
########

**spacedust** compares pairs of datapoints, and generates *feature values* for each pair,
based on what you tell it to look for. For example, you might look at this data and say, 
"well if the street number is the same, that's a good indicator ( although not a guarantee ) 
that these are the same places". So, you might come up with a feature like this :

.. code-block:: python

    def street_number_is_same(first, second):
        """Compares the street number and returns True if they're identical. Removes commas."""
        return first.split(" ")[0].replace(",", "") == second.split(" ")[0].replace(",", "")


Then, you might look at something that isn't boolean, maybe just a Levenshtein distance, using the ``fuzzywuzzy`` package.

.. code-block:: python

    from fuzzywuzzy import fuzz

    def street_name_is_same(first, second):
        return fuzz.ratio(first, second)


You can put together as many features as you like or need. Remember, a feature is a transformation on the data that allows your computer to understand the data better, or that highlights some salient feature of the data that helps inform you, the mere mortal, about whether two things are duplicates. These features here aren't particularly good, but they're a start, and we'll show that they are enough to work fairly well.

Because feature functions are required to accept two separate objects to compare, you can build a deduper around things that aren't Python primitives, or even serialisable. If you want to compare Django objects, go to town :

.. code-block:: python

    def commercial_properties_distance(first, second):
        lat_diff = first.primary_space.geography.latitude - second.primary_space.geography.latitude
        lon_diff = first.primary_space.geography.longitude - second.primary_space.geography.longitude
        return np.sqrt(lat_diff**2 + lon_diff**2)



Building the deduper
####################

The most basic deduper inherits from the `Dust` class, and wants a list of feature functions.

.. code-block:: python

    from spacedust import Dust

    class AddressDeduper(Dust):

        filename = "my_address_deduper"

        featureset = [
            street_number_is_same,
            street_name_is_same
        ]


You can pass in some hyperparameters for model tuning ( docs to come ), but for now, this will
get us started quite well.


Training the deduper
####################

To train, you just need your three lists, ``x1``, ``x2``, and ``y``. Instantiate your deduper and call
``fit()``.

.. code-block:: python

    deduper = AddressDeduper()
    deduper.fit(x1, x2, y)


Depending on the size of your training dataset, this can take anywhere from a second to several
minutes. Start small(ish) and increase your data size until you can't be bothered to wait any more.

When finished, you will get a print statement telling you the accuracy of your model. At this point, your model is fully trained and saved to disk, under the `filename` you provided. You're ready !


Using the deduper
#################

We're working on fully saving the entire object, including your featureset. Until then, we have two situations :

You've just finished training your model, and your class object ``deduper`` is still in RAM.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Great. You can just call ``.predict()``. Skip to the **Predicting** section.

You have a new Python kernel and you want to load your model into RAM.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At this point, you need to define your class and features again -- sorry ( working on it ). So,
you'll need to run the code in the **Building the deduper** section again; *however*, you will not need to train the model again, because on instantiation, we look for the model under the filename provided, and if it's there, we load that. So, whilst we need your featureset and filename once more, we don't need to spend all that time calling `.fit()`.


Making predictions
##################

At this point, we assume you have a `deduper` object in RAM. You can now feed it a bunch of data, and it will return some probabilities. 

.. code-block:: python

    deduper.predict(
        ["123 Main Street", "420 5th Ave", "123, Main St"],
        "123 Main Street"
    )


So, ``.predict()`` takes two arguments. They can be either lists, tuples, or np.ndarray iterables, *or* they can be single objects. If they're single objects ( as in the case of the second arg here ), we wrap them in a list for you.

``.predict()`` returns a np.ndarray of probabilities. If you pass in, as here, a list of three
elements, and then a single element, it will return a (3, 1)-shaped np.ndarray, containing the
probabilities of each possible combination of pairs between your arguments. If you pass in two lists of five, it will return a (5, 5)-shaped array. The (*i*, *j*) :sup:`th` element of this array is the probability that the *i*:sup:`th` element of your first list is a duplicate of the *j*:sup:`th` element of your second list.

We try not to be ( overly ) opinionated, despite the French heritage. As such, we return a 
*probability* and not a boolean as to whether things are duplicates. We leave it to you to specify a threshold above which something is a duplicate. If you're not sure where to start, 0.5 might be a good place, but this is not guaranteed.


Installation
############

.. code-block::

    pip install spacedust



TO DO
#####

1. Serialise featuresets and complete Dust model saving
2. Expand docs to describe hyperparameters
3. Put together complete notebook of examples
