NASA Kepler Exoplanet Candidate Classification
Overview
The practical goal of this challenge is train and test an optimized Classification model (using cross validation), and analyze its results.

Description
The Kepler space telescope is a now-retired telescope orbiting Earth. It was launched with a mission of discovering Earth-sized planets orbiting other stars - in other words, exoplanets. The way this was done was by monitoring stars for periodic small drops in intensity, which would signify something passing in front of them.

The dataset you'll use in this final project contains every candidate that Kepler spotted over its tenure, as well as whether or not it was confirmed to be an exoplanet. Your goal is to use the various attributes of each candidate to classify whether an object is an exoplanet or not.

You will be cross-validating the data through several models, and then doing the process again but with the data PCAed; comparing the results from both of these and selecting the most optimal one; cross-validating a third time and creating a final model from the results of the previous two steps; and finally analyzing that model.

From the final optimized model, you will produce:
A visualized Confusion Matrix
A classification report

Requirements:

- Properly and appropriately wrangle the data. See the Notes section below for more information about the data.
- Cross validate to find both the optimal [1] model, and its [2] hyperparams.
  -Process:
    1. Find the non-PCA optimal model using Randomized Search with the non-PCAed data. You assume the model found here has the optimal hyperparams in it. Make sure you are searching at least 10% of the total hyperparam space.
    2. Find the PCA optimal model using Randomized Search with the PCAed data. You assume the model found here has the optimal hyperparams in it. Make sure you are searching at least 10% of the total hyperparam space.
    3. Find the optimal hyperparams using Grid Search on the better of the two models from above, using the corresponding dataset (PCA or non-PCA).
  - Model and hyperparam requirements:
    1. You must use a Pipeline to cross validate.
    2. Validate against every Classifier we've used in class (except the Naive Bayes algorithms):
      1. Logistic Regression
        - No hyperparams
      2. K Nearest Neighbors
        - n_neighbors: 1 to 1.5* the square root of the number of training samples
      3. Decision Tree
        - criterion : 'entropy' and 'gini'
        - max_depth: 3 to 15 inclusive
        - min_samples_leaf: 1 to 10 inclusive
      4. SVC
        - kernel: 'rbf'
        - C: 0.1, 1, 10, and 100
        - gamma: 0.1, 1, and 10
- Train a final optimized model and produce the following outputs:
  - A visualized Confusion Matrix
  - A classification report

Attribute descriptions:
- koi_disposition: Exoplanet Archive *Disposition; not needed
- koi_pdisposition: *Disposition using Kepler data
- koi_period: orbital period (days) - how long does it take to orbit the star
- koi_eccen: eccentricity in the orbit (ratio) - how much of an ellipse is it
- koi_duration: transit duration (hours) - how long does it take to pass in front of the star
- koi_prad: planetary radius (earth radii) - half-width of the object
- koi_sma: orbit semi-major axis (au) - half-width of the widest part of the orbit ellipse
- koi_incl: inclination (deg) - how tilted is to orbital path
- koi_teq: equilibrium temp (K) - "average" temperature of the object
- koi_dor: planetary-star distance over star radius (ratio) - [distance between object and star] divided by [star radius]
- koi_steff: stellar effective temperature (K) - "average" temperature of the star
- koi_srad: stellar radius (solar radii) - half-width of the star
- koi_smass: stellar mass (solar mass) - mass of the star
- anything with err in it: error values (plus and minus) for values; not needed


