# Recommendation Filtering Service

 This project provides a simple solution and FastAPI service for creating, managing, and filtering collaborative filtering embeddings. Let's explore the key components and functionalities of this service.

## Table of Contents
1. [CF Embeddings](#cf-embeddings)
2. [Metrics](#metrics)
3. [Filtering Service](#filtering-service)
4. [Usage](#usage)

## CF Embeddings

The `cf_embeddings` folder contains essential components for creating and managing collaborative filtering embeddings. Here's a brief overview of the main files:

### create_embeddings.py

This script is the heart of our embedding creation process. It handles the creation of item embeddings using the Alternating Least Squares (ALS) algorithm.

Key functions:
- `normalize_matrix(ui_matrix, method='bm_25')`: Normalizes the user-item interaction matrix using various methods.
- `items_embeddings(ui_matrix, item_ids, dim, **kwargs)`: Creates item embeddings using the ALS algorithm.

### normalizations.py

This file contains various normalization techniques for the user-item matrix.

Class: `Normalization`
- Static methods: `by_column()`, `by_row()`, `tf_idf()`, `bm_25()`

### user_item_matrix.py

This file contains the `UserItemMatrix` class, responsible for creating the user-item interaction matrix from raw data.

## Metrics

The `metrics` folder includes various metrics for evaluating and filtering recommendations:

### group_diversity.py

Contains functions for calculating group diversity based on uniqueness metrics using either Kernel Density Estimation (KDE) or K-Nearest Neighbors (KNN) methods.

Key function:
- `group_diversity(embeddings, threshold, diversity_metric='kde', num_neighbors=5)`
  - Parameters:
    - `embeddings`: A list or array of item embeddings for the group
    - `threshold`: A float value representing the minimum acceptable diversity score
    - `diversity_metric`: A string specifying the method to use ('kde' or 'knn')
    - `num_neighbors`: An integer specifying the number of neighbors to consider for KNN (default is 5)
  - Function behavior:
    - Calculates the uniqueness of each item in the group using the specified metric
    - Computes the overall group diversity by averaging the uniqueness scores
    - Compares the calculated diversity against the provided threshold
    - Returns a tuple containing a boolean (True if diverse enough, False otherwise) and the calculated diversity score

### knn_uniqueness.py

Implements a method for estimating the uniqueness of items using the K-Nearest Neighbors algorithm.

Key function:
- `knn_uniqueness(embeddings, num_neighbors=5)`
  - Parameters:
    - `embeddings`: A list or array of item embeddings
    - `num_neighbors`: An integer specifying the number of nearest neighbors to consider (default is 5)
  - Function behavior:
    - For each item embedding, finds the K nearest neighbors in the embedding space
    - Calculates the average distance to these neighbors
    - Returns an array of uniqueness scores, where higher scores indicate more unique items

### kde_uniqueness.py

Implements a method for estimating item uniqueness using Kernel Density Estimation.

Presumed key function:
- `kde_uniqueness(embeddings)`
  - Parameters:
    - `embeddings`: A list or array of item embeddings
  - Likely function behavior:
    - Estimates the probability density function of the item embeddings in the feature space
    - Calculates the uniqueness of each item based on its position in the estimated density
    - Returns an array of uniqueness scores, where lower density areas correspond to higher uniqueness

These functions work together to provide a comprehensive approach to assessing and ensuring diversity in recommendation groups, allowing for flexible application in various recommendation scenarios.

## Filtering Service

The main component of this project is the `filtering_service.py`, which provides an API for filtering recommendations based on their diversity estimated via KNN or KDE (by default).

Key features:
- Loads item embeddings
- Filters recommendations based on group diversity
- Provides an API endpoint for recommendation filtering

## Usage

To use the Filtering Service, follow these steps:

1. Ensure you have all the required dependencies installed:
   - Install the necessary Python packages listed in the `requirements.txt` file using pip:
     ```
     pip install -r requirements.txt
     ```

2. Prepare your item embeddings using the `create_embeddings.py` script:
   - Run the script to generate embeddings based on your user-item interaction data:
     ```
     python cf_embeddings/create_embeddings.py
     ```
   - Make sure your data path and output path are correctly set in the `.env` file.
   - The generated embeddings should be stored as a dictionary where keys are item IDs and values are the corresponding embeddings.
   - Example files are provided in the `cf_embeddings` folder. You can dive into the code to understand how to generate embeddings for your own data.

3. Start the Filtering Service:
   - Launch the service by running the `filtering_service.py` script:
     ```
     python filtering_service.py
     ```
   - The service will start and listen for incoming requests on the specified port (default is usually 8000).

4. Use the Filtering Service API:
   - The main function is an endpoint `filtering_service.py` which takes a string of item IDs, a diversity threshold, and a diversity metric (KDE or KNN).
   - It returns a tuple where the first element is a boolean indicating if the recommendation is relevant (diverse enough), and the second element is the calculated diversity value.

5. Understanding the diversity metric:
   - The diversity metric is calculated using either KDE (Kernel Density Estimation) or KNN (K-Nearest Neighbors) method.
   - The function first retrieves the embeddings for the given item IDs from the pre-computed embeddings dictionary.
   - For each item in the group, the function calculates its uniqueness based on the chosen method:
     - KDE method estimates the density of item embeddings in the feature space.
     - KNN method calculates the mean Euclidean distance to K nearest neighbors for each item.
   - The overall diversity of the group is then calculated as the sum of individual item uniqueness values divided by the number of items in the group.
   - If the calculated diversity is below the specified threshold, the recommendation is considered not diverse enough (is_relevant = False).

6. Interpreting the results:
   - If is_relevant is True, the recommendation group is considered diverse enough.
   - The diversity_value provides a numerical measure of the group's diversity.
   - You can use these results to filter or adjust your recommendations as needed.

7. Customizing the service:
   - You can modify the diversity calculation methods in the respective files (kde_uniqueness.py and knn_uniqueness.py).
   - Adjust the threshold and metric choice based on your specific requirements and dataset characteristics.
   - Ensure that your embeddings dictionary is properly loaded and accessible to the filtering service.
