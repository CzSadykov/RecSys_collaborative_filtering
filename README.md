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

Calculates group diversity based on uniqueness metrics (KDE or KNN).

Key function:
- `group_diversity(embeddings, threshold, diversity_metric='kde', num_neighbors=5)`

### knn_uniqueness.py

Estimates uniqueness of items using K-nearest neighbors.

Key function:
- `knn_uniqueness(embeddings, num_neighbors=5)`

### kde_uniqueness.py

(Not shown in the context, but likely implements KDE-based uniqueness estimation)

## Filtering Service

The main component of this project is the `filtering_service.py`, which provides an API for filtering recommendations based on various criteria.

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
