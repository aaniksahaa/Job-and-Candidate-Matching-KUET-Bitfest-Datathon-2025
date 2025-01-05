# Winning Solution of KUET-Bitfest-Datathon-2025

**Team Sirius**

A Brief Overview of our solution:

We performed the following steps to approach the problem aiming at accurate job matching prediction.

- **Exploratory Data Analysis:**
    - **Missing Values:** First, inspecting the train and test data, we find that several columns have significant amount of missing values. These include address, languages, proficiency_levels, certification_providers, extra_curricular_activity_types and many more.
    - **Unimportant columns:** We also notice that some fields clearly do not contribute at all to the prediction problem, for instance, address, company_urls, online_links etc.
    - **Diverse Textual Data:** We notice that a lot of the columns contain textual data which have a rather high number of unique values. One exception is ‘educationaL_requirements’ which contains 20 unique values. Others like ‘skills’, ‘responsibilties’ seem to include highly diverse textual data
    - **Date Columns in Textual Form:** We note that, columns like ‘start_dates’, ‘end_dates’ actually present date values, but are given in text format, like ‘18/2022’ or ‘Current’ etc
    - **Numeric Columns in Textual Form:** We also note that, some of the columns like ‘experiencere_requirement’, ‘age_requirement’ etc actually present minimum and maximum allowed values, but are given in text format, like ‘2 to 5 years’

+

- **Data Manipulation**
    - **Feature Selection:** Based on our observations, we decide to ignore some features completely. These are, address, extra_curricular_activity_types, extra_curricular_organization_names, extra_curricular_organization_links, role_positions, languages, proficiency_levels, certification_providers, certification_skills, online_links, issue_dates, expiry_dates
    - **Textual Data Processing:** The textual columns present list formatted strings in any cases. We process them to obtain comma-separated plain-text format
    - **Data Extraction:** From the columns ‘start_dates’, ‘end_date’, we extract exact dates in (month, year) format using a Python based parser function.
    - **Numeric Data Extraction:** From the columns, ‘experiencere_requirement’, ‘age_requirement’, we extract minimum and maximum values using Python-based parser.
    - **Textual Data Embedding:** We embed the textual columns data using a grouped hybrid embedder.
    - **Hybrid Embedding:** We use a concatenation of the vectors obtained from each of these embeddings
        - **Word2Vec**: Semantic embeddings based on word contexts.
        - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Measures term importance within and across documents.
        - **Hashing Vectorizer**: Hash-based feature extraction for text representation.
        - **Count Vectorizer**: Frequency-based feature extraction for n-grams.
        - **Truncated SVD (Singular Value Decomposition)**: Reduces dimensionality of Count Vectorizer outputs.
        - **BM25**: A ranking function for text retrieval based on term frequencies and document lengths.
        - **Mean Pooling**: Aggregates features by their mean value.
        - **Max Pooling**: Aggregates features by their maximum value.
    - **Grouped Hybrid Embedding Training:** Please note here, we do not use any pre-trained embedding model. These are standard libraries that are available in Python, without any model weights. Therefore, we extract column-group-wise corpora from the provided dataset and fit embedders on those texts. For instance, we consider 'skills_required', 'related_skils_in_job', 'skills', 'responsibilities', these four features for one group. That means, we fit embedders on all the texts present in these columns over all rows. Then we infer embeddings for the test data.
- **Predictor Model Training**
    - **Input Feature Preparation:** We tried using only text embeddings as features and also merging text embedding and numeric values as input features.
        - We concatenated all the embeddings obtained from the following the columns, educationaL_requirements, ﻿job_position_name, responsibilities, skills_required, degree_names, major_field_of_studies, positions, related_skils_in_job, skills, career_objective, professional_company_names, experiencere_requirement
        - For vector sizes, different values were explored rangong from 50 to 224
        - Numeric features are extracted from the columns ‘experiencere_requirement’ and  ‘age_requirement’
        - Then we concatenate all the obtained features
    - **Model Exploration and Selection**
        - We tried the following models
            - ML Models:
                - Random Forest Regressor
                - XGBoost regressor
                - Catboot Regressor
                - LightGBM regressor
            - DL Models:
                - Feed Forward Deep Neural Network
                - Dual Encoder Neural Network
        - Among the ML Models, LIghtBGM performed best. Finally, among all models, Feed Forward NN with early stopping seemed to perform best, we go on with this
    - **Hyperparameter Tuning:** We search for different hyperparameters for different ML models. In case of DL models, we try changing number of layers, dropout rate and number of epochs.
    - **Model Architecture:** In the Feed Forward NN we tried, the notable properties are
        - 6 layers
        - Batch Normalization
        - Dropout
        - Residual Connection
    - **Training:**
        - **Early Stopping:** We keep saving the model every 100 iterations and keep track of the validation losses. Then we load the state where the loss dropped to the lowest. This helps us battle overfitting.
        - **K-Fold Cross Validation:** We divide the train data into folds with 80-20 split and train on it using a remaining part as validation set. For the value of K, we try values from 5 to 10.
- **Inference on Test Set**
    - We infer on test set with the same K-Fold scheme. That means, for each of the trained models on folds, we predict a for a part of the test set
    - Then we append all those predictions on the test set
    - Note that, this folded inference seemed to improve the performance over training a single model over the whole train set
