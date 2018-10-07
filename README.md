# Yelp_Data_Challenge

<img src="https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_and_Clustering_and_Sentiment_Classification_and_Recommender_System/blob/master/images/Yelp_image.jpg" width="660" height="240">

[**Code**](pending)

## Project Objectives

• Through this project, we had three main tasks:
- 1. **Classified negative and positive reviews**.
    - Used NPL techniques, such as **stemming**, **lemmatization** and **TF-IDF**, to extract features from unstructured review text data.
    - Used **Naive Bayes**, **Logistic Regression**, and **Random Forests** with a self-defined metric.
    - After the model fitting, gained insights about what words usually contribute to the negative or positive review. 
    - A restaurant can summarize what's the main aspects the customers care about, so that make corresponding improvement to attract more customers.
- 2. **Cluster reviews into groups**. 
    - Used NPL techniques, such as **stemming**, **lemmatization** and **TF-IDF**, to extract features from unstructured review text data.
    - Used **KMeans**. 
    - Clustered positive reviews of all business in "Las Vegas" into groups
        - By finding the top 10 features of centroid of each cluster, we can get information about the key features of different type of business, thus, if you’re going to start a new business in such type, you can pay more attention to these key features which customer care mostly.
    - Clustered reviews of the most reviewed restaurant in "Las Vegas" into groups
        - By inspecting the cluster centroids, identified and understood the common user preference within each of the group, providing information for market promotion strategy.
- 3. **Built a restaurant recommender system** 
    - Used **collaborative filtering** and **matrix factorization** based on users’ past visits and ratings.

## Dataset description
- Dataset is downloaded from [**Yelp Dataset Challenge**](https://www.yelp.com/dataset_challenge).
    - These files contain complete loan data for all loans issued in 2018Q1, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. 
    - The file containing loan data through the "present" contains complete loan data for all loans issued through the previous completed calendar quarter. 
    - 'LoanStats_2018Q1.csv' contains loan application records from January 2018 to March 2018.
        - 107866 loan applications.
        - 145 features.
    - [**Data is available here**](https://github.com/will-zw-wang/Lending_Club-Default_Prediction_and_Interest_Rate_Prediction_and_Loan_Grade_Prediction/blob/master/data/LoanStats_2018Q1.csv.zip)
- I classify all 145 features into eight types:
    - User feature (general)
    - User feature (financial specific)
        - Income
        - Credit scores
        - Credit lines
    - Loan general feature
    - Current loan payment feature
    - Potential response variables
    - Secondary applicant info
    - Hardship 
    - Settlement
    - [**Grouped features dictionary is available here**](https://github.com/will-zw-wang/Lending_Club-Default_Prediction_and_Interest_Rate_Prediction_and_Loan_Grade_Prediction/blob/master/data/LC_DataDictionary.xlsx)
