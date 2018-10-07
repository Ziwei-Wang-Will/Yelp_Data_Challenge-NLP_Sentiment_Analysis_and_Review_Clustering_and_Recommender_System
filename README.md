# Yelp_Data_Challenge

<img src="https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_and_Clustering_and_Sentiment_Classification_and_Recommender_System/blob/master/images/Yelp_image.jpg" width="660" height="200">

[**Code**](https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_Sentiment_Analysis_and_Review_Clustering_and_Recommender_System/tree/master/code)

## Project Objectives

• Through this project, we had three main tasks:
- 1. **Classified negative and positive reviews**.
    - Used NPL techniques, such as **stemming**, **lemmatization** and **TF-IDF**, to extract features from unstructured review text data.
    - Used **Naive Bayes**, **Logistic Regression**, and **Random Forests** with a self-defined metric.
    - After the model fitting, gained insights about what words usually contribute to the negative or positive review. 
    - A restaurant can summarize what's the main aspects the customers care about, so that make corresponding improvement to attract more customers.
- 2. **Clustered reviews into groups**. 
    - Used NPL techniques, such as **stemming**, **lemmatization** and **TF-IDF**, to extract features from unstructured review text data.
    - Used **KMeans**. 
    - Clustered positive reviews of all business in "Las Vegas" into groups
        - By finding the top 10 features of centroid of each cluster, we can get information about the key features of different type of business, thus, if you’re going to start a new business in such type, you can pay more attention to these key features which customer care mostly.
    - Clustered reviews of the most reviewed restaurant in "Las Vegas" into groups
        - By inspecting the cluster centroids, identified and understood the common user preference within each of the group, providing information for market promotion strategy.
- 3. **Built a restaurant recommender system** 
    - Used **collaborative filtering** and **matrix factorization** based on users’ past visits and ratings.

## Dataset description (Data vary by rounds, below is round 9)
- Dataset is downloaded from [**Yelp Dataset Challenge**](https://www.yelp.com/dataset_challenge).
    - The Challenge Dataset:
        - 4.1M reviews and 947K tips by 1M users for 144K businesses
        - 1.1M business attributes, e.g., hours, parking availability, ambience.
        - Aggregated check-ins over time for each of the 125K businesses
        - 200,000 pictures from the included businesses
    - Cities:
        - U.K.: Edinburgh
        - Germany: Karlsruhe
        - Canada: Montreal and Waterloo
        - U.S.: Pittsburgh, Charlotte, Urbana-Champaign, Phoenix, Las Vegas, Madison, Cleveland
    - Files:
        - yelp_academic_dataset_business.json
        - yelp_academic_dataset_checkin.json
        - yelp_academic_dataset_review.json
        - yelp_academic_dataset_tip.json
        - yelp_academic_dataset_user.json
        - **Note**: Each file is composed of a single object type, one json-object per-line.

## Analysis Structure
1. Data Preprocessing
2. NLP and Sentiment Classification


## Analysis Details

### 1. Data Preprocessing
- Extracted target data
    - Filtered business that are located in "Las Vegas" and contains "Restaurants" in their category.
    - Merged business information with review data based on "business_id".
    - Further filtered reviews by date between '2015-07-01' and '2018-01-01'
- Exploratory data analysis (EDA)
    - Bar plots findings
        - Most of business entities have reviewed from 1 to 250 times.
        - Most of business entities scored around 4.0 stars on average.
        - In most of time, the lengths for most reviews are within 1000.
        - 'Hash House A Go Go' is the top business entities with most comments in Las Vegas.
    - Line plots findings
        - We plotted the number of reviews by month
            - The plot had a general upward trend developed from Jan, 2015 to Dec, 2017. 
            - Moreover, three peaks were appeard on the same month (July), perhaps because of the summer vacation, more poeple went to restaurants and left reviews.
        - We plotted the distribution of text length of 5 stars and that of less than 5 stars
            - Two distribution are slightly different, longer texts appeared more when stars is less than 5 compared with when stars is 5, which means longer texts do not stand for good rating. Perhaps because customers write more to complain rather than praise.
- [**Detailed Code and Plotting**](https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_Sentiment_Analysis_and_Review_Clustering_and_Recommender_System/blob/master/code/Yelp_Data_Challenge%20-%20Data%20Preprocessing.ipynb)

### 2. NLP and Sentiment Classification
- Define feature variables and target variable
    - Feature variables: customer reviews
    - Target variable: 
        - positive review: 5 stars review
        - negative review: less than 5 stars review
- Get NLP representation of the documents
    - Tokenization
    - Remove stop words
    - Lemmatization
    - Generate vocabulary
    - Generate TF-IDF Matrix
- Classifying positive/negative review
    - **Basic model with cross validation**
        - We tried to perform cross validation to evaluate basic model.
        - <img src="https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_Sentiment_Analysis_and_Review_Clustering_and_Recommender_System/blob/master/images/sentiment%20classification/CV_basic_model_performance_comparison.png" width="500" height="140"> 
        - The performance of **KNN** is the weakest. 
        - The trainng score and the test score of the other three models are comparable, seems no negative impact of overfitting. 
        - In **KNN**, **Logistic regression** and **Naive-Bayes** models, the difference between training and testing score is extremely close. 
        - However, in the **Random Forest** model, the difference is more significant compared to the results from **Logistic Regression** and **Naive-Bayes models**. 
            - It indicated that Random Forest, in fact, can't generalized the results to the unseen data (or testing data) very well, and is a high-variance model for this project instance.
    - **Models with grid search**
        - We tried grid search for these models to find the best predictable classifier
        - <img src=" " width="400" height="160"> 
        - The performance of models with grid_search are very close to those of basic models, which indicate our baisc models have already performed greatly in this dataset and hard for the models with grid_search to perform better.
    - **Models with standardized tf-idf vectors and with (Stardardized + PCA) tf-idf vectors**
        -  We tried standardization and PCA to see if we can improve the model performace.
        - <img src=" " width="400" height="160"> 
        - 








