# Yelp_Data_Challenge

<img src="https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_and_Clustering_and_Sentiment_Classification_and_Recommender_System/blob/master/images/Yelp_image.jpg" width="660" height="200">

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
2. 


## Analysis Details

### 1. Data Preprocessing
- Extracted target data
    - created filters that selects business:
        - that are located in "Las Vegas"
        - that contains "Restaurants" in their category
    - merged with review data on "business_id"
    - Further filtered data by date: 
        - reviews between '2015-07-01' and '2018-01-01'
