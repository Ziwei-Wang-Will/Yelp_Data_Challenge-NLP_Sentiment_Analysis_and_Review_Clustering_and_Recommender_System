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
3. Review Clustering
4. Other user cases for clustering
    

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
            - The plot had a general upward trend developed from Jul, 2015 to Dec, 2017. 
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
### 3. Review Clustering
- **Clustered positive reviews of all business in "Las Vegas" into groups**
    - **Cluster reviews with KMeans, k = 8(default)**
        - Inspect the centroids
            - (1) Sort each centroid vector to find the top 10 features
            - (2) Go back to our vectorizer object to find out what words each of these features corresponds to
            - top 10 features for each cluster:
                - 0: burger, fries, burgers, good, great, place, cheese, best, shake, food
                - 1: food, good, place, best, vegas, amazing, delicious, time, service, just
                - 2: excellent, service, food, great, place, good, vegas, definitely, restaurant, best
                - 3: love, place, food, great, good, service, amazing, best, friendly, staff
                - 4: pizza, great, crust, place, good, best, vegas, cheese, service, delicious
                - 5: great, food, service, place, amazing, good, awesome, friendly, staff, definitely
                - 6: sushi, place, roll, rolls, great, fresh, ayce, service, best, fish
                - 7: chicken, fried, good, food, rice, place, delicious, great, ordered, amazing
        - We then tried different k, because:
            - Using eight clusters (default setting in kmeans), I found that several clusters are kind of similar to each other, such as in Cluster 0 and 7 might signify fast food restaurants. 
            - The rest of clusters have some significant meanings such as in Cluster 6, it mainly tell about Japanese restaurants.
    - **Cluster reviews with KMeans, k = 5**
        - Inspect the centroids
            - top 10 features for each cluster:
                - 0: good,food,really,place,service,great,nice,love,chicken,time
                - 1: place,food,best,vegas,delicious,amazing,time,love,ve,just
                - 2: sushi,place,roll,rolls,great,fresh,ayce,service,best,fish
                - 3: pizza,great,place,crust,good,best,love,service,vegas,cheese
                - 4: great,food,service,place,amazing,awesome,friendly,excellent,staff,definitely
        - Summary
            - Using five clusters, the difference among clusters stands out more significant than using eight clusters. Each cluster now has an unique topic, such as Cluster 0 is surrounding with the topic of chicken, Cluster 2 is relating to Japanese food, Cluster 3 is relating to the pizza, and Cluster 4 is mainly about service aspect in vegas.
            - However, the top features using five clusters seem to be highly overlapped with the default method. In fact, it's a good strategy to narrow down overlapped clusters into denser clusters.
- **Clustered reviews of the most reviewed restaurant in "Las Vegas" into groups**
    - **Cluster reviews with KMeans, k = 4**
        - Inspect the centroids
            - top 10 features for each cluster:
                - 0: chicken, waffles, fried, sage, bacon, benedict, good, food, place, huge
                - 1: food, minutes, wait, time, just, service, good, took, order, table
                - 2: hash, good, breakfast, food, house, eggs, pancake, place, potatoes, huge
                - 3: great, food, portions, place, service, huge, wait, good, vegas, amazing
        - Summary
            - Using four clusters, the difference among clusters stands out significantly and each cluster now has an unique topic, shows different aspects that customers care about:
                - Cluster 0 is surrounding with the topic of food, like chicken and waffles. 
                - Cluster 1 is surrounding with the topic of waiting time and service.
                - Cluster 2 is relating to the breakfast, like eggs and pancake. 
                - Cluster 3 is mainly about the taste and nutritional value.

### 4. Other user cases for clustering
- **Cluster restaurants by category information**
    - **Note:** a business may have mutiple categories, e.g. a restaurant can have both "Restaurants" and "Korean"
    - Inspect the centroids
            - top 10 features for each cluster:
                - 0: restaurants, food, mexican, chinese, thai, barbeque, asian, seafood, fusion, japanese
                - 1: bars, nightlife, sushi, restaurants, japanese, american, wine, new, cocktail, seafood
                - 2: pizza, italian, restaurants, sandwiches, wings, chicken, salad, food, seafood, delis
                - 3: breakfast, brunch, american, restaurants, traditional, sandwiches, food, new, buffets, diners
                - 4: american, traditional, new, burgers, restaurants, food, steakhouses, fast, seafood, southern
    - Summary
        - Cluster restaurants from their category information, the difference among clusters is significant. 
        - Each cluster now has an unique topic, such as Cluster 0 is mainly about Mexican and Chinese, Cluster 1 is Japanese, Cluster 2 is Italian,  Cluster 3 is American breakfast, and Cluster 4 is American(Traditional) in vegas.            
- **Cluster restaurants by restaurant names**
    - We clustered categories from business entities and tried to find the similarity between restaurant names.
    - Inspect the centroids
            - top 10 features for each cluster:
                - 0: restaurants, food, american, mexican, burgers, chinese, new, traditional, fast, seafood
                - 1: japanese, sushi, bars, restaurants, fusion, asian, ramen, noodles, seafood, poke
                - 2: bars, nightlife, american, restaurants, wine, new, cocktail, sports, traditional, mexican
                - 3: breakfast, brunch, american, restaurants, traditional, sandwiches, food, new, buffets, diners
                - 4: pizza, italian, restaurants, sandwiches, wings, salad, chicken, food, seafood, american
    - Summary
        - We notice the most used business names are very straight forword, telling the major business the entities are running.
        - While I don't think these clusters are meaningful in distinguishing each other.
- **Cluster restaurants by tips**
    - As we have data **"tip.json"**, we can cluster the tips business entities to customers, to see whether different business entities emphasis different aspects of their business.
    - Inspect the centroids
            - top 10 features for each cluster:
                - 0: great, food, service, place, staff, friendly, love, atmosphere, amazing, prices
                - 1: place, love, time, amazing, food, service, try, don, delicious, like
                - 2: awesome, food, service, place, great, staff, love, friendly, good, best
                - 3: best, town, ve, place, vegas, food, pizza, service, love, hands
                - 4: good, food, service, great, place, really, nice, pretty, friendly, prices
    - Summary
        - We notice that almost all business entities are using positive words in their tips, thus these clusters are not meaningful in distinguishing each other.









