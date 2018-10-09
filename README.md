# Yelp_Data_Challenge

<img src="https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_and_Clustering_and_Sentiment_Classification_and_Recommender_System/blob/master/images/Yelp_image.jpg" width="500" height="200">

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
- Dataset was downloaded from [**Yelp Dataset Challenge**](https://www.yelp.com/dataset_challenge).
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
2. NLP, Sentiment Classification Model Fitting and Models Comparison
3. Review Clustering
4. Other User Cases of Clustering
5. Recommender System Model Fitting and Models Comparison
6. Recommendation Results Analysis and Insights    

## Analysis Details

### 1. Data Preprocessing
- Extracted target data
    - Filtered business that are located in "Las Vegas" and contains "Restaurants" in their category.
    - Merged business information with review data based on "business_id".
    - Further filtered reviews by date between '2015-07-01' and '2018-01-01'
- Exploratory data analysis (EDA)
    - Bar plots findings
        - Most of business entities had reviewed from 1 to 250 times.
        - Most of business entities scored around 4.0 stars on average.
        - In most of time, the lengths for most reviews were within 1000.
        - 'Hash House A Go Go' was the top business entities with most comments in Las Vegas.
    - Line plots findings
        - We plotted the number of reviews by month
            - The plot had a general upward trend developed from Jul, 2015 to Dec, 2017. 
            - Moreover, three peaks were appeard on the same month (July), perhaps because of the summer vacation, more poeple went to restaurants and left reviews.
        - We plotted the distribution of text length of 5 stars and that of less than 5 stars
            - Two distribution were slightly different, longer texts appeared more when stars was less than 5 compared with when stars is 5, which meant longer texts do not stand for good rating. Perhaps because customers writed more to complain rather than praise.
- [**Detailed Code and Plotting**](https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_Sentiment_Analysis_and_Review_Clustering_and_Recommender_System/blob/master/code/Yelp_Data_Challenge%20-%20Data%20Preprocessing.ipynb)

### 2. NLP, Sentiment Classification Model Fitting and Models Comparison
- Defined feature variables and target variable
    - Feature variables: customer reviews
    - Target variable: 
        - positive review: 5 stars review
        - negative review: less than 5 stars review
- Got NLP representation of the documents
    - Tokenization
    - Removed stop words
    - Lemmatization
    - Generated vocabulary
    - Generated TF-IDF Matrix
- Classified positive/negative review
    - **Basic model with cross validation**
        - We tried to perform cross validation to evaluate basic model.
        - <img src="https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_Sentiment_Analysis_and_Review_Clustering_and_Recommender_System/blob/master/images/sentiment%20classification/CV_basic_model_performance_comparison.png" width="500" height="140"> 
        - The performance of **KNN** was the weakest. 
        - The trainng score and the test score of the other three models were comparable, seems no negative impact of overfitting. 
        - In **KNN**, **Logistic regression** and **Naive-Bayes** models, the difference between training and testing score was extremely close. 
        - However, in the **Random Forest** model, the difference was more significant compared to the results from **Logistic Regression** and **Naive-Bayes models**. 
            - It indicated that Random Forest, in fact, couldn't generalized the results to the unseen data (or testing data) very well, and was a high-variance model for this project instance.
    - **Models with grid search**
        - We tried grid search for these models to find the best predictable classifier
        - <img src="https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_Sentiment_Analysis_and_Review_Clustering_and_Recommender_System/blob/master/images/sentiment%20classification/Basic_vs_GridSearch_model_performance_comparison.png" width="600" height="160"> 
        - The performance of models with grid_search were very close to those of basic models, which indicated our baisc models had already performed greatly in this dataset and hard for the models with grid_search to perform better.
    - **Models with standardized tf-idf vectors and with (Stardardized + PCA) tf-idf vectors**
        -  We tried standardization and PCA to see if we could improve the model performace.
        - <img src=" " width="400" height="160"> 
        - 
- [**Detailed Code and Plotting**](pending)

### 3. Review Clustering
- **Clustered positive reviews of all business in "Las Vegas" into groups**
    - **Clustered reviews with KMeans, k = 8(default)**
        - Inspected the centroids
            - (1) Sorted each centroid vector to find the top 10 features
            - (2) Went back to our vectorizer object to find out what words each of these features corresponds to
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
            - Using eight clusters (default setting in kmeans), I found that several clusters were kind of similar to each other, such as in Cluster 0 and 7 might signified fast food restaurants. 
            - The rest of clusters had some significant meanings such as in Cluster 6, it mainly told about Japanese restaurants.
    - **Clustered reviews with KMeans, k = 5**
        - Inspected the centroids
            - top 10 features for each cluster:
                - 0: good,food,really,place,service,great,nice,love,chicken,time
                - 1: place,food,best,vegas,delicious,amazing,time,love,ve,just
                - 2: sushi,place,roll,rolls,great,fresh,ayce,service,best,fish
                - 3: pizza,great,place,crust,good,best,love,service,vegas,cheese
                - 4: great,food,service,place,amazing,awesome,friendly,excellent,staff,definitely
        - Summary
            - Using five clusters, the difference among clusters standed out more significant than using eight clusters. Each cluster now had an unique topic, such as Cluster 0 was surrounding with the topic of chicken, Cluster 2 was relating to Japanese food, Cluster 3 was relating to the pizza, and Cluster 4 was mainly about service aspect in vegas.
            - However, the top features using five clusters seemed to be highly overlapped with the default method. 
                - In fact, it was a good strategy to narrow down overlapped clusters into denser clusters.
- **Clustered reviews of the most reviewed restaurant in "Las Vegas" into groups**
    - **Clustered reviews with KMeans, k = 4**
        - Inspected the centroids
            - top 10 features for each cluster:
                - 0: chicken, waffles, fried, sage, bacon, benedict, good, food, place, huge
                - 1: food, minutes, wait, time, just, service, good, took, order, table
                - 2: hash, good, breakfast, food, house, eggs, pancake, place, potatoes, huge
                - 3: great, food, portions, place, service, huge, wait, good, vegas, amazing
        - Summary
            - Using four clusters, the difference among clusters standed out significantly and each cluster now had an unique topic, showed different aspects that customers care about:
                - Cluster 0 was surrounding with the topic of food, liked chicken and waffles. 
                - Cluster 1 was surrounding with the topic of waiting time and service.
                - Cluster 2 was relating to the breakfast, liked eggs and pancake. 
                - Cluster 3 was mainly about the taste and nutritional value.
- [**Detailed Code and Plotting**](https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_Sentiment_Analysis_and_Review_Clustering_and_Recommender_System/blob/master/code/Yelp_Data_Challenge%20-%20Clustering.ipynb)

### 4. Other User Cases of Clustering
- **Clustered restaurants by category information**
    - **Note:** a business may have mutiple categories, e.g. a restaurant can have both "Restaurants" and "Korean"
    - Inspected the centroids
        - top 10 features for each cluster:
            - 0: restaurants, food, mexican, chinese, thai, barbeque, asian, seafood, fusion, japanese
            - 1: bars, nightlife, sushi, restaurants, japanese, american, wine, new, cocktail, seafood
            - 2: pizza, italian, restaurants, sandwiches, wings, chicken, salad, food, seafood, delis
            - 3: breakfast, brunch, american, restaurants, traditional, sandwiches, food, new, buffets, diners
            - 4: american, traditional, new, burgers, restaurants, food, steakhouses, fast, seafood, southern
    - Summary
        - Clustered restaurants from their category information, the difference among clusters was significant. 
        - Each cluster now had an unique topic, such as Cluster 0 was mainly about Mexican and Chinese, Cluster 1 was Japanese, Cluster 2 was Italian,  Cluster 3 was American breakfast, and Cluster 4 was American(Traditional) in vegas.   
- **Clustered restaurants by restaurant names**
    - We clustered categories from business entities and tried to find the similarity between restaurant names.
    - Inspected the centroids
        - top 10 features for each cluster:
            - 0: restaurants, food, american, mexican, burgers, chinese, new, traditional, fast, seafood
            - 1: japanese, sushi, bars, restaurants, fusion, asian, ramen, noodles, seafood, poke
            - 2: bars, nightlife, american, restaurants, wine, new, cocktail, sports, traditional, mexican
            - 3: breakfast, brunch, american, restaurants, traditional, sandwiches, food, new, buffets, diners
            - 4: pizza, italian, restaurants, sandwiches, wings, salad, chicken, food, seafood, american
    - Summary
        - We noticed the most used business names were very straight forword, telling the major business the entities were running.
        - While I don't think these clusters are meaningful in distinguishing each other.
- **Cluster restaurants by tips**
    - As we had data **"tip.json"**, we clustered the tips business entities to customers, to see whether different business entities emphasis different aspects of their business.
    - Inspect the centroids
        - top 10 features for each cluster:
            - 0: great, food, service, place, staff, friendly, love, atmosphere, amazing, prices
            - 1: place, love, time, amazing, food, service, try, don, delicious, like
            - 2: awesome, food, service, place, great, staff, love, friendly, good, best
            - 3: best, town, ve, place, vegas, food, pizza, service, love, hands
            - 4: good, food, service, great, place, really, nice, pretty, friendly, prices
    - Summary
        - We noticed that almost all business entities are using positive words in their tips, thus these clusters were not meaningful in distinguishing each other.
- [**Detailed Code and Plotting**](https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_Sentiment_Analysis_and_Review_Clustering_and_Recommender_System/blob/master/code/Yelp_Data_Challenge%20-%20Clustering.ipynb)

### 5. Recommender System Model Fitting and Models Comparison
- **Cleaned data and created utility matrix**
  - We built utility matrix with only users reviewed more than four times.
  - For the removed or new users, we could recommend popular restaurants at first.
- **Build Recommender Systems**
  - **Popularity-based recommender**
    - We defined **popular** as restaurants with most reviewed records.
    - For every new user or user reviewed less than four times, we built a **Popularity-based recommender** to recommend most popular restaurants at first. 
    - Our **Popularity-based recommender** recommended top 10 restaurants: 
      - [972, 920, 736, 51, 871, 718, 729, 921, 784, 651].
  - **Neighborhood-based Approach Collaborative Filtering Recommender：Item-Item similarity recommender**
    - For user reviewed more than four times, we tried **Neighborhood-based approach** to build an **Item-Item similarity recommender** here. 
    - Given a user_id and recommend 10 restaurants which had the largest similarities with restaurants the user had reviewed before.
    - We tried to get final recommendations for a user: user_number = 100, and our **Item-Item similarity recommender** recommended top 10 restaurants: 
      - [1469, 421, 2350, 2102, 618, 551, 2429, 1874, 1653, 1988]
      - With an **average absolute error** of **0.3467**.
    - Then we tried to improve performance with **Matrix Factorization approach** to build recommender, because **matrix factorization models** always performs better than **neighborhood models** in **collaborative filtering**. 
      - **Reason**: 
        - The reason is when we factorize a ‘m*n’ matrix into two ‘m*k’ and ‘k*n’ matrices we are reducing our "n"items to "k"factors, which means that instead than having our 3000 restaurants, we now have 500 factors where each factor is a linear combination of restaurants. 
        - The key is that recommending based on factors is more robust than just using restaurant similarities, maybe a user hasn’t reviewed the restaurant ‘stay’ but the user might have reviewer other restaurants that are related to ‘stay’ via some latent factors and those can be used.
        - The factors are called latent because they are there in our data but are not really discovered until we run the reduced rank matrix factorization, then the factors emerge and hence the "latency".
  - **Matrix Factorization Approach Collaborative Filtering Recommender：NMF**
    - The **RMSE** of **NMF** was **2.0848**, and the **average absolute error** was **1.1285**, the performance was acceptable. 
    - We tried to get final recommendations for a user: user_number = 100, and our **NMF** recommender recommended top 10 restaurants: 
      - [168, 1463, 361, 1892, 1208, 1629, 940, 1500, 2108, 227]
      - With an **average absolute error** of **0.6795**.
    - The result was slightly different from what we discussed above, the average absolute error of **NMF** with 0.6795 for this specific user was slightly worse than 0.3467 of **Item-Item similarity** recommender, perhaps because the size of our dataset with 3000+ restaurants was too small to show the advantage of **NMF**.
    - Then we tried **UVD** to verify whether it performs better than **NMF**.
  - **Matrix Factorization Approach Collaborative Filtering Recommender：UVD**
    - The **RMSE** of **UVD** was **1.8733** and the **average absolute error** was **1.0795**, which were better than scores of **NMF**(**2.0848** and **1.1285**). 
      - **Reason**: **UVD** performed better because it has larger degree of freedom than **NMF**, to be specific, **NMF** is a specialization of **UVD**, all values of V, W, and H in **NMF** must be non-negative.
    - Then we tried to get final recommendations for a user: user_number = 100, and our **UVD recommender** recommended top 10 restaurants: 
      - [933, 599, 1083, 1752, 673, 2108, 1154, 1629, 1891, 374]
      - With an **average absolute error** of **0.5461**, which was slightly better than **0.6795** of **NMF**.
- [**Detailed Code**](https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_Sentiment_Analysis_and_Review_Clustering_and_Recommender_System/blob/master/code/Yelp_Data_Challenge%20-%20Restaurant%20Recommender.ipynb) 

### 6. Recommendation Results Analysis and Insights
- **Recommendation results Analysis between different recommendation systems**
  - We generated the overlap tables of the **top_10** and **top_100** results given by the four models for 'user with user_number = 100' as below:
    - <img src="https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_Sentiment_Analysis_and_Review_Clustering_and_Recommender_System/blob/master/images/The%20overlap%20of%20the%20top%2010%20recommendation%20generated%20by%20these%20four%20models.png">
    - <img src="https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_Sentiment_Analysis_and_Review_Clustering_and_Recommender_System/blob/master/images/The%20overlap%20of%20the%20top%20100%20recommendation%20generated%20by%20these%20four%20models.png">
  - From the overlap tables above, we notice that:
    - The recommended restaurants given by **Popularity-based**, **Neighborhood-based approach** and **Matrix Factorization approach** models were very different from each other, had no overlap in top 10 and only 2 overlaps in top 100 recommended restaurants.
    - While the recommended restaurants given by **NMF** and **UVD** had 2 overlaps in top 10 and 40 overlaps in top 100 recommended restaurants.
- **Conclusion**
  - 1. For **new user or user reviewed less than four times**, we can recommend most popular restaurants at first generated by our **Popularity-based recommender**.
  - 2. For **user reviewed more than four times**:
    - Given the performances of **NMF** and **UVD** are comparable, we can have the overlap of commendation results generated by these two models as the final recommendation.
    - As the results generated by **Popularity-based**, **Neighborhood-based Approach** and **Matrix Factorization Approach** models are totally different, we can allocate different weights to these models to construct the final recommendation. 
      - Like 0.2 for **Popularity-based**, 0.2 for **Neighborhood-based**, 0.6 for overlap of **NMF** and **UVD**.







