# Yelp_Data_Challenge

<img src="https://github.com/will-zw-wang/Yelp_Data_Challenge-NLP_and_Clustering_and_Sentiment_Classification_and_Recommender_System/blob/master/images/Yelp_image.jpg" width="660" height="240">

[**Code**](pending)

## Project Objectives

• Through this project, I solved three main problems:

(1) Classified negative and positive reviews by a self-defined metric. After the model fitting, gained insights about what words usually contribute to the negative or positive review. A restaurant can summarize what's the main aspect such as food or service that resulted in a lower rating.

(2) Used unsupervised learning to cluster users into groups. Identified and understood the common user preference within each of the group by inspecting the cluster centroids. 

(3) Built a restaurant recommender system using collaborative filtering and matrix factorization based on users’ past visits and ratings.

- For Lending club, it's extremely important to know how's the loan repayment capacity for each loan applicant, and how much interest rate should be assigned to each loan application appropriately. 
    - Firstly, knowing the loan repayment capacity for each loan applicant could help it decide whether to "accept" or "deny" a loan application. 
    - Second, evaluating interest rate precisely could bring an additional incentive for those who are willing to "lend" money and also attain a balance between demand (borrowers) and supply (lenders).
- As a result, some suitable metrics must be determined by looking at the dataset.
    - **Metrics**
        - **loan status**: To evaluate the loan repayment capacity for each loan applicant.
        - **interest rate**: A numerical feature playing a role of balancing demand and supply. 
        - **grade**: A good categorical index to know the loan repayment capacity.       
- In our work, we bulid models to predict **Default(loan status)**, **Interest Rate** and **Loan Grade**.
