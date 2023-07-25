
# Recommendation

### Data

- Popularity (hot items)
  topN product
- Product hashtags
  - Topic, pirce, ...
  - Similarity, e.g., Jaccard similarity coefficient
- Personalization
  User intetests
  User segmentations (geo, demo, new/existing, active/inactive)
  Geographic
  Time Stamp
  
### Model

To build an app that ranks the top 3 reasons for app stores based on reviews, you can follow these general steps:

Data Collection: Gather reviews from the app stores you want to analyze. You may need to use the app store APIs or web scraping techniques to retrieve the reviews.

- Business side
- Third party data

Preprocessing: Clean and preprocess the collected reviews to remove irrelevant information, such as special characters, emojis, or URLs. You may also need to handle text normalization tasks like lowercasing, stemming, or lemmatization.

Retrieval:

1. Association Rules (support, confidence)
   
2. Similarity/Clustering
- Product and contents
- User behavior (purchased items, last view items, etc)
- Contextual features
- Timestamp(morning, afternoon, evening), Geographic, weather, device, etc
- User clustering, naive bayesian 
  
Sim(V1, V2) = cosine similarity/jaccard similarity

3. CF
CF (user-based CF) and (item-based CF)
MF(matrix factoriaztion) = user profile latent factor * product profile latent factor

4.Factorization machine
The dot product of user vector and item vector represents the user's preference for the item

5.Embedding
Word2Vec(COBW and Sip-Gram)
Item2vec
FB Faiss

6. Deep Learning
DNN

Ranking: Calculate the frequency or importance of each identified topic or reason across the reviews. You can use metrics like term frequency (TF), term frequency-inverse document frequency (TF-IDF), or more advanced methods like TextRank or LDA topic coherence.

Visualization: Present the top-ranked reasons in an intuitive and visually appealing manner. You can use charts, graphs, word clouds, or other visualization techniques to display the results to the app users.

Interactive Interface: Design an interactive user interface for the app where users can select the app store, specify the time range of reviews to analyze, and view the ranked reasons. Provide options to filter results based on sentiment (positive/negative/neutral) or other criteria.

Real-time Updates: If desired, implement a mechanism to fetch and analyze new reviews periodically to keep the rankings up to date. This can be achieved by scheduling automatic data collection and analysis processes.

Testing and Optimization: Thoroughly test the app's functionality and performance. Gather user feedback and iterate on the design and features based on user responses. Optimize the app for speed, accuracy, and usability.

### System

Deployment: Deploy the app to the desired platform(s) such as web, mobile, or desktop. Ensure that it can handle the expected user load and provide a smooth user experience.

- https://amatriain.net/blog/RecsysArchitectures
- Recommendation system design: https://eugeneyan.com/writing/system-design-for-discovery/
- Alibaba Rec Sys paper: https://arxiv.org/pdf/1803.02349.pdf
- Netflix system architecture： https://netflixtechblog.com/system-architectures-for-personalization-and-recommendation-e081aa94b5d8
- Netflix more details on system:  https://medium.com/@narengowda/netflix-system-design-dbec30fede8d
- Tiktok https://towardsdatascience.com/why-tiktok-made-its-user-so-obsessive-the-ai-algorithm-that-got-you-hooked-7895bb1ab423

Remember to consider privacy and data security aspects when collecting and processing user reviews. Also, check the terms and conditions of the app stores you're analyzing to ensure compliance with their policies.

It's important to note that building such an app involves various technical components and considerations. Depending on your programming skills and expertise, you may need to leverage natural language processing (NLP) libraries, machine learning frameworks, web development tools, and data storage solutions.
