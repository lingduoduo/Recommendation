
The project is developed using Databricks Asset Bundle.

### Key Components

#### 1. Data Processing Pipeline
The `search_with_click.py` script in the `features` directory is responsible for data collection and preprocessing.

- **Input Data:** Includes user interactions (clicks, views) and item/user descriptions.
- **Output Data:** Produces model-specific files such as item vectors and similarity results.

#### 2. Workflow Orchestration
The `workflow.yml` file defines a Databricks workflow comprising multiple tasks.

- **Feature Engineering Task:** Converts raw data into features suitable for modeling.
- **Model Tasks:** Executes multiple recommendation models in parallel after feature engineering.

#### 3. Model Implementation
The project incorporates various recommendation algorithms, each implemented in separate Python files.

- **Techniques Used:** Collaborative filtering, content-based methods, and deep learning approaches.
- **Files:** Models are organized under the `model` directory, with each file corresponding to a specific algorithm.

### Project Structure
```
├── README.md
├── databricks.yml
├── project.json
├── resources
│   └── workflow.yml
└── src
    ├── data
    │   ├── input
    │   │   ├── item_desc.csv
    │   │   ├── item_desc_clientid.csv
    │   │   ├── search_click.csv
    │   │   ├── search_click_ts.csv
    │   │   ├── user_desc.csv
    │   │   └── view_click.csv
    │   └── output
    │       ├── item2vec_sim_result.csv
    │       ├── item_vector.csv
    │       └── sim_result.csv
    ├── features
    │   └── search_with_click.py
    ├── model
    │   ├── ContentBased.py
    │   ├── DSSM.py
    │   ├── DeepCrossing.py
    │   ├── DeepWide.py
    │   ├── Item2Vec.py
    │   ├── LFM.py
    │   ├── PR.py
    │   └── TreeBased.py
    ├── notebooks
    │   └── search_with_click.ipynb
    └── utils
```

### Machine Learning and Deep Learning Models

We are exploring the following ML/DL recommendation models:

#### 1. Content-Based Recommender
**File:** `ContentBased.py`  
**Approach:** Recommends items based on category preferences of users  

**Key Features:**
- Uses time-weighted scoring to prioritize recent interactions.
- Groups items by categories and recommends based on user's category preferences.
- Implements a ratio-based allocation of recommendations across categories.

---

#### 2. LFM (Latent Factor Model)
**File:** `LFM.py`  
**Approach:** Matrix factorization approach for collaborative filtering  

**Key Features:**
- Computes average ratings per item.
- Implementation appears to be partially complete in the provided code.

---

#### 3. PR (Personalized Ranking)
**File:** `PR.py`  
**Approach:** Graph-based recommendation using a user-item interaction graph  

**Key Features:**
- Builds a bipartite graph between users and items.
- Uses graph algorithms to compute personalized rankings.
- Implementation uses `scipy` sparse matrices for efficient computation.

---

#### 4. Item2Vec
**File:** `Item2Vec.py`  
**Approach:** Uses Word2Vec to learn item embeddings from user interaction sequences  

**Key Features:**
- Treats sequences of items clicked by the same user as "sentences."
- Learns 128-dimensional vector representations for items.
- Calculates item similarity using cosine similarity.
- Outputs top-k similar items for a given item.

#### 5. Two-Tower Model
**File:** `DSSM.py`  
**Approach:** Neural network-based recommendation using a two-tower architecture  

**Key Features:**
- Employs a two-tower architecture to learn separate embeddings for users and items.
- Computes relevance scores using cosine similarity between user and item embeddings.
- Optimized for large-scale recommendation systems.

---

#### 6. Tree-Based Model
**File:** `TreeBased.py`  
**Approach:** Recommendation using tree-based machine learning algorithms  

**Key Features:**
- Utilizes tree-based methods such as decision trees, random forests, and gradient boosting.
- Ideal for scenarios requiring interpretability and feature importance analysis.
- Efficiently handles structured data with both categorical and numerical features.

---

#### 7. Deep Crossing Model
**File:** `DeepCrossing.py`  
**Approach:** Deep learning-based recommendation leveraging residual networks  

**Key Features:**
- Combines user and item embeddings with residual connections to model feature interactions.
- Effectively processes both sparse and dense features.
- Designed for high-dimensional datasets with complex relationships.

---

#### 8. Deep & Wide Model
**File:** `DeepWide.py`  
**Approach:** Hybrid recommendation model integrating linear and deep components  

**Key Features:**
- Merges a wide linear model for memorization with a deep neural network for generalization.
- Captures both low-level and high-level feature interactions.
- Balances interpretability and predictive power for diverse recommendation tasks.

### Input Data Description for Models

|    Model       | Transaction Data |  Description Data  | User Data |
|:--------------:|:----------------:|:------------------:|:---------:|
| ContentBased   | search_click_ts  | item_desc_clientid |           |
|     LFM        |   search_click   |     item_desc      |           |
|      PR        |   search_click   |     item_desc      |           |
|   Item2Vec     |   search_click   |     item_desc      |           |
|     DSSM       |   search_click   |     item_desc      |           |
|  TreeBased     |    view_click    |     item_desc      |           |
| Deep Crossing  |    view_click    |     item_desc      | user_desc |
| Deep & Wide    |    view_click    |     item_desc      | user_desc |

- **search_click**: Contains user-item interactions with ratings (`user_id`, `item_id`, `rating`).
- **search_click_ts**: Similar to `search_click` but includes a timestamp (`user_id`, `item_id`, `unix_timestamp`, `rating`).
- **item_desc**: Provides item descriptions (`item_id`, `item_desc`).
- **item_desc_clientid**: Extends `item_desc` with additional client-specific information (`item_id`, `title`, `click_client_id`).
- **user_desc**: Contains user-specific metadata (`user_id`, `user_desc`).
- **view_click**: Captures user-item interactions based on views (`user_id`, `item_id`, `rating`).

Each model utilizes specific combinations of these input files to process data and generate recommendations.

### Test Run Locally

To test the project locally, follow these steps:

1. **Install `uv`**  
    Use Homebrew to install `uv`, a tool for managing Python environments:
    ```bash
    brew install uv
    ```

2. **Set Up the Environment and Run the Model**  
    The following commands will clean the cache, set up the environment, and execute the content-based recommendation model:
    ```bash
    uv cache clean
    uv run ContentBased.py
    ```
   
3. **Local Test using steamlit app**
    Test the project using a Streamlit app. Ensure you have Streamlit installed snf run the following commands:
    ```bash
    cd /Users/huanglin/Bitbucket/recommender-system-poc/src/demo
    streamlit run app.py
    ```

### Reference

To use this repo as is with customizations for model registration, model re-training, & Batch Inferencing, Please follow the documention from below link.

https://confluence.es.ad.adp.com/display/cdomlops/Unity+Catalog+Model+Registry

Note:

For documentation on the Databricks asset bundles format used for this project, and for CI/CD configuration, see

https://docs.databricks.com/dev-tools/bundles/index.html.
