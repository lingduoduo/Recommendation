# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/market-basket-analysis. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/recommendation-engines.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to explore how we might leverage association rules to build product recommendations. 

# COMMAND ----------

# DBTITLE 1,Get Configuration Values
# MAGIC %run "./01_Configuration"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pandas as pd
import itertools

from pyspark.sql.types import *
import pyspark.sql.functions as f

# COMMAND ----------

# MAGIC %md ## Step 1: Access Association Rules
# MAGIC
# MAGIC We now have a set of rules generated from our *prior* period transactions.  We can retrieve these along with scores that tell us something about the reliability of these them:

# COMMAND ----------

# DBTITLE 1,Retrieve Rules & Scores
rules = (
  spark
    .table('instacart.assoc_rules')
    .selectExpr('antecedent', 'consequent[0] as consequent', 'lift_norm', 'confidence')
  )

display(rules)

# COMMAND ----------

# MAGIC %md ## Step 2: Construct Recommendations
# MAGIC
# MAGIC We now should consider how we might use these rules to make recommendations. In general, we will recommend products based on items added to a shopper's cart:
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/mba_recommender.png' width=400>
# MAGIC
# MAGIC But will we make those recommendations based on the whole cart or just the last few items added to it? Will we require the whole cart or just some portion of it to match the antecedent to a rule?  Should we consider rules where only a portion of the antecedent is matched by the contents of the cart?  How will we prioritize rules when the antecedent only matches a subset of the cart or when only a subset of the antecedent is matched?  How should we consider the impact of multiple rules recommending the same products? While on the surface, the notion that the *if this, then that* rules generated by a market basket analysis could be used to make recommendations seems clear, the reality of how we go about assembling those recommendations gets complicated quickly.
# MAGIC
# MAGIC In this [white paper](https://www.sciencedirect.com/science/article/pii/S095741741830441X?via%3Dihub) a team of researchers explored some of these choices as well as a few others to construct a recommender of this type.  While they arrived at an ideal solution for their scenario, it's not clear that the particular approach they took is universally ideal. With this in mind, we'll borrow a few items from some of their algorithms and insert our own.  The basic algorithm we will use is as follows:</p>
# MAGIC
# MAGIC 1. Assemble a shopper's cart at a point in time during the shopping process
# MAGIC 2. Retrieve all rules with antecedents match some or all of the product or product combinations assembled in the previous step
# MAGIC 3. Discard any rules with consequents already in the shopping cart
# MAGIC 4. Assign a weight to each rule based on the number of products in the antecedent relative to the number of products in the original cart
# MAGIC 5. Multiply each rule's confidence by this weight to arrive at a rule-specific score
# MAGIC 6. Sum the rule-scores for each unique consequent product
# MAGIC 7. Present products in order from highest to lowest summed scores
# MAGIC
# MAGIC A number of variations on these rules including the use of normalized lift over confidence were explored before arriving at this algorithm.  This algorithm provided the best results based on our evaluation metric to be discussed in the next step.
# MAGIC
# MAGIC To tackle the first step, we'll assemble each basket as it existed at a point in time.  The dataset provides access to a field named *add_to_cart_order* with which we can assemble these carts. The items that will be later added to the cart to complete the order are captured as well to enable later evaluation: 

# COMMAND ----------

# DBTITLE 1,Retrieve Cart as of Point in Time
# retrieve basket contents
basket = (
  spark
  .table('instacart.orders')
  .filter(f.expr("eval_set='train'"))
  .join( spark.table('instacart.order_products'), on='order_id')
  .selectExpr('order_id','add_to_cart_order as position', 'product_id')
  )

# assemble basket as of point in time
basket_at_position = (
  basket.alias('x')
    .join(basket.alias('y'), on=f.expr('x.order_id=y.order_id AND x.position>=y.position'))
    .groupBy('x.order_id','x.position')
      .agg(f.collect_list('y.product_id').alias('basket'))
  )

# place downstream products in "next"
basket_and_next = (
  basket_at_position.alias('m')
    .join(basket.alias('n'), on=f.expr('m.order_id=n.order_id AND m.position<n.position')) # this forces final basket to drop out (must always 1+ next product)
    .groupBy('m.order_id', 'm.position')
      .agg(
        f.first('m.basket').alias('basket'),
        f.collect_list('n.product_id').alias('next_products')
        )
  )

display(basket_and_next.orderBy('m.order_id','m.position'))

# COMMAND ----------

# MAGIC %md Given the size of the dataset on which we wish to make recommendations, we will take a 10% random sample. We will aggressively spread the data around the cluster through partitioning to ensure we don't overrun memory on any executors. We will then match baskets to rules where the antecedent is completely addressed by the basket:

# COMMAND ----------

# DBTITLE 1,Match Baskets with Rules
baskets_and_rules = (
  basket_and_next
    .sample(fraction=0.10, withReplacement=False)
    .repartition(sc.defaultParallelism * 100)
    .join( 
      f.broadcast(
        spark
          .table('instacart.assoc_rules')
          .selectExpr('antecedent', 'consequent[0] as consequent', 'lift', 'lift_norm', 'confidence') # just needed fields to keep broadcast mem pressure low
        ), 
      on=f.expr('array_intersect(basket, antecedent) != array()') # any overlap creates match
      )
  )

display(baskets_and_rules)

# COMMAND ----------

# MAGIC %md Using confidence as our key metric, we will score each rule based on that metric weighted relative to amount of overlap between the rule's antecedent and the basket.  For each recommended product, we'll simply sum these scores to establish a ranked list of products to recommend:

# COMMAND ----------

# DBTITLE 1,Build Recommendations
# metric around which to score rules
metric = 'confidence'

# assemble recommendations
recommendations = (
  baskets_and_rules
    .filter(f.expr('not array_contains(basket, consequent)'))  # consequent not already in basket
    .withColumn('intr', f.expr('size(array_intersect(basket, antecedent))'))
    .withColumn('match_score', f.expr('power(intr,2)/(size(antecedent) * size(basket))'))
    .withColumn('rule_score', f.expr('{0} * match_score'.format(metric)))
    .groupBy('order_id','position','consequent')
      .agg(
        f.first('next_products').alias('next_products'),
        f.sum('rule_score').alias('consequent_score')
        )
    .withColumn('rec_rank', f.expr('row_number() over(partition by order_id, position order by consequent_score desc)'))
    .selectExpr('order_id','position','consequent as rec_product','rec_rank','next_products')
    )

# persist recommendations for evaluation
recommendations.write.format('delta').mode('overwrite').option('overwriteSchema','true').save(config['root_path'] + '/tmp/recommendations')

# present results
display(spark.table('DELTA.`{0}/tmp/recommendations`'.format(config['root_path'])))

# COMMAND ----------

# MAGIC %md ## Step 3: Evaluate Recommendations
# MAGIC
# MAGIC To evaluate our recommender, we can consider our recommended products relative to the products that will be added to the shopper's cart is it is completed.  While this doesn't tell us anything about the power of our recommender to drive behavior, it does help us understand how well aligned our recommendations are relative to a customer's interests.  An online A/B test would be needed to see how a customer actually responds to the recommendations, but at least we can make sure we aren't completely off-base before we test them in the wild.
# MAGIC
# MAGIC To score the recommendations, we'll use the [MAP@K](https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52) metric. MAP@K calculates the mean average precision of recommendations given some k-number of recommendations being presented to a user.  The higher the number of hits within the sequence and the higher in order those hits occur, the more the value moves from 0.0 to 1.0.
# MAGIC
# MAGIC For k, we'll use a value of 10.  Why 10?  Given the illustration above intended to suggest the manner in which this kind of recommendation may be presented, a list of 10 recommended products seems to make sense.  A different layout might suggest a different value for k:

# COMMAND ----------

# DBTITLE 1,Calculate MAP@10
k = 10

display(
  spark
    .table('DELTA.`{0}/tmp/recommendations`'.format(config['root_path']))
    .filter(f.expr('rec_rank <= {0}'.format(k))) # get top k recommendations 
    .withColumn('hit', f.expr('case when array_contains(next_products, rec_product) then 1.0 else 0.0 end')) # record a hit if recommended product in collection of next products
    .withColumn('tot_hits', f.expr('sum(hit) over(partition by order_id, position order by rec_rank)')) # calculate cumulative hits by recommendation position
    .withColumn('raw_score', f.expr('tot_hits/rec_rank')) # divide cumulative hits by recommendation position
    .groupBy('order_id','position') # group by basket
      .agg(
        f.first('next_products').alias('next_products'), # bring next products across
        f.sum('raw_score').alias('score') # sum scores across recommendations for this basket
        )
    .withColumn('apk', f.expr('score/least(size(next_products),{0})'.format(k))) # average precision at k is score / lesser of next selections or k
    .groupBy()
      .agg(f.avg('apk').alias('map@{0}'.format(k))) # average scores to get mean avg precision at k
  )

# COMMAND ----------

# MAGIC %md MAP@K is difficult to evaluate outright.  Instead, it's often used in comparison with other MAP@K values to identify which of two algorithms performs better. To provide us a means of comparison, let's build a naive model where we always suggest the most popular products and calculate MAP@K for it:

# COMMAND ----------

# DBTITLE 1,Assemble Naïve Recommendations
# calculate product "popularity" in prior period
most_popular_products = (
  spark
    .table('instacart.orders')
    .filter(f.expr("eval_set='prior'"))
    .join(spark.table('instacart.order_products'), on='order_id')
    .groupBy('product_id')
      .agg(f.count('*').alias('purchases'))
    )

# rank products for basket-specific recommendations
naive_recs = (
    basket_at_position
     .join( spark.table('DELTA.`{0}/tmp/recommendations`'.format(config['root_path'])), on=['order_id','position'], how='leftsemi')
     .crossJoin(most_popular_products)
     .filter(f.expr('not array_contains(basket, product_id)')) # product not already in basket
     .withColumn('rec_rank', f.expr('row_number() over(partition by order_id, position order by purchases DESC)'))
     .withColumnRenamed('product_id','rec_product')
     )

# merge with next_products to enable evaluation
baskets_with_naive_recs = (
  spark
    .table('DELTA.`{0}/tmp/recommendations`'.format(config['root_path']))
    .select('order_id', 'position', 'rec_rank', 'next_products')
    .join( naive_recs, on=['order_id','position','rec_rank'])
    .select('order_id','position','rec_rank','rec_product','next_products')
  )

display(baskets_with_naive_recs.orderBy('order_id', 'position', 'rec_rank'))

# COMMAND ----------

# DBTITLE 1,Evaluate Naïve Recommendations
k = 10

display(
  baskets_with_naive_recs
    .filter(f.expr('rec_rank <= {0}'.format(k))) # get top k recommendations 
    .withColumn('hit', f.expr('case when array_contains(next_products, rec_product) then 1.0 else 0.0 end')) # record a hit if recommended product in collection of next products
    .withColumn('tot_hits', f.expr('sum(hit) over(partition by order_id, position order by rec_rank)')) # calculate cumulative hits by recommendation position
    .withColumn('raw_score', f.expr('tot_hits/rec_rank')) # divide cumulative hits by recommendation position
    .groupBy('order_id','position') # group by basket
      .agg(
        f.first('next_products').alias('next_products'), # bring next products across
        f.sum('raw_score').alias('score') # sum scores across recommendations for this basket
        )
    .withColumn('apk', f.expr('score/least(size(next_products),{0})'.format(k))) # average precision at k is score / lesser of next selections or k
    .groupBy()
      .agg(f.avg('apk').alias('map@{0}'.format(k))) # average scores to get mean avg precision at k
  )

# COMMAND ----------

# MAGIC %md Our market basket recommender out performs a naive recommendation of the most popular products.  With such a large assortment of products and only a limited number of recommendations we can make, we wouldn't expect to perfectly address every need with the recommendation, but we are meeting some and more than if we had simply made a simple, *most-popular* recommendation.

# COMMAND ----------

# MAGIC %md ## Step 4: Deploy Recommender
# MAGIC
# MAGIC How might we deploy this recommender into a production environment?  The logic for assembling recommendations was spelled out earlier before it was implemented against a Spark DataFrame.  This kind of approach is fine for testing ideas but would not meet the performance expectations of an app or website.  Instead, we might need to develop a custom application written using a reasonably performant language such as Java which we could then deploy behind a REST API as part of a microservices layer.  This service would call out to a data store or database to retrieve rules that we would periodically retrain.
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/mba_deployment_01.png' width=600>
# MAGIC
# MAGIC To support this kind of deployment pattern, we need the ability to publish rules data to a relevant data store or database.  Documentation on how to write data to a variety of such destinations can be found [here](https://docs.databricks.com/data/data-sources/index.html).