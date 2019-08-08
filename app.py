import json
import numpy as np
import pandas as pd
from flask import Flask, request
from flask_cors import CORS, cross_origin
from sklearn.neighbors import NearestNeighbors

import scipy
import math
import random
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

from scipy import sparse

# IMPORT AND DROP
# Restaurants
restaurants_df = pd.read_csv('./geoplaces2.csv')
restaurants_df = restaurants_df.set_index(['placeID']).drop(columns=['the_geom_meter','address','country','fax','zip','smoking_area','url','Rambience','franchise','area','other_services'])
res_accepted_pay = pd.read_csv('./chefmozaccepts.csv')
res_parking = pd.read_csv('./chefmozparking.csv')
# Users
user_df = pd.read_csv('./userprofile.csv')
user_df = user_df.set_index(['userID']).drop(columns=['smoker','ambience','hijos','marital_status','birth_year','interest','personality','religion','activity','color','weight','height'])
usr_payment = pd.read_csv('./userpayment.csv')
# Ratings
# {userID*, placeID*, rating} (* means key)
ratings_df = pd.read_csv('./rating_final.csv')
ratings_df = ratings_df.drop(columns=['food_rating', 'service_rating'])
ratings_df = ratings_df.set_index(['userID', 'placeID'])

# MAPPING AND ENCODING (OHE)

# User Payment
# {userID*, PAYMENT_cash  PAYMENT_credit_card  PAYMENT_debit_card}
map_user_payment = {
   'cash': 'cash',
   'bank_debit_cards': 'debit_card', 
   'MasterCard-Eurocard': 'credit_card', 
   'VISA': 'credit_card',
   'American_Express': 'credit_card',  
}
usr_payment['payment'] = usr_payment['Upayment'].apply(lambda x: map_user_payment[x])
usr_payment = usr_payment[['userID']].join(pd.get_dummies(usr_payment['payment']).add_prefix('PAYMENT_')).groupby('userID').max()

# Restaurants
map_restaurant_payment = {
   'cash': 'cash',
   'bank_debit_cards': 'debit_card', 
   'MasterCard-Eurocard': 'credit_card', 
   'VISA': 'credit_card',
   'Visa': 'credit_card',
   'American_Express': 'credit_card',  
   'Japan_Credit_Bureau': 'credit_card',  
   'Carte_Blanche': 'credit_card',
   'Diners_Club': 'credit_card', 
   'Discover': 'credit_card', 
   'gift_certificates': 'other',  
   'checks': 'other',  
}
res_accepted_pay['payment'] = res_accepted_pay['Rpayment'].apply(lambda x: map_restaurant_payment[x])
res_accepted_pay = res_accepted_pay[['placeID']].join(pd.get_dummies(res_accepted_pay['payment']).add_prefix('PAYMENT_')).groupby('placeID').max()

map_restaurant_parking = {
   'none': 'no',
   'public': 'yes', 
   'yes': 'yes', 
   'valet parking': 'yes',
   'fee': 'yes',
   'street': 'street',  
   'validated parking': 'yes',  
}
res_parking['parking'] = res_parking['parking_lot'].apply(lambda x: map_restaurant_parking[x])
res_parking = res_parking[['placeID']].join(pd.get_dummies(res_parking['parking']).add_prefix('PARKING_')).groupby('placeID').max()

# MERGE
# Users
# {userID*, 'latitude', 'longitude', 'drink_level', 'dress_preference', 'transport', 'budget', 'PAYMENT_cash', 'PAYMENT_credit_card', 'PAYMENT_debit_card'}
user_df = pd.merge(user_df, usr_payment, how='left', on=['userID'])

# Map user feature values
user_df.drink_level = user_df.drink_level.map({'abstemious':1,'casual drinker':2,'social drinker':3})
user_df.dress_preference = user_df.dress_preference.map({'no preference':1,'informal':2,'formal':3,'elegant':4,'?':1})
user_df.transport = user_df.transport.map({'public':1,'on foot':2,'car owner':3,'?':1})
user_df.budget = user_df.budget.map({'low': 2, 'medium': 1, 'high': 3,'?':1})
user_df = user_df.fillna(0)

# Restaurants
# {placeID*,'latitude', 'longitude', 'name', 'city', 'state', 'alcohol', 'dress_code', 'accessibility', 'price', 'PAYMENT_cash', 'PAYMENT_credit_card', 'PAYMENT_debit_card', 'PAYMENT_other', 'PARKING_no', 'PARKING_street', 'PARKING_yes'}
restaurants_df = pd.merge(restaurants_df, res_accepted_pay, how='left', on=['placeID'])
restaurants_df = pd.merge(restaurants_df, res_parking, how='left', on=['placeID'])
# Map restaurant feature values
# restaurants_df.alcohol = restaurants_df.alcohol.map({'No_Alcohol_Served':1,'Wine-Beer':2,'Full_Bar':3})
# restaurants_df.dress_code = restaurants_df.dress_code.map({'informal':1,'casual':2,'formal':3})
# restaurants_df.accessibility = restaurants_df.accessibility.map({'no_accessibility':1,'completely':2,'partially':3})
# restaurants_df.price = restaurants_df.price.map({'low': 2, 'medium': 1, 'high': 3})
restaurants_df = restaurants_df.join(pd.get_dummies(restaurants_df['alcohol']).add_prefix('ALCOHOL_')).groupby('placeID').max().drop(columns=['alcohol'])
restaurants_df = restaurants_df.join(pd.get_dummies(restaurants_df['dress_code']).add_prefix('DRESS_CODE_')).groupby('placeID').max().drop(columns=['dress_code'])
restaurants_df = restaurants_df.join(pd.get_dummies(restaurants_df['accessibility']).add_prefix('ACCESSIBILITY_')).groupby('placeID').max().drop(columns=['accessibility'])
restaurants_df = restaurants_df.join(pd.get_dummies(restaurants_df['price']).add_prefix('PRICE_')).groupby('placeID').max().drop(columns=['price'])


# # At this point, ratings_df, user_df and restaurants_df are clean
# ratings_df.head()
# print(list(restaurants_df.head()))
# user_df.head()

ratings_df = ratings_df.reset_index()
restaurants_df = restaurants_df.reset_index()
restaurants_df = restaurants_df.fillna(0)

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
@app.route('/', methods = ['POST'])

def index():
  response = {
    'data': []
  }
  r_payment = request.json['Rpayment']
  accesibility = request.json['accesibility']
  alcohol = request.json['alcohol']
  dress_code = request.json['dress_code']
  parking_lot = request.json['parking_lot']
  price = request.json['price']
  lat = request.json['lat']
  lng = request.json['lng']
  payment_methods = [0, 0, 0]
  if r_payment == "1":
    payment_methods = [1, 0, 0]
  elif r_payment == "2":
    payment_methods = [0, 1, 0]
  elif r_payment == "3":
    payment_methods = [0, 0, 1]
  
  user_test = np.append([lat, lng, alcohol, dress_code, parking_lot, price], payment_methods)

  indices = similar_user(user_test, 2)
  indices = np.squeeze(indices)

  similar_users = print_similar_user(indices[0])
  # FOR GOD'S SAKE DON'T TOUCH THIS.
  print similar_users

  response = json.dumps(response)
  recommendations = 'qswdeq'
  status = 204
  if recommendations:
    status = 200
    # TODO: For recommendation in recommendations serialize() ...
    # for place in recommendations:
    #   serialize(place)
  resp = Flask.response_class(response = response, mimetype = 'application/json', status = status)

  return resp

def serialize(item):
  # TODO
  print(item)

def print_similar_user(users):
  user_df_mod = user_df.reset_index()
  user_df_mod = user_df_mod.iloc[users]['userID']
  return user_df_mod

def similar_user(user_test,k):
  nbrs = NearestNeighbors(n_neighbors = k, algorithm = 'ball_tree').fit(user_df)
  distances, indices = nbrs.kneighbors([user_test])
  return indices

#Indexing by userID to speed up the searches during evaluation
ratings_indexed_df = ratings_df.set_index('userID')
# ratings_train_indexed_df = ratings_train_df.set_index('userID')
# ratings_test_indexed_df = ratings_test_df.set_index('userID')

# # CONTENT-BASED
item_ids = restaurants_df['placeID'].tolist()
content_matrix = restaurants_df.set_index('placeID').fillna(0).drop(columns=['name','city','state'])

def get_person_items(person_id):
    items_per_person_list = list(ratings_df[ratings_df['userID']==person_id]['placeID'])
    item_list = restaurants_df.set_index('placeID').loc[items_per_person_list]
    item_list['userID'] = person_id
    item_list_cleaned = item_list.fillna(0).drop(columns=['name','city','state'])
    # TODO: Instead of a list, a profile with (1x18) shape should be returned as concensus of all user-rated items
    item_list_cleaned = item_list_cleaned.groupby(['userID']).mean()

    return item_list_cleaned

class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(get_person_items(person_id), content_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        # Commented for evaluation reasons
        # Remove user alrealy interacted items
        # user_interacted_items = get_person_items(user_id).index.values
        # similar_items_filtered = list(filter(lambda x: x[0] not in user_interacted_items, similar_items))
        # similar_items_filtered = list(set(similar_items_filtered))
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['placeID', 'recStrength']).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'placeID', 
                                                          right_on = 'placeID')[['recStrength', 'placeID', 'latitude', 'longitude', 'name', 'city', 'state', 'PAYMENT_cash', 'PAYMENT_credit_card', 'PAYMENT_debit_card', 'PAYMENT_other', 'PARKING_no', 'PARKING_street', 'PARKING_yes', 'ALCOHOL_Full_Bar', 'ALCOHOL_No_Alcohol_Served', 'ALCOHOL_Wine-Beer', 'DRESS_CODE_casual', 'DRESS_CODE_formal', 'DRESS_CODE_informal', 'ACCESSIBILITY_completely', 'ACCESSIBILITY_no_accessibility', 'ACCESSIBILITY_partially', 'PRICE_high', 'PRICE_low', 'PRICE_medium']]

        return recommendations_df
    
content_based_recommender_model = ContentBasedRecommender(restaurants_df)

# COLLABORATIVE FILTERING

#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = ratings_df.pivot(index='userID', 
                                                          columns='placeID', 
                                                          values='rating').fillna(0)

# users_items_pivot_matrix_df.head(10)

users_items_pivot_matrix = users_items_pivot_matrix_df.as_matrix()
# users_items_pivot_matrix[:10]

users_ids = list(users_items_pivot_matrix_df.index)
# users_ids[:10]

#The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15
#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)

# U.shape

# Vt.shape

sigma = np.diag(sigma)
# sigma.shape

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
# all_user_predicted_ratings

#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
# cf_preds_df.head(10)

len(cf_preds_df.columns)

class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['placeID'].isin(items_to_ignore)].sort_values('recStrength', ascending = False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'placeID', 
                                                          right_on = 'placeID')[['recStrength', 'placeID', 'latitude', 'longitude', 'name', 'city', 'state', 'PAYMENT_cash', 'PAYMENT_credit_card', 'PAYMENT_debit_card', 'PAYMENT_other', 'PARKING_no', 'PARKING_street', 'PARKING_yes', 'ALCOHOL_Full_Bar', 'ALCOHOL_No_Alcohol_Served', 'ALCOHOL_Wine-Beer', 'DRESS_CODE_casual', 'DRESS_CODE_formal', 'DRESS_CODE_informal', 'ACCESSIBILITY_completely', 'ACCESSIBILITY_no_accessibility', 'ACCESSIBILITY_partially', 'PRICE_high', 'PRICE_low', 'PRICE_medium']]

        return recommendations_df
    
cf_recommender_model = CFRecommender(cf_preds_df, restaurants_df)

# HYBRID

class HybridRecommender:
    
    MODEL_NAME = 'Hybrid'
    
    def __init__(self, cb_rec_model, cf_rec_model, items_df):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        #Getting the top-1000 Content-based filtering recommendations
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        
        #Getting the top-1000 Collaborative filtering recommendations
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose, 
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCF'})
        
        #Combining the results by contentId
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how = 'inner', 
                                   left_on = 'placeID', 
                                   right_on = 'placeID')
        
        #Computing a hybrid recommendation score based on CF and CB scores
        recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF']
        
        #Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'placeID', 
                                                          right_on = 'placeID')[['recStrengthHybrid', 'placeID', 'latitude', 'longitude', 'name', 'city', 'state', 'PAYMENT_cash', 'PAYMENT_credit_card', 'PAYMENT_debit_card', 'PAYMENT_other', 'PARKING_no', 'PARKING_street', 'PARKING_yes', 'ALCOHOL_Full_Bar', 'ALCOHOL_No_Alcohol_Served', 'ALCOHOL_Wine-Beer', 'DRESS_CODE_casual', 'DRESS_CODE_formal', 'DRESS_CODE_informal', 'ACCESSIBILITY_completely', 'ACCESSIBILITY_no_accessibility', 'ACCESSIBILITY_partially', 'PRICE_high', 'PRICE_low', 'PRICE_medium']]

        return recommendations_df
    
hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, restaurants_df)