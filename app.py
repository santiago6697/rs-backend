import json
import numpy as np
import pandas as pd
from flask import Flask, request
from flask_cors import CORS, cross_origin
from sklearn.neighbors import NearestNeighbors

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
restaurants_df.alcohol = restaurants_df.alcohol.map({'No_Alcohol_Served':1,'Wine-Beer':2,'Full_Bar':3})
restaurants_df.dress_code = restaurants_df.dress_code.map({'informal':1,'casual':2,'formal':3})
restaurants_df.accessibility = restaurants_df.accessibility.map({'no_accessibility':1,'completely':2,'partially':3})
restaurants_df.price = restaurants_df.price.map({'low': 2, 'medium': 1, 'high': 3})
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