from flask import Flask, request
from flask_cors import CORS, cross_origin
import json
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
@app.route('/', methods = ['POST'])

def index():
  # headers = {
  #   'Access-Control-Allow-Origin': '*', 
  #   'Access-Control-Allow-Headers': 'Origin, Content-Type, X-Auth-Token', 
  #   'Access-Control-Max-Age': '3600', 
  #   'Access-Control-Allow-Methods': 'POST'
  # }
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
  print(r_payment)
  print(accesibility)
  print(alcohol)
  print(dress_code)
  print(parking_lot)
  print(price)
  print(lat)
  print(lng)
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
  print(item)