import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
import json

app = Flask(__name__)

model = pickle.load(open('hotel_model.sav', 'rb'))
scaler = pickle.load(open('hotel_scaler.sav', 'rb'))
encoders = pickle.load(open('hotel_encoders.sav', 'rb'))



@app.route('/')
def Home():
    return render_template("index.html")



@app.route('/predict', methods=['POST'])
def predict():
    data = {}
    data['Booking_ID'] = request.form.get('Booking_ID')
    data['number of adults'] = request.form.get('number_of_adults')
    data['number of children'] = request.form.get('number_of_children')
    data['number of weekend nights'] = request.form.get('number_of_weekend_nights')
    data['number of week nights'] = request.form.get('number_of_week_nights')
    data['type of meal'] = request.form.get('type_of_meal')
    data['car parking space'] = request.form.get('car_parking_space')
    data['room type'] = request.form.get('room_type')
    data['lead time'] = request.form.get('lead_time')
    data['market segment type'] = request.form.get('market_segment_type')
    data['repeated'] = request.form.get('repeated')
    data['P-C'] = request.form.get('previosly_canceled')
    data['P-not-C'] = request.form.get('previosly_not_canceled')
    data['average price '] = request.form.get('average_price')
    data['special requests'] = request.form.get('special_requests')
    data['date of reservation'] = request.form.get('date')
    if data['car parking space'] == 'on':
        data['car parking space'] = 1
    else:
        data['car parking space'] = 0

    if data['repeated'] == 'on':
        data['repeated'] = 1
    else:
        data['repeated'] = 0

    df = pd.DataFrame([data])
    
    df['date of reservation'] = df['date of reservation'].replace({'2018-2-29':'3/1/2018'})    
    df['year'] = pd.to_datetime(df['date of reservation']).dt.year
    df['total_nights']= df['number of weekend nights'] + df['number of week nights']
    
    
    for i in encoders['type of meal'].categories_[0]:
        df['type of meal' + '_' + i] = 0.0
    df['type of meal' + '_' + df['type of meal']] = 1.0
    df.drop(columns='type of meal', inplace=True)

    for i in encoders['room type'].categories_[0]:
        df['room type' + '_' + i] = 0.0
    df['room type' + '_' + df['room type']] = 1.0
    df.drop(columns='room type', inplace=True)


    for i in encoders['market segment type'].categories_[0]:
        df['market segment type' + '_' + i] = 0.0
    df['market segment type' + '_' + df['market segment type']] = 1.0
    df.drop(columns='market segment type', inplace=True)
    
    
    df.drop(columns=['Booking_ID', 'date of reservation'], inplace=True)
    
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    print(df.head())
    pred = model.predict(df)
    return render_template('index.html', predection_text="predection is: "+encoders['booking status'].inverse_transform(pred)[0].replace('_', ' '))

if __name__ == "__main__":
    app.run(debug=True)