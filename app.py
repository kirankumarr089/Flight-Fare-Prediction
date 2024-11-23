from flask import Flask, request, render_template, jsonify
from flask_cors import cross_origin
#import sklearn
import pickle
import pandas as pd
# import pyrebase
import os

app = Flask(__name__)
model = pickle.load(open("flight_rf.pkl", "rb"))

# config = {
#   'apiKey': "AIzaSyCo783lem5bYK_vzOjBHeXRVYFvKWIyaS0",
#   'authDomain': "flightpriceprediction-d1c0d.firebaseapp.com",
#   'databaseURL': "https://flightpriceprediction-d1c0d-default-rtdb.firebaseio.com",
#   'projectId': "flightpriceprediction-d1c0d",
#   'storageBucket': "flightpriceprediction-d1c0d.appspot.com",
#   'messagingSenderId': "269936056053",
#   'appId': "1:269936056053:web:f20cc207d84ab5215d57bc",
#   'measurementId': "G-N07LPG0MWQ"
# }

# firebase = pyrebase.initialize_app(config)
# db = firebase.database()

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        # Date_of_Journey
        date_of_journey = request.form["Date_of_Journey"]
        Journey_day = int(pd.to_datetime(date_of_journey, format="%Y-%m-%d").day)
        Journey_month = int(pd.to_datetime(date_of_journey, format="%Y-%m-%d").month)

        # Departure
        dep_time = request.form["Dep_Time"]
        Dep_hour = int(pd.to_datetime(dep_time, format="%H:%M").hour)
        Dep_min = int(pd.to_datetime(dep_time, format="%H:%M").minute)

        # Arrival
        arr_time = request.form["Arrival_Time"]
        Arrival_hour = int(pd.to_datetime(arr_time, format="%H:%M").hour)
        Arrival_min = int(pd.to_datetime(arr_time, format="%H:%M").minute)

        # Duration
        dur_hour = abs(Arrival_hour - Dep_hour)
        dur_min = abs(Arrival_min - Dep_min)

        # Total Stops
        Total_stops = int(request.form["stops"])

        # Airline
        airline = request.form['airline']
        airlines = ['Jet Airways', 'IndiGo', 'Air India', 'Multiple carriers', 'SpiceJet', 
                    'Vistara', 'GoAir', 'Multiple carriers Premium economy', 
                    'Jet Airways Business', 'Vistara Premium economy', 'Trujet']
        airline_features = {f'Airline_{name}': 1 if airline == name else 0 for name in airlines}

        # Source
        source = request.form["Source"]
        sources = ['Delhi', 'Kolkata', 'Mumbai', 'Chennai']
        source_features = {f'Source_{name}': 1 if source == name else 0 for name in sources}

        # Destination
        destination = request.form["Destination"]
        destinations = ['Cochin', 'Delhi', 'New_Delhi', 'Hyderabad', 'Kolkata']
        destination_features = {f'Destination_{name}': 1 if destination == name else 0 for name in destinations}

        # Combine features
        features = {
            'Total_Stops': Total_stops,
            'Journey_day': Journey_day,
            'Journey_month': Journey_month,
            'Dep_hour': Dep_hour,
            'Dep_min': Dep_min,
            'Arrival_hour': Arrival_hour,
            'Arrival_min': Arrival_min,
            'Duration_hours': dur_hour,
            'Duration_mins': dur_min,
        }
        features.update(airline_features)
        features.update(source_features)
        features.update(destination_features)

        # Make prediction
        prediction = model.predict([list(features.values())])
        total_price = round(float(prediction[0]), 2)

        model_name = str(model)
        # Create JSON response
        data = {
            "Total_stops": Total_stops,
            "Journey_day": Journey_day,
            "Journey_month": Journey_month,
            "Dep_hour": Dep_hour,
            "Dep_min": Dep_min,
            "Arrival_hour": Arrival_hour,
            "Arrival_min": Arrival_min,
            "traveling_flight": airline,
            "source": source,
            "destination": destination,
            "total_price": total_price,
            "model_name": model_name,
            "date_of_journey": date_of_journey,  # Add date of journey
            "dep_time": dep_time,  # Add departure time
            "arr_time": arr_time   # Add arrival time
        }
        # result = db.child("TravellingDetails").push(data)
        # print("Data sent to Firebase Realtime Database:", result)
        return render_template('result.html', data=data)


@app.route("/images")
@cross_origin()
def images():
    image_folder = 'static/images'  # Folder where images are stored
    images = os.listdir(image_folder)  # List all images in the folder
    image_paths = [os.path.join(image_folder, image) for image in images]  # Create full path for each image
    return render_template('images.html', images=image_paths)

if __name__ == "__main__":
    app.run(debug=True)
