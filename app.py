import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
import plotly
import plotly.express as px 

from flask import Flask, render_template, request

#global vars
#df = pd.read_csv('cleaned_data.csv', index_col="Unnamed: 0")
model = joblib.load("classifier.pkl")

road_features = ["Junction", "Stop", "Traffic_Signal", "Station", "Give_Way", "Crossing", "Railway"]
con_features =  [["Visibility(mi)",0,100], ["Precipitation(in)",0,100], ["Temperature(F)",-150,150], ["Wind_Speed(mph)",0,100]]
weather_types = ['cloudy', 'fair', 'rain', 'fog', 'snow', 'storm', 'smoke', 'windy',
       'dust', 'hail']

days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
hours = list(range(0,24))

cols = ['Junction_ohe', 'Stop_ohe', 'Traffic_Signal_ohe',
       'Hour_ohe', 'Station_ohe', 'Give_Way_ohe', 'Crossing_ohe',
       'Railway_ohe', 'Visibility(mi)', 'Precipitation(in)', 'Temperature(F)', 'Wind_Speed(mph)',
       'weather_cloudy', 'weather_dust', 'weather_fair', 'weather_fog',
       'weather_hail', 'weather_rain', 'weather_smoke', 'weather_snow',
       'weather_storm', 'weather_windy', 'Sunrise_Sunset_Day',
       'Sunrise_Sunset_Night', 'Day_Friday', 'Day_Monday', 'Day_Saturday',
       'Day_Sunday', 'Day_Thursday', 'Day_Tuesday', 'Day_Wednesday']

weights = pd.Series(data=np.round(model.feature_importances_,3),index=cols).sort_values(ascending=False).head(10)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", road_features=road_features, weather_types=weather_types, con_features=con_features,
    hours=hours, days=days, weights=weights)

@app.route("/prediction", methods=["POST"])
def make_prediction():
    #list of all features
    features = road_features + [con_feature[0] for con_feature in con_features] + ["weather", "Day", "Hour", "TimeOfDay"] 
    #object to store in
    feature_vals = {}
    for feature in features:
        feature_vals[feature] = request.form.get(feature)
    feat_df = pd.DataFrame(data=np.zeros((1,31)), columns=cols)
    #messy, i know
    for col in cols:
        for key in feature_vals:
            if key in col:
                #road features handling
                if feature_vals[key] == 'on':
                    feat_df[col] = 1
                elif feature_vals[key] == None:
                    feat_df[col] = 0

                #weather handling
                elif key == 'weather':
                    if feature_vals[key] in col:
                        feat_df[col] = 1

                #day & daytime handling
                elif key == 'Day':
                    if feature_vals[key] in col:
                        feat_df[col] = 1

                #con value handling        
                else:
                    if feature_vals[key]:
                        feat_df[col] = float(feature_vals[key])
    
    pred = model.predict_proba(feat_df)[0]
    #define prediction graph
    fig = px.bar(pred)
    fig.update_layout(
    title="Accident Duration Classification Results",
        xaxis_title="Accident Duration (minutes)",
        yaxis_title="Probability",
        showlegend=False,
        paper_bgcolor = "#EEEEEE",
    )

    fig.update_xaxes(
        ticktext=["0-15", "16-30", "31-60", "61-180","181-360","361+"],
        tickvals=[0,1,2,3,4,5]
    )
    # encode plotly graphs in JSON
    #ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #Summary output
    #define map of category to max minutes
    category_map = {0:15,1:30,2:60,3:180,4:360,5:"over 360"}
    #define map of adverb to probability
    adv = ""
    adv_map = {0.1:"could potentially",0.3:"may",0.5:"will likely",0.7:"will most likely"}
    for key in adv_map:
        if np.max(pred) > key:
            adv = adv_map[key]
    text_out = f"This accident {adv} be clear in {category_map[np.argmax(pred)]} minutes."

    params = ""
    for k,v in feature_vals.items():
        if v:
            params += k + ": " + v + ", "

    return render_template("index.html", road_features=road_features, weather_types=weather_types, con_features=con_features,
    hours=hours, days=days, weights=weights, graphJSON=graphJSON, text_out=text_out, params=params)
