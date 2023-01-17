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

cols = model.feature_names_in_.tolist()
states_ordered = pd.read_csv("states_ordered.csv")


road_features = cols[5:8] + cols[9:11]
weather_types = [x[8:] for x in cols[11:21]]

#manually defining continuous features to define min/max vals in form input
con_features =  [["Visibility(mi)",0,100], ["Precipitation(in)",0,100], ["Temperature(F)",-150,150], ["Wind_Speed(mph)",0,100]]

days = [x[4:] for x in cols[-7:]]
hours = list(range(0,24))

con_vals = cols[0:4]

cat_vals = ["state_ordered","weather", "Junction", "Stop", "Traffic_Signal",
"Sunrise_Sunset", "Day", "Hour", "Station", "Crossing"]




weights = pd.Series(data=np.round(model.feature_importances_,3),index=cols).sort_values(ascending=False).head(10)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", road_features=road_features, weather_types=weather_types, con_features=con_features,
    hours=hours, days=days, weights=weights, states_ordered=states_ordered)

@app.route("/prediction", methods=["POST"])
def make_prediction():
    #list of all features
    features = road_features + [con_feature[0] for con_feature in con_features] + ["weather", "Day", "Hour", "TimeOfDay", "state_ordered"] 
    
    #object to store form response in
    feature_vals = {}
    for feature in features:
        #loop through features and get form value for each
        feature_vals[feature] = request.form.get(feature)
    
    #dataframe to hold response converted to appropriate data for model
    feat_df = pd.DataFrame(data=np.zeros((1,30)), columns=cols)
    #loop through our columns and assign values in dataframe
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

                #state handling
                elif key == "state_ordered":
                    if feature_vals[key]:
                        #assign the index (rank) of the state to df
                        feat_df[col] = states_ordered[states_ordered["State"] == feature_vals[key]].index[0]

                #con, ordinal value handling        
                else:
                    if feature_vals[key]:
                        feat_df[col] = float(feature_vals[key])
    
    #query model with dataframe
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
    hours=hours, days=days, weights=weights, graphJSON=graphJSON, text_out=text_out, params=params, states_ordered=states_ordered)
