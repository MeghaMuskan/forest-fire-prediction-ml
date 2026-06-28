import pickle
from flask import Flask,request,jsonify,render_template

import numpy as np
import pandas as pf
from sklearn.preprocessing import StandardScaler 

application = Flask(__name__)
app = application

## import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('home.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature',0))
        RH = float(request.form.get('RH',0))
        Ws = float(request.form.get('Ws',0))
        Rain = float(request.form.get('Rain',0))
        FFMC = float(request.form.get('FFMC',0))
        DMC = float(request.form.get('DMC',0))
        ISI = float(request.form.get('ISI',0))
        Classes = float(request.form.get('Classes',0))
        Region = float(request.form.get('Region',0))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        result = ridge_model.predict(new_data_scaled)

        prediction = round(result[0],2)
        
        if prediction < 5:
            status = "Low Fire Risk"
            color = "low"

        elif prediction < 15:
            status = "Moderate Fire Risk"
            color = "moderate"

        else:
            status = "High Fire Risk"
            color = "high"

        # Scale the gauge fill against a realistic FWI ceiling (30) rather than
        # using the raw score as a literal percentage, and cap at 100.
        gauge_pct = round(min((prediction / 30) * 100, 100.0), 1)

        return render_template("home.html",
                                result=prediction,
                                status=status,
                                color=color,
                                gauge_pct=gauge_pct
                            )

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")