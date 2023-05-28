
from flask import Flask ,render_template ,request , jsonify

from helpers.dummies import *
import joblib

model=joblib.load('models/gb.pkl')
scaler=joblib.load('models/scaler.pkl')

app=Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    
    all_data=request.args
    
    SeniorCitizen=int(all_data['SeniorCitizen'])
    
    tenure=int(all_data['tenure'])
   
    MonthlyCharges=int(all_data['monthlycharges'])
    TotalCharges=int(all_data['totalcharges'])
    x=[SeniorCitizen , tenure , MonthlyCharges , TotalCharges]
    # Dummies_Features
   
    Internet=Internet_services_dummies[all_data['Internet']]
    OnlineSecurity=OnlineSecurity_dummies[all_data['OnlineSecurity']]
    OnlineBackup=OnlineBackup_dummies[all_data['OnlineBackup']]
    DeviceProtection=DeviceProtection_dummies[all_data['DeviceProtection']]
    TechSupport=TechSupport_dummies[all_data['TechSupport']]
    StreamingTV=StreamingTV_dummies[all_data['StreamingTV']]
    StremingMovies=StreamingMovies_dummies[all_data['StremingMovies']]
    Contract=Contract_dummies[all_data['Contract']]
    PaymentMethod=PaymentMethod_dummies[all_data['PaymentMethod']]

    x+=Internet+OnlineSecurity+OnlineBackup+DeviceProtection+TechSupport+StreamingTV+StremingMovies+Contract+PaymentMethod
    x=scaler.transform([x])
    pred=model.predict(x)
    
    # Retrieve the value from the prediction array and convert it to an integer
    pred_value = int(pred[0])

    if pred_value == 0:
        result = "Not Churn"
    else:
        result = "Churn"

    return  render_template('prediction.html' , prediction= result)   



if __name__ =='__main__':
    app.run()