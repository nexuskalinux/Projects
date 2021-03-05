from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


model = pickle.load(open('model/finance.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')

def main():
    return render_template('home.html')



@app.route('/predict' , methods=['POST'])
def home() :
    data1 = request.form['country']
    data2 = request.form['location_type']
    data3 = request.form['cellphone_access']
    data4 = request.form['household_size']
    data5 = request.form['age_of_respondent']
    data6 = request.form['gender_of_respondent']
    data7 = request.form['relationship_with_head']
    data8 = request.form['marital_status']
    data9 = request.form['education_level']
    data10 = request.form['job_type']
   
    
    df = pd.DataFrame([data1,data2,data3,data4,data5,data6,data7,data8,data9,data10])
    
    #categ = [data7, data8, data9, data10]
    #df = pd.get_dummies(df, prefix_sep='_', columns = categ)


    #le = LabelEncoder()
    #df[data2] = le.fit_transform(df[data])
    #df[data3] = le.fit_transform(df[data3])
    #df[data6] = le.fit_transform(df[data6])
    

    prediction = model.predict(df)
    return (render_template('after.html', data=prediction))

if __name__ == "__main__":
    app.run(debug = True)