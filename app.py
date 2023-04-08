import os
import pickle
import cv2
from flask import Flask,render_template,request
import sqlite3
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from werkzeug.utils import secure_filename
import keras
from sklearn.model_selection import train_test_split

import numpy as np

app=Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/heart_disease')
def heart_disease():
    return render_template('heart_disease.html')

@app.route('/heart_disease_result',methods=['POST','GET'])
def heart_disease_result():
    df = pd.read_csv('model/Heart_train.csv')
    df["sex"] = df["sex"] .map({"female" : 1, "male" : 0})
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1:]

    value =''
    
    if request.method == 'POST':
        name = str(request.form['name'])
        age = float(request.form['age'])
        sex = str(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])
        all_data=[name,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        if sex == 'male':
            sex = 0
        else :
            sex =1

        user_data = np.array(
            (age,
             sex,
             cp,
             trestbps,
             chol,
             fbs,
             restecg,
             thalach,
             exang,
             oldpeak,
             slope,
             ca,
             thal)
        ).reshape(1, 13)

        rf = RandomForestClassifier(
            n_estimators=16,
            criterion='entropy',
            max_depth=9
        )

        rf.fit(np.nan_to_num(X), Y)
        rf.score(np.nan_to_num(X), Y)
        predictions = rf.predict(user_data)

        if int(predictions[0]) == 1:
            data = 'Have a heart attack'
        elif int(predictions[0]) == 0:
            data = "don't have a heart attack"
    return render_template('heart_disease_result.html',data=data,all_data=all_data)


@app.route('/blood_analysis')
def blood_analysis():
    return render_template('blood_analysis.html')


@app.route('/blood_analysis_result',methods=['POST','GET'])
def blood_analysis_result():
    if request.method == 'POST' :
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        glucouse=float(request.form['glucouse'])
        insuline=float(request.form['insuline'])
        homa=float(request.form['homa'])
        leptin=float(request.form['leptin'])
        adiponcetin=float(request.form['adiponcetin'])
        resistiin=float(request.form['resistiin'])
        mcp=float(request.form['mcp'])
        all_data=[age,bmi,glucouse,insuline,homa,leptin,adiponcetin,resistiin,mcp]

        # ML Part
        loaded_model=joblib.load(open(r"model/bloodmodelRBF", 'rb'))
        clf=loaded_model.predict([[age,bmi,glucouse,insuline,homa,leptin,adiponcetin,resistiin,mcp]])
        for i in range(1):
            if(clf[i]==0):
                data = "No Cancer" 
            elif(clf[i]==1):
                data = "Cancer" 
    return render_template('blood_analysis_result.html',data=data,all_data=all_data)


@app.route('/Chest_Exploration')
def Chest_Exploration():
    return render_template('Chest_Exploration.html')

@app.route('/Chest_Exploration_result',methods=['POST','GET'])
def Chest_Exploration_result():
    if request.method == 'POST':
        name = str(request.form['name'])
        age = int(request.form['age'])
        file = request.files['image']
        all_data=[name,age]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # print('upload_image filename: ' + filename)
            file_path = os.path.join( os.getcwd(),f'static/uploads/{filename}' )

        modedl_path = r"model\chestExploration.hdf5"
        model = keras.models.load_model(modedl_path)
        gray_image = cv2.imread(file_path, 0)
        resized_image = cv2.resize(gray_image, (100, 100))
        scaled_image = resized_image.astype("float32") / 255.0
        sample_batch = scaled_image.reshape(1, 100, 100, 1)  # 1 image, 100, 100 dim , 1 no of chanels
        result = model.predict(sample_batch)
        result[result >= 0.5] = 1  # Normal
        result[result < 0.5] = 0  # Pneimonia
        if result[0][0] == 1:
            result = "Normal"
        else:
            result = "Pneimonia"
        return render_template('Chest_Exploration_result.html' ,result=result,all_data=all_data)


@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')


@app.route('/alzheimer_result',methods=['POST','GET'])
def alzheimer_result():
    if request.method == 'POST' :
        gender=str(request.form['gender'])
        Age=float(request.form['Age'])
        EDUC = int(request.form['EDUC'])
        SES = float(request.form['SES'])
        MMSE=float(request.form['MMSE'])
        eTIV=float(request.form['eTIV'])
        nWBV=float(request.form['nWBV'])
        ASF=float(request.form['ASF'])
        if gender == 'female':
            gender = 0
        else :
            gender = 1
        all_data=[gender,Age ,EDUC, SES, MMSE, eTIV, nWBV, ASF]
        import pickle 
        scaler = pickle.load(open("model/alzheimer.scl","rb"))
        model = pickle.load(open("model/alzheimer.model","rb"))
        # ML Part
        scaled_feature = scaler.transform([[gender,Age ,EDUC, SES, MMSE, eTIV, nWBV, ASF]])
        clf=model.predict(scaled_feature)
    
        if(clf==0):
            data = "Nondemented	" 
        elif(clf==1):
            data = "Demented" 
    return render_template('alzheimer_result.html',data=data,all_data=all_data)


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/diabetes_result',methods=['POST'])
def diabetes_result():
    import pickle
    model = pickle.load(open("model/diabetes-prediction-rfc-model.pkl","rb"))
    if request.method == 'POST':
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bloodpressure = request.form['bloodpressure']
        skinthickness = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        all_data = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]
        prediction = model.predict([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])
        if int(prediction[0]) == 1:
            data = 'diabetic'
        elif int(prediction[0]) == 0:
            data = "not diabetic"

    return render_template('diabetes_result.html', data=data,all_data=all_data)


@app.route('/parkinson')
def parkinson():
    return render_template('parkinson.html')


@app.route('/parkinson_result', methods=['POST'])
def parkinson_result():
    if request.method == 'POST':
        MDVP_Fo_Hz = float(request.form['MDVP_Fo_Hz'])
        MDVP_Fhi_Hz = float(request.form['MDVP_Fhi_Hz'])
        MDVP_Flo_Hz = float(request.form['MDVP_Flo_Hz'])
        MDVP_Jitter = float(request.form['MDVP_Jitter'])
        MDVP_Jitter_Abs = float(request.form['MDVP_Jitter_Abs'])
        MDVP_RAP = float(request.form['MDVP_RAP'])
        MDVP_PPQ = float(request.form['MDVP_PPQ'])
        Jitter_DDP = float(request.form['Jitter_DDP'])
        MDVP_Shimmer = float(request.form['MDVP_Shimmer'])
        MDVP_Shimmer_dB = float(request.form['MDVP_Shimmer_dB'])
        Shimmer_APQ3 = float(request.form['Shimmer_APQ3'])
        Shimmer_APQ5 = float(request.form['Shimmer_APQ5'])
        MDVP_APQ = float(request.form['MDVP_APQ'])
        Shimmer_DDA = float(request.form['Shimmer_DDA'])
        NHR = float(request.form['NHR'])
        HNR = float(request.form['HNR'])
        RPDE = float(request.form['RPDE'])
        DFA = float(request.form['DFA'])
        spread1 = float(request.form['spread1'])
        spread2 = float(request.form['spread2'])
        D2 = float(request.form['D2'])
        PPE = float(request.form['PPE'])
        #ML Part
        all_data = [MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        import joblib
        model = joblib.load("model/Predict Parkinson.model")
        feature= [[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]
        data = model.predict(feature)
        if data == 0:
            data = "Uninfected"
        else:
            data = "infected"
        
    return render_template('parkinson_result.html',data=data,all_data=all_data)





if __name__ == '__main__':
	app.run(debug=True)
