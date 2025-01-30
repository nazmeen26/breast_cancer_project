import pickle
from flask import Flask, request, jsonify, render_template
from flask import Response
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

model=pickle.load(open("models/randomf.pkl", "rb"))
scaler=pickle.load(open("models/standardscalar.pkl", "rb"))


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    result=""

    if request.method=="POST":
        AgeatDiagnosis=float(request.form.get('AgeatDiagnosis'))
        TypeofBreastSurgery=float(request.form.get('TypeofBreastSurgery'))
        CancerType=float(request.form.get('CancerType'))
        Cellularity=float(request.form.get('Cellularity'))
        Chemotherapy=float(request.form.get('Chemotherapy'))
        Cohort=float(request.form.get('Cohort'))
        ERstatusmeasuredbyIHC=float(request.form.get('ERstatusmeasuredbyIHC'))
        ERStatus=float(request.form.get('ERStatus'))
        NeoplasmHistologicGrade=float(request.form.get('NeoplasmHistologicGrade'))
        HER2statusmeasuredbySNP6=float(request.form.get('HER2statusmeasuredbySNP6'))
        HER2Status=float(request.form.get('HER2Status'))
        HormoneTherapy=float(request.form.get('HormoneTherapy'))
        InferredMenopausalState=float(request.form.get('InferredMenopausalState'))
        PrimaryTumorLaterality=float(request.form.get('PrimaryTumorLaterality'))
        Lymphnodesexaminedpositive=float(request.form.get('Lymphnodesexaminedpositive'))
        MutationCount=float(request.form.get('MutationCount'))
        Nottinghamprognosticindex=float(request.form.get('Nottinghamprognosticindex'))
        OverallSurvival_Months=float(request.form.get('OverallSurvival_Months'))
        PRStatus=float(request.form.get('PRStatus'))
        RadioTherapy=float(request.form.get('RadioTherapy'))
        RelapseFreeStatus_Months=float(request.form.get('RelapseFreeStatus_Months'))
        RelapseFreeStatus=float(request.form.get('RelapseFreeStatus'))
        Geneclassifiersubtype=float(request.form.get('Geneclassifiersubtype'))
        TumorSize=float(request.form.get('TumorSize'))
        TumorStage=float(request.form.get('TumorStage'))
        PatientsVitalStatus=float(request.form.get('PatientsVitalStatus'))
        Cancer_Type_Detailed=float(request.form.get('Cancer_Type_Detailed'))
        CPam50low_subtype=float(request.form.get('CPam50low_subtype'))
        Tumor_Other_Histologic_Subtype=float(request.form.get('Tumor_Other_Histologic_Subtype'))
        Integrative_Cluster=float(request.form.get('Integrative_Cluster'))
        Oncotree_Code=float(request.form.get('Oncotree_Code'))
        
        



        new_data=scaler.transform([[AgeatDiagnosis, TypeofBreastSurgery, CancerType, Cellularity,
                                    Chemotherapy, Cohort, ERstatusmeasuredbyIHC, ERStatus,
                                    NeoplasmHistologicGrade, HER2statusmeasuredbySNP6, HER2Status,
                                    HormoneTherapy, InferredMenopausalState, PrimaryTumorLaterality,
                                    Lymphnodesexaminedpositive, MutationCount,
                                    Nottinghamprognosticindex, OverallSurvival_Months,PRStatus, RadioTherapy,
                                    RelapseFreeStatus_Months, RelapseFreeStatus,
                                    Geneclassifiersubtype, TumorSize, TumorStage,
                                    PatientsVitalStatus, Cancer_Type_Detailed, CPam50low_subtype,
                                    Tumor_Other_Histologic_Subtype, Integrative_Cluster, Oncotree_Code]])

        predict=model.predict(new_data)
        
        if predict[0] ==1 :
            result = 'Deceased'
        else:
            result ='Living'

        return render_template('single_prediction.html',result=result)


    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")




