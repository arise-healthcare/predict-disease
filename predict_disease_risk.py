from flask import Flask, jsonify, request
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import json
import urllib
import pickle
import glob

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# @app.route('/')
# def hello_world():
#     return "Hello, World!"
#
# @app.route('/about')
# def about():
#     return "<h1>About:</h1>"
#
# @app.route('/about/<whom>')
# def hello(whom):
#     return f"<h1>Hello {whom}</h1>"

@app.route('/predict/<string:model_dir_path>', methods=['POST'])
# @app.route('/predict', methods=['POST'])
def predict_lgb(model_dir_path):
    if request.method == 'POST':
        data = request.get_json()

        #decode url encoding by utf-8
        med_list = data["ismedicine"]
        data["ismedicine"] = []
        for med in med_list:
            med_name = urllib.parse.unquote(med)
            data["ismedicine"].append(med_name)

        #change to pd.DataFrame
        d2 = {}
        for k, v in data.items():
            d2[k] = pd.Series(v)
        df=pd.DataFrame(d2)
        print(df)

        #create input values
        #age
        df['H25_age'] = df['age']
        #sex
        df['H25_sex']=df['sex'].replace({"man": 1,"woman":2})
        print(df['H25_sex'].unique())
        #BMI
        df['H25_bmi']=df['weight']/df['height']*df['height']
        #dbp
        df['H25_dbp'] = df['dbp']
        #sbp
        df['H25_sbp'] = df['sbp']
        #LHrate
        df['H25_lhrate']=df['ldl']/df['hdl']
        #HbA1c
        df['H25_a1c'] = df['hba1c']
        # taking medicine for dm
        df['H25_dm'] = 1
        # taking medicine for dl
        df['H25_dl'] = 1
        # taking medicine for ht
        df['H25_ht'] = 1

        print(model_dir_path)

        cols = np.load(model_dir_path+'/lgb_columns.npy', allow_pickle=True)
        in_df = df[cols].iloc[0]
        print(in_df)

        n_split = 5
        test_pred = pd.DataFrame()

        # get pkl models from directory
        files = glob.glob(model_dir_path +'/*.pkl')
        files = [os.path.basename(x) for x in files]
        print(files)

        for fold in range(n_split):
            print(fold)
            with open(model_dir_path +'/'+ files[fold], "rb") as f:
                print(model_dir_path +'/'+ files[fold])
                clf = pickle.load(f)
            print(clf.feature_name())
            print(fold)

            test_pred['fold_{}'.format(fold + 1)] = clf.predict(in_df)

        test_pred['average'] = test_pred[['fold_{}'.format(fold + 1) for fold in range(n_split)]].mean(axis=1)
        prediction = test_pred['average']
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0,0,0,0', port=8080)
