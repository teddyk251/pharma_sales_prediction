import sys
sys.path.append("../scripts/")
sys.path.append("../dashboard/")
import dvc.api


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st


import scripts.feature_engineering as feature_engineering
from scripts.preprocess import Preprocess


def plot_predictions(date, sales):
    fig = plt.figure(figsize=(20, 7))
    ax = sns.lineplot(x=date, y=sales)
    ax.set_title("Predicted Sales", fontsize=24)
    ax.set_xlabel("Row index", fontsize=18)
    ax.set_ylabel("Sales", fontsize=18)

    return fig      

def prediction_graph(res_dataframe):

    fig = plt.figure(figsize=(18, 5))
    ax = sns.lineplot(x=res_dataframe.index,
                    y=res_dataframe["Actual Sales"], label='Actual')
    ax.set_xlabel(xlabel="Day", fontsize=16)
    ax.set_ylabel(ylabel="Sales", fontsize=16)
    # plt.show()
    return fig

@st.cache
def load_data_dvc(tag:str, data_path: str, repo:str) -> pd.DataFrame:
        """
        Load data from a file.
        """
        df = None
        try:
            with dvc.api.open(
                repo=repo, 
                path=data_path, 
                rev=tag,
                mode="r"
            ) as fd:
                df = pd.read_csv(fd)
                print("Data loaded from DVC.")
        except Exception:
            print(f"{Exception}")
        return df



def app():

    # Load Saved Results Data
    # data = load_data()
    # columns = data.columns.to_list()
    cols = ['Store', 'Customers', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 
    'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promos2SinceWeek', 'Promo2SinceYear',
    'PromoInterval', 'Open','Date', 'DayOfWeek'  ]
    


    st.title("Rossmann Pharmaceuticals Sales Forecaster")
    input_data = pd.DataFrame()
    submitted = False
    preprocess = Preprocess()
    df = None

    uploaded_file = st.file_uploader("Upload CSV to predict sales", type=".csv")
    manual_input = st.checkbox('Manually Input Values')
    data_added = False
    if manual_input:
        data_added = True
        with st.form(key='inference', clear_on_submit=True):
            
            for feature in cols:
                values = []
                value = st.text_input(feature)
                values.append(value)
                input_data[feature] = values

            submitted = st.form_submit_button("Predict!")
       
        if submitted:
            print(input_data)
            st.dataframe(input_data)

        # input_data = preprocess.prepare_data(input_data)
        feature_eng = feature_engineering.FeatureEngineering(input_data)
        feature_eng.preprocess()
    elif uploaded_file is not None:
        print("File uploaded")
        data_added = True
        data = pd.read_csv(uploaded_file)
        data['Date'] = pd.to_datetime(data['Date'])
        # data = preprocess.prepare_data(data)
        feature_eng = feature_engineering.FeatureEngineering(data)
        feature_eng.preprocess()
    
    else:
        st.write("Cannot use both options")
    if data_added:
        df = feature_eng.df

        model = load_data_dvc('model-v1-random_reg', 'models/28-05-2022-15-52-45.pkl','https://github.com/teddyk251/pharma_sales_prediction')
        prediction = model.predict(df)
        fig = plot_predictions([*range(len(df['Date']))], prediction)
        st.pyplot(fig)


