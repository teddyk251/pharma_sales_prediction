import sys
sys.path.append("../scripts/")
sys.path.append("../dashboard/")
import dvc.api

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

def load_data_dvc(tag:str, data_path: str, repo:str) -> pd.DataFrame:
        """
        Load data from a csv file.
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



@st.cache
def load_data():
    # merged_df = load_data_dvc('v1.1-merged','data/train_merged_latest.csv','https://github.com/teddyk251/pharma_sales_prediction')
    df = pd.read_csv('data/merged_latest_dashboard.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['day'] = df.index.day
    df['month'] =df.index.month
    df['year'] = df.index.year

    return df


def get_promoVsSales_df(df):
    promo_sales = df.query('Promo == 1')
    no_promo_sales = df.query('Promo == 0')
    promo_sales_agg = promo_sales.groupby("month").agg({"Sales":  "mean"})
    promo_sales_agg = promo_sales_agg.rename(columns={"Sales": "Avg Promotion Sales"})

    no_promo_sales_agg = no_promo_sales.groupby("month").agg({"Sales":  "mean"})
    no_promo_sales_agg = no_promo_sales_agg.rename(columns={"Sales": "Avg Non-Promotion Sales"})


    promo_non_promo_sales_df = pd.merge(promo_sales_agg, no_promo_sales_agg, on="month")

    sales_increase_df = promo_non_promo_sales_df["Avg Promotion Sales"] - promo_non_promo_sales_df["Avg Non-Promotion Sales"]

    promo_non_promo_sales_df["increase percent"] = (sales_increase_df/promo_non_promo_sales_df["Avg Non-Promotion Sales"]) * 100
    return promo_non_promo_sales_df

def app():

    # Load Saved Results Data
    data = load_data()
    
    
# plots.plot_bar(promo_non_promo_sales_df, promo_non_promo_sales_df.index,
#                "increase percent", "The overall sales increase percentage for each month due to promotion",
#                "Month", "Sales increase in percent")

    st.title("Rossmann Pharmaceuticals Data Analysis")

    st.header("Promo vs Sales")
    st.dataframe(get_promoVsSales_df(data))

    # st.subheader("Top 10 users per Session duration")
    # st.dataframe(data['top_ten_per_duration'])


    # st.subheader(f"Top 10 users per Session Frequency")
    # st.dataframe(data['top_ten_per_freq'])
    
    # st.subheader(f"Top 10 users per engagement metric ")
    # st.dataframe(data['top_ten_customers_per_metric'])

