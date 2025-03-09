import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
@st.cache_data
def load_data():
    orders = pd.read_csv("orders_dataset.csv")
    payments = pd.read_csv("order_payments_dataset.csv")
    return orders, payments

orders, payments = load_data()

# Preprocessing Data
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
rfm_data = orders[['customer_id', 'order_id', 'order_purchase_timestamp']]

# RFM Calculation
max_date = rfm_data['order_purchase_timestamp'].max()
recency = rfm_data.groupby('customer_id').order_purchase_timestamp.max().reset_index()
recency['Recency'] = (max_date - recency['order_purchase_timestamp']).dt.days

frequency = rfm_data.groupby('customer_id').order_id.nunique().reset_index()
frequency.columns = ['customer_id', 'Frequency']

monetary = payments.groupby('customer_id').payment_value.sum().reset_index()
monetary.columns = ['customer_id', 'Monetary']

df_rfm = recency.merge(frequency, on='customer_id').merge(monetary, on='customer_id')

df_rfm['R_Score'] = pd.qcut(df_rfm['Recency'], q=4, labels=[4, 3, 2, 1])
df_rfm['F_Score'] = pd.qcut(df_rfm['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4])
df_rfm['M_Score'] = pd.qcut(df_rfm['Monetary'], q=4, labels=[1, 2, 3, 4])
df_rfm['RFM_Score'] = df_rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

# RFM Segmentation
def rfm_segment(score):
    if score >= 10:
        return 'Best Customers'
    elif score >= 8:
        return 'Loyal Customers'
    elif score >= 6:
        return 'Potential Loyalist'
    elif score >= 4:
        return 'At Risk'
    else:
        return 'Lost Customers'

df_rfm['Segment'] = df_rfm['RFM_Score'].astype(int).apply(rfm_segment)

# Dashboard Layout
st.title("📊 RFM Analysis Dashboard")
st.sidebar.header("Filter Data")

segment_filter = st.sidebar.multiselect("Pilih Segment Pelanggan:", df_rfm['Segment'].unique(), default=df_rfm['Segment'].unique())
filtered_data = df_rfm[df_rfm['Segment'].isin(segment_filter)]

st.subheader("📌 RFM Summary")
st.write(filtered_data.describe())

# Visualization
st.subheader("📈 Distribusi Recency")
fig, ax = plt.subplots()
sns.histplot(filtered_data['Recency'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

st.subheader("📈 Distribusi Frequency")
fig, ax = plt.subplots()
sns.histplot(filtered_data['Frequency'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

st.subheader("📈 Distribusi Monetary")
fig, ax = plt.subplots()
sns.histplot(filtered_data['Monetary'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

st.subheader("📊 Segmentasi Pelanggan")
seg_counts = filtered_data['Segment'].value_counts()
st.bar_chart(seg_counts)

st.subheader("📋 Data Pelanggan")
st.dataframe(filtered_data)
