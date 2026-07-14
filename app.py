# ==============================
# Crypto Price Prediction and Market Segmentation App
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Pennyworth Crypto Price Prediction App",
    page_icon="📈",
    layout="wide"
)


# ==============================
# Load Data and Model
# ==============================

@st.cache_data
def load_data():
    crypto_df = pd.read_csv("data/crypto_feature_engineered_data.csv")
    prediction_results = pd.read_csv("data/crypto_prediction_results.csv")
    coin_segments = pd.read_csv("data/crypto_coin_segments.csv")
    feature_importance = pd.read_csv("data/crypto_feature_importance.csv")

    crypto_df['date'] = pd.to_datetime(crypto_df['date'])
    prediction_results['date'] = pd.to_datetime(prediction_results['date'])

    return crypto_df, prediction_results, coin_segments, feature_importance


@st.cache_resource
def load_model():
    model = joblib.load("models/crypto_return_model.pkl")
    feature_columns = joblib.load("models/crypto_feature_columns.pkl")

    return model, feature_columns


crypto_df, prediction_results, coin_segments, feature_importance = load_data()
model, feature_columns = load_model()


# ==============================
# App Title
# ==============================

st.title("Pennyworth Crypto Price Prediction and Market Segmentation App")

st.write(
    """
    This app uses machine learning to predict the next-day return of top cryptocurrencies,
    converts the predicted return into an estimated next-day price, and segments cryptocurrencies
    based on their market behavior.
    """
)

st.warning(
    "This project is for educational machine learning practice only. It is not financial advice. DYOR (Do Your Own Research) before making any investment decisions."
)


# ==============================
# Sidebar
# ==============================

st.sidebar.header("Select Cryptocurrency")

coin_options = crypto_df[['coin_id', 'symbol', 'name']].drop_duplicates()

coin_display_names = coin_options['name'].tolist()

selected_coin_name = st.sidebar.selectbox(
    "Choose a cryptocurrency",
    coin_display_names
)

selected_coin_id = coin_options[
    coin_options['name'] == selected_coin_name
]['coin_id'].values[0]

selected_symbol = coin_options[
    coin_options['name'] == selected_coin_name
]['symbol'].values[0]


# ==============================
# Filter Selected Coin Data
# ==============================

selected_coin_data = crypto_df[
    crypto_df['coin_id'] == selected_coin_id
].sort_values(by='date')

latest_row = selected_coin_data.iloc[-1]


# ==============================
# Main Metrics
# ==============================

st.subheader(f"{selected_coin_name} Market Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest Price",
    f"${latest_row['price']:,.4f}"
)

col2.metric(
    "Market Cap",
    f"${latest_row['market_cap']:,.0f}"
)

col3.metric(
    "Volume",
    f"${latest_row['volume']:,.0f}"
)

col4.metric(
    "Market Cap Rank",
    int(latest_row['market_cap_rank'])
)


# ==============================
# Price Chart
# ==============================

st.subheader(f"{selected_coin_name} Historical Price Movement")

price_chart = px.line(
    selected_coin_data,
    x='date',
    y='price',
    title=f"{selected_coin_name} Price History"
)

st.plotly_chart(price_chart, use_container_width=True)


# ==============================
# Volume Chart
# ==============================

st.subheader(f"{selected_coin_name} Trading Volume")

volume_chart = px.line(
    selected_coin_data,
    x='date',
    y='volume',
    title=f"{selected_coin_name} Volume History"
)

st.plotly_chart(volume_chart, use_container_width=True)


# ==============================
# Prepare Latest Row for Prediction
# ==============================

def prepare_latest_features(latest_row, selected_coin_id, feature_columns):
    # Create empty dataframe with all training feature columns
    latest_features = pd.DataFrame(0.0, index=[0], columns=feature_columns).astype(float)

    # Add one empty row
    latest_features.loc[0] = 0

    # Fill numerical features if they exist in the training columns
    for col in feature_columns:
        if col in latest_row.index:
            latest_features.loc[0, col] = latest_row[col]

    # Set selected coin dummy column to 1
    coin_column = "coin_" + selected_coin_id

    if coin_column in latest_features.columns:
        latest_features.loc[0, coin_column] = 1

    return latest_features


latest_features = prepare_latest_features(
    latest_row,
    selected_coin_id,
    feature_columns
)


# ==============================
# Predict Next-Day Return and Price
# ==============================

predicted_return = model.predict(latest_features)[0]

estimated_next_day_price = latest_row['price'] * (1 + predicted_return)


st.subheader("Next-Day Prediction")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Price",
    f"${latest_row['price']:,.4f}"
)

col2.metric(
    "Predicted Next-Day Return",
    f"{predicted_return * 100:.2f}%"
)

col3.metric(
    "Estimated Next-Day Price",
    f"${estimated_next_day_price:,.4f}"
)


# ==============================
# Prediction Interpretation
# ==============================

if predicted_return > 0:
    st.success(
        f"The model predicts a possible positive next-day movement for {selected_coin_name}."
    )
elif predicted_return < 0:
    st.error(
        f"The model predicts a possible negative next-day movement for {selected_coin_name}."
    )
else:
    st.info(
        f"The model predicts almost no next-day movement for {selected_coin_name}."
    )


# ==============================
# Segment Information
# ==============================

st.subheader("Market Segmentation Result")

selected_segment = coin_segments[
    coin_segments['coin_id'] == selected_coin_id
]

if len(selected_segment) > 0:
    selected_segment = selected_segment.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "K-Means Cluster",
        int(selected_segment['KMeans_Cluster'])
    )

    col2.metric(
        "Hierarchical Cluster",
        int(selected_segment['Hierarchical_Cluster'])
    )

    col3.metric(
        "Segment Name",
        selected_segment['Segment_Name']
    )

    st.write(
        f"""
        **{selected_coin_name}** belongs to the **{selected_segment['Segment_Name']}**.
        In the clustering comparison, Hierarchical Clustering performed better based on Silhouette Score.
        """
    )
else:
    st.info("Segment information is not available for this coin.")


# ==============================
# Show All Coin Segments
# ==============================

st.subheader("All Crypto Market Segments")

segment_table = coin_segments[
    [
        'coin_id',
        'symbol',
        'name',
        'latest_market_cap_rank',
        'average_return',
        'average_volatility',
        'return_30d',
        'return_90d',
        'KMeans_Cluster',
        'Hierarchical_Cluster',
        'Segment_Name'
    ]
].sort_values(by='latest_market_cap_rank')

st.dataframe(segment_table, use_container_width=True)


# ==============================
# Cluster Visualization
# ==============================

st.subheader("Crypto Segments Visualization")

cluster_chart = px.scatter(
    coin_segments,
    x='average_volatility',
    y='return_30d',
    color='Segment_Name',
    hover_data=['name', 'symbol', 'latest_market_cap_rank'],
    title="Crypto Market Segments by Volatility and 30-Day Return"
)

st.plotly_chart(cluster_chart, use_container_width=True)


# ==============================
# Model Performance Section
# ==============================

st.subheader("Model Prediction Results")

st.write(
    """
    The model was trained to predict next-day return. The estimated next-day price is calculated
    by applying the predicted return to the current price.
    """
)

prediction_table = prediction_results[
    [
        'date',
        'coin_id',
        'symbol',
        'price',
        'next_day_return',
        'predicted_next_day_return',
        'actual_next_day_price',
        'predicted_next_day_price'
    ]
].copy()

st.dataframe(prediction_table.tail(20), use_container_width=True)


# ==============================
# Actual vs Predicted Return Chart
# ==============================

st.subheader("Actual vs Predicted Next-Day Return")

return_chart = px.scatter(
    prediction_results,
    x='next_day_return',
    y='predicted_next_day_return',
    color='coin_id',
    title="Actual vs Predicted Next-Day Return"
)

st.plotly_chart(return_chart, use_container_width=True)


# ==============================
# Feature Importance
# ==============================

st.subheader("Top Important Features")

top_features = feature_importance.head(15)

feature_chart = px.bar(
    top_features,
    x='Importance',
    y='Feature',
    orientation='h',
    title="Top 15 Features Used by the Model"
)

st.plotly_chart(feature_chart, use_container_width=True)


# ==============================
# Final Note
# ==============================

st.info(
    """
    Note: Crypto next-day return prediction is tricky because the market is highly volatile at this time.
    The model's predictions are based on historical data and may not always be accurate.
    """
)



