import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

# 🌟 Stylish Page Heading
st.markdown(
    """
    <style>
        .header {
            background-color: black;
            color: white;
            text-align: center;
            padding: 30px;
            font-size: 36px;
            font-weight: bold;
            border-radius: 10px;
        }
        .stSelectbox, .stButton>button {
            font-size: 18px !important;
        }
        .stAlert {
            font-size: 16px;
        }
        .table-container {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="header">📊 Stock Price Prediction</div>', unsafe_allow_html=True)

# -------------------- Fetch Stock Data --------------------
def fetch_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="5y")
    if data.empty:
        return None
    return data[['Close']]

# -------------------- Data Preprocessing --------------------
def normalize_data(df):
    if df is None or df.empty:
        return None, None
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    return scaled_data, scaler

# -------------------- Model Training --------------------
def train_lstm(data):
    X, y = [], []
    for i in range(50, len(data)):
        X.append(data[i-50:i, 0])
        y.append(data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)
    return model

# -------------------- Streamlit UI --------------------
stock_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
selected_stock = st.selectbox('💡 Select a Stock:', stock_list)

data = fetch_data(selected_stock)

if data is None:
    st.error("⚠️ No data available for the selected stock. Please try another.")
else:
    st.markdown(f"📉 **Latest Price:** ${data['Close'].iloc[-1]:.2f}")

    if st.button("🚀 Train and Predict"):
        st.markdown("🔄 **Processing...** Please wait...")

        scaled_data, scaler = normalize_data(data)
        if scaled_data is None:
            st.error("⚠️ Not enough data for training. Try a different stock.")
        else:
            model = train_lstm(scaled_data)
            predictions = model.predict(scaled_data[-50:].reshape(1, 50, 1))

            predicted_price = scaler.inverse_transform(predictions)[0, 0]

            st.success(f"📈 **Predicted Next Price:** ${predicted_price:.2f}")

            # -------------------- Show Table --------------------
            df_display = data.tail(10).reset_index()
            df_display['Close'] = df_display['Close'].apply(lambda x: f"${x:.2f}")

            st.markdown('<div class="table-container">', unsafe_allow_html=True)
            st.write("📜 **Recent Data:**")
            st.dataframe(df_display.style.set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#212529'), ('color', 'white'), ('font-size', '16px')]},
                {'selector': 'td', 'props': [('font-size', '14px')]},
            ]))
            st.markdown('</div>', unsafe_allow_html=True)

            # -------------------- Plot Graph --------------------
            st.markdown("📊 **Stock Price Trend:**")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data.index, data['Close'], label="Actual Price", color="#0077b6")
            ax.axhline(y=predicted_price, color='r', linestyle="dashed", label="Predicted Price")
            ax.legend()
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)
