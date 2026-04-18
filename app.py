import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# -----------------------------
# USER INPUT
# -----------------------------
st.title("Stock Prediction ANN")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA)", "MSFT")

start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))


if ticker:

    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("Invalid ticker or no data found.")
    else:

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)


        st.subheader("Stock Price Chart")

        fig, ax = plt.subplots()
        ax.plot(df["Close"])
        ax.set_title(f"{ticker} Closing Price")
        st.pyplot(fig)


        df["Tomorrow"] = df["Close"].shift(-1)
        df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

        df = df.dropna()


        predictors = ["Close"]
        X = df[predictors]
        y = df["Target"]


        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # -----------------------------
        # MODEL
        # -----------------------------
        model = Sequential([
            Input(shape=(X_scaled.shape[1],)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # -----------------------------
        # TRAIN
        # -----------------------------
        model.fit(X_scaled, y, epochs=5, batch_size=32, verbose=0)

        # -----------------------------
        # PREDICTION
        # -----------------------------
        if st.button("Predict Next Day"):

            latest = X.tail(1)
            latest_scaled = scaler.transform(latest)

            prediction = model.predict(latest_scaled)[0][0]

            st.subheader("Prediction Result")

            if prediction > 0.5:
                st.success("📈 Price will go UP tomorrow")
            else:
                st.error("📉 Price will go DOWN tomorrow")