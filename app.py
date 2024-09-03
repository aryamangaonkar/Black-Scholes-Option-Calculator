import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm

def calculate_option_values(s, sk, rf_rate, sigma, t):
    dt = t / 365
    d1 = (np.log(s / sk) + ((rf_rate + (0.5 * sigma**2)) * dt)) / (sigma * np.sqrt(dt))
    d2 = d1 - (sigma * np.sqrt(dt))
    
    call_val = s * norm.cdf(d1) - (sk * norm.cdf(d2) * np.exp(-rf_rate * dt))
    put_val = sk * np.exp(-rf_rate * dt) * norm.cdf(-d2) - s * norm.cdf(-d1)
    
    return round(call_val, 3), round(put_val, 3)

def calculate_profit_loss(option_type, option_val, s, sk, ask_price):
    if option_type == 'call':
        intrinsic_val_exp = max(0, s - sk)
        profit_loss = option_val - (intrinsic_val_exp - ask_price)
    elif option_type == 'put':
        intrinsic_val_exp = max(0, sk - s)
        profit_loss = (intrinsic_val_exp - ask_price) - option_val
    else:
        raise ValueError("Invalid option type")
    
    return profit_loss

# Custom color maps
def create_custom_colormap():
    return LinearSegmentedColormap.from_list(
        'red_green', ['red', 'white', 'green'], N=256
    )

# Streamlit app
def main():
    st.title("Black Scholes Option Pricing Calculator")
    st.sidebar.header("Created by : Arya Mangaonkar")
    st.write("This app uses the Black Scholes model to calculate the value of a European call")

    # Move inputs to the sidebar
    ticker = st.sidebar.text_input("Enter the ticker symbol:", max_chars=5)

    if ticker:
        stock = yf.Ticker(ticker)
        s = stock.history(period='1d')['Close'][0]
        st.sidebar.write(f"Current Stock Price: ${s:.2f}")

        sk = st.sidebar.number_input("Enter the strike price:", min_value=0.0, format="%.2f")
        rf_rate = st.sidebar.number_input("Enter the risk-free rate (per annum, %):", min_value=0.0, format="%.2f") / 100
        sigma = st.sidebar.number_input("Enter the volatility (per annum, %):", min_value=0.0, format="%.2f") / 100
        t = st.sidebar.number_input("Enter the time to expiry (days):", min_value=0.0, format="%.2f")
        
        min_buy_price = st.sidebar.number_input("Enter minimum buy price:", value=s-20, min_value=0.0, format="%.2f")
        max_buy_price = st.sidebar.number_input("Enter maximum buy price:", value=s+20, min_value=0.0, format="%.2f")

        if st.sidebar.button("Calculate"):
            call_val, put_val = calculate_option_values(s, sk, rf_rate, sigma, t)
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                    <div style="flex: 1; border-radius:10px; background-color:green; padding: 15px; margin-right: 10px;">
                        <p style="text-align: center;">Call Value: <b>${call_val:.2f}</b></p>
                    </div>
                    <div style="flex: 1; border-radius:10px; background-color:red; padding: 15px;">
                        <p style="text-align: center;">Put Value: <b>${put_val:.2f}</b></p>        
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

            st.write("The Plots show the change in the profits with a change in the spot prices ")

            buy_prices = np.linspace(min_buy_price, max_buy_price, num=100)
            
            call_profits = [calculate_profit_loss('call', call_val, s, sk, price) for price in buy_prices]
            put_profits = [calculate_profit_loss('put', put_val, s, sk, price) for price in buy_prices]
            
            cmap = create_custom_colormap()
            
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            # Call Option Heatmap
            cax1 = ax[0].imshow(np.array(call_profits).reshape(-1, 1), cmap=cmap, aspect='auto')
            ax[0].set_title('Call Option Profits')
            ax[0].set_xlabel('Buy Prices')
            ax[0].set_ylabel('Range')
            ax[0].set_yticks([0, 25, 50, 75, 99])
            ax[0].set_yticklabels(np.round(np.linspace(min_buy_price, max_buy_price, num=5), 2))
            fig.colorbar(cax1, ax=ax[0], orientation='vertical', label='Profit/Loss')

            cax2 = ax[1].imshow(np.array(put_profits).reshape(-1, 1), cmap=cmap, aspect='auto')
            ax[1].set_title('Put Option Profits')
            ax[1].set_xlabel('Buy Prices')
            ax[1].set_ylabel('Range')
            ax[1].set_yticks([0, 25, 50, 75, 99])
            ax[1].set_yticklabels(np.round(np.linspace(min_buy_price, max_buy_price, num=5), 2))
            fig.colorbar(cax2, ax=ax[1], orientation='vertical', label='Profit/Loss')

            st.pyplot(fig)

if __name__ == "__main__":
    main()
