import streamlit as st
import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
# import quantstats as qs

st.set_page_config(
    page_title="Crypto Trend Dashboard",  # Title of the page
    layout="wide",              # Use wide layout
    initial_sidebar_state="expanded"  # Sidebar starts expanded
)
# Streamlit dashboard
st.title("Crypto Trend Dashboard")

#################################################################################################
# Helper Functions
# def qs_stats_light(returns, periods=12):

#     cagr = returns.apply(qs.stats.cagr)
#     er = returns.apply(qs.stats.expected_return) * periods
#     vol = returns.apply(qs.stats.volatility, periods=periods)
#     sharpe = returns.apply(qs.stats.sharpe, periods=periods)
#     sortino = returns.apply(qs.stats.sortino, periods=periods)
#     maxdd = returns.apply(qs.stats.max_drawdown)
#     calmar = returns.apply(qs.stats.calmar)
#     # greeks = sreturns.apply(qs.stats.greeks, benchmark=sreturns.SPX, periods=12)
#     result = pd.concat([cagr, er, vol, sharpe, sortino, maxdd, calmar, 1/calmar], axis=1)
#     result.columns =["CAGR", "Mean Return", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar", "InvCalmar"]
#     return result.sort_values("Sharpe", ascending=False).round(3)

#################################################################################################
# Data & Inputs
# Load api data
file_path = 'crypto_ohlc_since_2019-01-01_1d_200.csv'
prices = pd.read_csv(file_path, index_col=0)
prices['date'] = pd.to_datetime(prices['date'])

coins = prices.ticker.unique()

# Fetch the list of top coins
last_date = prices.date.max()
coins_sorted_volume = prices[prices.date == last_date].sort_values(by="volume").ticker
coins_sorted_volume = coins_sorted_volume[coins_sorted_volume.isin(["USDC/USDT", "EURUSDT", "FDUSD/USDT", "WBTC/USDT", "WBETH/USDT"]) == False]

# Input Token
st.sidebar.header("Select a token")
token_options = coins_sorted_volume.tolist()
selected_coin = st.sidebar.selectbox(label="Token", options=token_options, index = token_options.index("BTC/USDT"))

col1, col2 = st.columns(2)
with col1:
    # Input Date
    # Select start date for strategy
    min_date_coin = prices[prices.ticker == selected_coin].date.min()
    selected_startd = st.date_input("Select start date", "2023/01/01")
    selected_startd = str(selected_startd)[:10]

with col2:
    # Input trend signal lookback period
    periods = [20, 40, 60, 180]
    selected_lookback = st.selectbox(
        "Select trend lookback period (days):",
        tuple(periods), index = 1,
)

#################################################################################################

# Calculate momentum signal 20 days
momo = prices.sort_values(by=['ticker', 'date'])
momo["next_day_log_return"] = momo.groupby("ticker").apply(lambda df: np.log(df.close.shift(-1).div(df.close)), include_groups=False).values

for i in periods: #
    momo["momo{}".format(i)] = momo.groupby("ticker").apply(lambda df: np.log(df.close.div(df.close.shift(i))), include_groups=False).values
    momo[f"trend_quintile{i}"] = momo.groupby("ticker")[f"momo{i}"].transform(
            lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1 if len(x.dropna()) >= 5 else np.nan
        )    
    momo['trend{}'.format(i)] = momo['momo{}'.format(i)] > 0

quintile_dict = dict()
for i in periods:
    quintile_dict[i] = momo.groupby(['ticker', f'trend_quintile{i}']).next_day_log_return.mean().reset_index()

# Calculate scaled_trend
def rolling_std(series, window):
    return series.rolling(window=window, min_periods=window).std()

for i in periods: #
    momo['scaled_trend_{}'.format(i)] = momo.groupby("ticker")["momo{}".format(i)].apply(lambda s: s / rolling_std(s, window=20), include_groups=False).values

# Add a `weight` column with the scaled and clipped values
for i in periods: # 
    momo['weight_{}'.format(i)] = 0.75 * momo['scaled_trend_{}'.format(i)].clip(-2, 2)


#################################################################################################

# Top / Bottom trend signals
last_signal = momo.groupby("ticker").last(1)[f"scaled_trend_{selected_lookback}"].sort_values().dropna()#.drop(index=["FUSDUSDT", "EURUSDT"])
bottom = last_signal[:50]
top = last_signal[-50:]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top Signal")
    st.dataframe(top.sort_values(ascending=False), use_container_width=True)

with col2:
    st.subheader("Bottom Signal")
    st.dataframe(bottom, use_container_width=True)

# Fetch historical data for the selected coin
if selected_coin:
    st.subheader(f"Trend Monitor for {selected_coin}")

    trend_factor20 = momo[(momo.date >= selected_startd) & (momo.ticker == selected_coin)]
    trend_factor20["strat_return"] = trend_factor20["weight_{}".format(selected_lookback)] * trend_factor20['next_day_log_return']
    trend_factor20["strat_return"] = (np.exp(trend_factor20["strat_return"]) - 1).add(1) 
    trend_factor20["cumreturn"] = trend_factor20.groupby("ticker")['strat_return'].cumprod().sub(1)

    avg_exposures20 = trend_factor20.groupby("ticker", as_index=False).agg(avg_exposure=('weight_{}'.format(selected_lookback), 'mean'))

    # Merge dataframes and compute derived columns
    trend_strat_long = (
        trend_factor20.reset_index().merge(avg_exposures20, on="ticker")
        .assign(
            long_trend_weight=lambda df: 0.75 * np.clip(df["scaled_trend_{}".format(selected_lookback)], 0, 2), # WHY SCALE BY 0.75?
            # long_trend_weight=lambda df: np.clip(df["scaled_trend_20"], 0, 1),
            scaled_long=lambda df: df["next_day_log_return"] * df["avg_exposure"],
            long_100=lambda df: df["next_day_log_return"],
            trend_strat=lambda df: df["weight_{}".format(selected_lookback)] * df["next_day_log_return"],
            long_trend=lambda df: df["long_trend_weight"] * df["next_day_log_return"],
        )
    )

    # Reshape and calculate cumulative returns
    trend_strat_long1 = (
        trend_strat_long[["date", "ticker", "scaled_long", "long_100", "trend_strat", "long_trend"]]
        .melt(id_vars=["date", "ticker"], var_name="strategy", value_name="returns")
        .sort_values(["ticker", "strategy", "date"])
        .assign(
            cumreturns=lambda df: df.groupby(["ticker", "strategy"])["returns"].cumsum()
        )
    )

    rdata = trend_strat_long1[trend_strat_long1.ticker == selected_coin]
    rdata = rdata.pivot(index="date", columns="strategy", values="returns")

    #ss = qs_stats_light(rdata.dropna(), periods=365)

    col1, col2 = st.columns(2)
    with col1:
        # Plot the data
        fig = px.line(rdata.cumsum().apply(np.exp).sub(1).add(1))
        fig.update_layout(
            title=f"{selected_coin} Trend Strategy since {selected_startd}",
            xaxis_title="Date",
            yaxis_title="NAV (USDT)",
            yaxis_tickformat=".0%",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

         # Heatmap monthly returns
        mrdata = rdata.resample("ME").apply(lambda df: df.add(1).prod().sub(1))
        mrdata["year"] = mrdata.index.year
        mrdata["month"] = mrdata.index.month

        fig_monthly_rets = go.Figure(data=go.Heatmap(
            z=mrdata["long_trend"].values,
            y=mrdata["year"].to_numpy(),
            x=mrdata["month"].to_numpy(),
            colorscale="deep",
            text = [f"{val * 100:.2f}%" for val in mrdata["long_trend"].values],
            texttemplate="%{text}",
            textfont={"size": 12}  # Adjust font size for annotations

            ),
        )
        fig_monthly_rets.update_layout(
            title='Monthly Returns Trend Long Only Strategy',
            xaxis_tickmode = "linear",
            yaxis_tickmode = "linear",
            yaxis_nticks=mrdata["year"].nunique(),
            xaxis_nticks=mrdata["month"].nunique(),
            xaxis_title="Month",
            yaxis_title="Year",
        )
        st.plotly_chart(fig_monthly_rets, use_container_width=True)

    with col2:
        wdata = trend_strat_long[trend_strat_long.ticker == selected_coin].set_index("date")

        fig_w_lo = px.line(wdata.long_trend_weight)
        fig_w_lo.update_layout(
            title="Long Only Trend Signal",
            xaxis_title="Date",
            yaxis_title="Scaled Signal [0, 1.5]",
            #template="plotly_dark",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig_w_lo, use_container_width=True)

        st.text("Average Net Exposure L/S Trend")
        #st.dataframe(ss, use_container_width=True)
        st.dataframe(avg_exposures20, use_container_width=True)


    # Plot scaled trend signal
    signal = momo[momo.ticker == selected_coin].set_index("date")
    fig_signal = px.line(signal["scaled_trend_{}".format(selected_lookback)])
    fig_signal.update_layout(
            title="Long / Short Trend Signal",
            xaxis_title="Date",
            yaxis_title="Unscaled Signal",
            #template="plotly_dark",
            xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_signal, use_container_width=True)

    # Rolling sharpe heatmap
    selected_sharpe_rolling_period = st.selectbox(
        "Select sharpe lookback period (days):",
        tuple(periods), index = 2
    )

    #roll_sharpe = qs.stats.rolling_sharpe(rdata.dropna(), periods_per_year=365, rolling_period=selected_sharpe_rolling_period)
    #m_roll_sharpe = roll_sharpe.asfreq("W")

    # Heatmap 
    # fig_rsharpe = go.Figure(data=go.Heatmap(
    #     z=m_roll_sharpe.T.values,
    #     y=m_roll_sharpe.T.index.to_numpy(),
    #     x=m_roll_sharpe.T.columns.to_numpy(),
    #     colorscale='RdBu_r')
    # )
    # fig_rsharpe.update_layout(
    #         title='Rolling sharpe heatmap (weekly)',
    #         yaxis_nticks=m_roll_sharpe.shape[1], 
    #         xaxis_nticks=m_roll_sharpe.shape[0],
    #         xaxis_title="Date",
    #         yaxis_title="Sharpe Ratio",
    #         height=400
    #         )
    # st.plotly_chart(fig_rsharpe, use_container_width=True)

    st.subheader(f"Trend Statistics for {selected_coin} for selected period")

    trend_quintile_summary = quintile_dict[selected_lookback]
    trend_quintile_summary = trend_quintile_summary[trend_quintile_summary.ticker == selected_coin]

    col1, col2 = st.columns(2)
    with col1: 
        fig_quintiles = px.bar(trend_quintile_summary, x=f"trend_quintile{selected_lookback}", y="next_day_log_return")
        fig_quintiles.update_layout(
            title='Mean next day return quintiles based on trend signal strenght',
            xaxis_title="Date",
            yaxis_title="Mean Return",
            yaxis_tickformat=".2%",
            height=400
            )
        st.plotly_chart(fig_quintiles, use_container_width=True)
    
    with col2:
        fig_scatter = px.scatter(momo[momo.ticker == selected_coin], x=f'momo{selected_lookback}', y='next_day_log_return')
        fig_scatter.update_layout(
            title=f"Previous {selected_lookback} days return vs next day",
            xaxis_title=f"Previous {selected_lookback} days return",
            yaxis_title="Next Day Log Return",
            yaxis_tickformat=".2%",
            xaxis_tickformat=".2%",
            height=400
            )
        st.plotly_chart(fig_scatter, use_container_width=True)
    

################################################################################
st.subheader("Data Management")

def fetch_top_coins_by_volume(limit=200):
    """
    Fetch the top coins ranked by trading volume on Binance.

    Parameters:
        limit (int): Number of top coins to return (default: 200).

    Returns:
        pd.DataFrame: DataFrame containing the top coins with their trading volume.
    """
    exchange = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True,
    })

    # Fetch all tickers from Binance
    tickers = exchange.fetch_tickers()

    # Extract trading pairs and their volume information
    data = []
    for symbol, ticker in tickers.items():
        # Filter out non-spot pairs (e.g., futures or indices)
        if not symbol.endswith("/USDT"):
            continue
        data.append({
            "symbol": symbol,
            "base_currency": symbol.split("/")[0],
            "quote_currency": symbol.split("/")[1],
            "volume": ticker['quoteVolume'],  # Volume in terms of the quote currency
            "last_price": ticker['last'],    # Last traded price
        })

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Rank by volume and select the top 'limit' coins
    top_coins = df.sort_values(by="volume", ascending=False).head(limit).reset_index(drop=True)

    return top_coins

def fetch_binance_ohlc(symbol, start_date, timeframe='1d'):
    """
    Fetch historical OHLCV data from Binance for a single symbol.
    
    Parameters:
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT').
        start_date (str): Start date for fetching data (format: 'YYYY-MM-DD').
        timeframe (str): Timeframe for the data (default: '1d').
    
    Returns:
        pd.DataFrame: DataFrame containing OHLCV data.
    """
    exchange = ccxt.binance({
        'rateLimit': 1200,
        'enableRateLimit': True,
    })

    since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)

    df = pd.DataFrame(
        ohlcv, 
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.rename(columns={'timestamp': 'date'})
    df['ticker'] = symbol#.replace('/', '')  # Add ticker column for coin pair
    return df

def update_csv_for_all_symbols(file_path, timeframe='1d', top_n_coins=200):
    """
    Update the existing CSV file with new OHLC data for all symbols present in the file.
    
    Parameters:
        file_path (str): Path to the CSV file to be updated.
        timeframe (str): Timeframe for the data (default: '1d').
    """
    # Load existing data
    try:
        existing_data = pd.read_csv(file_path)
        existing_data['date'] = pd.to_datetime(existing_data['date'])
    except FileNotFoundError:
        print("CSV file not found. No data to update.")
        return

    # Extract unique symbols (tickers) from the existing data
    existing_symbols = existing_data['ticker'].unique()
    print(f"Symbols found in the file: {len(existing_symbols)}")

    # Fetch the top 200 symbols by volume
    top_symbols = fetch_top_coins_by_volume(limit=top_n_coins)
    print(f"Fetched {len(top_symbols)} top symbols by volume.")

    # Combine existing and new symbols
    all_symbols = set(existing_symbols).union(set(top_symbols))
    print(f"Updating data for {len(all_symbols)} symbols.")

    updated_data = existing_data.copy()

    for symbol in all_symbols:
        print(f"Updating data for {symbol}...")
        # Find the trading pair format for Binance API
        trading_pair = symbol #if '/' in symbol else f"{symbol[:3]}/{symbol[3:]}"  # Convert BTCUSDT to BTC/USDT
        # Determine the start date for fetching new data
        if existing_data.empty or symbol not in existing_symbols:
            start_date = '2019-01-01'  # Default start date for new symbols
        else:
            last_date = existing_data[existing_data['ticker'] == symbol]['date'].max()
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')

        # Fetch new data
        try:
            new_data = fetch_binance_ohlc(trading_pair, start_date, timeframe)
            if not new_data.empty:
                # Append new data and remove duplicates
                updated_data = pd.concat([updated_data, new_data]).drop_duplicates(subset=['date', 'ticker'])
            else:
                print(f"No new data found for {symbol}.")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    # Save updated data back to CSV
    updated_data = updated_data.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    updated_data.to_csv(file_path, index=False)
    print(f"Data successfully updated in {file_path}.")
    return updated_data


# Timeframe for the data
timeframe = '1d'

# Button to trigger the update
if st.button("Update Database"):
    with st.spinner("Updating data..."):
        try:
            # Run the update
            updated_data = update_csv_for_all_symbols(file_path, timeframe)
            st.success("Data successfully updated!")
            st.write("Here are the last entries for BTC/USDT of the updated dataset:")
            st.dataframe(updated_data[updated_data.ticker == "BTC/USDT"].sort_values(by=['date']).tail())
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.write(f"The updated data is saved to {file_path}.")




# Additional Info
st.sidebar.markdown("### About")
st.sidebar.info("This dashboard uses the Binance API to fetch end of day OHLC price data.")
st.sidebar.info("Data updated as of: {}".format(last_date))
