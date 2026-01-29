#%pip install yfinance
import yfinance as yf
from matplotlib import pyplot as plt

def plot_gas_prices():
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

    ttf = yf.Ticker("TTF=F")
    ttf_hist = ttf.history(period="10y")
    ttf_price = ttf_hist["Close"]

    fig, ax = plt.subplots(figsize=(10, 10/1.608))

    ax.plot(ttf_price, color='#56b4e9', linewidth=1.5)

    # Title and subtitle using ax.text (coordinates in axes fraction)
    ax.text(0.0, 1.06,
        "TTF Natural Gas Prices (Last 10 Years)",
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        family='Arial',
        ha='left',
        va='bottom',
        clip_on=False)

    ax.text(0.0, 1.02,
        "(EUR/MWh)",
        transform=ax.transAxes,
        fontsize=12,
        family='Arial',
        ha='left',
        va='bottom',
        color='gray',
        clip_on=False)

    ax.spines[['top', 'right']].set_visible(False)

    ax.set_xlim(ttf_price.index.min(), ttf_price.index.max())
    ax.set_ylim(0, 350)

    ax.grid(visible=True, which='major', linestyle=':', color='gray', alpha=0.6)

    plt.show()

plot_gas_prices()