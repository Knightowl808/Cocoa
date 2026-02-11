import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Plot style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def load_data():
    """Load both datasets."""
    daily = pd.read_csv('00_Data/Daily.csv', parse_dates=['DateTime'])
    fivemin = pd.read_csv('00_Data/5min.csv', parse_dates=['DateTime'])

    # Sort by date ascending
    daily = daily.sort_values('DateTime').reset_index(drop=True)
    fivemin = fivemin.sort_values('DateTime').reset_index(drop=True)

    return daily, fivemin

def compute_returns(df):
    """Compute log returns."""
    df = df.copy()
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Return'] = df['Close'].pct_change()
    return df

def descriptive_stats(df, name):
    """Compute descriptive statistics."""
    returns = df['LogReturn'].dropna()

    stats_dict = {
        'Observations': len(returns),
        'Mean (%)': returns.mean() * 100,
        'Std Dev (%)': returns.std() * 100,
        'Min (%)': returns.min() * 100,
        'Max (%)': returns.max() * 100,
        'Skewness': stats.skew(returns),
        'Kurtosis': stats.kurtosis(returns),
        'Jarque-Bera': stats.jarque_bera(returns)[0],
        'JB p-value': stats.jarque_bera(returns)[1],
    }

    print(f"\n{'='*50}")
    print(f"Descriptive Statistics: {name}")
    print('='*50)
    for key, val in stats_dict.items():
        if 'p-value' in key:
            print(f"{key:20s}: {val:.4e}")
        else:
            print(f"{key:20s}: {val:.4f}")

    return stats_dict

def compute_realized_volatility(fivemin):
    """Compute daily realized volatility from 5-min returns."""
    df = fivemin.copy()
    df['Date'] = df['DateTime'].dt.date

    # RV = sum of squared returns per day
    rv = df.groupby('Date').apply(lambda x: (x['LogReturn']**2).sum())
    rv = rv.reset_index()
    rv.columns = ['Date', 'RV']
    rv['RV_sqrt'] = np.sqrt(rv['RV'])  # Annualize: * np.sqrt(252) if needed
    rv['Date'] = pd.to_datetime(rv['Date'])

    return rv

def save_statistics_markdown(daily, fivemin, rv):
    """Save statistics to markdown file."""
    daily_ret = daily['LogReturn'].dropna()
    fivemin_ret = fivemin['LogReturn'].dropna()

    md = """# Cocoa Volatility Analysis - Descriptive Statistics

## Data Overview

| Dataset | Observations | Date Range |
|---------|--------------|------------|
| Daily | {:,} | {} to {} |
| 5-Minute | {:,} | {} to {} |

## Daily Returns Statistics

| Statistic | Value |
|-----------|-------|
| Mean | {:.4f}% |
| Std Dev | {:.4f}% |
| Min | {:.4f}% |
| Max | {:.4f}% |
| Skewness | {:.4f} |
| Kurtosis | {:.4f} |
| Jarque-Bera | {:.2f} |
| JB p-value | {:.2e} |

## 5-Minute Returns Statistics

| Statistic | Value |
|-----------|-------|
| Mean | {:.6f}% |
| Std Dev | {:.4f}% |
| Min | {:.4f}% |
| Max | {:.4f}% |
| Skewness | {:.4f} |
| Kurtosis | {:.4f} |
| Jarque-Bera | {:.2f} |
| JB p-value | {:.2e} |

## Realized Volatility (from 5-min data)

| Statistic | Value |
|-----------|-------|
| Trading Days | {:,} |
| Mean RV | {:.6f} |
| Mean sqrt(RV) | {:.4f}% |
| Std Dev sqrt(RV) | {:.4f}% |
| Min sqrt(RV) | {:.4f}% |
| Max sqrt(RV) | {:.4f}% |

""".format(
        len(daily), daily['DateTime'].min().strftime('%Y-%m-%d'), daily['DateTime'].max().strftime('%Y-%m-%d'),
        len(fivemin), fivemin['DateTime'].min().strftime('%Y-%m-%d'), fivemin['DateTime'].max().strftime('%Y-%m-%d'),
        daily_ret.mean()*100, daily_ret.std()*100, daily_ret.min()*100, daily_ret.max()*100,
        stats.skew(daily_ret), stats.kurtosis(daily_ret), stats.jarque_bera(daily_ret)[0], stats.jarque_bera(daily_ret)[1],
        fivemin_ret.mean()*100, fivemin_ret.std()*100, fivemin_ret.min()*100, fivemin_ret.max()*100,
        stats.skew(fivemin_ret), stats.kurtosis(fivemin_ret), stats.jarque_bera(fivemin_ret)[0], stats.jarque_bera(fivemin_ret)[1],
        len(rv), rv['RV'].mean(), rv['RV_sqrt'].mean()*100, rv['RV_sqrt'].std()*100, rv['RV_sqrt'].min()*100, rv['RV_sqrt'].max()*100
    )

    with open('02_Output/00_descriptive_statistics.md', 'w') as f:
        f.write(md)
    print("Saved: 02_Output/00_descriptive_statistics.md")

def plot_price_series(daily, fivemin):
    """Plot price time series."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(daily['DateTime'], daily['Close'], 'k-', linewidth=0.8)
    axes[0].set_title('Daily Cocoa Prices')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price')

    axes[1].plot(fivemin['DateTime'], fivemin['Close'], 'k-', linewidth=0.5)
    axes[1].set_title('5-Minute Cocoa Prices')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Price')

    plt.tight_layout()
    plt.savefig('02_Output/01_price_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 02_Output/01_price_series.png")

def plot_returns(daily, fivemin):
    """Plot return series."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(daily['DateTime'], daily['LogReturn']*100, 'k-', linewidth=0.8)
    axes[0].set_title('Daily Log Returns')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Return (%)')
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    axes[1].plot(fivemin['DateTime'], fivemin['LogReturn']*100, 'k-', linewidth=0.3)
    axes[1].set_title('5-Minute Log Returns')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Return (%)')
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('02_Output/02_returns_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 02_Output/02_returns_series.png")

def plot_return_distribution(daily, fivemin):
    """Plot return distributions with normal comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (df, name) in enumerate([(daily, 'Daily'), (fivemin, '5-Minute')]):
        returns = df['LogReturn'].dropna() * 100

        # Histogram
        ax = axes[idx, 0]
        ax.hist(returns, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')

        # Normal fit
        x = np.linspace(returns.min(), returns.max(), 100)
        ax.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()),
                'r-', linewidth=2, label='Normal')
        ax.set_title(f'{name} Return Distribution')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Density')
        ax.legend()

        # QQ Plot
        ax = axes[idx, 1]
        stats.probplot(returns, dist="norm", plot=ax)
        ax.set_title(f'{name} Q-Q Plot')
        ax.get_lines()[0].set_markerfacecolor('steelblue')
        ax.get_lines()[0].set_markeredgecolor('steelblue')
        ax.get_lines()[0].set_markersize(3)

    plt.tight_layout()
    plt.savefig('02_Output/03_return_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 02_Output/03_return_distribution.png")

def plot_acf(daily, fivemin):
    """Plot ACF of returns and squared returns."""
    from statsmodels.tsa.stattools import acf

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    nlags = 40

    # Daily
    returns_daily = daily['LogReturn'].dropna()

    # Returns ACF (skip lag 0)
    acf_vals, confint = acf(returns_daily, nlags=nlags, alpha=0.05)
    lags = np.arange(1, nlags + 1)
    axes[0, 0].bar(lags, acf_vals[1:], color='steelblue', width=0.6)
    axes[0, 0].axhline(y=confint[1, 1] - acf_vals[1], color='r', linestyle='--', linewidth=0.8)
    axes[0, 0].axhline(y=confint[1, 0] - acf_vals[1], color='r', linestyle='--', linewidth=0.8)
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 0].set_title('Daily Returns ACF')
    axes[0, 0].set_xlabel('Lag')
    axes[0, 0].set_ylabel('ACF')

    # Squared returns ACF (skip lag 0)
    acf_vals, confint = acf(returns_daily**2, nlags=nlags, alpha=0.05)
    axes[0, 1].bar(lags, acf_vals[1:], color='steelblue', width=0.6)
    axes[0, 1].axhline(y=confint[1, 1] - acf_vals[1], color='r', linestyle='--', linewidth=0.8)
    axes[0, 1].axhline(y=confint[1, 0] - acf_vals[1], color='r', linestyle='--', linewidth=0.8)
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 1].set_title('Daily Squared Returns ACF')
    axes[0, 1].set_xlabel('Lag')
    axes[0, 1].set_ylabel('ACF')

    # 5-min
    returns_5min = fivemin['LogReturn'].dropna()

    acf_vals, confint = acf(returns_5min, nlags=nlags, alpha=0.05)
    axes[1, 0].bar(lags, acf_vals[1:], color='steelblue', width=0.6)
    axes[1, 0].axhline(y=confint[1, 1] - acf_vals[1], color='r', linestyle='--', linewidth=0.8)
    axes[1, 0].axhline(y=confint[1, 0] - acf_vals[1], color='r', linestyle='--', linewidth=0.8)
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].set_title('5-Minute Returns ACF')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('ACF')

    acf_vals, confint = acf(returns_5min**2, nlags=nlags, alpha=0.05)
    axes[1, 1].bar(lags, acf_vals[1:], color='steelblue', width=0.6)
    axes[1, 1].axhline(y=confint[1, 1] - acf_vals[1], color='r', linestyle='--', linewidth=0.8)
    axes[1, 1].axhline(y=confint[1, 0] - acf_vals[1], color='r', linestyle='--', linewidth=0.8)
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 1].set_title('5-Minute Squared Returns ACF')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('ACF')

    plt.tight_layout()
    plt.savefig('02_Output/04_acf_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 02_Output/04_acf_plots.png")

def plot_realized_volatility(rv):
    """Plot realized volatility time series."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(rv['Date'], rv['RV_sqrt']*100, 'k-', linewidth=0.8)
    axes[0].set_title('Daily Realized Volatility (sqrt)')
    axes[0].set_ylabel('RV (%)')

    axes[1].plot(rv['Date'], np.log(rv['RV']), 'k-', linewidth=0.8)
    axes[1].set_title('Log Realized Volatility')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('log(RV)')

    plt.tight_layout()
    plt.savefig('02_Output/06_realized_volatility.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 02_Output/06_realized_volatility.png")

def plot_volatility_clustering(daily):
    """Plot volatility clustering evidence."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    returns = daily['LogReturn'].dropna() * 100
    dates = daily.loc[returns.index, 'DateTime']

    # Returns
    axes[0].plot(dates, returns, 'k-', linewidth=0.8)
    axes[0].set_title('Daily Log Returns')
    axes[0].set_ylabel('Return (%)')
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # Absolute returns (proxy for volatility)
    axes[1].plot(dates, np.abs(returns), 'k-', linewidth=0.8)
    axes[1].set_title('Absolute Returns (Volatility Proxy)')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('|Return| (%)')

    plt.tight_layout()
    plt.savefig('02_Output/05_volatility_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 02_Output/05_volatility_clustering.png")

def main():
    import os
    os.makedirs('02_Output', exist_ok=True)

    print("Loading data...")
    daily, fivemin = load_data()

    # Compute returns
    daily = compute_returns(daily)
    fivemin = compute_returns(fivemin)

    # Compute realized volatility
    print("\nComputing Realized Volatility...")
    rv = compute_realized_volatility(fivemin)
    rv.to_csv('02_Output/realized_volatility.csv', index=False)
    print(f"  Saved RV to 02_Output/realized_volatility.csv ({len(rv)} days)")

    # Descriptive statistics
    descriptive_stats(daily, "Daily Data")
    descriptive_stats(fivemin, "5-Minute Data")

    # Save markdown
    save_statistics_markdown(daily, fivemin, rv)

    # Generate plots
    print("\nGenerating plots...")
    plot_price_series(daily, fivemin)
    plot_returns(daily, fivemin)
    plot_return_distribution(daily, fivemin)
    plot_acf(daily, fivemin)
    plot_volatility_clustering(daily)
    plot_realized_volatility(rv)

    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
