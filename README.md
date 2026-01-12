# Cocoa Volatility Analysis

Analysis of cocoa futures volatility using daily and high-frequency (5-minute) data.

## Data

- **Daily data**: OHLCV prices for cocoa futures (LCCc1)
- **5-minute data**: Intraday OHLCV prices

## Methodology

### Returns

Log returns are computed as:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

where $P_t$ is the closing price at time $t$.

### Realized Volatility

Daily realized volatility is computed from 5-minute returns as:

$$RV_t = \sum_{i=1}^{N} r_{t,i}^2$$

where $r_{t,i}$ denotes the $i$-th intraday return on day $t$, and $N$ is the number of intraday observations.

### Descriptive Statistics

- **Skewness**: $\frac{1}{T}\sum_{t=1}^{T}\left(\frac{r_t - \bar{r}}{\sigma}\right)^3$
- **Kurtosis**: $\frac{1}{T}\sum_{t=1}^{T}\left(\frac{r_t - \bar{r}}{\sigma}\right)^4 - 3$ (excess kurtosis)
- **Jarque-Bera test**: Tests normality based on skewness and kurtosis

### Autocorrelation

ACF plots show autocorrelation of returns $r_t$ and squared returns $r_t^2$. Significant autocorrelation in squared returns indicates volatility clustering (ARCH effects).

## Output

- `00_descriptive_statistics.md`: Summary statistics
- `realized_volatility.csv`: Daily RV series
- Plots: Price series, returns, distributions, ACF, volatility clustering
