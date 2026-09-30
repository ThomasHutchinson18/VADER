import logging
import warnings
from datetime import datetime
from itertools import islice

import lseg.data as ld
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from configuration.config import all_tickers, rics_by_year, config

logger = logging.getLogger(__name__)

# statsmodels' RollingOLS emits "divide by zero encountered in log" when a
# rolling window has zero return variance. This happens for firms that IPO
# partway through the sample and have stale or repeated prices in their early
# trading history
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in log",
    category=RuntimeWarning,
    module="statsmodels.regression.rolling",
)
# Some duplicate instruments because companies change names/merge and retain ticker.
# This is handled in code, but mapping is retained in all_tickers.json for clarity.
warnings.filterwarnings(
    "ignore",
    message="You have duplicated instruments",
    category=UserWarning,
    module=r"lseg\..*",
)


class VaderData:
    """Construct the VADER measure of geopolitical risk.

        VADER is the exponentially weighted moving average (EWMA) conditional
        volatility of the market-filtered returns of a value-weighted portfolio of
        listed defence firms. The construction proceeds in three steps:

        1. Market filtering. Daily returns of each firm are regressed on
           contemporaneous S&P 500 returns using a rolling window of
           regression_window trading days, and betas are lagged by one day so
           that the residual at time t uses only information available at
           t - 1.
        2. Portfolio aggregation. Residuals are aggregated into a value-weighted
           portfolio. Membership is rebalanced annually to reflect updates to the
           SIPRI database, and weights are updated monthly based on market
           capitalisation.
        3. Conditional volatility. VADER is the EWMA standard deviation of the
           portfolio residual with decay factor `lambda = 0.94` (alpha=0.06), consistent with
           RiskMetrics conventions.

        All inputs and intermediate series are loaded lazily and cached on the
        instance. Constructing a ``VaderData`` object performs no network calls;
        data are fetched the first time a property is accessed.

        Parameters
        ----------
        start : str, optional
            Start date of the sample in YYYY-MM-DD format. Defaults to
            "2001-01-01".
        end : str, optional
            End date of the sample in YYYY-MM-DD format. Defaults to today's
            date.
        regression_window : int, optional
            Length of the rolling window, in trading days, used to estimate market
            betas. Defaults to 252.

        Attributes
        ----------
        start : str
            Sample start date.
        end : str
            Sample end date.
        regression_window : int
            Rolling-window length in trading days.
        config_path : str
            Path to the LSEG Workspace configuration, read from
            config["lseg_config_path"].

        Examples
        --------
        >>> vader_data = VaderData(start="2002-01-01", end="2026-06-01")
        >>> vader = vader_data.vader

        Notes
        -----
        The ticker universe and annual SIPRI membership are defined in
        ``VADER/configuration/config.py`` as ``all_tickers`` and ``rics_by_year``.
        A valid LSEG Workspace session is required for all data loading.
        """


    def __init__(self, start=None, end=None, regression_window=252):

        self.start = start or "2001-01-01"
        self.end = end or datetime.now().strftime("%Y-%m-%d")
        self.regression_window = regression_window
        self.config_path = config["lseg_config_path"]

        # Cached data
        self._prices = None
        self._market_caps = None
        self._market_proxy = None
        self._residual_series = None
        self._value_weighted_returns = None
        self._vader = None

    # ------------------------------------------------------------------
    # Raw data
    # ------------------------------------------------------------------

    @property
    def prices(self):
        """Daily close prices for the defence universe.

                Prices are loaded from one year before start so that residuals can
                be computed from the first day of the sample.

            Returns
            -------
            pandas.DataFrame
                Daily USD close prices, indexed by date, one column per RIC.
        """

        if self._prices is None:
            self._prices = self.load_prices()
        return self._prices

    @property
    def market_caps(self):
        """Monthly market capitalisations for the defence universe.

        Loaded from one month before start so that weights for January of
        the first sample year can be computed from the previous month-end.

        Returns
        -------
        pandas.DataFrame
            Monthly USD market capitalisations, indexed by date, one column
            per RIC.
        """

        if self._market_caps is None:
            self._market_caps = self.load_market_caps()
        return self._market_caps

    @property
    def market_proxy(self):
        """Daily close prices of the S&P 500 market proxy (``.SPX``).

        Returns
        -------
        pandas.Series
            Daily USD close prices of the S&P 500, indexed by date.
        """

        if self._market_proxy is None:
            self._market_proxy = self.load_market_proxy()
        return self._market_proxy

    # ------------------------------------------------------------------
    # Derived data
    # ------------------------------------------------------------------

    @property
    def residual_series(self):
        """Market-filtered residual returns.

        For each firm, daily returns are regressed on contemporaneous S&P 500
        returns using a rolling window of regression_window trading days.
        Betas are lagged by one day, so the residual at time t uses only
        information available at t - 1.

        Returns
        -------
        pandas.DataFrame
            Residual returns, indexed by date, one column per RIC. Columns
            for firms with insufficient observations are all-NaN.
        """

        if self._residual_series is None:
            self._residual_series = self.rolling_regression()
        return self._residual_series

    @property
    def value_weighted_returns(self):
        """Value-weighted portfolio residual series.

        The residual series is aggregated into a value-weighted portfolio.
        Membership is rebalanced annually based on rics_by_year; weights
        are updated monthly based on market capitalisation and renormalised
        daily to the set of firms with a valid residual.

        Returns
        -------
        pandas.Series
            Daily value-weighted portfolio residuals, restricted to weekdays.
        """

        if self._value_weighted_returns is None:
            self._value_weighted_returns = (
                self.construct_value_weighted_portfolio()
            )
        return self._value_weighted_returns

    @property
    def vader(self):
        """VADER: EWMA conditional volatility of the portfolio residual.

        Computed as the exponentially weighted moving standard deviation of
        value_weighted_returns with decay factor lambda = 0.94
        (alpha = 0.06), consistent with RiskMetrics conventions.

        Returns
        -------
        pandas.Series
            Daily VADER series, indexed by date.
        """

        if self._vader is None:
            self._vader = self.construct_vader()
        return self._vader

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_prices(self):

        logger.info("Loading prices...")
        ld.open_session(config_name=self.config_path)

        results = []
        price_start = (
                pd.Timestamp(self.start) - pd.DateOffset(years=1)
        ).strftime("%Y-%m-%d")
        try:
            batches = list(self.chunks(all_tickers.values(), 10))
            total = len(batches)

            for i, batch in enumerate(batches, 1):

                logger.info(
                    f"Batch {i}/{total} ({len(batch)} securities)..."
                )

                try:
                    df = ld.get_history(
                        universe=batch,
                        fields=["TR.PriceClose"],
                        start=price_start,
                        end=self.end,
                        interval="daily",
                        parameters={"Curn": "USD"},
                    )

                    results.append(df)
                    logger.info(f"Batch {i} complete.")

                except Exception as e:
                    logger.warning(
                        f"Batch {i} FAILED ({e})"
                    )

            if not results:
                raise RuntimeError("No price data was retrieved.")

            prices = pd.concat(results, axis=1)
            if prices.columns.duplicated().any():
                logger.warning(
                    "Duplicate instruments found. Removing duplicate columns."
                )
                prices = prices.loc[:, ~prices.columns.duplicated()]
            logger.info(
                f"Price data loaded: "
                f"{len(prices)} observations, "
                f"{len(prices.columns)} securities."
            )

            return prices

        finally:
            ld.close_session()

    def load_market_caps(self):

        logger.info("Loading market caps...")
        ld.open_session(config_name=self.config_path)

        market_start = (
                pd.Timestamp(self.start) - pd.DateOffset(months=1)
        ).strftime("%Y-%m-%d")

        try:
            return ld.get_history(
                universe=all_tickers.values(),
                fields=["TR.CompanyMarketCap"],
                start=market_start,
                end=self.end,
                interval="monthly",
                parameters={"Curn": "USD"},
            )

        finally:
            ld.close_session()

    def load_market_proxy(self):

        logger.info("Loading market proxy...")
        ld.open_session(config_name=self.config_path)
        market_start = (
                pd.Timestamp(self.start) - pd.DateOffset(years=1)
        ).strftime("%Y-%m-%d")

        try:
            return ld.get_history(
                universe=".SPX",
                fields=["TR.PriceClose"],
                start=market_start,
                end=self.end,
                interval="daily",
                parameters={"Curn": "USD"},
            ).iloc[:, 0]

        finally:
            ld.close_session()

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    def rolling_regression(self):

        logger.info("Rolling regression...")

        returns = self.prices.ffill().pct_change(fill_method=None)
        market_returns = self.market_proxy.ffill().pct_change(fill_method=None)

        residuals_df = pd.DataFrame(
            index=returns.index,
            columns=returns.columns,
            dtype=float
        )

        for ticker in returns.columns:

            reg_df = pd.DataFrame({
                "Y": returns[ticker],
                "X": market_returns
            })

            reg_df = (
                reg_df
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
            )

            if len(reg_df) < self.regression_window:
                logger.warning(
                    f"Skipping {ticker}: insufficient observations "
                    f"({len(reg_df)} < {self.regression_window})"
                )
                continue

            Y = reg_df["Y"]
            X = sm.add_constant(reg_df["X"])

            model = RollingOLS(
                endog=Y,
                exog=X,
                window=self.regression_window
            ).fit()

            # Use coefficients estimated through t-1
            params_lagged = model.params.shift(1)

            fitted = (params_lagged * X).sum(axis=1)
            residuals = Y - fitted

            residuals_df.loc[
                reg_df.index, ticker
            ] = residuals

        self._residual_series = residuals_df

        return residuals_df

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def construct_value_weighted_portfolio(self):

        returns = self.residual_series
        market_caps = self.market_caps.sort_index()

        yearly_returns = []

        for year in sorted(rics_by_year.keys(), key=int):

            year_int = int(year)
            previous_year = str(year_int - 1)
            rics = (rics_by_year[previous_year]
                    if previous_year in rics_by_year
                    else rics_by_year[year])

            year_returns = returns.loc[
                returns.index.year == year_int,
                returns.columns.intersection(rics)
            ]
            if year_returns.empty:
                continue

            caps = (
                market_caps
                .reindex(year_returns.index, method="ffill")
                .reindex(columns=year_returns.columns)
            )

            # monthly weights from previous month-end caps
            monthly_caps = caps.groupby(caps.index.to_period("M")).first()
            monthly_weights = monthly_caps.div(
                monthly_caps.sum(axis=1), axis=0
            )
            daily_weights = monthly_weights.loc[
                year_returns.index.to_period("M")
            ]
            daily_weights.index = year_returns.index

            # renormalise each day to the available universe
            mask = year_returns.notna() & daily_weights.notna()
            ret_m = year_returns.where(mask)
            w_m = daily_weights.where(mask)
            total = w_m.sum(axis=1)
            w_norm = w_m.div(total, axis=0).where(total > 0, 0)

            yearly_returns.append((w_norm * ret_m).sum(axis=1))

        out = pd.concat(yearly_returns).sort_index()
        return out[out.index.weekday < 5]
    # ------------------------------------------------------------------
    # VADER
    # ------------------------------------------------------------------

    def construct_vader(self):

        logger.info("Constructing VADER...")

        return (
            (self.value_weighted_returns ** 2)
            .ewm(alpha=0.06, adjust=False)
            .mean()
            .pow(0.5)
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def chunks(iterable, size):

        it = iter(iterable)

        while True:

            batch = list(islice(it, size))

            if not batch:
                break

            yield batch