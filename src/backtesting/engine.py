import numpy as np
import pandas as pd

class BacktestEngine:
    def __init__(self, transaction_cost=0.001, slippage=0.0005, risk_free_rate=0.06):
        """
        transaction_cost: friction per trade (e.g., 0.1% = 0.001)
        slippage: execution slippage (e.g., 0.05% = 0.0005)
        risk_free_rate: annual risk-free rate (e.g., 6% = 0.06)
        """
        self.friction = transaction_cost + slippage
        self.rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1

    def run_backtest(self, df: pd.DataFrame, predictions: pd.Series) -> dict:
        """
        df: DataFrame containing daily prices and returns
        predictions: Series of binary predictions (1 = UP, 0 = DOWN) matched with the df index
        """
        results = df.copy()
        results['predicted_trend'] = predictions
        
        # Verify index alignment
        if 'target_return' not in results.columns:
            results['target_return'] = results['daily_return_pct'] / 100
        else:
            results['target_return'] = results['target_return'] / 100

        # Implement strategies (positions: 1 = Long, 0 = Cash)
        positions = {}
        
        # 1. Buy and Hold (Always Long)
        positions['Buy_Hold'] = np.ones(len(results))
        
        # 2. AI Strategy
        positions['AI_Strategy'] = results['predicted_trend'].fillna(0).values
        
        # 3. Always Bullish (same as Buy and Hold)
        positions['Always_Bullish'] = np.ones(len(results))
        
        # 4. Momentum Baseline (Long if 5-day return > 0, else cash)
        ma5_ret = results['daily_return_pct'].rolling(window=5).mean().shift(1)
        positions['Momentum'] = np.where(ma5_ret > 0, 1, 0)
        
        # 5. Moving Average Crossover (Long if SMA10 > SMA30, else cash)
        close = results['close_price']
        sma10 = close.rolling(window=10).mean().shift(1)
        sma30 = close.rolling(window=30).mean().shift(1)
        positions['MA_Crossover'] = np.where(sma10 > sma30, 1, 0)
        
        # 6. Previous-Day Direction (Long if yesterday was positive, else cash)
        prev_ret = results['daily_return_pct'].shift(1)
        positions['Prev_Day_Dir'] = np.where(prev_ret > 0, 1, 0)

        # 7. AI Kelly Sizing Strategy (Scale position using model confidence)
        probs = results.get('predicted_prob', pd.Series(0.5, index=results.index)).fillna(0.5).values
        kelly_fraction = np.clip(2 * probs - 1, 0.0, 1.0)
        positions['AI_Strategy_Kelly'] = results['predicted_trend'].fillna(0).values * kelly_fraction

        # 8. AI Volatility Target Strategy (Target constant annualized volatility of 15%)
        # Extract realized_volatility or lag1 (from technical indicators)
        realized_vol_col = 'realized_volatility_lag1' if 'realized_volatility_lag1' in results.columns else 'realized_volatility'
        if realized_vol_col in results.columns:
            realized_vol = results[realized_vol_col].fillna(15.0).values / 100.0
        else:
            realized_vol = np.ones(len(results)) * 0.15
        target_vol = 0.15  # Target 15% annualized volatility
        vol_scalar = np.clip(target_vol / (realized_vol + 1e-9), 0.1, 1.5)
        positions['AI_Strategy_VolTarget'] = results['predicted_trend'].fillna(0).values * vol_scalar

        strategy_curves = {}
        strategy_metrics = {}

        for name, pos in positions.items():
            # Calculate daily returns with transaction costs
            daily_ret = results['target_return'].values
            
            # Position changes (trades)
            pos_shifted = np.roll(pos, 1)
            pos_shifted[0] = 0 # Assume starting in cash
            trades = np.abs(pos - pos_shifted)
            
            # Apply transaction costs on trades
            trade_costs = trades * self.friction
            
            # Strategy daily returns
            strat_ret = pos * daily_ret - trade_costs
            
            # Cumulative returns
            cum_ret = np.cumprod(1 + strat_ret)
            strategy_curves[name] = cum_ret
            
            # Metrics
            metrics = self.calculate_metrics(strat_ret)
            strategy_metrics[name] = metrics

        # Generate comparative dataframe
        curves_df = pd.DataFrame(strategy_curves, index=results.index)
        curves_df['date'] = results['date']
        
        return {
            "curves": curves_df,
            "metrics": strategy_metrics
        }

    def calculate_metrics(self, daily_returns: np.ndarray) -> dict:
        n_days = len(daily_returns)
        if n_days == 0:
            return {}

        total_return = np.prod(1 + daily_returns) - 1
        
        # CAGR (assuming 252 trading days per year)
        years = n_days / 252
        cagr = (total_return + 1) ** (1 / years) - 1 if years > 0 and total_return > -1 else -1.0
        
        # Volatility
        vol = np.std(daily_returns) * np.sqrt(252)
        
        # Sharpe Ratio (with Rf = 0 for daily returns)
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns)
        sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(252)
        
        # Sortino Ratio (downside deviation standard deviation * sqrt(252))
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-9
        sortino = (mean_ret / (np.std(daily_returns[daily_returns < 0]) + 1e-9)) * np.sqrt(252) if len(daily_returns[daily_returns < 0]) > 0 else sharpe
        
        # Max Drawdown
        cum_returns = np.cumprod(1 + daily_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / (running_max + 1e-9)
        max_dd = np.min(drawdowns)

        # Calmar Ratio
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

        # Value at Risk (VaR 95%) and Conditional VaR (CVaR 95%)
        var_95 = np.percentile(daily_returns, 5)
        cvar_95 = np.mean(daily_returns[daily_returns <= var_95]) if len(daily_returns[daily_returns <= var_95]) > 0 else var_95
        
        # Win Rate
        win_rate = np.sum(daily_returns > 0) / np.sum(daily_returns != 0) if np.sum(daily_returns != 0) > 0 else 0.0
        
        # 95% Confidence Interval for Daily Returns (via standard error of mean)
        sem = std_ret / np.sqrt(n_days)
        ci_lower = mean_ret - 1.96 * sem
        ci_upper = mean_ret + 1.96 * sem
        
        return {
            "Total Return": float(total_return),
            "CAGR": float(cagr),
            "Annualized Volatility": float(vol),
            "Sharpe Ratio": float(sharpe),
            "Sortino Ratio": float(sortino),
            "Calmar Ratio": float(calmar),
            "Max Drawdown": float(max_dd),
            "VaR_95": float(var_95),
            "CVaR_95": float(cvar_95),
            "Win Rate": float(win_rate),
            "CI_Lower_Daily": float(ci_lower),
            "CI_Upper_Daily": float(ci_upper)
        }
