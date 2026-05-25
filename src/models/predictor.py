import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)

class PricePredictor:
    def __init__(self):
        self.trend_model = None
        self.return_model = None
        self.metrics = {}
        self.trained = False
        self.feature_names = []

    def prepare_data(self, df: pd.DataFrame, sector_index: str = None) -> tuple:
        """
        Calculates advanced technical indicators and lags them to prevent lookahead bias.
        """
        df = df.copy()
        df = df.sort_values('date')
        df['date'] = pd.to_datetime(df['date'])
        
        if 'daily_return_pct' not in df.columns:
            df['daily_return_pct'] = df['close_price'].pct_change() * 100

        # --- TECHNICAL INDICATORS ---
        
        # 1. RSI (14-day)
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)

        # 2. MACD
        ema12 = df['close_price'].ewm(span=12, adjust=False).mean()
        ema26 = df['close_price'].ewm(span=26, adjust=False).mean()
        df['MACD_line'] = ema12 - ema26
        df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']

        # 3. Bollinger Bands
        df['BB_middle'] = df['close_price'].rolling(window=20).mean()
        df['BB_std'] = df['close_price'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (2 * df['BB_std'])
        df['BB_lower'] = df['BB_middle'] - (2 * df['BB_std'])
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (df['BB_middle'] + 1e-9)
        df['BB_width'] = df['BB_width'].fillna(0)

        # 4. ATR (Average True Range)
        if 'high_price' in df.columns and 'low_price' in df.columns:
            tr1 = df['high_price'] - df['low_price']
            tr2 = (df['high_price'] - df['close_price'].shift(1)).abs()
            tr3 = (df['low_price'] - df['close_price'].shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=14).mean()
        else:
            df['ATR'] = df['close_price'].rolling(window=14).std()
        df['ATR'] = df['ATR'].fillna(0)

        # 5. Moving Average Crossover (SMA20/SMA50)
        df['SMA20'] = df['close_price'].rolling(window=20).mean()
        df['SMA50'] = df['close_price'].rolling(window=50).mean()
        df['MA_crossover'] = np.where(df['SMA20'] > df['SMA50'], 1, -1)

        # 6. Sector Momentum
        df['momentum_5d'] = df['close_price'].pct_change(5) * 100
        df['momentum_21d'] = df['close_price'].pct_change(21) * 100

        # 7. Volume Z-score
        if 'volume' in df.columns and df['volume'].std() > 0:
            df['volume_zscore'] = (df['volume'] - df['volume'].rolling(window=10).mean()) / (df['volume'].rolling(window=10).std() + 1e-9)
        else:
            df['volume_zscore'] = 0.0
        df['volume_zscore'] = df['volume_zscore'].fillna(0)

        # 8. Drawdown
        rolling_max = df['close_price'].cummax()
        df['drawdown'] = (df['close_price'] - rolling_max) / (rolling_max + 1e-9)

        # 9. Rolling Sharpe (20-day)
        rolling_mean_ret = df['daily_return_pct'].rolling(window=20).mean()
        rolling_std_ret = df['daily_return_pct'].rolling(window=20).std()
        df['rolling_sharpe'] = (rolling_mean_ret / (rolling_std_ret + 1e-9)) * np.sqrt(252)
        df['rolling_sharpe'] = df['rolling_sharpe'].fillna(0)

        # --- NEW TECHNICAL INDICATORS (TASK 2) ---
        
        # 10. Stochastic Oscillator
        low_14 = df['low_price'].rolling(window=14).min()
        high_14 = df['high_price'].rolling(window=14).max()
        df['stochastic_k'] = ((df['close_price'] - low_14) / (high_14 - low_14 + 1e-9)) * 100
        df['stochastic_d'] = df['stochastic_k'].rolling(window=3).mean()
        df['stochastic_k'] = df['stochastic_k'].fillna(50)
        df['stochastic_d'] = df['stochastic_d'].fillna(50)

        # 11. VWAP (rolling 20-day proxy)
        typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        pv = typical_price * df['volume']
        df['vwap'] = pv.rolling(window=20).sum() / (df['volume'].rolling(window=20).sum() + 1e-9)
        df['vwap'] = df['vwap'].fillna(df['close_price'])

        # 12. Sector Relative Strength (to SENSEX) and Rolling Beta (60-day)
        from src.database.connection import get_connection
        try:
            conn = get_connection()
            sensex_df = pd.read_sql("""
                SELECT date(date) as date, close_price as sensex_close
                FROM bse_sector_prices
                WHERE sector_index = 'BSE_SENSEX'
            """, conn)
            conn.close()
            if not sensex_df.empty:
                sensex_df['date'] = pd.to_datetime(sensex_df['date'])
                df = pd.merge(df, sensex_df, on='date', how='left')
                df['sensex_close'] = df['sensex_close'].ffill().bfill()
                df['relative_strength'] = df['close_price'] / (df['sensex_close'] + 1e-9)
                df['sensex_return'] = df['sensex_close'].pct_change() * 100
                ret_cov = df['daily_return_pct'].rolling(window=60).cov(df['sensex_return'])
                mkt_var = df['sensex_return'].rolling(window=60).var()
                df['rolling_beta'] = ret_cov / (mkt_var + 1e-9)
            else:
                df['relative_strength'] = 1.0
                df['rolling_beta'] = 1.0
        except Exception as e:
            logger.warning(f"Failed to calculate beta or relative strength: {e}")
            df['relative_strength'] = 1.0
            df['rolling_beta'] = 1.0
            
        df['relative_strength'] = df['relative_strength'].fillna(1.0)
        df['rolling_beta'] = df['rolling_beta'].fillna(1.0)

        # 13. Rolling Sortino Ratio (20-day)
        rolling_mean_ret = df['daily_return_pct'].rolling(window=20).mean()
        downside_diff = df['daily_return_pct'].clip(upper=0)
        rolling_downside_std = downside_diff.rolling(window=20).std()
        df['rolling_sortino'] = (rolling_mean_ret / (rolling_downside_std + 1e-9)) * np.sqrt(252)
        df['rolling_sortino'] = df['rolling_sortino'].fillna(0.0)

        # 14. Realized Volatility (20-day)
        df['realized_volatility'] = df['daily_return_pct'].rolling(window=20).std() * np.sqrt(252)
        df['realized_volatility'] = df['realized_volatility'].fillna(0.0)

        # 15. Skewness & Kurtosis (20-day)
        df['rolling_skew'] = df['daily_return_pct'].rolling(window=20).skew().fillna(0.0)
        df['rolling_kurt'] = df['daily_return_pct'].rolling(window=20).kurt().fillna(0.0)

        # 16. Momentum Factors
        df['momentum_10d'] = df['close_price'].pct_change(10) * 100
        df['momentum_10d'] = df['momentum_10d'].fillna(0.0)

        # 17. Volatility Regime
        vol_threshold = df['realized_volatility'].rolling(window=100, min_periods=20).quantile(0.7)
        df['vol_regime'] = np.where(df['realized_volatility'] > vol_threshold.fillna(999.0), 1, 0)

        # --- MACRO INDICATORS ---
        try:
            conn = get_connection()
            macro_dfs = []
            for macro_idx in ['USD_INR', 'CRUDE_OIL', 'INDIA_VIX', 'BOND_YIELD_10Y', 'INFLATION_CPI', 'REPO_RATE']:
                m_df = pd.read_sql("""
                    SELECT date(date) as date, close_price as {col}
                    FROM bse_sector_prices
                    WHERE sector_index = ?
                """.format(col=macro_idx.lower()), conn, params=(macro_idx,))
                if not m_df.empty:
                    m_df['date'] = pd.to_datetime(m_df['date'])
                    macro_dfs.append(m_df)
            conn.close()
            for m_df in macro_dfs:
                df = pd.merge(df, m_df, on='date', how='left')
                df[m_df.columns[1]] = df[m_df.columns[1]].ffill().bfill()
        except Exception as e:
            logger.warning(f"Failed to fetch macro features: {e}")

        # Ensure macro columns exist
        for col in ['usd_inr', 'crude_oil', 'india_vix', 'bond_yield_10y', 'inflation_cpi', 'repo_rate']:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

        # --- NLP SENTIMENT INTEGRATION ---
        if sector_index:
            try:
                conn = get_connection()
                sent_df = pd.read_sql("""
                    SELECT date(published_at) as date, AVG(sentiment_score) as avg_sentiment
                    FROM raw_news r
                    JOIN news_sector_mapping m ON r.id = m.news_id
                    WHERE m.sector_index = ?
                    GROUP BY date(published_at)
                """, conn, params=(sector_index,))
                conn.close()
                if not sent_df.empty:
                    sent_df['date'] = pd.to_datetime(sent_df['date'])
                    df = pd.merge(df, sent_df, on='date', how='left')
                    df['avg_sentiment'] = df['avg_sentiment'].fillna(0.0)
                else:
                    df['avg_sentiment'] = 0.0
            except Exception as e:
                logger.warning(f"Failed to fetch daily sentiment feature: {e}")
                df['avg_sentiment'] = 0.0
        else:
            df['avg_sentiment'] = 0.0

        # --- LAG FEATURES BY 1 DAY (No Lookahead Bias) ---
        feature_cols = [
            'daily_return_pct', 'RSI', 'MACD_line', 'MACD_signal', 'MACD_hist',
            'BB_width', 'ATR', 'MA_crossover', 'momentum_5d', 'momentum_21d',
            'volume_zscore', 'drawdown', 'rolling_sharpe', 'avg_sentiment',
            'stochastic_k', 'stochastic_d', 'vwap', 'relative_strength',
            'rolling_beta', 'rolling_sortino', 'realized_volatility',
            'rolling_skew', 'rolling_kurt', 'momentum_10d', 'vol_regime',
            'usd_inr', 'crude_oil', 'india_vix', 'bond_yield_10y', 'inflation_cpi', 'repo_rate'
        ]
        
        for col in feature_cols:
            df[f'{col}_lag1'] = df[col].shift(1)

        lagged_features = [f'{col}_lag1' for col in feature_cols]

        # Target: t+1 daily return and binary trend direction
        df['target_return'] = df['daily_return_pct'].shift(-1)
        df['target_trend'] = np.where(df['target_return'] > 0, 1, 0)

        df = df.dropna(subset=lagged_features)
        
        return df, lagged_features

    def train_and_evaluate(self, df: pd.DataFrame, sector_index: str = None) -> tuple:
        """
        Performs walk-forward time-series splitting to train and validate model performance.
        """
        if len(df) < 50:
            logger.warning("Insufficient historical data for robust ML model training.")
            return False, None

        df_clean, features = self.prepare_data(df, sector_index)
        self.feature_names = features
        
        df_train_eval = df_clean.iloc[:-1]
        if len(df_train_eval) < 20:
            return False, None

        X = df_train_eval[features]
        y_trend = df_train_eval['target_trend']
        y_return = df_train_eval['target_return']

        # Time-based split (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        yt_train, yt_test = y_trend.iloc[:split_idx], y_trend.iloc[split_idx:]
        yr_train, yr_test = y_return.iloc[:split_idx], y_return.iloc[split_idx:]

        # --- MODEL INITIALIZATION & STACKED ENSEMBLE (TASK 3) ---
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, VotingClassifier, VotingRegressor
        from sklearn.calibration import CalibratedClassifierCV

        # Ensemble Classification Setup
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        et_clf = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)
        estimators = [('rf', rf_clf), ('et', et_clf)]
        
        try:
            import xgboost as xgb
            estimators.append(('xgb', xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42, eval_metric='logloss')))
        except ImportError:
            pass
        try:
            import lightgbm as lgb
            estimators.append(('lgb', lgb.LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1)))
        except ImportError:
            pass

        ensemble_clf = VotingClassifier(estimators=estimators, voting='soft')
        # Platt Scaling / Probability Calibration via CalibratedClassifierCV
        self.trend_model = CalibratedClassifierCV(estimator=ensemble_clf, method='sigmoid', cv=3)

        # Ensemble Regression Setup
        rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        et_reg = ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=42)
        reg_estimators = [('rf', rf_reg), ('et', et_reg)]

        try:
            import xgboost as xgb
            reg_estimators.append(('xgb', xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)))
        except ImportError:
            pass
        try:
            import lightgbm as lgb
            reg_estimators.append(('lgb', lgb.LGBMRegressor(n_estimators=50, max_depth=3, random_state=42, verbose=-1)))
        except ImportError:
            pass

        self.return_model = VotingRegressor(estimators=reg_estimators)

        # Train models
        self.trend_model.fit(X_train, yt_train)
        self.return_model.fit(X_train, yr_train)

        # Generate predictions
        trend_preds = self.trend_model.predict(X_test)
        trend_probs = self.trend_model.predict_proba(X_test)[:, 1]
        return_preds = self.return_model.predict(X_test)

        self.trained = True

        # Calculate standard ML classification & regression metrics
        self.metrics = {
            "accuracy": accuracy_score(yt_test, trend_preds),
            "precision": precision_score(yt_test, trend_preds, zero_division=0),
            "recall": recall_score(yt_test, trend_preds, zero_division=0),
            "f1": f1_score(yt_test, trend_preds, zero_division=0),
            "rmse": np.sqrt(mean_squared_error(yr_test, return_preds))
        }

        # Build test results DataFrame
        test_results = df_train_eval.iloc[split_idx:].copy()
        test_results['predicted_trend'] = trend_preds
        test_results['predicted_prob'] = trend_probs
        test_results['predicted_return'] = return_preds

        return True, test_results

    def predict_next_day(self, df: pd.DataFrame, sector_index: str = None) -> tuple:
        """
        Uses the trained model to predict the market direction and expected return for t+1.
        """
        if not self.trained:
            logger.error("Predictor model has not been trained yet.")
            return None, None, None

        df_clean, features = self.prepare_data(df, sector_index)
        if df_clean.empty:
            return None, None, None

        # Predict based on the latest available row (which contains feature values shifted to t)
        last_row = df_clean.iloc[-1]
        X_latest = pd.DataFrame([last_row[features]])

        pred_trend = self.trend_model.predict(X_latest)[0]
        trend_probs = self.trend_model.predict_proba(X_latest)[0]
        confidence = max(trend_probs) * 100
        
        pred_return = self.return_model.predict(X_latest)[0]
        pred_price = last_row['close_price'] * (1 + pred_return / 100)

        return int(pred_trend), float(pred_price), float(confidence)
