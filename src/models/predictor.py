import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

class PricePredictor:
    def __init__(self):
        self.trend_model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.return_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.accuracy = 0.0
        self.rmse = 0.0
        self.trained = False

    def prepare_data(self, df):
        """
        Prepare features and targets.
        Features: lag1_return, lag2_return, rolling_mean_return_5, volatility_5
        Targets: target_return (next day return), target_trend (1 for UP, 0 for DOWN)
        """
        df = df.copy()
        df = df.sort_values('date')
        
        # We need daily_return_pct to calculate things safely
        if 'daily_return_pct' not in df.columns:
            df['daily_return_pct'] = df['close_price'].pct_change() * 100

        # Features
        df['lag1_return'] = df['daily_return_pct'].shift(1)
        df['lag2_return'] = df['daily_return_pct'].shift(2)
        df['rolling_mean_return_5'] = df['daily_return_pct'].rolling(window=5).mean().shift(1)
        df['volatility_5'] = df['daily_return_pct'].rolling(window=5).std().shift(1)
        
        # Targets
        # Predict the percentage return, NOT the absolute price
        df['target_return'] = df['daily_return_pct'].shift(-1)
        df['target_trend'] = np.where(df['target_return'] > 0, 1, 0)

        # Drop NaNs
        df = df.dropna()
        
        features = ['daily_return_pct', 'lag1_return', 'lag2_return', 'rolling_mean_return_5', 'volatility_5']
        
        return df, features

    def train_and_evaluate(self, df):
        if len(df) < 20:
            return False, None
            
        df_clean, features = self.prepare_data(df)
        if len(df_clean) < 10:
            return False, None

        X = df_clean[features]
        y_trend = df_clean['target_trend']
        y_return = df_clean['target_return']

        # Simple time-series split (last 20% for testing)
        split_idx = int(len(X) * 0.8)
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        yt_train, yt_test = y_trend.iloc[:split_idx], y_trend.iloc[split_idx:]
        yr_train, yr_test = y_return.iloc[:split_idx], y_return.iloc[split_idx:]

        # Train Trend Model
        self.trend_model.fit(X_train, yt_train)
        trend_preds = self.trend_model.predict(X_test)
        self.accuracy = accuracy_score(yt_test, trend_preds) * 100

        # Train Return Model
        self.return_model.fit(X_train, yr_train)
        return_preds = self.return_model.predict(X_test)
        
        # RMSE on percentage returns
        self.rmse = np.sqrt(mean_squared_error(yr_test, return_preds))
        
        self.trained = True
        
        # --- Backtesting Logic ---
        # "If prediction UP -> Buy -> Sell next day"
        test_results = df_clean.iloc[split_idx:].copy()
        test_results['predicted_return'] = return_preds
        test_results['predicted_trend'] = trend_preds
        
        # Calculate derived price from predicted return
        # next_day_price = today_price * (1 + predicted_return / 100)
        test_results['predicted_price'] = test_results['close_price'] * (1 + test_results['predicted_return'] / 100)
        test_results['actual_price'] = test_results['close_price'].shift(-1) # For visualization

        # Buy & Hold strategy return over test period
        bnh_returns = test_results['target_return'] / 100
        test_results['bnh_cumulative'] = (1 + bnh_returns).cumprod()

        # Model strategy return: If predicted trend == 1, we capture target_return, else 0.
        strategy_returns = np.where(test_results['predicted_trend'] == 1, test_results['target_return'] / 100, 0)
        test_results['strategy_cumulative'] = (1 + strategy_returns).cumprod()

        return True, test_results

    def predict_next_day(self, df):
        if not self.trained or len(df) < 5:
            return None, None
            
        # Manually construct features for the very last known day
        df = df.copy()
        if 'daily_return_pct' not in df.columns:
            df['daily_return_pct'] = df['close_price'].pct_change() * 100
            
        last_row = df.iloc[-1]
        
        feature_dict = {
            'daily_return_pct': last_row['daily_return_pct'],
            'lag1_return': df.iloc[-2]['daily_return_pct'] if len(df) >= 2 else 0,
            'lag2_return': df.iloc[-3]['daily_return_pct'] if len(df) >= 3 else 0,
            'rolling_mean_return_5': df['daily_return_pct'].tail(5).mean(),
            'volatility_5': df['daily_return_pct'].tail(5).std()
        }
        
        X_latest = pd.DataFrame([feature_dict])
        
        pred_trend = self.trend_model.predict(X_latest)[0]
        trend_probs = self.trend_model.predict_proba(X_latest)[0]
        confidence = max(trend_probs) * 100
        
        pred_return = self.return_model.predict(X_latest)[0]
        
        # Derive price
        pred_price = last_row['close_price'] * (1 + pred_return / 100)
        
        return pred_trend, pred_price, confidence
