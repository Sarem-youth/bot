import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class GoldPricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.trained = False
        
    def prepare_features(self, data):
        """Convert raw candle data to feature matrix"""
        features = []
        targets = []
        
        for i in range(len(data) - 1):
            candle = data[i]
            next_candle = data[i + 1]
            
            # Technical indicators
            rsi = candle['rsi']
            macd = candle['macd']
            bb_width = candle['bb_width']
            volatility = candle['volatility']
            
            features.append([rsi, macd, bb_width, volatility])
            
            # Target: Price direction (1 for up, -1 for down)
            price_direction = 1 if next_candle['close'] > candle['close'] else -1
            targets.append(price_direction)
            
        return np.array(features), np.array(targets)
        
    def train(self, training_data):
        """Train the model on historical data"""
        X, y = self.prepare_features(training_data)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.trained = True
        
        # Calculate accuracy
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        return {
            'train_score': train_score,
            'test_score': test_score
        }
        
    def predict(self, features):
        """Make predictions for new data"""
        if not self.trained:
            return None
            
        features_scaled = self.scaler.transform([features])
        return self.model.predict(features_scaled)[0]
        
    def save_model(self, path):
        """Save model to file"""
        if not self.trained:
            return False
            
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler
            }, path)
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
            
    def load_model(self, path):
        """Load model from file"""
        try:
            saved_model = joblib.load(path)
            self.model = saved_model['model']
            self.scaler = saved_model['scaler']
            self.trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
