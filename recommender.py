import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.preprocessing import LabelEncoder
import json
from typing import Dict, List, Optional, Tuple
import logging
import datetime
from scipy.sparse import coo_matrix, csr_matrix
import sqlite3
import pickle
import os
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelMetrics:
    """Data class for model metrics."""
    version: str
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    train_size: Optional[int] = None
    validation_accuracy: Optional[float] = None
    average_rating: Optional[float] = None
    created_at: datetime.datetime = datetime.datetime.now()


class DatabaseManager:
    def __init__(self, db_path: str = "toy_recommender.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    age INTEGER,
                    gender TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # User preferences table (many-to-many relationships)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    preference_type TEXT,  -- 'material', 'color', 'brand', 'toy_type', 'size'
                    preference_value TEXT,
                    preference_score FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

            # Feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    item_attributes TEXT,  -- JSON string of item attributes
                    rating FLOAT,
                    price_range TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

            # Model versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    version TEXT PRIMARY KEY,
                    accuracy FLOAT,
                    loss FLOAT,
                    train_dataset_size INTEGER,
                    validation_accuracy FLOAT,
                    average_rating FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 0
                )
            """)

            conn.commit()

    def add_user(self, user_id: int, age: int, gender: str):
        """Add a new user to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO users (user_id, age, gender) VALUES (?, ?, ?)",
                (user_id, age, gender)
            )
            conn.commit()

    def add_user_preference(
            self,
            user_id: int,
            preference_type: str,
            preference_value: str,
            preference_score: float
    ):
        """Add a user preference."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO user_preferences 
                   (user_id, preference_type, preference_value, preference_score)
                   VALUES (?, ?, ?, ?)""",
                (user_id, preference_type, preference_value, preference_score)
            )
            conn.commit()

    def get_user_preferences(self, user_id: int) -> Dict:
        """Get all preferences for a user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT preference_type, preference_value, preference_score 
                   FROM user_preferences WHERE user_id = ?""",
                (user_id,)
            )
            preferences = {}
            for pref_type, value, score in cursor.fetchall():
                if pref_type not in preferences:
                    preferences[pref_type] = []
                preferences[pref_type].append({
                    'value': value,
                    'score': score
                })
            return preferences

    def add_feedback(
            self,
            user_id: int,
            item_attributes: Dict,
            rating: float,
            price_range: str
    ):
        """Record user feedback."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO feedback 
                   (user_id, item_attributes, rating, price_range)
                   VALUES (?, ?, ?, ?)""",
                (user_id, json.dumps(item_attributes), rating, price_range)
            )
            conn.commit()

    def get_all_feedback(self) -> pd.DataFrame:
        """Get all feedback data."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM feedback", conn)

    def save_model_metrics(self, metrics: ModelMetrics):
        """Save model metrics to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO model_versions 
                   (version, accuracy, loss, train_dataset_size,
                    validation_accuracy, average_rating, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (metrics.version, metrics.accuracy, metrics.loss,
                 metrics.train_size, metrics.validation_accuracy,
                 metrics.average_rating, metrics.created_at)
            )
            conn.commit()

    def get_active_model_version(self) -> Optional[str]:
        """Get the currently active model version."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT version FROM model_versions WHERE is_active = 1"
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def set_active_model(self, version: str):
        """Set the active model version."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE model_versions SET is_active = 0")
            cursor.execute(
                "UPDATE model_versions SET is_active = 1 WHERE version = ?",
                (version,)
            )
            conn.commit()


class ModelManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

    def save_model(self, model: LightFM, version: str):
        """Save a model to disk."""
        model_path = self.models_dir / f"model_{version}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, version: str) -> Optional[LightFM]:
        """Load a model from disk."""
        model_path = self.models_dir / f"model_{version}.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None

class ToyRecommenderEnhanced:
    def __init__(self):
        self.db = DatabaseManager()
        self.model_manager = ModelManager()

        # Load the active model if it exists
        active_version = self.db.get_active_model_version()
        self.model = (self.model_manager.load_model(active_version)
                      if active_version else self._create_initial_model())

        # Initialize other components
        self.dataset = Dataset()
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        # Define possible values for each attribute
        self.attribute_values = {
            'type': ['doll', 'constructor', 'vehicle', 'furniture', 'educational',
                     'arts_crafts', 'outdoor', 'electronic'],
            'size': ['small', 'medium', 'large'],
            'material': ['plastic', 'wood', 'metal', 'fabric', 'leather'],
            'color': ['red', 'blue', 'green', 'yellow', 'pink', 'white', 'black',
                      'multicolor'],
            'brand': ['LEGO', 'Fisher-Price', 'Mattel', 'Hasbro', 'Melissa & Doug',
                      'Little Tikes']
        }

        # Generate synthetic items
        self.items_df = self._generate_item_combinations()

    def _create_initial_model(self) -> LightFM:
        """Create and save initial model."""
        model = LightFM(
            learning_rate=0.05,
            loss='warp',
            no_components=64,
            user_alpha=1e-6,
            item_alpha=1e-6
        )

        # Save initial model metrics
        metrics = ModelMetrics(
            version="v0.1",
            train_size=0
        )
        self.db.save_model_metrics(metrics)
        self.model_manager.save_model(model, "v0.1")
        self.db.set_active_model("v0.1")

        return model

    def train_model(self, validate: bool = True) -> ModelMetrics:
        """Train a new model version."""
        # Get all feedback data
        feedback_data = self.db.get_all_feedback()
        if feedback_data.empty:
            raise ValueError("No feedback data available for training")

        # Prepare training data
        train_data, val_data = train_test_split(
            feedback_data, test_size=0.2 if validate else 0.0
        )

        # Create new model
        new_model = LightFM(
            learning_rate=0.05,
            loss='warp',
            no_components=64,
            user_alpha=1e-6,
            item_alpha=1e-6
        )

        # Train model and compute metrics
        train_interactions, train_weights = self._prepare_interactions(train_data)
        metrics = self._train_and_evaluate(
            new_model,
            train_interactions,
            train_weights,
            val_data if validate else None
        )

        # Save new model and metrics
        self.model_manager.save_model(new_model, metrics.version)
        self.db.save_model_metrics(metrics)
        self.db.set_active_model(metrics.version)

        # Update current model
        self.model = new_model

        return metrics

    def _prepare_interactions(self, feedback_data: pd.DataFrame) -> Tuple[coo_matrix, coo_matrix]:
        """Prepare interaction matrices from feedback data."""
        # Implementation details...
        pass

    def _train_and_evaluate(
            self,
            model: LightFM,
            train_interactions: coo_matrix,
            train_weights: coo_matrix,
            val_data: Optional[pd.DataFrame] = None
    ) -> ModelMetrics:
        """Train model and compute metrics."""
        # Implementation details...
        pass

    def _generate_item_combinations(self) -> pd.DataFrame:
        """Generate realistic combinations of toy attributes."""
        items = []

        # Define some realistic constraints
        valid_combinations = [
            # Dolls
            {'type': 'doll', 'material': ['plastic', 'fabric'],
             'size': ['small', 'medium']},
            # Construction toys
            {'type': 'constructor', 'material': ['plastic', 'wood'],
             'size': ['small', 'medium']},
            # Vehicles
            {'type': 'vehicle', 'material': ['plastic', 'metal'],
             'size': ['small', 'medium', 'large']},
            # Furniture
            {'type': 'furniture', 'material': ['plastic', 'wood'],
             'size': ['medium', 'large']}
        ]

        item_id = 0
        for combo in valid_combinations:
            toy_type = combo['type']
            for material in combo['material']:
                for size in combo['size']:
                    for color in self.attribute_values['color']:
                        for brand in self.attribute_values['brand']:
                            # Add some realistic constraints
                            if (toy_type == 'constructor' and brand == 'LEGO' and
                                    material == 'plastic'):
                                price_range = np.random.choice(['medium', 'high'])
                                items.append({
                                    'item_id': item_id,
                                    'type': toy_type,
                                    'size': size,
                                    'material': material,
                                    'color': color,
                                    'brand': brand,
                                    'price_range': price_range
                                })
                                item_id += 1
                            elif toy_type != 'constructor':
                                price_range = np.random.choice(
                                    ['low', 'medium', 'high'],
                                    p=[0.3, 0.4, 0.3]
                                )
                                items.append({
                                    'item_id': item_id,
                                    'type': toy_type,
                                    'size': size,
                                    'material': material,
                                    'color': color,
                                    'brand': brand,
                                    'price_range': price_range
                                })
                                item_id += 1

        return pd.DataFrame(items)

    def _create_user_features(self, user_data: Dict) -> np.ndarray:
        """Create user feature vector from user data."""
        features = []

        # Age feature (normalized)
        age = user_data.get('age', 5)  # default age
        features.append(age / 12)  # normalize by maximum age (12 years)

        # Gender feature (one-hot)
        gender_map = {'male': [1, 0], 'female': [0, 1], 'other': [0, 0]}
        features.extend(gender_map.get(user_data.get('gender', 'other')))

        # Preferences (normalized scores)
        preferences = user_data.get('preferences', {})
        pref_features = [
            preferences.get('educational', 0.5),
            preferences.get('creative', 0.5),
            preferences.get('active', 0.5),
            preferences.get('social', 0.5)
        ]
        features.extend(pref_features)

        return np.array(features)

    def _create_item_features(self, item: pd.Series) -> np.ndarray:
        """Create item feature vector from item attributes."""
        features = []

        # One-hot encode categorical features
        for attr in ['type', 'size', 'material', 'color', 'brand']:
            values = self.attribute_values[attr]
            feature = [1 if item[attr] == val else 0 for val in values]
            features.extend(feature)

        # Add price range feature
        price_map = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
        features.append(price_map[item['price_range']])

        return np.array(features)

    def _initial_fit(self):
        """Perform initial model fitting with empty data."""
        n_users = 1
        n_items = len(self.items_df)

        # Create sparse interaction matrix
        row = np.array([0])  # Single user
        col = np.array([0])  # Single item
        data = np.array([0.0])  # Single interaction value

        interactions = coo_matrix(
            (data, (row, col)),
            shape=(n_users, n_items)
        )

        # Create feature matrices
        user_features = csr_matrix((n_users, 8))  # 8 is the number of user features
        item_features = csr_matrix(np.array([
            self._create_item_features(item)
            for _, item in self.items_df.iterrows()
        ]))

        # Fit the model
        self.model.fit(
            interactions,
            user_features=user_features,
            item_features=item_features,
            epochs=1,
            num_threads=4
        )

    def _cold_start_scoring(self, user_data: Dict, items: pd.DataFrame) -> np.ndarray:
        """Simple rule-based scoring for cold start recommendations."""
        scores = np.zeros(len(items))

        age = user_data.get('age', 5)
        gender = user_data.get('gender', 'other')
        preferences = user_data.get('preferences', {})

        for i in range(len(items)):
            item = items.iloc[i]
            score = 0.5  # base score

            # Age-based scoring
            if age < 3:
                if item['size'] == 'small':
                    score += 0.2
            elif 3 <= age <= 7:
                if item['size'] == 'medium':
                    score += 0.2
            else:  # age > 7
                if item['size'] in ['medium', 'large']:
                    score += 0.2

            # Educational preference
            if preferences.get('educational', 0.5) > 0.7:
                if item['type'] in ['educational', 'constructor']:
                    score += 0.3

            # Creative preference
            if preferences.get('creative', 0.5) > 0.7:
                if item['type'] in ['arts_crafts', 'constructor']:
                    score += 0.3

            # Active preference
            if preferences.get('active', 0.5) > 0.7:
                if item['type'] in ['outdoor', 'vehicle']:
                    score += 0.3

            # Social preference
            if preferences.get('social', 0.7) > 0.7:
                if item['type'] in ['doll', 'furniture']:
                    score += 0.3

            # Age safety adjustment
            if age < 3 and item['material'] in ['metal', 'small']:
                score -= 0.4

            scores[i] = max(0.1, min(1.0, score))  # Clamp between 0.1 and 1.0

        return scores

    def recommend(
            self,
            user_data: Dict,
            price_constraint: Optional[float] = None,
            n_recommendations: int = 5
    ) -> List[Dict]:
        """Get toy recommendations based on user data and constraints."""
        # Create user features
        user_features = csr_matrix(self._create_user_features(user_data).reshape(1, -1))

        # Filter items by price constraint
        if price_constraint is not None:
            price_ranges = {
                'low': (0, 30),
                'medium': (30, 70),
                'high': (70, float('inf'))
            }
            valid_ranges = [
                range_name for range_name, (min_price, max_price)
                in price_ranges.items()
                if min_price <= price_constraint
            ]
            valid_items = self.items_df[
                self.items_df['price_range'].isin(valid_ranges)
            ]
        else:
            valid_items = self.items_df

        # Get predictions for all valid items
        item_features = csr_matrix(np.array([
            self._create_item_features(item)
            for _, item in valid_items.iterrows()
        ]))

        # If we're in cold start (no feedback data)
        if self.feedback_db.empty:
            # Use simple rule-based scoring
            scores = self._cold_start_scoring(user_data, valid_items)
        else:
            # Use trained model for predictions
            scores = [
                self.model.predict(
                    user_ids=[0],
                    item_ids=[i],
                    user_features=user_features,
                    item_features=item_features[i].reshape(1, -1)
                )[0]
                for i in range(len(valid_items))
            ]

        # Get top N recommendations
        top_indices = np.argsort(scores)[-n_recommendations:][::-1]
        recommendations = []

        for idx in top_indices:
            item = valid_items.iloc[idx]
            recommendations.append({
                'type': item['type'],
                'size': item['size'],
                'material': item['material'],
                'color': item['color'],
                'brand': item['brand'],
                'confidence_score': float(scores[idx]),
                'price_range': item['price_range']
            })

        return recommendations

    def record_feedback(
            self,
            user_id: int,
            recommendation: Dict,
            rating: float,
            price_range: str
    ):
        """Record user feedback for a recommendation."""
        # Create a unique item ID from the recommendation attributes
        item_key = json.dumps({
            k: recommendation[k]
            for k in ['type', 'size', 'material', 'color', 'brand']
        })

        # Add to feedback database
        self.feedback_db = pd.concat([
            self.feedback_db,
            pd.DataFrame([{
                'user_id': user_id,
                'item_id': item_key,
                'rating': rating,
                'timestamp': datetime.datetime.now(),
                'price_range': price_range
            }])
        ], ignore_index=True)

        # Retrain model with new feedback
        self.fit()

    def fit(self):
        """Train the recommendation model with feedback data."""
        if not self.feedback_db.empty:
            # Create interaction matrix
            n_users = len(self.feedback_db['user_id'].unique())
            n_items = len(self.items_df)

            # Create sparse matrix in COO format
            rows = []
            cols = []
            data = []

            for idx, row in self.feedback_db.iterrows():
                user_idx = self.user_encoder.fit_transform([row['user_id']])[0]
                item_idx = self.item_encoder.fit_transform([row['item_id']])[0]
                rows.append(user_idx)
                cols.append(item_idx)
                data.append(row['rating'])

            interactions = coo_matrix(
                (data, (rows, cols)),
                shape=(n_users, n_items)
            )

            # Create feature matrices
            user_features = csr_matrix((n_users, 8))
            item_features = csr_matrix(np.array([
                self._create_item_features(item)
                for _, item in self.items_df.iterrows()
            ]))

            # Fit model
            self.model.fit(
                interactions,
                user_features=user_features,
                item_features=item_features,
                epochs=30,
                num_threads=4
            )


# Example usage
if __name__ == "__main__":
    recommender = ToyRecommenderEnhanced()

    # Add a user
    recommender.db.add_user(1, 6, "male")
    recommender.db.add_user(2, 11, "female")

    # Add user preferences
    recommender.db.add_user_preference(1, "material", "wood", 0.8)
    recommender.db.add_user_preference(2, "type", "educational", 0.9)

    # Get recommendations based on user preferences
    user_data = {
        'age': 6,
        'gender': 'female',
        'preferences': recommender.db.get_user_preferences(1)
    }

    recommendations = recommender.recommend(
        user_data=user_data,
        price_constraint=50,
        n_recommendations=3
    )

    # Record feedback
    recommender.db.add_feedback(
        user_id=1,
        item_attributes=recommendations[0],
        rating=4.5,
        price_range="medium"
    )

    # Train new model version
    try:
        metrics = recommender.train_model()
        print(f"New model trained: {metrics}")
    except ValueError as e:
        print(f"Training failed: {e}")