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

            # Users table (unchanged)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    age INTEGER,
                    gender TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Modified user preferences table - removed score
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    preference_type TEXT,  -- 'material', 'color', 'brand', 'type', 'size'
                    preference_value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

            # Other tables remain unchanged
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    item_attributes TEXT,
                    rating FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

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

    def add_user_preferences(
            self,
            user_id: int,
            preferences: Dict[str, List[str]]
    ):
        """
        Add multiple user preferences.

        Args:
            user_id: User identifier
            preferences: Dictionary where keys are preference types
                        (color, brand, etc.) and values are lists of
                        preferred values
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # First delete existing preferences for this user
            cursor.execute(
                "DELETE FROM user_preferences WHERE user_id = ?",
                (user_id,)
            )

            # Add new preferences
            for pref_type, values in preferences.items():
                for value in values:
                    cursor.execute(
                        """INSERT INTO user_preferences 
                           (user_id, preference_type, preference_value)
                           VALUES (?, ?, ?)""",
                        (user_id, pref_type, value)
                    )

            conn.commit()

    def get_user_preferences(self, user_id: int) -> Dict[str, List[str]]:
        """
        Get all preferences for a user.

        Returns:
            Dictionary where keys are preference types and values are
            lists of preferred values
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT preference_type, preference_value 
                   FROM user_preferences WHERE user_id = ?""",
                (user_id,)
            )

            preferences = {}
            for pref_type, value in cursor.fetchall():
                if pref_type not in preferences:
                    preferences[pref_type] = []
                preferences[pref_type].append(value)

            return preferences

    def add_feedback(
            self,
            user_id: int,
            item_attributes: Dict,
            rating: float,
    ):
        """Record user feedback."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO feedback 
                   (user_id, item_attributes, rating)
                   VALUES (?, ?, ?)""",
                (user_id, json.dumps(item_attributes), rating)
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

        # Initialize feedback database as DataFrame
        self.feedback_db = pd.DataFrame(columns=[
            'user_id', 'item_attributes', 'rating', 'timestamp'
        ])

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
                      'Little Tikes', 'LOL', 'Hot Wheels']
        }

        # Generate synthetic items
        self.items_df = self._generate_item_combinations()

        # Load existing feedback from database if any exists
        existing_feedback = self.db.get_all_feedback()
        if not existing_feedback.empty:
            self.feedback_db = existing_feedback

    def _create_initial_model(self) -> LightFM:
        """Create and save initial model."""
        model = LightFM(
            learning_rate=0.05,
            loss='warp',
            no_components=64,
            user_alpha=1e-6,
            item_alpha=1e-6
        )

        # Create initial empty interaction matrix
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

        # Fit the model with empty data
        model.fit(
            interactions,
            user_features=user_features,
            item_features=item_features,
            epochs=1,
            num_threads=4
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

        # Only split if we have enough data
        if len(feedback_data) > 5 and validate:  # Only split if we have more than 5 ratings
            train_data, val_data = train_test_split(
                feedback_data, test_size=0.2, random_state=42
            )
        else:
            train_data = feedback_data
            val_data = None if validate else None

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
            val_data
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
        """Prepare interaction matrices from feedback data."""
        n_users = len(feedback_data['user_id'].unique())
        n_items = len(self.items_df)

        # Create sparse matrix in COO format
        rows = []
        cols = []
        data = []

        for idx, row in feedback_data.iterrows():
            user_idx = self.user_encoder.fit_transform([row['user_id']])[0]
            item_attr = json.loads(row['item_attributes'])

            # Find the corresponding item in items_df
            matching_items = self.items_df[
                (self.items_df['type'] == item_attr['type']) &
                (self.items_df['size'] == item_attr['size']) &
                (self.items_df['material'] == item_attr['material']) &
                (self.items_df['color'] == item_attr['color']) &
                (self.items_df['brand'] == item_attr['brand'])
                ]

            if not matching_items.empty:
                item_idx = matching_items.index[0]
                rows.append(user_idx)
                cols.append(item_idx)
                data.append(row['rating'])

        # Create interactions matrix
        interactions = coo_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items)
        )

        # Create weights matrix (all 1.0 for now)
        weights = coo_matrix(
            (np.ones_like(data), (rows, cols)),
            shape=(n_users, n_items)
        )

        return interactions, weights

    def _train_and_evaluate(
            self,
            model: LightFM,
            train_interactions: coo_matrix,
            train_weights: coo_matrix,
            val_data: Optional[pd.DataFrame] = None
    ) -> ModelMetrics:
        """Train model and compute metrics."""
        """Train model and compute metrics."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v{timestamp}"

        metrics = ModelMetrics(version=version)

        # Create feature matrices
        n_users = train_interactions.shape[0]
        user_features = csr_matrix((n_users, 8))
        item_features = csr_matrix(np.array([
            self._create_item_features(item)
            for _, item in self.items_df.iterrows()
        ]))

        # Train the model
        model.fit(
            train_interactions,
            user_features=user_features,
            item_features=item_features,
            sample_weight=train_weights,
            epochs=30,
            num_threads=4
        )

        # Calculate training metrics
        train_predictions = model.predict(
            user_ids=train_interactions.row,
            item_ids=train_interactions.col,
            user_features=user_features,
            item_features=item_features
        )

        # Calculate RMSE instead of simple accuracy
        actual_ratings = train_interactions.data
        rmse = np.sqrt(np.mean((train_predictions - actual_ratings) ** 2))
        mae = np.mean(np.abs(train_predictions - actual_ratings))

        # Calculate validation metrics if validation data is provided
        validation_rmse = None
        if val_data is not None and not val_data.empty:
            val_interactions, val_weights = self._prepare_interactions(val_data)
            val_predictions = model.predict(
                user_ids=val_interactions.row,
                item_ids=val_interactions.col,
                user_features=csr_matrix((val_interactions.shape[0], 8)),
                item_features=item_features
            )
            validation_rmse = np.sqrt(np.mean((val_predictions - val_interactions.data) ** 2))

        # Update metrics
        metrics.accuracy = float(mae)  # Using MAE instead of binary accuracy
        metrics.loss = float(rmse)  # Using RMSE as loss
        metrics.train_size = len(actual_ratings)  # Using actual number of ratings
        metrics.validation_accuracy = float(validation_rmse) if validation_rmse is not None else None
        metrics.average_rating = float(np.mean(actual_ratings))

        return metrics

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
        interactions = coo_matrix(
            ([0.0], ([0], [0])),
            shape=(n_users, n_items)
        )

        # Create feature matrices for all items
        user_features = csr_matrix((n_users, 8))
        item_features = csr_matrix(np.array([
            self._create_item_features(item)
            for _, item in self.items_df.iterrows()
        ]))

        # Fit the model with empty data
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
        preferences = user_data.get('preferences', {})

        # Preference type weights
        preference_weights = {
            'color': 0.15,
            'brand': 0.25,
            'material': 0.2,
            'type': 0.25,
            'size': 0.15
        }

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

            # Preference matching
            for pref_type, pref_values in preferences.items():
                if pref_type in preference_weights:
                    weight = preference_weights[pref_type]
                    # If item attribute matches any of the user's preferences for this type
                    if item[pref_type] in pref_values:
                        score += weight

            # Safety adjustments
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

        # If we're in cold start (no feedback data) or model hasn't been fitted
        if len(self.feedback_db) == 0:
            # Use simple rule-based scoring
            scores = self._cold_start_scoring(user_data, valid_items)
        else:
            try:
                # Create feature matrices for all items
                item_features = csr_matrix(np.array([
                    self._create_item_features(item)
                    for _, item in self.items_df.iterrows()  # Use full items_df
                ]))

                # Get indices of valid items
                valid_indices = valid_items.index.values

                # Try to use the trained model
                scores = [
                    self.model.predict(
                        user_ids=[0],
                        item_ids=[idx],
                        user_features=user_features,
                        item_features=item_features
                    )[0]
                    for idx in valid_indices
                ]
            except ValueError:
                # If model isn't fitted, use cold start scoring
                scores = self._cold_start_scoring(user_data, valid_items)

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
                'item_attributes': json.dumps(recommendation),
                'rating': rating,
                'timestamp': datetime.datetime.now(),
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

            # Create mapping of item_id to index
            item_to_idx = {str(item_id): idx for idx, item_id in enumerate(self.items_df.index)}

            for idx, row in self.feedback_db.iterrows():
                user_idx = self.user_encoder.fit_transform([row['user_id']])[0]
                item_attr = json.loads(row['item_attributes'])
                # Find the corresponding item in items_df
                matching_items = self.items_df[
                    (self.items_df['type'] == item_attr['type']) &
                    (self.items_df['size'] == item_attr['size']) &
                    (self.items_df['material'] == item_attr['material']) &
                    (self.items_df['color'] == item_attr['color']) &
                    (self.items_df['brand'] == item_attr['brand'])
                    ]
                if not matching_items.empty:
                    item_idx = matching_items.index[0]
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

    # # Add a user
    # recommender.db.add_user(1, 6, "male")
    # recommender.db.add_user(2, 11, "female")
    #
    # # Add user preferences in the new format
    # user1_preferences = {
    #     'color': ['blue', 'yellow', 'white'],
    #     'brand': ['LEGO', 'Hot Wheels'],
    #     'material': ['plastic', 'wood'],
    #     'type': ['vehicle', 'constructor', 'outdoor']
    # }
    #
    # user2_preferences = {
    #     'color': ['blue', 'gold', 'black'],
    #     'brand': ['LOL', 'LEGO'],
    #     'material': ['plastic', 'fabric'],
    #     'type': ['arts_crafts', 'electronic']
    # }

    # recommender.db.add_user_preferences(1, user1_preferences)
    # recommender.db.add_user_preferences(2, user2_preferences)

    # Get recommendations based on user preferences
#     user_data = {
#         'age': 6,
#         'gender': 'male',
#         'preferences': recommender.db.get_user_preferences(1)
#     }
#
#     recommendations = recommender.recommend(
#         user_data=user_data,
#         price_constraint=50,
#         n_recommendations=3
#     )
#
#     for i, rec in enumerate(recommendations, 1):
#         print(f"\nRecommendation {i}:")
#         print(f"Type: {rec['type']}")
#         print(f"Size: {rec['size']}")
#         print(f"Material: {rec['material']}")
#         print(f"Color: {rec['color']}")
#         print(f"Brand: {rec['brand']}")
#         print(f"Price Range: {rec['price_range']}")
#         print(f"Confidence Score: {rec['confidence_score']:.2f}")
#
# # # Record feedback
#     recommender.db.add_feedback(
#         user_id=1,
#         item_attributes=recommendations[2],
#         rating=4.5
#     )
#
    # recommender.fit()

    #
    # Train new model version
    try:
        metrics = recommender.train_model()
        print(f"New model trained: {metrics}")
    except ValueError as e:
        print(f"Training failed: {e}")