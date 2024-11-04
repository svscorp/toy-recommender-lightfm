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


class ToyAttributeRecommender:
    def __init__(self):
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

        # Generate synthetic items from attribute combinations
        self.items_df = self._generate_item_combinations()

        # Initialize feedback storage
        self.feedback_db = pd.DataFrame(columns=[
            'user_id', 'item_id', 'rating', 'timestamp', 'price_range'
        ])

        # Initialize the model
        self.model = LightFM(
            learning_rate=0.05,
            loss='warp',
            no_components=64,
            user_alpha=1e-6,
            item_alpha=1e-6
        )

        # Perform initial fit
        self._initial_fit()

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
            user_id: str,
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
    # Initialize recommender
    recommender = ToyAttributeRecommender()

    # Example user data
    user_data = {
        'age': 6,
        'gender': 'female',
        'preferences': {
            'educational': 0.8,
            'creative': 0.9,
            'active': 0.5,
            'social': 0.7
        }
    }

    # Get recommendations
    recommendations = recommender.recommend(
        user_data=user_data,
        price_constraint=50,
        n_recommendations=3
    )

    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\nRecommendation {i}:")
        print(f"Type: {rec['type']}")
        print(f"Size: {rec['size']}")
        print(f"Material: {rec['material']}")
        print(f"Color: {rec['color']}")
        print(f"Brand: {rec['brand']}")
        print(f"Price Range: {rec['price_range']}")
        print(f"Confidence Score: {rec['confidence_score']:.2f}")

    # # Example of recording feedback
    # recommender.record_feedback(
    #     user_id="1",
    #     recommendation=recommendations[0],
    #     rating=2,
    #     price_range="medium"
    # )