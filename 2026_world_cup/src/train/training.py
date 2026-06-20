import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


def prepare_binary_classification_data(df_match_enriched, feature_cols, test_size=0.2, random_state=42):
    """
    Prepare data for binary classification (home win vs away win, excluding draws)
    
    Args:
        df_match_enriched: DataFrame with match data and team statistics
        feature_cols: List of feature column names
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # Remove rows with missing target values
    df_train = df_match_enriched.dropna(subset=['home_team_win'])
    
    # Prepare features and target
    X = df_train[feature_cols]
    y_continuous = df_train['home_team_win']
    
    # Convert to binary classification: exclude draws (0.5)
    # 0.0 -> 'away_win', 1.0 -> 'home_win'
    binary_mask = (y_continuous == 0.0) | (y_continuous == 1.0)
    X_binary = X[binary_mask]
    y_binary = y_continuous[binary_mask]
    
    # Convert to categorical labels
    y = y_binary.map({0.0: 'away_win', 1.0: 'home_win'})
    
    # Display class distribution
    print(f"Binary Classification - Class distribution:")
    print(f"  Away wins: {(y == 'away_win').sum()}")
    print(f"  Home wins: {(y == 'home_win').sum()}")
    print(f"  Total matches (excluding draws): {len(y)}")
    print(f"  Draws excluded: {(~binary_mask).sum()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_logistic_regression(X_train, y_train, X_test, y_test, max_iter=2000):
    """Train and evaluate Logistic Regression"""
    print("\n" + "="*50)
    print("LOGISTIC REGRESSION")
    print("="*50)
    
    model = LogisticRegression(
        random_state=42,
        max_iter=max_iter,
        solver="lbfgs"
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=['away_win', 'home_win']))
    
    return model, y_pred, accuracy


def train_random_forest(X_train, y_train, X_test, y_test, feature_cols, 
                       n_estimators=100, max_depth=12):
    """Train and evaluate Random Forest"""
    print("\n" + "="*50)
    print("RANDOM FOREST")
    print("="*50)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, y_pred, accuracy


def train_xgboost(X_train, y_train, X_test, y_test, feature_cols,
                 n_estimators=100, max_depth=6, learning_rate=0.1):
    """Train and evaluate XGBoost"""
    print("\n" + "="*50)
    print("XGBOOST")
    print("="*50)
    
    # Encode labels for XGBoost
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train_encoded)
    y_pred_encoded = model.predict(X_test)
    
    # Decode predictions
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=['away_win', 'home_win']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop Features:")
    print(feature_importance)
    
    return model, y_pred, accuracy


def train_svm(X_train, y_train, X_test, y_test, C=10.0):
    """Train and evaluate SVM"""
    print("\n" + "="*50)
    print("SVM")
    print("="*50)
    
    model = SVC(
        kernel='rbf',
        C=C,
        gamma='scale',
        probability=True,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, y_pred, accuracy


def train_knn(X_train, y_train, X_test, y_test, n_neighbors=3):
    """Train and evaluate KNN"""
    print("\n" + "="*50)
    print("KNN")
    print("="*50)
    
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights='distance',
        metric='euclidean'
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, y_pred, accuracy


def train_all_models(df_match_enriched, feature_cols, test_size=0.2, random_state=42):
    """
    Train all models and compare results
    
    Args:
        df_match_enriched: DataFrame with match data and team statistics
        feature_cols: List of feature column names
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Dictionary with all models and results
    """
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_binary_classification_data(
        df_match_enriched, feature_cols, test_size, random_state
    )
    
    results = {}
    
    # Train all models
    lr_model, lr_pred, lr_acc = train_logistic_regression(X_train, y_train, X_test, y_test)
    results['Logistic Regression'] = {'model': lr_model, 'predictions': lr_pred, 'accuracy': lr_acc}
    
    rf_model, rf_pred, rf_acc = train_random_forest(X_train, y_train, X_test, y_test, feature_cols)
    results['Random Forest'] = {'model': rf_model, 'predictions': rf_pred, 'accuracy': rf_acc}
    
    xgb_model, xgb_pred, xgb_acc = train_xgboost(X_train, y_train, X_test, y_test, feature_cols)
    results['XGBoost'] = {'model': xgb_model, 'predictions': xgb_pred, 'accuracy': xgb_acc}
    
    svm_model, svm_pred, svm_acc = train_svm(X_train, y_train, X_test, y_test)
    results['SVM'] = {'model': svm_model, 'predictions': svm_pred, 'accuracy': svm_acc}
    
    knn_model, knn_pred, knn_acc = train_knn(X_train, y_train, X_test, y_test)
    results['KNN'] = {'model': knn_model, 'predictions': knn_pred, 'accuracy': knn_acc}
    
    # Model comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results.keys()]
    }).sort_values('Accuracy', ascending=False)
    
    print(comparison.to_string(index=False))
    
    best_model = comparison.iloc[0]['Model']
    best_accuracy = comparison.iloc[0]['Accuracy']
    print(f"\n🏆 Best Model: {best_model} with accuracy: {best_accuracy:.4f}")
    
    return {
        'results': results,
        'comparison': comparison,
        'test_data': (X_test, y_test),
        'scaler': scaler
    }


if __name__ == "__main__":
    print("Training Module")
    print("=" * 50)
    print("\nThis module provides binary classification training for match prediction.")
    print("Excludes draws - only predicts home win vs away win.")
    print("\nUsage:")
    print("  from training import train_all_models")
    print("  results = train_all_models(df_match_enriched, feature_cols)")
