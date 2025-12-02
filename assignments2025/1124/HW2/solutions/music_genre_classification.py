"""
Music Genre Classification using GTZAN Dataset
===============================================
This project classifies music genres using machine learning algorithms
based on audio features extracted from the GTZAN dataset.

Dataset: GTZAN (10 genres × 100 audio files each)
Features: Audio features from features_3_sec.csv (3-second segments)
Test Set: Last 20 audio tracks per genre (files 80-99) as per requirement
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, GroupKFold
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    classification_report,
    precision_recall_fscore_support
)
from sklearn.pipeline import Pipeline

# ML Models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

import warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, '../input/gtzan-dataset-music-genre-classification/Data')
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')

# Genre list
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_data():
    """Load features_3_sec.csv dataset"""
    csv_path = os.path.join(DATA_PATH, 'features_3_sec.csv')
    data = pd.read_csv(csv_path)
    print(f"Loaded data shape: {data.shape}")
    print(f"Columns: {list(data.columns[:5])}... ({len(data.columns)} total)")
    return data


def extract_file_number(filename):
    """
    Extract the original file number from filename.
    Example: 'blues.00045.3.wav' -> 45
    """
    parts = filename.split('.')
    if len(parts) >= 2:
        return int(parts[1])
    return -1


def split_train_test(data):
    """
    Split data according to requirement:
    - Train: files 00-79 (first 80 files per genre)
    - Test: files 80-99 (last 20 files per genre)
    
    Since features_3_sec.csv has 10 segments per original audio file,
    this splits by the original audio file number.
    
    Also returns groups for GroupKFold cross-validation.
    """
    data = data.copy()
    data['file_number'] = data['filename'].apply(extract_file_number)
    
    # Extract genre from filename for unique group ID
    # e.g., 'blues.00045.3.wav' -> 'blues_45'
    data['genre'] = data['filename'].apply(lambda x: x.split('.')[0])
    # Create unique group ID: genre_index * 100 + file_number
    genre_mapping = {g: i for i, g in enumerate(sorted(data['genre'].unique()))}
    data['group_id'] = data['genre'].map(genre_mapping) * 100 + data['file_number']
    
    # Train: files 0-79, Test: files 80-99
    train_mask = data['file_number'] < 80
    test_mask = data['file_number'] >= 80
    
    train_data = data[train_mask].copy()
    test_data = data[test_mask].copy()
    
    # Extract groups for CV (before dropping columns)
    train_groups = train_data['group_id'].values
    
    # Drop the helper columns
    train_data = train_data.drop(columns=['file_number', 'genre', 'group_id'])
    test_data = test_data.drop(columns=['file_number', 'genre', 'group_id'])
    
    print(f"Train set size: {len(train_data)} samples")
    print(f"Test set size: {len(test_data)} samples")
    print(f"Train genres distribution:\n{train_data['label'].value_counts()}")
    print(f"Number of unique groups in train set: {len(np.unique(train_groups))}")
    
    return train_data, test_data, train_groups


def prepare_features(train_data, test_data, scaler_type='standard'):
    """
    Prepare features and labels for training and testing.
    - Extract features (X) and labels (y)
    - Encode labels to numeric
    - Normalize features
    
    Args:
        scaler_type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
    """
    # Drop filename and extract labels
    X_train = train_data.drop(columns=['filename', 'label', 'length'])
    y_train = train_data['label']
    X_test = test_data.drop(columns=['filename', 'label', 'length'])
    y_test = test_data['label']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Normalize features
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
        
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns
    )
    
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, label_encoder


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single model"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred


def run_all_models(X_train, X_test, y_train, y_test):
    """Run multiple classification models and compare results"""
    
    models = {
        'Naive Bayes': GaussianNB(),
        'SGD': SGDClassifier(max_iter=5000, random_state=42, loss='modified_huber'),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'KNN (k=10)': KNeighborsClassifier(n_neighbors=10, weights='distance'),
        'Decision Tree': DecisionTreeClassifier(max_depth=15, min_samples_split=5, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=2, 
            n_jobs=-1, random_state=42
        ),
        'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        'SVM (Linear)': SVC(kernel='linear', C=1.0, random_state=42),
        'Logistic Regression': LogisticRegression(
            max_iter=2000, C=1.0, random_state=42, 
            multi_class='multinomial', solver='lbfgs'
        ),
        'MLP Neural Network': MLPClassifier(
            hidden_layer_sizes=(512, 256, 128), 
            max_iter=1000, 
            early_stopping=True,
            random_state=42,
            learning_rate='adaptive'
        ),
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=500, 
            learning_rate=0.1, 
            max_depth=8,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1
        )
    
    results = {}
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...", end=" ")
        accuracy, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results[name] = {'accuracy': accuracy, 'predictions': y_pred, 'model': model}
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return results


def plot_model_comparison(results):
    """Plot accuracy comparison of all models"""
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = plt.barh(models, accuracies, color=colors)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(acc + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{acc:.2%}', va='center', fontsize=10)
    
    plt.xlabel('Accuracy', fontsize=12)
    plt.title('Model Accuracy Comparison - Music Genre Classification', fontsize=14)
    plt.xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'model_comparison.png'), dpi=150)
    plt.close()
    print(f"\nSaved: {os.path.join(OUTPUT_PATH, 'model_comparison.png')}")


def plot_confusion_matrix(y_test, y_pred, label_encoder, model_name='Best Model'):
    """Plot confusion matrix for the predictions"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_PATH, 'confusion_matrix.png')}")


def plot_correlation_heatmap(data):
    """Plot correlation heatmap for mean features"""
    # Select only mean features
    mean_cols = [col for col in data.columns if 'mean' in col]
    corr = data[mean_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=(16, 12))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
    plt.title('Correlation Heatmap (Mean Features)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_PATH, 'correlation_heatmap.png')}")


def plot_bpm_boxplot(data):
    """Plot BPM distribution by genre"""
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='label', y='tempo', data=data, palette='husl')
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('BPM (Tempo)', fontsize=12)
    plt.title('BPM Distribution by Music Genre', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'bpm_boxplot.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_PATH, 'bpm_boxplot.png')}")


def plot_pca_visualization(X, y, label_encoder):
    """Visualize data using PCA (2D)"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=y, cmap='tab10', alpha=0.6, s=30)
    plt.colorbar(scatter, ticks=range(10), label='Genre')
    
    # Add legend
    handles = [plt.scatter([], [], c=plt.cm.tab10(i/10), label=label_encoder.classes_[i]) 
               for i in range(10)]
    plt.legend(handles=handles, title='Genre', loc='best')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.title('PCA Visualization of Music Genres', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'pca_visualization.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_PATH, 'pca_visualization.png')}")
    
    print(f"PCA Explained Variance: PC1={pca.explained_variance_ratio_[0]:.2%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.2%}, "
          f"Total={sum(pca.explained_variance_ratio_):.2%}")


def generate_classification_report(y_test, y_pred, label_encoder):
    """Generate and save detailed classification report"""
    report = classification_report(y_test, y_pred, 
                                  target_names=label_encoder.classes_,
                                  output_dict=True)
    
    # Convert to DataFrame
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(OUTPUT_PATH, 'classification_report.csv'))
    
    # Print report
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return report_df


def run_cross_validation(X_train, y_train, best_model, model_name, groups=None, cv=5):
    """
    Run cross-validation on training set.
    
    If groups is provided, uses GroupKFold to ensure segments from the same 
    audio file stay in the same fold (prevents data leakage).
    """
    print(f"\nRunning {cv}-fold cross-validation on {model_name}...")
    
    if groups is not None:
        # Use GroupKFold to prevent data leakage
        print("Using GroupKFold (segments from same song stay together)")
        group_kfold = GroupKFold(n_splits=cv)
        scores = cross_val_score(
            best_model, X_train, y_train, 
            cv=group_kfold, 
            groups=groups,
            scoring='accuracy', 
            n_jobs=-1
        )
    else:
        # Standard KFold (may have data leakage for audio segments)
        print("Using standard KFold (WARNING: may have data leakage)")
        scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"CV Scores: {scores}")
    print(f"CV Mean: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    return scores


def main():
    """Main execution function"""
    print("="*60)
    print("MUSIC GENRE CLASSIFICATION - GTZAN DATASET")
    print("="*60)
    
    # Step 1: Load data
    print("\n[1/7] Loading data...")
    data = load_data()
    
    # Step 2: Split train/test according to requirements
    print("\n[2/7] Splitting data (Train: files 0-79, Test: files 80-99)...")
    print("NOTE: This is the PROPER split without data leakage!")
    print("(Different 3-sec segments of same audio stay in same set)")
    train_data, test_data, train_groups = split_train_test(data)
    
    # Step 3: Prepare features
    print("\n[3/7] Preparing features (using StandardScaler)...")
    X_train, X_test, y_train, y_test, label_encoder = prepare_features(train_data, test_data, scaler_type='standard')
    print(f"Feature dimensions: Train={X_train.shape}, Test={X_test.shape}")
    
    # Step 4: EDA Visualizations
    print("\n[4/7] Generating EDA visualizations...")
    plot_correlation_heatmap(train_data)
    plot_bpm_boxplot(train_data)
    plot_pca_visualization(X_train, y_train, label_encoder)
    
    # Step 5: Train and evaluate models
    print("\n[5/7] Training and evaluating models...")
    results = run_all_models(X_train, X_test, y_train, y_test)
    
    # Step 6: Find best model and run cross-validation
    print("\n[6/7] Analyzing results...")
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    best_predictions = results[best_model_name]['predictions']
    best_model = results[best_model_name]['model']
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    # Cross-validation on training set using GroupKFold
    # Clone model for CV to avoid refitting issues
    from sklearn.base import clone
    cv_model = clone(best_model)
    run_cross_validation(X_train, y_train, cv_model, best_model_name, groups=train_groups)
    
    # Step 7: Generate final reports and plots
    print("\n[7/7] Generating final reports and visualizations...")
    plot_model_comparison(results)
    plot_confusion_matrix(y_test, best_predictions, label_encoder, best_model_name)
    report_df = generate_classification_report(y_test, best_predictions, label_encoder)
    
    # Save results summary
    summary = {
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results.keys()]
    }
    summary_df = pd.DataFrame(summary).sort_values('Accuracy', ascending=False)
    summary_df.to_csv(os.path.join(OUTPUT_PATH, 'model_summary.csv'), index=False)
    print(f"Saved: {os.path.join(OUTPUT_PATH, 'model_summary.csv')}")
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("ALL OUTPUTS SAVED TO:", OUTPUT_PATH)
    print("="*60)
    
    return results, label_encoder


if __name__ == "__main__":
    results, label_encoder = main()

