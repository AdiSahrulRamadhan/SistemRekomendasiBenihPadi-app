import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

def create_balanced_training_data():
    """Create synthetic balanced training data since we don't have access to original data"""
    print("🔄 Creating balanced synthetic training data...")
    
    # Define variety characteristics based on domain knowledge
    variety_profiles = {
        'IR-64': {
            'UmurTanaman': (110, 120), 'Kerebahan': (4, 7), 'TeksturNasi': (5, 7),
            'PotensiHasil': (3, 6), 'KetahananTerhadapHama': (4, 7), 
            'Kerontokan': (4, 7), 'WarnaGabah': (0, 1), 'PHTanah': (5.0, 6.5)
        },
        'Ciherang': {
            'UmurTanaman': (115, 125), 'Kerebahan': (7, 9), 'TeksturNasi': (7, 8),
            'PotensiHasil': (5, 7), 'KetahananTerhadapHama': (7, 9), 
            'Kerontokan': (7, 9), 'WarnaGabah': (0, 1), 'PHTanah': (5.5, 6.5)
        },
        'Inpari 30': {
            'UmurTanaman': (95, 105), 'Kerebahan': (8, 9), 'TeksturNasi': (7, 8),
            'PotensiHasil': (6, 8), 'KetahananTerhadapHama': (8, 9), 
            'Kerontokan': (8, 9), 'WarnaGabah': (1, 2), 'PHTanah': (5.5, 7.0)
        },
        'Inpari 32': {
            'UmurTanaman': (100, 110), 'Kerebahan': (7, 9), 'TeksturNasi': (6, 8),
            'PotensiHasil': (6, 8), 'KetahananTerhadapHama': (7, 9), 
            'Kerontokan': (7, 9), 'WarnaGabah': (0, 2), 'PHTanah': (5.0, 6.5)
        },
        'Inpari 42': {
            'UmurTanaman': (120, 130), 'Kerebahan': (8, 9), 'TeksturNasi': (7, 9),
            'PotensiHasil': (7, 9), 'KetahananTerhadapHama': (8, 9), 
            'Kerontokan': (8, 9), 'WarnaGabah': (1, 2), 'PHTanah': (6.0, 7.5)
        },
        'Inpari 46': {
            'UmurTanaman': (125, 135), 'Kerebahan': (7, 9), 'TeksturNasi': (6, 8),
            'PotensiHasil': (6, 9), 'KetahananTerhadapHama': (7, 9), 
            'Kerontokan': (7, 9), 'WarnaGabah': (0, 2), 'PHTanah': (5.5, 7.0)
        },
        'Mekongga': {
            'UmurTanaman': (130, 140), 'Kerebahan': (6, 8), 'TeksturNasi': (5, 7),
            'PotensiHasil': (5, 7), 'KetahananTerhadapHama': (6, 8), 
            'Kerontokan': (6, 8), 'WarnaGabah': (0, 1), 'PHTanah': (5.5, 6.5)
        },
        'Sembada B9': {
            'UmurTanaman': (105, 115), 'Kerebahan': (5, 7), 'TeksturNasi': (5, 7),
            'PotensiHasil': (4, 6), 'KetahananTerhadapHama': (5, 7), 
            'Kerontokan': (5, 7), 'WarnaGabah': (0, 1), 'PHTanah': (5.0, 6.0)
        },
        'Situ Bagendit': {
            'UmurTanaman': (100, 110), 'Kerebahan': (4, 6), 'TeksturNasi': (4, 6),
            'PotensiHasil': (3, 5), 'KetahananTerhadapHama': (4, 6), 
            'Kerontokan': (4, 6), 'WarnaGabah': (0, 2), 'PHTanah': (4.5, 6.0)
        }
    }
    
    # Mapping varieties to indices
    variety_to_idx = {
        'IR-64': 0, 'Ciherang': 1, 'Inpari 30': 2, 'Inpari 32': 3, 'Inpari 42': 4,
        'Inpari 46': 5, 'Mekongga': 6, 'Sembada B9': 7, 'Situ Bagendit': 8
    }
    
    # Feature columns
    feature_cols = ['UmurTanaman', 'Kerebahan', 'TeksturNasi', 'PotensiHasil', 
                    'KetahananTerhadapHama', 'Kerontokan', 'WarnaGabah', 'PHTanah']
    
    # Generate balanced data (same number of samples per variety)
    samples_per_variety = 100
    np.random.seed(42)
    
    data = []
    labels = []
    
    for variety, characteristics in variety_profiles.items():
        variety_idx = variety_to_idx[variety]
        
        for _ in range(samples_per_variety):
            sample = {}
            for feature in feature_cols:
                min_val, max_val = characteristics[feature]
                if feature in ['UmurTanaman', 'PHTanah']:
                    # Continuous features
                    value = np.random.uniform(min_val, max_val)
                else:
                    # Discrete features
                    value = np.random.randint(min_val, max_val + 1)
                sample[feature] = value
            
            data.append([sample[col] for col in feature_cols])
            labels.append(variety_idx)
    
    X = np.array(data)
    y = np.array(labels)
    
    print(f"✅ Generated {len(X)} balanced samples")
    print(f"   Samples per variety: {samples_per_variety}")
    print(f"   Total varieties: {len(variety_profiles)}")
    
    # Check distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"   Class distribution: {dict(zip(unique, counts))}")
    
    return X, y, feature_cols

def train_balanced_models():
    """Train new models with balanced data"""
    print("🎯 Training new balanced models...")
    
    # Generate balanced data
    X, y, feature_cols = create_balanced_training_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Define models
    models = {
        'Decision Tree': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', DecisionTreeClassifier(
                random_state=42, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced'
            ))
        ]),
        'Gaussian Naive Bayes': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95, random_state=42)),
            ('clf', GaussianNB())
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf', 
                C=1.0, 
                gamma='scale', 
                probability=True, 
                random_state=42,
                class_weight='balanced'
            ))
        ])
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n📚 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        # Test predictions
        y_pred = model.predict(X_test)
        test_accuracy = (y_pred == y_test).mean()
        
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy
        }
        
        trained_models[name] = model
        
        print(f"   CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"   Test Accuracy: {test_accuracy:.3f}")
        
        # Show classification report
        idx_to_variety = {
            0: 'IR-64', 1: 'Ciherang', 2: 'Inpari 30', 3: 'Inpari 32', 4: 'Inpari 42',
            5: 'Inpari 46', 6: 'Mekongga', 7: 'Sembada B9', 8: 'Situ Bagendit'
        }
        
        variety_names = [idx_to_variety[i] for i in range(9)]
        print(f"\\n   Classification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=variety_names, zero_division=0))
    
    # Save best models
    print("\\n💾 Saving improved models...")
    
    # Save individual models
    for name, model in trained_models.items():
        filename_map = {
            'Decision Tree': 'decision_tree_balanced.joblib',
            'Gaussian Naive Bayes': 'gaussian_naive_bayes_balanced.joblib',
            'SVM (RBF)': 'svm_rbf_balanced.joblib'
        }
        filename = filename_map[name]
        joblib.dump(model, filename)
        print(f"   ✅ {name} saved as {filename}")
    
    # Create and save new preprocessing artifacts
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Create dummy target encoder (not used in this version)
    class DummyEncoder:
        def __init__(self):
            pass
    
    preprocess_bundle = {
        "scaler": scaler,
        "target_encoder": DummyEncoder(),
        "feature_cols": feature_cols
    }
    
    joblib.dump(preprocess_bundle, "preprocess_artifacts_balanced.pkl")
    print(f"   ✅ Preprocessing artifacts saved as preprocess_artifacts_balanced.pkl")
    
    return results, trained_models

def test_improved_models():
    """Test the improved models with diverse inputs"""
    print("\\n🧪 Testing improved models...")
    
    # Load improved models
    models = {}
    model_files = {
        'Decision Tree': 'decision_tree_balanced.joblib',
        'Gaussian Naive Bayes': 'gaussian_naive_bayes_balanced.joblib', 
        'SVM (RBF)': 'svm_rbf_balanced.joblib'
    }
    
    for name, filename in model_files.items():
        try:
            model = joblib.load(filename)
            models[name] = model
            print(f"✅ {name} loaded")
        except:
            print(f"❌ Failed to load {name}")
            return
    
    # Load preprocessing
    try:
        preprocess_bundle = joblib.load("preprocess_artifacts_balanced.pkl")
        scaler = preprocess_bundle["scaler"]
        feature_cols = preprocess_bundle["feature_cols"]
        print(f"✅ Balanced preprocessing loaded")
    except:
        print(f"❌ Failed to load balanced preprocessing")
        return
    
    # Test cases
    test_cases = [
        {
            'name': 'Ciherang-like characteristics',
            'values': [118, 8, 8, 6, 8, 8, 0, 6.0],
            'expected': 'Ciherang'
        },
        {
            'name': 'IR-64-like characteristics', 
            'values': [115, 5, 6, 4, 5, 5, 1, 5.5],
            'expected': 'IR-64'
        },
        {
            'name': 'Inpari 30-like (early, high yield)',
            'values': [100, 9, 8, 7, 9, 9, 2, 6.5],
            'expected': 'Inpari 30'
        },
        {
            'name': 'Mekongga-like (late maturity)',
            'values': [135, 7, 6, 6, 7, 7, 1, 6.0],
            'expected': 'Mekongga'
        },
        {
            'name': 'Situ Bagendit-like (low performance)',
            'values': [105, 5, 5, 4, 5, 5, 1, 5.0],
            'expected': 'Situ Bagendit'
        }
    ]
    
    # Variety mapping
    idx_to_variety = {
        0: 'IR-64', 1: 'Ciherang', 2: 'Inpari 30', 3: 'Inpari 32', 4: 'Inpari 42',
        5: 'Inpari 46', 6: 'Mekongga', 7: 'Sembada B9', 8: 'Situ Bagendit'
    }
    
    all_predictions = []
    
    for test_case in test_cases:
        print(f"\\n🔬 Test: {test_case['name']}")
        print(f"Expected: {test_case['expected']}")
        
        # Prepare input
        input_array = np.array(test_case['values']).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        
        # Test each model
        for model_name, model in models.items():
            pred = model.predict(input_scaled)[0]
            pred_variety = idx_to_variety[pred]
            all_predictions.append(pred_variety)
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[0]
                confidence = proba[pred] * 100
                print(f"   {model_name}: {pred_variety} ({confidence:.1f}%)")
            else:
                print(f"   {model_name}: {pred_variety}")
    
    # Check prediction diversity
    prediction_counts = Counter(all_predictions)
    print(f"\\n📊 Prediction distribution:")
    total = len(all_predictions)
    for variety, count in prediction_counts.most_common():
        percentage = (count / total) * 100
        print(f"   {variety}: {count}/{total} ({percentage:.1f}%)")
    
    # Check if diversity improved
    max_percentage = (prediction_counts.most_common(1)[0][1] / total) * 100
    if max_percentage < 60:
        print(f"\\n✅ SUCCESS: Prediction diversity improved! No single variety dominates.")
    else:
        print(f"\\n⚠️ Still some bias toward {prediction_counts.most_common(1)[0][0]}")

if __name__ == "__main__":
    print("🔧 FIXING RICE VARIETY PREDICTION MODELS")
    print("=" * 60)
    
    # Step 1: Train balanced models
    results, models = train_balanced_models()
    
    # Step 2: Test improved models
    test_improved_models()
    
    print("\\n" + "=" * 60)
    print("🎉 Model improvement process completed!")
    print("\\nTo use the improved models in your Streamlit app:")
    print("1. Replace the old .joblib files with the new _balanced.joblib files")
    print("2. Update preprocess_artifacts.pkl with preprocess_artifacts_balanced.pkl")
    print("3. Restart your Streamlit application")