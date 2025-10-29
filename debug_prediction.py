import joblib
import numpy as np
import pandas as pd
from collections import Counter

def debug_prediction_issue():
    print("🔍 DEBUG: Analyzing prediction consistency issue")
    print("=" * 60)
    
    # Load all artifacts
    try:
        preprocess_bundle = joblib.load('preprocess_artifacts.pkl')
        scaler = preprocess_bundle['scaler']
        feature_cols = preprocess_bundle['feature_cols']
        target_encoder = preprocess_bundle['target_encoder']
        
        print(f"✅ Preprocessing artifacts loaded")
        print(f"Feature columns: {feature_cols}")
        print(f"Feature order: {list(range(len(feature_cols)))}")
        
    except Exception as e:
        print(f"❌ Error loading preprocessing: {e}")
        return
    
    # Load models
    models = {}
    model_files = [
        ('Decision Tree', 'decision_tree_final.joblib'),
        ('Gaussian Naive Bayes', 'gaussian_naive_bayes_final.joblib'),
        ('SVM (RBF)', 'svm_rbf_final.joblib')
    ]
    
    for name, filename in model_files:
        try:
            model = joblib.load(filename)
            models[name] = model
            print(f"✅ {name} loaded: {type(model)}")
        except:
            print(f"❌ Failed to load {name}")
    
    print("\n" + "=" * 60)
    
    # Check scaler parameters
    print("📊 SCALER ANALYSIS:")
    print(f"Scaler mean: {scaler.mean_}")
    print(f"Scaler scale: {scaler.scale_}")
    
    # Check if scaler values are reasonable
    for i, (col, mean, scale) in enumerate(zip(feature_cols, scaler.mean_, scaler.scale_)):
        print(f"  {i}: {col} -> mean={mean:.2f}, scale={scale:.2f}")
    
    print("\n" + "=" * 60)
    
    # Test multiple diverse inputs
    print("🧪 TESTING DIVERSE INPUTS:")
    
    # Test cases that should produce different varieties
    test_cases = [
        {
            'name': 'Default (should be balanced)',
            'values': [116, 9, 8, 5, 9, 9, 0, 5.6],  # UmurTanaman, Kerebahan, TeksturNasi, PotensiHasil, KetahananTerhadapHama, Kerontokan, WarnaGabah, PHTanah
            'expected': 'Mixed predictions'
        },
        {
            'name': 'Low resistance (favor IR-64 or others)',
            'values': [110, 4, 5, 3, 4, 4, 1, 5.0],
            'expected': 'Not Ciherang'
        },
        {
            'name': 'High performance (favor Inpari varieties)',
            'values': [125, 9, 8, 9, 9, 9, 2, 6.5],
            'expected': 'Inpari varieties'
        },
        {
            'name': 'Early maturity (favor early varieties)',
            'values': [95, 7, 7, 6, 7, 7, 0, 6.0],
            'expected': 'Early varieties'
        },
        {
            'name': 'Late maturity (favor late varieties)',
            'values': [140, 8, 7, 7, 8, 8, 1, 6.2],
            'expected': 'Late varieties'
        }
    ]
    
    # Varietas mapping
    idx2name = {
        0: 'IR-64', 1: 'Ciherang', 2: 'Inpari 30', 3: 'Inpari 32', 
        4: 'Inpari 42', 5: 'Inpari 46', 6: 'Mekongga', 7: 'Sembada B9', 8: 'Situ Bagendit'
    }
    
    all_predictions = []
    
    for test_case in test_cases:
        print(f"\n🔬 Test: {test_case['name']}")
        print(f"Input: {test_case['values']}")
        print(f"Expected: {test_case['expected']}")
        
        # Create input array
        input_array = np.array(test_case['values']).reshape(1, -1)
        
        # Scale input
        try:
            input_scaled = scaler.transform(input_array)
            print(f"Scaled: {input_scaled[0]}")
        except Exception as e:
            print(f"❌ Scaling error: {e}")
            continue
        
        # Test each model
        case_predictions = []
        for model_name, model in models.items():
            try:
                pred = model.predict(input_scaled)
                pred_idx = int(pred[0])
                pred_name = idx2name.get(pred_idx, f"Unknown({pred_idx})")
                case_predictions.append(pred_name)
                all_predictions.append(pred_name)
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_scaled)[0]
                    top_3_idx = np.argsort(proba)[::-1][:3]
                    top_3 = [(idx2name.get(idx, f"Unknown({idx})"), proba[idx]*100) for idx in top_3_idx]
                    prob_str = " | ".join([f"{name}: {prob:.1f}%" for name, prob in top_3])
                    print(f"  {model_name}: {pred_name} ({prob_str})")
                else:
                    print(f"  {model_name}: {pred_name}")
                    
            except Exception as e:
                print(f"  {model_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    
    # Analyze prediction distribution
    print("📈 PREDICTION DISTRIBUTION ANALYSIS:")
    prediction_counts = Counter(all_predictions)
    total_predictions = len(all_predictions)
    
    print(f"Total predictions made: {total_predictions}")
    for variety, count in prediction_counts.most_common():
        percentage = (count / total_predictions) * 100
        print(f"  {variety}: {count} predictions ({percentage:.1f}%)")
    
    # Check if dominated by one variety
    if prediction_counts.most_common(1)[0][1] > total_predictions * 0.8:
        most_common_variety = prediction_counts.most_common(1)[0][0]
        print(f"\n⚠️  WARNING: {most_common_variety} dominates {prediction_counts.most_common(1)[0][1]}/{total_predictions} predictions!")
        print("This indicates a severe class imbalance or model bias issue.")
        
        # Suggest solutions
        print("\n💡 SUGGESTED SOLUTIONS:")
        print("1. Check training data distribution - likely heavily skewed toward Ciherang")
        print("2. Apply data balancing techniques (SMOTE, oversampling)")
        print("3. Retrain models with balanced dataset")
        print("4. Check feature scaling - extreme values might indicate data issues")
        print("5. Verify feature encoding consistency between training and prediction")
    
    print("\n" + "=" * 60)
    
    # Check feature importance
    print("🎯 FEATURE IMPORTANCE ANALYSIS:")
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        
        # Handle pipeline models
        if hasattr(model, 'steps'):
            classifier = model.steps[-1][1]
        else:
            classifier = model
        
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            for i, (col, imp) in enumerate(zip(feature_cols, importances)):
                print(f"  {col}: {imp:.4f}")
                if imp == 0:
                    print(f"    ⚠️ Zero importance - this feature is ignored!")
        else:
            print("  No feature importance available")

if __name__ == "__main__":
    debug_prediction_issue()