import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def create_simple_balanced_artifacts():
    """Create simple preprocessing artifacts that are compatible"""
    print("🔧 Creating improved preprocessing artifacts...")
    
    # Feature columns order (same as original)
    feature_cols = ['UmurTanaman', 'Kerebahan', 'TeksturNasi', 'PotensiHasil', 
                    'KetahananTerhadapHama', 'Kerontokan', 'WarnaGabah', 'PHTanah']
    
    # Create balanced synthetic data for fitting scaler
    np.random.seed(42)
    
    # Generate more realistic training data based on variety characteristics
    data = []
    
    # Each variety gets equal representation
    variety_ranges = {
        # [UmurTanaman, Kerebahan, TeksturNasi, PotensiHasil, KetahananTerhadapHama, Kerontokan, WarnaGabah, PHTanah]
        0: [(110, 120), (4, 7), (5, 7), (3, 6), (4, 7), (4, 7), (0, 1), (5.0, 6.5)],  # IR-64
        1: [(115, 125), (7, 9), (7, 8), (5, 7), (7, 9), (7, 9), (0, 1), (5.5, 6.5)],  # Ciherang
        2: [(95, 105), (8, 9), (7, 8), (6, 8), (8, 9), (8, 9), (1, 2), (5.5, 7.0)],   # Inpari 30
        3: [(100, 110), (7, 9), (6, 8), (6, 8), (7, 9), (7, 9), (0, 2), (5.0, 6.5)],  # Inpari 32
        4: [(120, 130), (8, 9), (7, 9), (7, 9), (8, 9), (8, 9), (1, 2), (6.0, 7.5)],  # Inpari 42
        5: [(125, 135), (7, 9), (6, 8), (6, 9), (7, 9), (7, 9), (0, 2), (5.5, 7.0)],  # Inpari 46
        6: [(130, 140), (6, 8), (5, 7), (5, 7), (6, 8), (6, 8), (0, 1), (5.5, 6.5)],  # Mekongga
        7: [(105, 115), (5, 7), (5, 7), (4, 6), (5, 7), (5, 7), (0, 1), (5.0, 6.0)],  # Sembada B9
        8: [(100, 110), (4, 6), (4, 6), (3, 5), (4, 6), (4, 6), (0, 2), (4.5, 6.0)]   # Situ Bagendit
    }
    
    # Generate 100 samples per variety
    for variety_idx in range(9):
        ranges = variety_ranges[variety_idx]
        for _ in range(100):
            sample = []
            for i, (min_val, max_val) in enumerate(ranges):
                if i in [0, 7]:  # UmurTanaman and PHTanah - continuous
                    value = np.random.uniform(min_val, max_val)
                else:  # Discrete values
                    value = np.random.randint(min_val, max_val + 1)
                sample.append(value)
            data.append(sample)
    
    X_balanced = np.array(data)
    
    # Fit new scaler with balanced data
    scaler = StandardScaler()
    scaler.fit(X_balanced)
    
    print(f"✅ Scaler fitted with {len(X_balanced)} balanced samples")
    print("New scaler parameters:")
    for i, col in enumerate(feature_cols):
        print(f"  {col}: mean={scaler.mean_[i]:.2f}, scale={scaler.scale_[i]:.2f}")
    
    # Create simple preprocessing bundle (without complex objects)
    preprocess_bundle = {
        "scaler": scaler,
        "target_encoder": None,  # Simplified
        "feature_cols": feature_cols
    }
    
    # Save the new artifacts
    joblib.dump(preprocess_bundle, "preprocess_artifacts_balanced.pkl")
    print("✅ Balanced preprocessing artifacts saved")
    
    return scaler, feature_cols

def test_balanced_models():
    """Test the balanced models with the new preprocessing"""
    print("\\n🧪 Testing balanced models...")
    
    # Load balanced models
    model_files = {
        'Decision Tree': 'decision_tree_balanced.joblib',
        'Gaussian Naive Bayes': 'gaussian_naive_bayes_balanced.joblib',
        'SVM (RBF)': 'svm_rbf_balanced.joblib'
    }
    
    models = {}
    for name, filename in model_files.items():
        try:
            model = joblib.load(filename)
            models[name] = model
            print(f"✅ {name} loaded")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    if not models:
        print("❌ No models loaded, stopping test")
        return
    
    # Load balanced preprocessing
    try:
        preprocess_bundle = joblib.load("preprocess_artifacts_balanced.pkl")
        scaler = preprocess_bundle["scaler"]
        feature_cols = preprocess_bundle["feature_cols"]
        print(f"✅ Balanced preprocessing loaded")
    except Exception as e:
        print(f"❌ Failed to load preprocessing: {e}")
        return
    
    # Variety mapping
    idx_to_variety = {
        0: 'IR-64', 1: 'Ciherang', 2: 'Inpari 30', 3: 'Inpari 32', 4: 'Inpari 42',
        5: 'Inpari 46', 6: 'Mekongga', 7: 'Sembada B9', 8: 'Situ Bagendit'
    }
    
    # Test cases that should favor different varieties
    test_cases = [
        {
            'name': 'Ciherang characteristics',
            'values': [120, 8, 8, 6, 8, 8, 0, 6.0],
            'expected': 'Ciherang'
        },
        {
            'name': 'IR-64 characteristics',
            'values': [115, 5, 6, 4, 5, 5, 1, 5.5],
            'expected': 'IR-64'
        },
        {
            'name': 'Inpari 30 characteristics (early, high performance)',
            'values': [100, 9, 8, 7, 9, 9, 2, 6.5],
            'expected': 'Inpari 30'
        },
        {
            'name': 'Mekongga characteristics (late maturity)',
            'values': [135, 7, 6, 6, 7, 7, 1, 6.0],
            'expected': 'Mekongga'
        },
        {
            'name': 'Situ Bagendit characteristics (low performance)',
            'values': [105, 5, 5, 4, 5, 5, 1, 5.2],
            'expected': 'Situ Bagendit'
        },
        {
            'name': 'Inpari 42 characteristics (high performance, late)',
            'values': [125, 9, 8, 8, 9, 9, 2, 7.0],
            'expected': 'Inpari 42'
        }
    ]
    
    all_predictions = []
    correct_predictions = 0
    
    for test_case in test_cases:
        print(f"\\n🔬 Test: {test_case['name']}")
        print(f"Expected: {test_case['expected']}")
        print(f"Input: {test_case['values']}")
        
        # Prepare input (following the exact order from feature_cols)
        input_array = np.array(test_case['values']).reshape(1, -1)
        
        try:
            input_scaled = scaler.transform(input_array)
            print(f"Scaled: {input_scaled[0][:4]}... (showing first 4)")
        except Exception as e:
            print(f"❌ Scaling error: {e}")
            continue
        
        # Test with each model
        test_predictions = []
        for model_name, model in models.items():
            try:
                pred = model.predict(input_scaled)[0]
                pred_variety = idx_to_variety[pred]
                test_predictions.append(pred_variety)
                all_predictions.append(pred_variety)
                
                # Check if correct
                is_correct = pred_variety == test_case['expected']
                if is_correct:
                    correct_predictions += 1
                
                # Get confidence if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_scaled)[0]
                    confidence = proba[pred] * 100
                    status = "✅" if is_correct else "❌"
                    print(f"   {model_name}: {pred_variety} ({confidence:.1f}%) {status}")
                else:
                    status = "✅" if is_correct else "❌"
                    print(f"   {model_name}: {pred_variety} {status}")
                    
            except Exception as e:
                print(f"   {model_name}: ERROR - {e}")
        
        # Consensus for this test
        from collections import Counter
        consensus = Counter(test_predictions)
        if consensus:
            winner, votes = consensus.most_common(1)[0]
            print(f"   Consensus: {winner} ({votes}/{len(test_predictions)} votes)")
    
    # Overall results
    total_tests = len(test_cases) * len(models)
    accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"\\n📊 OVERALL RESULTS:")
    print(f"Accuracy: {correct_predictions}/{total_tests} ({accuracy:.1f}%)")
    
    # Check prediction diversity
    from collections import Counter
    pred_distribution = Counter(all_predictions)
    print(f"\\nPrediction diversity:")
    for variety, count in pred_distribution.most_common():
        percentage = (count / len(all_predictions)) * 100
        print(f"   {variety}: {count} ({percentage:.1f}%)")
    
    # Success criteria
    max_percentage = (pred_distribution.most_common(1)[0][1] / len(all_predictions)) * 100
    if max_percentage < 50 and len(pred_distribution) >= 5:
        print(f"\\n🎉 SUCCESS: Good prediction diversity! Multiple varieties predicted.")
    elif max_percentage < 70:
        print(f"\\n✅ IMPROVED: Better diversity than before.")
    else:
        print(f"\\n⚠️ Still needs improvement: {pred_distribution.most_common(1)[0][0]} dominates predictions.")

if __name__ == "__main__":
    print("🔧 CREATING BALANCED PREDICTION SYSTEM")
    print("=" * 60)
    
    # Step 1: Create balanced preprocessing
    scaler, feature_cols = create_simple_balanced_artifacts()
    
    # Step 2: Test the balanced models
    test_balanced_models()
    
    print("\\n" + "=" * 60)
    print("🎯 INSTRUCTIONS TO USE BALANCED MODELS:")
    print("\\n1. Replace files in your Streamlit app:")
    print("   - Copy decision_tree_balanced.joblib → decision_tree_final.joblib")
    print("   - Copy gaussian_naive_bayes_balanced.joblib → gaussian_naive_bayes_final.joblib") 
    print("   - Copy svm_rbf_balanced.joblib → svm_rbf_final.joblib")
    print("   - Copy preprocess_artifacts_balanced.pkl → preprocess_artifacts.pkl")
    print("\\n2. Restart your Streamlit application")
    print("\\n3. Test with different variety characteristics to see improved diversity!")