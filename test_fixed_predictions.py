import requests
import numpy as np
from collections import Counter

def test_fixed_predictions():
    """Test the fixed Streamlit app predictions"""
    print("🧪 TESTING FIXED STREAMLIT PREDICTIONS")
    print("=" * 60)
    
    # Manual mappings (same as app.py)
    def _norm(s):
        return str(s).strip().lower().replace(" ", "").replace("_", "")

    map_kerebahan = {_norm(k): v for k, v in {
        "Tahan": 9, "Sedang": 7, "Tidak Tahan": 4
    }.items()}

    map_teksturnasi = {_norm(k): v for k, v in {
        "Pulen": 8, "Agak Pulen": 7, "Agak Pera": 6, "Pera": 5
    }.items()}

    map_ketahanan = {_norm(k): v for k, v in {
        "Tahan": 9, "Agak Tahan": 7, "Sedang": 7, "Agak Rentan": 6, "Rentan": 4, "Tidak Tahan": 4
    }.items()}

    map_kerontokan = {_norm(k): v for k, v in {
        "Tahan": 9, "Toleran": 9, "Kuat": 9, "Sedang": 7, "Agak Tahan": 7, "Mudah": 4, "Rentan": 4
    }.items()}

    map_warnagabah = {_norm(k): v for k, v in {
        "Kuning Bersih": 0, "Kuning Jerami": 1, "Kuning": 2
    }.items()}

    MANUAL_MAP = {
        "Kerebahan": map_kerebahan,
        "TeksturNasi": map_teksturnasi,
        "KetahananTerhadapHama": map_ketahanan,
        "Kerontokan": map_kerontokan,
        "WarnaGabah": map_warnagabah,
    }
    
    # Load current models and preprocessing directly
    import joblib
    
    try:
        preprocess_bundle = joblib.load("preprocess_artifacts.pkl")
        scaler = preprocess_bundle["scaler"]
        feature_cols = preprocess_bundle["feature_cols"]
        print("✅ Preprocessing loaded")
    except Exception as e:
        print(f"❌ Error loading preprocessing: {e}")
        return
    
    # Load models
    models = {}
    model_files = {
        'Decision Tree': 'decision_tree_final.joblib',
        'Gaussian Naive Bayes': 'gaussian_naive_bayes_final.joblib',
        'SVM (RBF)': 'svm_rbf_final.joblib'
    }
    
    for name, filename in model_files.items():
        try:
            model = joblib.load(filename)
            models[name] = model
            print(f"✅ {name} loaded")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    # Fixed preprocessing function (same as the updated app.py)
    def preprocess_input_fixed(input_dict, feature_cols):
        processed = {}
        
        # Apply manual mappings
        for col, mapper in MANUAL_MAP.items():
            if col in input_dict:
                val = input_dict[col]
                mapped_val = mapper.get(_norm(str(val)), 0)
                processed[col] = float(mapped_val)
        
        # Handle numeric columns
        numeric_cols = ["UmurTanaman", "PHTanah", "PotensiHasil"]
        for col in numeric_cols:
            if col in input_dict:
                processed[col] = float(input_dict[col])
        
        # Create input array in CORRECT ORDER
        input_row = []
        for col in feature_cols:
            value = processed.get(col, 0.0)
            input_row.append(value)
        
        return np.array(input_row).reshape(1, -1)
    
    # Test cases with diverse characteristics
    test_cases = [
        {
            'name': 'Ciherang-like (high resistance, good texture)',
            'input': {
                "Kerebahan": "Tahan",
                "TeksturNasi": "Pulen", 
                "KetahananTerhadapHama": "Tahan",
                "Kerontokan": "Tahan",
                "WarnaGabah": "Kuning Bersih",
                "UmurTanaman": 120,
                "PHTanah": 6.0,
                "PotensiHasil": 6
            },
            'expected_variety': 'Ciherang'
        },
        {
            'name': 'IR-64-like (medium resistance, early)',
            'input': {
                "Kerebahan": "Sedang",
                "TeksturNasi": "Agak Pera", 
                "KetahananTerhadapHama": "Sedang",
                "Kerontokan": "Sedang",
                "WarnaGabah": "Kuning Jerami",
                "UmurTanaman": 115,
                "PHTanah": 5.5,
                "PotensiHasil": 4
            },
            'expected_variety': 'IR-64'
        },
        {
            'name': 'Inpari 30-like (early, high yield)',
            'input': {
                "Kerebahan": "Tahan",
                "TeksturNasi": "Pulen", 
                "KetahananTerhadapHama": "Tahan",
                "Kerontokan": "Tahan",
                "WarnaGabah": "Kuning",
                "UmurTanaman": 100,
                "PHTanah": 6.5,
                "PotensiHasil": 7
            },
            'expected_variety': 'Inpari 30'
        },
        {
            'name': 'Mekongga-like (late maturity)',
            'input': {
                "Kerebahan": "Sedang",
                "TeksturNasi": "Agak Pera", 
                "KetahananTerhadapHama": "Sedang",
                "Kerontokan": "Sedang",
                "WarnaGabah": "Kuning Bersih",
                "UmurTanaman": 135,
                "PHTanah": 6.0,
                "PotensiHasil": 6
            },
            'expected_variety': 'Mekongga'
        },
        {
            'name': 'Situ Bagendit-like (low performance)',
            'input': {
                "Kerebahan": "Tidak Tahan",
                "TeksturNasi": "Pera", 
                "KetahananTerhadapHama": "Rentan",
                "Kerontokan": "Mudah",
                "WarnaGabah": "Kuning Jerami",
                "UmurTanaman": 105,
                "PHTanah": 5.0,
                "PotensiHasil": 3
            },
            'expected_variety': 'Situ Bagendit'
        },
        {
            'name': 'Inpari 42-like (high performance, late)',
            'input': {
                "Kerebahan": "Tahan",
                "TeksturNasi": "Pulen", 
                "KetahananTerhadapHama": "Tahan",
                "Kerontokan": "Tahan",
                "WarnaGabah": "Kuning",
                "UmurTanaman": 125,
                "PHTanah": 7.0,
                "PotensiHasil": 8
            },
            'expected_variety': 'Inpari 42'
        }
    ]
    
    idx_to_variety = {
        0: 'IR-64', 1: 'Ciherang', 2: 'Inpari 30', 3: 'Inpari 32', 4: 'Inpari 42',
        5: 'Inpari 46', 6: 'Mekongga', 7: 'Sembada B9', 8: 'Situ Bagendit'
    }
    
    all_predictions = []
    correct_predictions = 0
    total_tests = 0
    
    print("\\n🔬 TESTING DIVERSE INPUT CHARACTERISTICS:")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"\\n📝 Test: {test_case['name']}")
        print(f"Expected: {test_case['expected_variety']}")
        
        # Process input
        try:
            X_input = preprocess_input_fixed(test_case['input'], feature_cols)
            X_scaled = scaler.transform(X_input)
            
            print(f"Processed input: {X_input[0]}")
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            continue
        
        # Test each model
        case_predictions = []
        for model_name, model in models.items():
            try:
                pred = model.predict(X_scaled)[0]
                pred_variety = idx_to_variety[pred]
                case_predictions.append(pred_variety)
                all_predictions.append(pred_variety)
                total_tests += 1
                
                # Check if prediction matches expectation
                is_correct = pred_variety == test_case['expected_variety']
                if is_correct:
                    correct_predictions += 1
                
                # Get confidence
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[0]
                    confidence = proba[pred] * 100
                    status = "✅" if is_correct else "❌"
                    print(f"   {model_name}: {pred_variety} ({confidence:.1f}%) {status}")
                else:
                    status = "✅" if is_correct else "❌"
                    print(f"   {model_name}: {pred_variety} {status}")
                    
            except Exception as e:
                print(f"   {model_name}: ERROR - {e}")
        
        # Consensus for this test case
        if case_predictions:
            consensus = Counter(case_predictions)
            winner, votes = consensus.most_common(1)[0]
            is_consensus_correct = winner == test_case['expected_variety']
            status = "✅" if is_consensus_correct else "❌"
            print(f"   🗳️ Consensus: {winner} ({votes}/{len(case_predictions)} votes) {status}")
    
    # Overall analysis
    print("\\n" + "=" * 60)
    print("📊 OVERALL PREDICTION ANALYSIS:")
    
    # Accuracy
    accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
    print(f"\\n🎯 Accuracy: {correct_predictions}/{total_tests} ({accuracy:.1f}%)")
    
    # Diversity analysis
    pred_counts = Counter(all_predictions)
    print(f"\\n🌈 Prediction Diversity:")
    total_predictions = len(all_predictions)
    unique_varieties = len(pred_counts)
    
    for variety, count in pred_counts.most_common():
        percentage = (count / total_predictions) * 100
        print(f"   {variety}: {count}/{total_predictions} ({percentage:.1f}%)")
    
    print(f"\\n📈 Diversity Metrics:")
    print(f"   Unique varieties predicted: {unique_varieties}/9")
    
    # Check if the fix worked
    max_percentage = (pred_counts.most_common(1)[0][1] / total_predictions) * 100
    most_common_variety = pred_counts.most_common(1)[0][0]
    
    print(f"   Most common prediction: {most_common_variety} ({max_percentage:.1f}%)")
    
    if max_percentage < 50 and unique_varieties >= 4:
        print(f"\\n🎉 SUCCESS: Excellent diversity! No single variety dominates.")
        success_level = "EXCELLENT"
    elif max_percentage < 70 and unique_varieties >= 3:
        print(f"\\n✅ GOOD: Good diversity improvement.")
        success_level = "GOOD"
    elif unique_varieties >= 2:
        print(f"\\n🔄 IMPROVED: Better than before, but can be improved further.")
        success_level = "IMPROVED"
    else:
        print(f"\\n❌ STILL PROBLEMATIC: Limited diversity.")
        success_level = "PROBLEMATIC"
    
    print(f"\\n🏆 FINAL RESULT: {success_level}")
    
    if success_level in ["EXCELLENT", "GOOD"]:
        print("\\n🎯 The input order fix successfully resolved the prediction diversity issue!")
        print("Your Streamlit app should now predict different rice varieties based on input characteristics.")
    
    return success_level, accuracy, unique_varieties, max_percentage

if __name__ == "__main__":
    test_fixed_predictions()