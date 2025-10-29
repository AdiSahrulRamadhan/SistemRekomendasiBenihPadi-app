import joblib
import numpy as np
import pandas as pd
from collections import Counter

def analyze_current_models():
    """Analyze what's wrong with current models"""
    print("🔍 DEEP ANALYSIS OF CURRENT MODELS")
    print("=" * 60)
    
    # Load current artifacts
    try:
        preprocess_bundle = joblib.load("preprocess_artifacts.pkl")
        scaler = preprocess_bundle["scaler"]
        feature_cols = preprocess_bundle["feature_cols"]
        print("✅ Current preprocessing loaded")
    except Exception as e:
        print(f"❌ Error loading preprocessing: {e}")
        return
    
    # Load current models  
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
    
    # Check what the scaler is doing
    print("\\n📊 CURRENT SCALER ANALYSIS:")
    print(f"Feature order: {feature_cols}")
    print(f"Scaler means: {scaler.mean_}")
    print(f"Scaler scales: {scaler.scale_}")
    
    # The problem: let's test with original Colab-style input order
    print("\\n🧪 TESTING INPUT ORDER ISSUE:")
    
    # Test the exact input from your default
    test_input_original = [9, 8, 9, 9, 0, 116, 5.6, 5]  # Original order from Streamlit
    test_input_correct = [116, 9, 8, 5, 9, 9, 0, 5.6]   # Correct order: UmurTanaman first
    
    print(f"Original input (wrong order): {test_input_original}")
    print(f"Correct input (right order):  {test_input_correct}")
    
    # Test both
    for label, test_input in [("WRONG ORDER", test_input_original), ("CORRECT ORDER", test_input_correct)]:
        print(f"\\n--- {label} ---")
        
        input_array = np.array(test_input).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        print(f"Scaled: {input_scaled[0]}")
        
        for model_name, model in models.items():
            try:
                pred = model.predict(input_scaled)[0]
                idx_to_variety = {
                    0: 'IR-64', 1: 'Ciherang', 2: 'Inpari 30', 3: 'Inpari 32', 4: 'Inpari 42',
                    5: 'Inpari 46', 6: 'Mekongga', 7: 'Sembada B9', 8: 'Situ Bagendit'
                }
                pred_variety = idx_to_variety[pred]
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_scaled)[0]
                    confidence = proba[pred] * 100
                    print(f"   {model_name}: {pred_variety} ({confidence:.1f}%)")
                else:
                    print(f"   {model_name}: {pred_variety}")
            except Exception as e:
                print(f"   {model_name}: ERROR - {e}")

def fix_streamlit_preprocessing():
    """Fix the preprocessing order in Streamlit app"""
    print("\\n🔧 FIXING STREAMLIT PREPROCESSING")
    print("=" * 60)
    
    # The issue is in the input preprocessing order
    # Current Streamlit uses: [Kerebahan, TeksturNasi, KetahananTerhadapHama, Kerontokan, WarnaGabah, UmurTanaman, PHTanah, PotensiHasil]
    # But scaler expects: ['UmurTanaman', 'Kerebahan', 'TeksturNasi', 'PotensiHasil', 'KetahananTerhadapHama', 'Kerontokan', 'WarnaGabah', 'PHTanah']
    
    print("The problem is INPUT ORDER in Streamlit preprocessing!")
    print("\\nCurrent Streamlit input order (WRONG):")
    print("  [Kerebahan, TeksturNasi, KetahananTerhadapHama, Kerontokan, WarnaGabah, UmurTanaman, PHTanah, PotensiHasil]")
    
    print("\\nRequired scaler input order (CORRECT):")
    print("  ['UmurTanaman', 'Kerebahan', 'TeksturNasi', 'PotensiHasil', 'KetahananTerhadapHama', 'Kerontokan', 'WarnaGabah', 'PHTanah']")
    
    print("\\n💡 SOLUTION:")
    print("We need to fix the preprocess_input function in app.py to reorder inputs correctly.")
    
    # Create the correct mapping
    print("\\n📝 CORRECT PREPROCESSING CODE:")
    
    code = '''
def preprocess_input_fixed(input_dict, feature_cols):
    """Fixed preprocessing that orders inputs correctly"""
    
    # Apply manual mappings
    processed = {}
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
    
    # Create input array in the CORRECT ORDER that scaler expects
    # feature_cols = ['UmurTanaman', 'Kerebahan', 'TeksturNasi', 'PotensiHasil', 
    #                 'KetahananTerhadapHama', 'Kerontokan', 'WarnaGabah', 'PHTanah']
    
    input_row = []
    for col in feature_cols:
        value = processed.get(col, 0.0)
        input_row.append(value)
    
    return np.array(input_row).reshape(1, -1)
    '''
    
    print(code)
    
    return True

def test_with_corrected_order():
    """Test predictions with corrected input order"""
    print("\\n🧪 TESTING WITH CORRECTED INPUT ORDER")
    print("=" * 60)
    
    # Load current artifacts
    preprocess_bundle = joblib.load("preprocess_artifacts.pkl")
    scaler = preprocess_bundle["scaler"]
    feature_cols = preprocess_bundle["feature_cols"]
    
    # Load models
    models = {}
    model_files = {
        'Decision Tree': 'decision_tree_final.joblib',
        'Gaussian Naive Bayes': 'gaussian_naive_bayes_final.joblib',
        'SVM (RBF)': 'svm_rbf_final.joblib'
    }
    
    for name, filename in model_files.items():
        try:
            models[name] = joblib.load(filename)
        except:
            continue
    
    # Manual mappings (same as in Streamlit)
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
    
    # Correct preprocessing function
    def preprocess_input_corrected(input_dict, feature_cols):
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
    
    # Test cases
    test_cases = [
        {
            'name': 'Default Streamlit input',
            'input_dict': {
                "Kerebahan": "Tahan",
                "TeksturNasi": "Pulen", 
                "KetahananTerhadapHama": "Tahan",
                "Kerontokan": "Tahan",
                "WarnaGabah": "Kuning Bersih",
                "UmurTanaman": 116,
                "PHTanah": 5.6,
                "PotensiHasil": 5
            }
        },
        {
            'name': 'IR-64 like characteristics',
            'input_dict': {
                "Kerebahan": "Sedang",
                "TeksturNasi": "Agak Pera", 
                "KetahananTerhadapHama": "Sedang",
                "Kerontokan": "Sedang",
                "WarnaGabah": "Kuning Jerami",
                "UmurTanaman": 115,
                "PHTanah": 5.5,
                "PotensiHasil": 4
            }
        },
        {
            'name': 'Inpari 30 like (early, high performance)',
            'input_dict': {
                "Kerebahan": "Tahan",
                "TeksturNasi": "Pulen", 
                "KetahananTerhadapHama": "Tahan",
                "Kerontokan": "Tahan",
                "WarnaGabah": "Kuning",
                "UmurTanaman": 100,
                "PHTanah": 6.5,
                "PotensiHasil": 7
            }
        }
    ]
    
    idx_to_variety = {
        0: 'IR-64', 1: 'Ciherang', 2: 'Inpari 30', 3: 'Inpari 32', 4: 'Inpari 42',
        5: 'Inpari 46', 6: 'Mekongga', 7: 'Sembada B9', 8: 'Situ Bagendit'
    }
    
    all_predictions = []
    
    for test_case in test_cases:
        print(f"\\n🔬 Test: {test_case['name']}")
        
        # Process input correctly
        X_input = preprocess_input_corrected(test_case['input_dict'], feature_cols)
        print(f"Processed input: {X_input[0]}")
        
        # Scale
        X_scaled = scaler.transform(X_input)
        print(f"Scaled input: {X_scaled[0][:4]}... (first 4)")
        
        case_predictions = []
        for model_name, model in models.items():
            try:
                pred = model.predict(X_scaled)[0]
                pred_variety = idx_to_variety[pred]
                case_predictions.append(pred_variety)
                all_predictions.append(pred_variety)
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[0]
                    confidence = proba[pred] * 100
                    print(f"   {model_name}: {pred_variety} ({confidence:.1f}%)")
                else:
                    print(f"   {model_name}: {pred_variety}")
            except Exception as e:
                print(f"   {model_name}: ERROR - {e}")
        
        # Consensus
        consensus = Counter(case_predictions)
        if consensus:
            winner, votes = consensus.most_common(1)[0]
            print(f"   Consensus: {winner} ({votes}/{len(case_predictions)} votes)")
    
    # Check diversity
    print(f"\\n📊 PREDICTION DIVERSITY:")
    pred_counts = Counter(all_predictions)
    total = len(all_predictions)
    for variety, count in pred_counts.most_common():
        percentage = (count / total) * 100
        print(f"   {variety}: {count}/{total} ({percentage:.1f}%)")
    
    # Check if this fixes the issue
    max_percentage = (pred_counts.most_common(1)[0][1] / total) * 100
    if max_percentage < 70 and len(pred_counts) >= 2:
        print(f"\\n🎉 SUCCESS: Input order fix improved diversity!")
        print(f"   No single variety dominates (max: {max_percentage:.1f}%)")
        return True
    else:
        print(f"\\n⚠️ Still needs work: {pred_counts.most_common(1)[0][0]} has {max_percentage:.1f}%")
        return False

if __name__ == "__main__":
    # Step 1: Analyze current issue
    analyze_current_models()
    
    # Step 2: Show the fix needed
    fix_streamlit_preprocessing()
    
    # Step 3: Test with corrected order
    success = test_with_corrected_order()
    
    print("\\n" + "=" * 60)
    if success:
        print("🎯 SOLUTION FOUND: Fix input order in Streamlit app!")
        print("\\nNext step: Update the preprocess_input function in app.py")
    else:
        print("🔍 Need to investigate further...")