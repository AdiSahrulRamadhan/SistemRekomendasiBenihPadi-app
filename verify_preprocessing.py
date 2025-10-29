#!/usr/bin/env python3
"""
Verify current preprocessing artifacts and model status
"""

import joblib
import pickle
import os
import time
import numpy as np

def verify_files():
    print("🔍 VERIFYING PREPROCESSING & MODEL FILES")
    print("=" * 50)
    
    # Check preprocessing artifacts
    filename = "preprocess_artifacts.pkl"
    if os.path.exists(filename):
        mtime = os.path.getmtime(filename)
        age_minutes = (time.time() - mtime) / 60
        
        # Load with pickle (same as app)
        with open(filename, 'rb') as f:
            artifacts = pickle.load(f)
        
        scaler = artifacts['scaler']
        scaler_mean = scaler.mean_[:5] if hasattr(scaler, 'mean_') else None
        
        print(f"📄 Preprocessing File: {filename}")
        print(f"⏰ Age: {age_minutes:.1f} minutes")
        print(f"🔢 Scaler signature: {scaler_mean}")
        print(f"📊 Features: {len(artifacts['feature_names'])}")
        
        # Check if it's balanced version (balanced has different signature)
        if scaler_mean is not None:
            if abs(scaler_mean[0] - 116.06) < 0.1:  # Balanced signature
                print("✅ BALANCED preprocessing detected!")
            elif abs(scaler_mean[0] - 116.19) < 0.1:  # Original signature  
                print("⚠️  ORIGINAL (biased) preprocessing detected!")
            else:
                print(f"❓ Unknown preprocessing (mean[0] = {scaler_mean[0]:.2f})")
                
    else:
        print(f"❌ {filename} not found!")
    
    print()
    
    # Check model files
    model_files = [
        "decision_tree_final.joblib",
        "gaussian_naive_bayes_final.joblib", 
        "svm_rbf_final.joblib"
    ]
    
    print("🤖 MODEL FILES:")
    for model_file in model_files:
        if os.path.exists(model_file):
            mtime = os.path.getmtime(model_file)
            age_minutes = (time.time() - mtime) / 60
            
            # Quick prediction test
            model = joblib.load(model_file)
            test_input = np.array([[100, 4, 5, 3, 4, 4, 1, 5.0]])  # Low performance
            
            try:
                if hasattr(scaler, 'transform'):
                    test_scaled = scaler.transform(test_input)
                    pred = model.predict(test_scaled)[0]
                    
                    variety_names = {
                        0: "IR-64", 1: "Ciherang", 2: "Inpari 30", 3: "Inpari 32",
                        4: "Inpari 42", 5: "Inpari 46", 6: "Mekongga", 
                        7: "Sembada B9", 8: "Situ Bagendit"
                    }
                    
                    predicted_variety = variety_names.get(pred, f"Unknown ({pred})")
                    print(f"  📦 {model_file}: age {age_minutes:.1f}min → predicts: {predicted_variety}")
                else:
                    print(f"  📦 {model_file}: age {age_minutes:.1f}min → (scaler not available)")
                    
            except Exception as e:
                print(f"  📦 {model_file}: age {age_minutes:.1f}min → Error: {str(e)}")
        else:
            print(f"  ❌ {model_file}: Not found!")

if __name__ == "__main__":
    verify_files()