import joblib
import numpy as np

def test_model_responsiveness():
    """Test apakah model responsif terhadap perubahan input"""
    
    print("🧪 TESTING MODEL RESPONSIVENESS")
    print("=" * 50)
    
    # Load models
    models = {
        'Decision Tree': joblib.load('decision_tree_final.joblib'),
        'Gaussian NB': joblib.load('gaussian_naive_bayes_final.joblib'),
        'SVM': joblib.load('svm_rbf_final.joblib')
    }
    
    preprocess_bundle = joblib.load('preprocess_artifacts.pkl')
    scaler = preprocess_bundle['scaler']
    
    idx2name = {0: 'IR-64', 1: 'Ciherang', 2: 'Inpari 30', 3: 'Inpari 32', 4: 'Inpari 42', 
                5: 'Inpari 46', 6: 'Mekongga', 7: 'Sembada B9', 8: 'Situ Bagendit'}
    
    # Test cases yang sangat berbeda
    test_cases = [
        {
            'name': 'EXTREME LOW (preset)',
            'input': [100, 4, 5, 2, 4, 4, 1, 4.5],
            'expected_variety': 'Situ Bagendit'
        },
        {
            'name': 'IR-64 LIKE (preset)', 
            'input': [115, 7, 6, 4, 7, 7, 1, 5.5],
            'expected_variety': 'IR-64 or others'
        },
        {
            'name': 'SEMBADA B9 (preset)',
            'input': [135, 7, 6, 6, 7, 7, 0, 6.0],
            'expected_variety': 'Sembada B9'
        },
        {
            'name': 'HIGH PERFORMANCE',
            'input': [125, 9, 8, 8, 9, 9, 2, 7.0],
            'expected_variety': 'Inpari varieties'
        },
        {
            'name': 'DEFAULT (original)',
            'input': [116, 9, 8, 5, 9, 9, 0, 5.6],
            'expected_variety': 'Mixed'
        }
    ]
    
    all_predictions = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\\nTest {i+1}: {test_case['name']}")
        print(f"Input: {test_case['input']}")
        print(f"Expected: {test_case['expected_variety']}")
        
        input_scaled = scaler.transform(np.array(test_case['input']).reshape(1, -1))
        
        case_predictions = []
        for model_name, model in models.items():
            pred = model.predict(input_scaled)[0]
            variety = idx2name[pred]
            case_predictions.append(variety)
            all_predictions.append(variety)
            
            # Get confidence if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[0]
                confidence = proba[pred] * 100
                print(f"  {model_name}: {variety} ({confidence:.1f}%)")
            else:
                print(f"  {model_name}: {variety}")
        
        # Check consensus
        from collections import Counter
        consensus = Counter(case_predictions)
        winner, votes = consensus.most_common(1)[0]
        print(f"  🗳️ Consensus: {winner} ({votes}/3 votes)")
        
        # Check if all same (bad)
        if len(set(case_predictions)) == 1:
            print(f"  ⚠️ All models agree on: {winner}")
        else:
            print(f"  ✅ Diversity: {len(set(case_predictions))} different predictions")
    
    # Overall analysis
    print("\\n" + "=" * 50)
    print("OVERALL ANALYSIS:")
    
    from collections import Counter
    pred_counts = Counter(all_predictions)
    total = len(all_predictions)
    
    print(f"\\nPrediction distribution:")
    for variety, count in pred_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {variety}: {count}/{total} ({percentage:.1f}%)")
    
    # Check responsiveness
    unique_varieties = len(pred_counts)
    max_percentage = (pred_counts.most_common(1)[0][1] / total) * 100
    
    print(f"\\nResponsiveness Check:")
    print(f"  Unique varieties predicted: {unique_varieties}/9")
    print(f"  Most common variety: {pred_counts.most_common(1)[0][0]} ({max_percentage:.1f}%)")
    
    if unique_varieties >= 4 and max_percentage < 60:
        print(f"\\n🟢 EXCELLENT: Models are highly responsive to input changes")
        return "EXCELLENT"
    elif unique_varieties >= 3 and max_percentage < 70:
        print(f"\\n🟡 GOOD: Models show decent responsiveness")
        return "GOOD"
    elif unique_varieties >= 2:
        print(f"\\n🟠 FAIR: Limited but some responsiveness")
        return "FAIR"
    else:
        print(f"\\n🔴 POOR: Models not responsive to input changes")
        return "POOR"

if __name__ == "__main__":
    result = test_model_responsiveness()
    print(f"\\nFINAL ASSESSMENT: {result}")