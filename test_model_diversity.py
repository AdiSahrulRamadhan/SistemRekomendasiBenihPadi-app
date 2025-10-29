import joblib
import numpy as np

def test_model_diversity():
    """Test apakah model bisa prediksi beragam dengan input ekstrem"""
    
    # Load models
    models = {
        'Decision Tree': joblib.load('decision_tree_final.joblib'),
        'Gaussian NB': joblib.load('gaussian_naive_bayes_final.joblib'),
        'SVM': joblib.load('svm_rbf_final.joblib')
    }
    preprocess_bundle = joblib.load('preprocess_artifacts.pkl')
    scaler = preprocess_bundle['scaler']

    idx2name = {0: 'IR-64', 1: 'Ciherang', 2: 'Inpari 30', 3: 'Inpari 32', 4: 'Inpari 42', 5: 'Inpari 46', 6: 'Mekongga', 7: 'Sembada B9', 8: 'Situ Bagendit'}

    # Test input yang sangat berbeda
    test_cases = [
        ([100, 4, 5, 2, 4, 4, 1, 4.5], 'EXTREME LOW (should be Situ Bagendit)'),
        ([115, 7, 6, 4, 7, 7, 1, 5.5], 'IR-64 LIKE'),
        ([135, 7, 6, 6, 7, 7, 0, 6.0], 'LATE MATURITY (should be Mekongga/Sembada)'),
        ([140, 9, 8, 9, 9, 9, 2, 7.0], 'HIGH PERFORMANCE (should be Inpari)'),
        ([90, 4, 4, 1, 4, 4, 2, 4.0], 'ULTRA LOW PERFORMANCE')
    ]

    print('=== MODEL DIVERSITY TEST ===')
    all_predictions = []
    
    for test_input, description in test_cases:
        print(f'\\nTest: {description}')
        print(f'Input: {test_input}')
        
        input_scaled = scaler.transform(np.array(test_input).reshape(1, -1))
        
        case_predictions = []
        for name, model in models.items():
            pred = model.predict(input_scaled)[0]
            variety = idx2name[pred]
            case_predictions.append(variety)
            all_predictions.append(variety)
            print(f'  {name}: {variety}')
        
        # Count unique predictions for this case
        unique_preds = set(case_predictions)
        print(f'  Unique predictions: {len(unique_preds)} varieties: {unique_preds}')
        
        if len(unique_preds) == 1:
            print(f'  ⚠️ ALL MODELS predict same: {list(unique_preds)[0]}')
        else:
            print(f'  ✅ Good diversity!')
    
    # Overall analysis
    print('\\n' + '='*50)
    print('OVERALL ANALYSIS:')
    
    from collections import Counter
    pred_counts = Counter(all_predictions)
    total = len(all_predictions)
    
    print(f'\\nPrediction distribution across all tests:')
    for variety, count in pred_counts.most_common():
        percentage = (count / total) * 100
        print(f'  {variety}: {count}/{total} ({percentage:.1f}%)')
    
    # Check if Ciherang dominates
    ciherang_count = pred_counts.get('Ciherang', 0)
    ciherang_percentage = (ciherang_count / total) * 100
    
    print(f'\\nCiherang bias check:')
    print(f'  Ciherang predictions: {ciherang_count}/{total} ({ciherang_percentage:.1f}%)')
    
    if ciherang_percentage > 70:
        print(f'  🔴 SEVERE BIAS: Ciherang dominates {ciherang_percentage:.1f}% of predictions')
        print(f'  💡 SOLUTION NEEDED: Models are still heavily biased')
    elif ciherang_percentage > 50:
        print(f'  🟡 MODERATE BIAS: Ciherang still preferred but improving')
    else:
        print(f'  🟢 GOOD DIVERSITY: Ciherang bias under control')
    
    # Unique varieties predicted
    unique_varieties = len(pred_counts)
    print(f'\\nVariety diversity:')
    print(f'  Unique varieties predicted: {unique_varieties}/9')
    
    if unique_varieties >= 5:
        print(f'  🟢 EXCELLENT: Good variety coverage')
    elif unique_varieties >= 3:
        print(f'  🟡 DECENT: Reasonable variety coverage')
    else:
        print(f'  🔴 POOR: Limited variety coverage')

if __name__ == "__main__":
    test_model_diversity()