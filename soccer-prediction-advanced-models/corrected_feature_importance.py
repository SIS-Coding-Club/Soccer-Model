# -*- coding: utf-8 -*-
"""
Corrected Feature Importance Analysis - Excluding Target Leakage
================================================================

This script properly calculates feature importance by excluding the target variable
itself from each target's feature importance calculation.

The problem: When predicting aerials_duels_won, if aerials_duels_won is in the features,
it will naturally be the most important feature - but that's circular/not useful.

The solution: For each target, exclude that target from the feature importance calculation,
then calculate global importance across all targets.
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime


def load_shap_values(model_name='xgboost'):
    """Load pre-computed SHAP values from shap_analysis.py"""
    base_dir = os.path.dirname(__file__)
    shap_dir = os.path.join(base_dir, 'outputs', 'shap_analysis', model_name)
    
    # Find latest SHAP pickle file
    if not os.path.exists(shap_dir):
        print(f"\nWarning: SHAP directory not found: {shap_dir}")
        print("   Please run shap_analysis.py first to generate SHAP values.")
        return None, None
        
    shap_files = [f for f in os.listdir(shap_dir) if f.startswith('shap_values_dict') and f.endswith('.pkl')]
    
    if not shap_files:
        print(f"\nWarning: No saved SHAP values found in {shap_dir}")
        print("   Looking for SHAP importance matrix CSV instead...")
        
        # Try to load from CSV matrices instead
        csv_files = [f for f in os.listdir(shap_dir) if 'importance_matrix' in f and f.endswith('.csv')]
        if not csv_files:
            print("   No SHAP data found. Please run shap_analysis.py first.")
            return None, None
        
        # Load most recent CSV
        latest_csv = sorted(csv_files)[-1]
        csv_path = os.path.join(shap_dir, latest_csv)
        print(f"   [CSV] Loading from: {latest_csv}")
        
        importance_df = pd.read_csv(csv_path, index_col=0)
        feature_names = importance_df.columns.tolist()
        
        # Note: This gives us importance directly, not raw SHAP values
        # So we return it in a compatible format
        print(f"   [OK] Loaded importance matrix: {importance_df.shape}")
        return importance_df, feature_names
    
    latest_file = sorted(shap_files)[-1]
    filepath = os.path.join(shap_dir, latest_file)
    
    print(f"[PKL] Loading SHAP values from: {latest_file}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data['shap_values_dict'], data['feature_names']


def compute_corrected_feature_importance(data, feature_names, output_dir, from_shap_values=True):
    """
    Compute feature importance excluding target leakage.
    
    Args:
        data: Either shap_values_dict (from_shap_values=True) or importance_df (from_shap_values=False)
        from_shap_values: If True, data is raw SHAP values; if False, data is pre-computed importance
    """
    print("\n" + "="*80)
    print("[ANALYSIS] CORRECTED FEATURE IMPORTANCE")
    print("="*80)
    print("\nExcluding target leakage (target variable from its own importance)...\n")
    
    if from_shap_values:
        # Build corrected importance matrix from raw SHAP values
        shap_values_dict = data
        corrected_importance_matrix = []
        target_names = list(shap_values_dict.keys())
        
        for target_name in target_names:
            shap_vals = shap_values_dict[target_name]
            
            # Calculate mean absolute SHAP for all features
            feature_importance = np.mean(np.abs(shap_vals), axis=0)
            
            # Check if target is in features - exclude it
            excluded_features = []
            for i, feat_name in enumerate(feature_names):
                # Check for exact match or close match
                if target_name == feat_name or feat_name in target_name or target_name in feat_name:
                    excluded_features.append(feat_name)
                    feature_importance[i] = 0  # Zero out the leaked feature
            
            if excluded_features:
                print(f"  Target: {target_name:30s} -> Excluded: {', '.join(excluded_features)}")
            
            corrected_importance_matrix.append(feature_importance)
        
        # Create DataFrame
        importance_df = pd.DataFrame(
            corrected_importance_matrix,
            columns=feature_names,
            index=target_names
        )
    else:
        # Data is already an importance matrix (loaded from CSV)
        importance_df = data.copy()
        target_names = importance_df.index.tolist()
        
        # Zero out leaked features
        for target_name in target_names:
            excluded_features = []
            for feat_name in feature_names:
                if target_name == feat_name or feat_name in target_name or target_name in feat_name:
                    excluded_features.append(feat_name)
                    importance_df.loc[target_name, feat_name] = 0
            
            if excluded_features:
                print(f"  Target: {target_name:30s} -> Excluded: {', '.join(excluded_features)}")
    
    # Global importance (averaged across all targets, excluding leakage)
    global_importance = importance_df.mean(axis=0).sort_values(ascending=False)
    
    # Remove features with zero importance (these were all leaked)
    global_importance = global_importance[global_importance > 0]
    
    return importance_df, global_importance


def generate_corrected_report(importance_df, global_importance, output_dir):
    """Generate corrected feature importance report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'corrected_feature_importance_{timestamp}.txt')
    
    print(f"\n[REPORT] Generating corrected report...")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CORRECTED FEATURE IMPORTANCE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Targets: {importance_df.shape[0]}\n")
        f.write(f"Number of Features: {importance_df.shape[1]}\n")
        f.write("Method: SHAP values with target leakage removed\n\n")
        
        f.write("="*80 + "\n")
        f.write("WHAT THIS REPORT SHOWS\n")
        f.write("="*80 + "\n\n")
        f.write("This analysis answers: 'Which features are most important for predicting\n")
        f.write("OTHER variables?' by excluding each target from its own feature importance.\n\n")
        f.write("Example: When finding importance of 'aerials_duels_won', we exclude\n")
        f.write("aerials_duels_won from the features so we see what ELSE predicts it.\n\n")
        
        f.write("="*80 + "\n")
        f.write("GLOBAL FEATURE IMPORTANCE (Averaged Across All Targets, No Leakage)\n")
        f.write("="*80 + "\n\n")
        f.write("Features ranked by average impact when predicting OTHER metrics:\n\n")
        
        for rank, (feature, importance) in enumerate(global_importance.items(), 1):
            f.write(f"{rank:2d}. {feature:40s} | Importance: {importance:.6f}\n")
        
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("FEATURE IMPORTANCE PER TARGET (Excluding Self-Prediction)\n")
        f.write("="*80 + "\n\n")
        
        for target_name in importance_df.index:
            target_importance = importance_df.loc[target_name].sort_values(ascending=False)
            # Show only non-zero importance
            target_importance = target_importance[target_importance > 0].head(10)
            
            f.write(f"\nTarget: {target_name}\n")
            f.write("-" * 80 + "\n")
            f.write("Top 10 features (excluding self):\n")
            
            for rank, (feature, importance) in enumerate(target_importance.items(), 1):
                f.write(f"  {rank:2d}. {feature:40s} | Impact: {importance:.6f}\n")
            
            f.write("\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("="*80 + "\n\n")
        f.write("GLOBAL IMPORTANCE:\n")
        f.write("- Features at the top are the most predictive across ALL targets\n")
        f.write("- These are the 'overall most important' features for your model\n")
        f.write("- High importance = strong influence on predicting OTHER metrics\n\n")
        f.write("PER-TARGET IMPORTANCE:\n")
        f.write("- Shows what features predict each specific target (excluding itself)\n")
        f.write("- Useful for understanding what drives each performance metric\n\n")
        f.write("KEY INSIGHT:\n")
        f.write("- The features with highest global importance are the most versatile\n")
        f.write("  predictors - they help predict many different outcomes\n")
        f.write("- Features with high importance for only 1-2 targets are specialists\n\n")
    
    print(f"  [OK] Report saved: {report_path}")
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f'corrected_importance_matrix_{timestamp}.csv')
    importance_df.to_csv(csv_path)
    print(f"  [OK] Matrix saved: {csv_path}")
    
    # Save global importance separately
    global_csv_path = os.path.join(output_dir, f'corrected_global_importance_{timestamp}.csv')
    global_importance.to_csv(global_csv_path, header=['Importance'])
    print(f"  [OK] Global importance saved: {global_csv_path}")
    
    return report_path


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("[START] CORRECTED FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    print("\nThis analysis fixes the target leakage problem by excluding each target")
    print("from its own feature importance calculation.\n")
    
    # Setup
    base_dir = os.path.dirname(__file__)
    model_name = 'xgboost'  # or 'lightgbm'
    output_dir = os.path.join(base_dir, 'outputs', 'shap_analysis', model_name)
    
    try:
        # Load SHAP values
        data, feature_names = load_shap_values(model_name)
        
        if data is None:
            print("\n[ERROR] Cannot proceed without SHAP data.")
            print("   Please run: python shap_analysis.py --model xgboost")
            return
        
        # Determine if we have raw SHAP values or pre-computed importance
        from_shap_values = isinstance(data, dict)
        
        if from_shap_values:
            print(f"[OK] Loaded raw SHAP values for {len(data)} targets")
        else:
            print(f"[OK] Loaded importance matrix for {len(data)} targets")
        
        print(f"[OK] Features: {len(feature_names)}\n")
        
        # Compute corrected importance
        importance_df, global_importance = compute_corrected_feature_importance(
            data, feature_names, output_dir, from_shap_values=from_shap_values
        )
        
        # Generate report
        report_path = generate_corrected_report(importance_df, global_importance, output_dir)
        
        # Display top results
        print("\n" + "="*80)
        print("[COMPLETE] ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nTop 15 Most Important Features (Overall, No Leakage):")
        print("-" * 80)
        for rank, (feature, importance) in enumerate(global_importance.head(15).items(), 1):
            print(f"{rank:2d}. {feature:40s} | Importance: {importance:.6f}")
        
        print(f"\n[OUTPUT] Full report saved to: {report_path}")
        print("\n[INSIGHT] Key Insight:")
        print("   These features are the most predictive ACROSS ALL targets,")
        print("   excluding self-prediction. They're your 'overall most important' features.")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
