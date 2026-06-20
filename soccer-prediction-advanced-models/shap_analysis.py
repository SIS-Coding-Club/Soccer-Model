#!/usr/bin/env python3
"""
SHAP Explainability Analysis for Soccer Player Performance Models

This script applies SHAP (SHapley Additive exPlanations) to determine which 
statistical features most strongly influence player performance predictions.

Standard: Model Output Standard
- Improvement measured as increase in model's predicted performance score
- SHAP shows which features drive higher predictions for each target metric
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.preprocessing import preprocess_data

# Set style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_trained_model(model_path):
    """Load a trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"📦 Loading model: {model_path}")
    model = joblib.load(model_path)
    print(f"✓ Model loaded successfully")
    return model


def create_shap_explainer(model, X_sample, model_type='xgboost'):
    """
    Create appropriate SHAP explainer for the model type.
    
    Args:
        model: Trained model
        X_sample: Sample data for background (TreeExplainer) or full data (others)
        model_type: 'xgboost', 'lightgbm', or 'neural_network'
    """
    print(f"\n🔍 Creating SHAP explainer for {model_type}...")
    
    if model_type in ['xgboost', 'lightgbm']:
        # Tree-based models: Use TreeExplainer (fast and exact)
        # For MultiOutputRegressor, we need to explain each estimator separately
        print("  Using TreeExplainer (optimized for gradient boosting)")
        explainer = shap.TreeExplainer(model.model.estimators_[0])  # First target as example
        print(f"✓ Explainer created (will analyze each of {len(model.model.estimators_)} targets)")
        return explainer, 'tree'
    
    elif model_type == 'neural_network':
        # Neural network: Use DeepExplainer or GradientExplainer
        print("  Using DeepExplainer (for neural networks)")
        # Need background data for DeepExplainer
        explainer = shap.DeepExplainer(model.model, X_sample[:100])
        print(f"✓ Explainer created with background sample (100 samples)")
        return explainer, 'deep'
    
    else:
        # Fallback: KernelExplainer (model-agnostic, slower)
        print("  Using KernelExplainer (model-agnostic)")
        explainer = shap.KernelExplainer(model.predict, X_sample[:100])
        print(f"✓ Explainer created")
        return explainer, 'kernel'


def compute_shap_values_multi_target(model, X, target_names, model_type='xgboost', max_samples=1000):
    """
    Compute SHAP values for multi-output model (all targets).
    
    Returns:
        dict: {target_name: shap_values} for each target
    """
    print(f"\n🧮 Computing SHAP values for {len(target_names)} targets...")
    print(f"   Analyzing {min(len(X), max_samples)} samples")
    
    X_sample = X.iloc[:max_samples] if len(X) > max_samples else X
    shap_values_dict = {}
    
    if model_type in ['xgboost', 'lightgbm']:
        # For tree models, compute SHAP for each target separately
        for i, target in enumerate(target_names):
            print(f"  [{i+1}/{len(target_names)}] Computing SHAP for: {target}")
            
            # Create explainer for this specific estimator
            explainer = shap.TreeExplainer(model.model.estimators_[i])
            
            # Compute SHAP values
            shap_values = explainer.shap_values(X_sample)
            shap_values_dict[target] = shap_values
            
        print(f"✓ SHAP values computed for all {len(target_names)} targets")
    
    else:
        # For other models, compute once and split by target
        print("  Computing SHAP values (this may take a few minutes)...")
        explainer, _ = create_shap_explainer(model, X_sample, model_type)
        shap_values = explainer.shap_values(X_sample)
        
        # Split by target
        for i, target in enumerate(target_names):
            shap_values_dict[target] = shap_values[:, i]
        
        print(f"✓ SHAP values computed")
    
    return shap_values_dict, X_sample


def plot_shap_summary_all_targets(shap_values_dict, X, feature_names, output_dir, timestamp):
    """
    Create SHAP summary plot showing feature importance across all targets.
    """
    print("\n📊 Creating global SHAP summary plot...")
    
    # Aggregate SHAP values across all targets (mean absolute values)
    all_shap_values = []
    for target, shap_vals in shap_values_dict.items():
        all_shap_values.append(np.abs(shap_vals))
    
    # Average across targets
    mean_abs_shap = np.mean(all_shap_values, axis=0)
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    shap.summary_plot(
        mean_abs_shap, 
        X, 
        feature_names=feature_names,
        plot_type='bar',
        show=False,
        max_display=20
    )
    
    plt.title('Global Feature Importance (Averaged Across All Targets)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Mean |SHAP Value| (Average Impact on Predictions)', fontsize=12)
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'shap_global_summary_{timestamp}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {filepath}")


def plot_shap_summary_per_target(shap_values_dict, X, feature_names, output_dir, timestamp, top_n=5):
    """
    Create SHAP summary plots for top N most important targets.
    """
    print(f"\n📊 Creating SHAP summary plots for top {top_n} targets...")
    
    # Calculate total SHAP importance per target
    target_importance = {}
    for target, shap_vals in shap_values_dict.items():
        target_importance[target] = np.mean(np.abs(shap_vals))
    
    # Sort targets by importance
    sorted_targets = sorted(target_importance.items(), key=lambda x: x[1], reverse=True)
    top_targets = sorted_targets[:top_n]
    
    print(f"  Top {top_n} most predictable targets:")
    for i, (target, importance) in enumerate(top_targets, 1):
        print(f"    {i}. {target:30s} (importance: {importance:.4f})")
    
    # Create plots for top targets
    for target, _ in top_targets:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        shap.summary_plot(
            shap_values_dict[target],
            X,
            feature_names=feature_names,
            show=False,
            max_display=15
        )
        
        plt.title(f'Feature Impact on {target}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save
        safe_target = target.replace('/', '_').replace(' ', '_')
        filepath = os.path.join(output_dir, f'shap_summary_{safe_target}_{timestamp}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  💾 Saved: {filepath}")


def plot_shap_dependence_plots(shap_values_dict, X, feature_names, output_dir, timestamp, 
                                target_name, top_features=4):
    """
    Create SHAP dependence plots showing how each top feature affects predictions.
    """
    print(f"\n📊 Creating dependence plots for {target_name}...")
    
    shap_vals = shap_values_dict[target_name]
    
    # Find top features for this target
    feature_importance = np.mean(np.abs(shap_vals), axis=0)
    top_indices = np.argsort(feature_importance)[-top_features:][::-1]
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, feat_idx in enumerate(top_indices[:4]):
        feat_name = feature_names[feat_idx]
        
        # Create dependence plot
        plt.sca(axes[idx])
        shap.dependence_plot(
            feat_idx,
            shap_vals,
            X,
            feature_names=feature_names,
            show=False,
            ax=axes[idx]
        )
        axes[idx].set_title(f'Impact of {feat_name}', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'Feature Dependence Plots for {target_name}', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save
    safe_target = target_name.replace('/', '_').replace(' ', '_')
    filepath = os.path.join(output_dir, f'shap_dependence_{safe_target}_{timestamp}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {filepath}")


def plot_shap_waterfall(shap_values_dict, X, feature_names, output_dir, timestamp, 
                        target_name, sample_idx=0):
    """
    Create waterfall plot showing how features contribute to a single prediction.
    """
    print(f"\n📊 Creating waterfall plot for {target_name} (sample {sample_idx})...")
    
    shap_vals = shap_values_dict[target_name]
    
    # Create waterfall plot for single sample
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create Explanation object for waterfall plot
    explanation = shap.Explanation(
        values=shap_vals[sample_idx],
        base_values=shap_vals[sample_idx].mean() if hasattr(shap_vals[sample_idx], 'mean') else 0,
        data=X.iloc[sample_idx].values,
        feature_names=feature_names
    )
    
    shap.plots.waterfall(explanation, show=False)
    
    plt.title(f'Feature Contributions to {target_name} Prediction\n(Player: {X.index[sample_idx]})', 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save
    safe_target = target_name.replace('/', '_').replace(' ', '_')
    filepath = os.path.join(output_dir, f'shap_waterfall_{safe_target}_sample{sample_idx}_{timestamp}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {filepath}")


def generate_feature_importance_report(shap_values_dict, feature_names, output_dir, timestamp, exclude_target_leakage=True):
    """
    Generate text report of feature importance across all targets.
    
    Args:
        exclude_target_leakage: If True, excludes each target from its own feature importance
                               to avoid circular importance (e.g., aerials_won predicting aerials_won)
    """
    print("\n📝 Generating feature importance report...")
    
    if exclude_target_leakage:
        print("   ⚠️  Excluding target leakage (each target excluded from its own features)")
    
    # Calculate mean absolute SHAP per feature per target
    importance_matrix = []
    
    for target, shap_vals in shap_values_dict.items():
        feature_importance = np.mean(np.abs(shap_vals), axis=0)
        
        # Exclude target leakage: zero out features that match the target name
        if exclude_target_leakage:
            for i, feat_name in enumerate(feature_names):
                # Check for exact match or if target/feature names contain each other
                if target == feat_name or feat_name in target or target in feat_name:
                    feature_importance[i] = 0  # Zero out leaked feature
        
        importance_matrix.append(feature_importance)
    
    importance_df = pd.DataFrame(
        importance_matrix,
        columns=feature_names,
        index=shap_values_dict.keys()
    )
    
    # Global importance (averaged across targets)
    global_importance = importance_df.mean(axis=0).sort_values(ascending=False)
    
    # If excluding leakage, remove zero-importance features from global ranking
    if exclude_target_leakage:
        global_importance = global_importance[global_importance > 0]
    
    # Create report
    report_path = os.path.join(output_dir, f'shap_feature_importance_report_{timestamp}.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SHAP FEATURE IMPORTANCE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Targets: {len(shap_values_dict)}\n")
        f.write(f"Number of Features: {len(feature_names)}\n")
        f.write(f"Standard: Model Output Standard (SHAP on predicted values)\n")
        f.write(f"Target Leakage Excluded: {'Yes' if exclude_target_leakage else 'No'}\n\n")
        
        if exclude_target_leakage:
            f.write("="*80 + "\n")
            f.write("NOTE: TARGET LEAKAGE EXCLUDED\n")
            f.write("="*80 + "\n\n")
            f.write("This report excludes each target from its own feature importance.\n")
            f.write("This answers: 'What features predict OTHER metrics?'\n")
            f.write("Example: When predicting aerials_duels_won, aerials_duels_won feature is excluded.\n\n")
        
        f.write("="*80 + "\n")
        f.write("GLOBAL FEATURE IMPORTANCE (Averaged Across All Targets)\n")
        f.write("="*80 + "\n\n")
        f.write("Features ranked by average impact on predictions:\n\n")
        
        for rank, (feature, importance) in enumerate(global_importance.items(), 1):
            f.write(f"{rank:2d}. {feature:35s} | Importance: {importance:.6f}\n")
        # If excluding leakage, show only non-zero features
            if exclude_target_leakage:
                target_importance = target_importance[target_importance > 0]
            
            f.write(f"\nTarget: {target}\n")
            f.write("-" * 80 + "\n")
            f.write("Top 10 features" + (" (excluding self)" if exclude_target_leakage else "") + "
        f.write("FEATURE IMPORTANCE PER TARGET\n")
        f.write("="*80 + "\n\n")
        
        for target in shap_values_dict.keys():
            target_importance = importance_df.loc[target].sort_values(ascending=False)
            
            f.write(f"\nTarget: {target}\n")
            f.write("-" * 80 + "\n")
            f.write("Top 10 features:\n")
            
            for rank, (feature, importance) in enumerate(target_importance.head(10).items(), 1):
                f.write(f"  {rank:2d}. {feature:35s} | Impact: {importance:.6f}\n")
            
            f.write("\n")
        
        f.write("\n")
        f.write("="*80 + "\n")")
        
        if exclude_target_leakage:
            f.write("\n")
            f.write("GLOBAL IMPORTANCE (No Leakage):\n")
            f.write("- These features are most predictive ACROSS ALL targets\n")
            f.write("- They represent 'overall most important' features for predicting OTHER metrics\n")
            f.write("- High global importance = versatile predictor useful for many outcomes\n")
        
        f.write("
        f.write("INTERPRETATION GUIDE\n")
        f.write("="*80 + "\n\n")
        f.write("SHAP values represent the contribution of each feature to the prediction:\n\n")
        f.write("- Positive SHAP: Feature increases the predicted value\n")
        f.write("- Negative SHAP: Feature decreases the predicted value\n")
        f.write("- Magnitude: How strongly the feature affects the prediction\n\n")
        f.write("The importance values shown are mean absolute SHAP values,\n")
        f.write("representing the average magnitude of each feature's impact.\n\n")
        f.write("Key Insights:\n")
        f.write("- Features at the top have the strongest influence on predictions\n")
        f.write("- Features with high importance across multiple targets are globally important\n")
        f.write("- Features with high importance for specific targets are specialists\n")
        f.write("- Low importance suggests the feature may be redundant or uninformative\n\n")
    
    print(f"  💾 Report saved: {report_path}")
    
    # Also save as CSV for further analysis
    csv_path = os.path.join(output_dir, f'shap_importance_matrix_{timestamp}.csv')
    importance_df.to_csv(csv_path)
    print(f"  💾 Importance matrix saved: {csv_path}")
    
    return importance_df, global_importance


def main():
    parser = argparse.ArgumentParser(
        description='SHAP Explainability Analysis for Soccer Prediction Models'
    )
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm', 'nn'],
                       help='Which model to analyze')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model (if not specified, uses latest in outputs/models/)')
    parser.add_argument('--data', type=str, 
                       default='../model_2/data/final/final.csv',
                       help='Path to data')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum samples for SHAP analysis (smaller = faster)')
    parser.add_argument('--top-targets', type=int, default=5,
                       help='Number of top targets to analyze in detail')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔍 SHAP EXPLAINABILITY ANALYSIS")
    print("="*80)
    print(f"\nModel: {args.model.upper()}")
    print(f"Standard: Model Output Standard (feature impact on predicted values)")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load data
    print(f"\n📊 Loading data from: {args.data}")
    try:
        data = pd.read_csv(args.data)
        X, y = preprocess_data(data)
        print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    feature_names = X.columns.tolist()
    target_names = y.columns.tolist()
    
    # Find or load model
    if args.model_path is None:
        # Find latest model
        model_dir = 'outputs/models'
        model_files = [f for f in os.listdir(model_dir) if f.startswith(args.model) and f.endswith('.pkl')]
        if not model_files:
            print(f"❌ No trained {args.model} model found in {model_dir}")
            print(f"   Please train a model first using: python3 train.py --model {args.model}")
            return
        
        # Get most recent
        model_files.sort(reverse=True)
        args.model_path = os.path.join(model_dir, model_files[0])
    
    # Load model
    try:
        model = load_trained_model(args.model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Compute SHAP values
    try:
        shap_values_dict, X_sample = compute_shap_values_multi_target(
            model, X, target_names, args.model, args.max_samples
        )
    except Exception as e:
        print(f"❌ Error computing SHAP values: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create output directory
    output_dir = f'outputs/shap_analysis/{args.model}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("📊 GENERATING SHAP VISUALIZATIONS")
    print("="*80)
    
    try:
        # 1. Global summary
        plot_shap_summary_all_targets(shap_values_dict, X_sample, feature_names, output_dir, timestamp)
        
        # 2. Per-target summaries (top N targets)
        plot_shap_summary_per_target(shap_values_dict, X_sample, feature_names, output_dir, 
                                     timestamp, args.top_targets)
        
        # 3. Dependence plots for most important target
        target_importance = {t: np.mean(np.abs(v)) for t, v in shap_values_dict.items()}
        most_important_target = max(target_importance, key=target_importance.get)
        
        plot_shap_dependence_plots(shap_values_dict, X_sample, feature_names, output_dir, 
                                   timestamp, most_important_target)
        
        # 4. Waterfall plot example
        plot_shap_waterfall(shap_values_dict, X_sample, feature_names, output_dir, 
                           timestamp, most_important_target, sample_idx=0)
        
        # 5. Feature importance report (with and without target leakage)
        print("\n📊 Generating feature importance reports...")
        
        # Report WITH target leakage (original)
        importance_df_with_leak, global_importance_with_leak = generate_feature_importance_report(
            shap_values_dict, feature_names, output_dir, timestamp + "_with_leakage", exclude_target_leakage=False
        )
        
        # Report WITHOUT target leakage (corrected)
        importance_df, global_importance = generate_feature_importance_report(
            shap_values_dict, feature_names, output_dir, timestamp, exclude_target_leakage=True
        )
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "="*80)
    print("✅ SHAP ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nTop 10 Most Influential Features (Global):")
    print("-" * 80)
    for rank, (feature, importance) in enumerate(global_importance.head(10).items(), 1):
        print(f"{rank:2d}. {feature:35s} | Importance: {importance:.6f}")
    
    print(f"\n📁 All outputs saved to: {output_dir}/")
    print(f"\n💡 Key Insights:")
    print(f"   - SHAP analysis reveals which features drive model predictions")
    print(f"   - High importance = strong influence on predicted performance")
    print(f"   - Use these insights to focus on key performance indicators")
    print(f"   - Features with consistent importance across targets are globally critical")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
