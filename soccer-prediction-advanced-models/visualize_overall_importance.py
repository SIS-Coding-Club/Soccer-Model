# -*- coding: utf-8 -*-
"""
Visualize Overall Feature Importance
=====================================

Creates visualizations showing:
1. Most influential features overall (global importance)
2. Heatmap showing which features affect which targets
3. Top features for maintaining high overall performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')


def load_corrected_importance(model_name='xgboost'):
    """Load the corrected importance data."""
    base_dir = os.path.dirname(__file__)
    shap_dir = os.path.join(base_dir, 'outputs', 'shap_analysis', model_name)
    
    # Find latest corrected files
    csv_files = [f for f in os.listdir(shap_dir) if f.startswith('corrected_importance_matrix') and f.endswith('.csv')]
    
    if not csv_files:
        print("Error: No corrected importance files found.")
        print(f"Please run: python corrected_feature_importance.py")
        return None, None
    
    # Load importance matrix
    latest_matrix = sorted(csv_files)[-1]
    matrix_path = os.path.join(shap_dir, latest_matrix)
    importance_df = pd.read_csv(matrix_path, index_col=0)
    
    # Load global importance
    global_files = [f for f in os.listdir(shap_dir) if f.startswith('corrected_global_importance') and f.endswith('.csv')]
    latest_global = sorted(global_files)[-1]
    global_path = os.path.join(shap_dir, latest_global)
    global_importance = pd.read_csv(global_path, index_col=0)
    
    print(f"[OK] Loaded corrected importance data")
    print(f"     Matrix: {importance_df.shape}")
    print(f"     Global features: {len(global_importance)}")
    
    return importance_df, global_importance


def plot_top_features_bar(global_importance, output_dir, top_n=15):
    """Bar chart showing top N most influential features overall."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_features = global_importance.head(top_n)
    
    # Create gradient colors (darker = more important)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    
    bars = ax.barh(range(len(top_features)), top_features['Importance'].values, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index, fontsize=11)
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['Importance'] + 0.5, i, f"{row['Importance']:.2f}", 
                va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Global Importance Score\n(Average Impact Across All Targets)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Influential Features for Overall Performance\n(Target Leakage Excluded)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add annotation
    ax.text(0.98, 0.02, 'Higher score = More influence on overall performance', 
            transform=ax.transAxes, fontsize=9, ha='right', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'top_features_overall_importance_{timestamp}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved: {filepath}")
    return filepath


def plot_importance_heatmap(importance_df, global_importance, output_dir, top_features=15):
    """Heatmap showing how top features impact different targets."""
    
    # Get top features by global importance
    top_feat_names = global_importance.head(top_features).index.tolist()
    
    # Filter matrix for top features
    heatmap_data = importance_df[top_feat_names].T
    
    # Normalize by row (each feature) for better visualization
    heatmap_data_norm = heatmap_data.div(heatmap_data.max(axis=1), axis=0)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create heatmap manually
    im = ax.imshow(heatmap_data_norm.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(heatmap_data_norm.columns)))
    ax.set_yticks(np.arange(len(heatmap_data_norm.index)))
    ax.set_xticklabels(heatmap_data_norm.columns)
    ax.set_yticklabels(heatmap_data_norm.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Importance\n(1.0 = Highest impact for that feature)', fontsize=10)
    
    ax.set_xlabel('Target Variables', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features (Ranked by Global Importance)', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Impact Across Different Performance Metrics\n(Top {top_features} Most Influential Features)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add annotation
    fig.text(0.99, 0.01, 'Darker color = Stronger influence on that target', 
             fontsize=9, ha='right', style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'feature_impact_heatmap_{timestamp}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved: {filepath}")
    return filepath


def plot_progression_dominance(global_importance, output_dir):
    """Highlight why progression metrics dominate."""
    
    # Categorize features
    progression_features = [f for f in global_importance.index if 'Prg' in f or 'Carries' in f]
    playing_time_features = [f for f in global_importance.index if 'Playing_Time' in f or '90s' in f]
    xg_features = [f for f in global_importance.index if 'xG' in f or 'xAG' in f or 'npxG' in f]
    defensive_features = [f for f in global_importance.index if 'Tkl' in f or 'Blocks' in f or 'Int' in f or 'Aerial' in f]
    other_features = [f for f in global_importance.index 
                     if f not in progression_features + playing_time_features + xg_features + defensive_features]
    
    categories = {
        'Progression\nMetrics': progression_features,
        'Playing\nTime': playing_time_features,
        'Expected Goals\n(xG/xAG)': xg_features,
        'Defensive\nActions': defensive_features,
        'Other\nStats': other_features
    }
    
    category_importance = {}
    for cat_name, features in categories.items():
        if features:
            category_importance[cat_name] = global_importance.loc[features, 'Importance'].sum()
        else:
            category_importance[cat_name] = 0
    
    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Pie chart
    colors_pie = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#dfe6e9']
    explode = [0.1 if cat == 'Progression\nMetrics' else 0 for cat in category_importance.keys()]
    
    wedges, texts, autotexts = ax1.pie(
        category_importance.values(),
        labels=category_importance.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=colors_pie,
        explode=explode,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    ax1.set_title('Total Importance by Feature Category\n(Why Progression Metrics Dominate)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Bar chart showing top features with categories
    top_n = 12
    top_features = global_importance.head(top_n)
    
    feature_categories = []
    for feat in top_features.index:
        if feat in progression_features:
            feature_categories.append('Progression')
        elif feat in playing_time_features:
            feature_categories.append('Playing Time')
        elif feat in xg_features:
            feature_categories.append('xG/xAG')
        elif feat in defensive_features:
            feature_categories.append('Defensive')
        else:
            feature_categories.append('Other')
    
    color_map = {
        'Progression': '#ff6b6b',
        'Playing Time': '#4ecdc4',
        'xG/xAG': '#45b7d1',
        'Defensive': '#96ceb4',
        'Other': '#dfe6e9'
    }
    
    bar_colors = [color_map[cat] for cat in feature_categories]
    
    bars = ax2.barh(range(len(top_features)), top_features['Importance'].values, color=bar_colors)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features.index, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel('Global Importance', fontsize=11, fontweight='bold')
    ax2.set_title(f'Top {top_n} Features (Colored by Category)', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[cat], label=cat) for cat in color_map.keys()]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'progression_dominance_{timestamp}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved: {filepath}")
    return filepath


def plot_versatility_chart(importance_df, global_importance, output_dir, top_n=10):
    """Show which features are versatile (affect many targets) vs specialist."""
    
    top_features = global_importance.head(top_n).index
    
    # Calculate versatility: how many targets does each feature significantly affect?
    # "Significantly" = above 10% of feature's max importance
    versatility_scores = {}
    for feat in top_features:
        feat_importances = importance_df[feat]
        threshold = feat_importances.max() * 0.1
        num_targets_affected = (feat_importances > threshold).sum()
        versatility_scores[feat] = num_targets_affected
    
    # Create scatter plot: Global Importance vs Versatility
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = [versatility_scores[f] for f in top_features]
    y = [global_importance.loc[f, 'Importance'] for f in top_features]
    
    scatter = ax.scatter(x, y, s=200, c=y, cmap='viridis', alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add labels for each point
    for i, feat in enumerate(top_features):
        ax.annotate(feat, (x[i], y[i]), fontsize=9, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('Versatility Score\n(Number of Targets Significantly Affected)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Global Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Versatility vs Global Importance\n(Top {top_n} Features)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Importance', fontsize=10)
    
    # Add quadrant lines
    ax.axvline(x=np.median(x), color='red', linestyle='--', alpha=0.3, linewidth=2)
    ax.axhline(y=np.median(y), color='red', linestyle='--', alpha=0.3, linewidth=2)
    
    # Add text annotations for quadrants
    ax.text(0.95, 0.95, 'High Impact\nHigh Versatility\n(BEST)', 
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'versatility_chart_{timestamp}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved: {filepath}")
    return filepath


def create_summary_infographic(global_importance, output_dir):
    """Create a clean infographic-style summary."""
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('WHAT STATS SHOULD PLAYERS KEEP HIGH FOR OVERALL PERFORMANCE?', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Top 3 features - Large display
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    top_3 = global_importance.head(3)
    medals = ['🥇', '🥈', '🥉']
    y_positions = [0.7, 0.4, 0.1]
    
    for i, (feat, row) in enumerate(top_3.iterrows()):
        ax1.text(0.5, y_positions[i], 
                f"{medals[i]} #{i+1}: {feat}\nImportance: {row['Importance']:.2f}", 
                fontsize=16, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=1', facecolor=['gold', 'silver', '#cd7f32'][i], alpha=0.3))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('Top 3 Most Influential Stats', fontsize=14, fontweight='bold', pad=20)
    
    # Why these matter
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    
    why_text = """WHY PROGRESSION METRICS?

✓ Predicts attacking output
✓ Predicts creativity/assists  
✓ Predicts defensive positioning
✓ Fundamental to how soccer works

Moving the ball forward is the
core action that enables all
other performance metrics."""
    
    ax2.text(0.5, 0.5, why_text, fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Top 10 list
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    top_10 = global_importance.head(10)
    list_text = "TOP 10 OVERALL:\n\n"
    for i, (feat, row) in enumerate(top_10.iterrows(), 1):
        short_name = feat.replace('_', ' ').replace('/', '|')[:25]
        list_text += f"{i:2d}. {short_name:25s}\n"
    
    ax3.text(0.5, 0.5, list_text, fontsize=10, ha='center', va='center', family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.3))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Key insight
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    insight_text = """KEY INSIGHT FOR PLAYERS:

To maximize overall performance across all dimensions (attacking, defending, creating), 
focus on PROGRESSIVE ACTIONS - moving the ball forward through passes and carries.

This has the broadest positive impact on all performance metrics."""
    
    ax4.text(0.5, 0.5, insight_text, fontsize=13, ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.4))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'summary_infographic_{timestamp}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  [OK] Saved: {filepath}")
    return filepath


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("[START] VISUALIZING OVERALL FEATURE IMPORTANCE")
    print("="*80)
    
    model_name = 'xgboost'
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, 'outputs', 'shap_analysis', model_name)
    
    # Load data
    importance_df, global_importance = load_corrected_importance(model_name)
    
    if importance_df is None:
        return
    
    print("\n[GENERATING] Creating visualizations...\n")
    
    # Generate all visualizations
    viz_files = []
    
    print("[1/5] Top features bar chart...")
    viz_files.append(plot_top_features_bar(global_importance, output_dir, top_n=15))
    
    print("[2/5] Impact heatmap...")
    viz_files.append(plot_importance_heatmap(importance_df, global_importance, output_dir, top_features=15))
    
    print("[3/5] Progression dominance analysis...")
    viz_files.append(plot_progression_dominance(global_importance, output_dir))
    
    print("[4/5] Versatility chart...")
    viz_files.append(plot_versatility_chart(importance_df, global_importance, output_dir, top_n=10))
    
    print("[5/5] Summary infographic...")
    viz_files.append(create_summary_infographic(global_importance, output_dir))
    
    print("\n" + "="*80)
    print("[COMPLETE] ALL VISUALIZATIONS CREATED")
    print("="*80)
    print(f"\nGenerated {len(viz_files)} visualizations:")
    for vf in viz_files:
        print(f"  - {os.path.basename(vf)}")
    
    print(f"\n[OUTPUT] All files saved to: {output_dir}/")
    print("\n[ANSWER] The most influential stat for overall performance:")
    top_1 = global_importance.index[0]
    top_1_score = global_importance.iloc[0]['Importance']
    print(f"         {top_1} (Importance: {top_1_score:.2f})")
    print("\n         This feature has the broadest impact across all performance metrics.")


if __name__ == '__main__':
    main()
