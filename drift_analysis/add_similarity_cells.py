"""
Script to add similarity analysis cells to the drift analysis visualization notebook.
Run this to append similarity analysis sections to the notebook.
"""

import json
from pathlib import Path

# Path to notebook
notebook_path = Path('../notebooks/drift_analysis_visualization.ipynb')

# New cells to add
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## Visualization 7: Stereotype Similarity Analysis\n\n### Response Similarity to Original Stereotypes\n\nThis section analyzes how similar bias-injected responses are to the original stereotype sentences using HEARTS scores."
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "# Filter entries with similarity scores\nif 'similarity_stereotype_score' in valid_df.columns:\n    similarity_df = valid_df[valid_df['similarity_hearts_available'] == True].copy()\n    print(f\"Entries with similarity scores: {len(similarity_df)}\")\n    \n    fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n    \n    # 1. Distribution of stereotype similarity scores\n    axes[0, 0].hist(similarity_df['similarity_stereotype_score'], bins=40, \n                    edgecolor='black', alpha=0.7, color='mediumpurple')\n    axes[0, 0].axvline(0.5, color='red', linestyle='--', linewidth=2, \n                       label='Classification threshold (0.5)')\n    axes[0, 0].set_xlabel('HEARTS Stereotype Score (Response)')\n    axes[0, 0].set_ylabel('Count')\n    axes[0, 0].set_title('Distribution of Response Stereotype Scores', fontweight='bold')\n    axes[0, 0].legend()\n    axes[0, 0].grid(alpha=0.3)\n    \n    # Add statistics\n    mean_sim = similarity_df['similarity_stereotype_score'].mean()\n    stereo_rate = similarity_df['similarity_is_stereotype'].mean() if 'similarity_is_stereotype' in similarity_df.columns else 0\n    stats_text = f\"Mean: {mean_sim:.3f}\\\\nStereotype Rate: {stereo_rate:.1%}\"\n    axes[0, 0].text(0.98, 0.97, stats_text, transform=axes[0, 0].transAxes,\n                    verticalalignment='top', horizontalalignment='right',\n                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\n    \n    # 2. Stereotype score vs drift score\n    if 'drift_drift_score' in similarity_df.columns:\n        axes[0, 1].scatter(similarity_df['drift_drift_score'], \n                          similarity_df['similarity_stereotype_score'],\n                          alpha=0.4, s=20, c=similarity_df['drift_significant_drift'].map({True: 'red', False: 'blue'}))\n        axes[0, 1].set_xlabel('HEARTS Drift Score')\n        axes[0, 1].set_ylabel('Response Stereotype Score')\n        axes[0, 1].set_title('Drift vs Stereotype Similarity', fontweight='bold')\n        axes[0, 1].axvline(0.1, color='red', linestyle='--', linewidth=1, alpha=0.5)\n        axes[0, 1].axhline(0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)\n        axes[0, 1].grid(alpha=0.3)\n        \n        # Calculate correlation\n        corr = similarity_df['drift_drift_score'].corr(similarity_df['similarity_stereotype_score'])\n        axes[0, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', \n                       transform=axes[0, 1].transAxes,\n                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))\n    \n    # 3. Stereotype score by model\n    if 'model_id_short' in similarity_df.columns:\n        model_sim_stats = similarity_df.groupby('model_id_short')['similarity_stereotype_score'].mean().sort_values(ascending=False)\n        model_sim_stats.plot(kind='barh', ax=axes[1, 0], color='mediumpurple', edgecolor='black')\n        axes[1, 0].set_xlabel('Mean Response Stereotype Score')\n        axes[1, 0].set_ylabel('Model')\n        axes[1, 0].set_title('Mean Stereotype Score by Model', fontweight='bold')\n        axes[1, 0].axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)\n        axes[1, 0].grid(alpha=0.3, axis='x')\n    \n    # 4. Stereotype detection rate by model\n    if 'similarity_is_stereotype' in similarity_df.columns and 'model_id_short' in similarity_df.columns:\n        stereo_rate_by_model = similarity_df.groupby('model_id_short')['similarity_is_stereotype'].mean().sort_values(ascending=False)\n        stereo_rate_by_model.plot(kind='barh', ax=axes[1, 1], color='coral', edgecolor='black')\n        axes[1, 1].set_xlabel('Stereotype Detection Rate')\n        axes[1, 1].set_ylabel('Model')\n        axes[1, 1].set_title('Stereotype Detection Rate by Model', fontweight='bold')\n        axes[1, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))\n        axes[1, 1].grid(alpha=0.3, axis='x')\n    \n    plt.tight_layout()\n    plt.savefig(RESULTS_DIR / 'stereotype_similarity_analysis.png', dpi=300, bbox_inches='tight')\n    plt.show()\n    \n    print(\"[SUCCESS] Saved: stereotype_similarity_analysis.png\")\nelse:\n    print(\"[WARNING] Similarity scores not available\")"
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## Visualization 8: Trait Mention Analysis\n\nAnalyzing whether the stereotypical trait is explicitly mentioned in responses."
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "# Trait mention analysis\nif 'similarity_trait_mentioned' in valid_df.columns:\n    fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n    \n    # 1. Overall trait mention rate\n    trait_mention_rate = valid_df['similarity_trait_mentioned'].mean()\n    labels = ['Trait Mentioned', 'Trait Not Mentioned']\n    sizes = [trait_mention_rate, 1 - trait_mention_rate]\n    colors = ['#ff9999', '#66b3ff']\n    explode = (0.1, 0)\n    \n    axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',\n               shadow=True, startangle=90)\n    axes[0].set_title('Overall Trait Mention Rate', fontweight='bold', pad=20)\n    \n    # 2. Trait mention rate by model\n    if 'model_id_short' in valid_df.columns:\n        trait_by_model = valid_df.groupby('model_id_short')['similarity_trait_mentioned'].mean().sort_values(ascending=False)\n        trait_by_model.plot(kind='barh', ax=axes[1], color='lightcoral', edgecolor='black')\n        axes[1].set_xlabel('Trait Mention Rate')\n        axes[1].set_ylabel('Model')\n        axes[1].set_title('Trait Mention Rate by Model', fontweight='bold')\n        axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))\n        axes[1].grid(alpha=0.3, axis='x')\n    \n    plt.tight_layout()\n    plt.savefig(RESULTS_DIR / 'trait_mention_analysis.png', dpi=300, bbox_inches='tight')\n    plt.show()\n    \n    print(\"[SUCCESS] Saved: trait_mention_analysis.png\")\n    print(f\"\\\\nOverall trait mention rate: {trait_mention_rate:.2%}\")\nelse:\n    print(\"[WARNING] Trait mention data not available\")"
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## Visualization 9: Semantic Similarity Metrics\n\nCosine similarity between responses and original stereotype sentences."
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "# Semantic similarity analysis\nif 'similarity_semantic_similarity' in valid_df.columns:\n    fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n    \n    # 1. Distribution of semantic similarity\n    axes[0].hist(valid_df['similarity_semantic_similarity'].dropna(), bins=40, \n                edgecolor='black', alpha=0.7, color='skyblue')\n    axes[0].set_xlabel('Semantic Similarity (Cosine)')\n    axes[0].set_ylabel('Count')\n    axes[0].set_title('Distribution of Semantic Similarity to Stereotype', fontweight='bold')\n    axes[0].grid(alpha=0.3)\n    \n    # Add statistics\n    mean_sem = valid_df['similarity_semantic_similarity'].mean()\n    median_sem = valid_df['similarity_semantic_similarity'].median()\n    stats_text = f\"Mean: {mean_sem:.3f}\\\\nMedian: {median_sem:.3f}\"\n    axes[0].text(0.98, 0.97, stats_text, transform=axes[0].transAxes,\n                verticalalignment='top', horizontalalignment='right',\n                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\n    \n    # 2. Semantic similarity by bias type\n    if 'bias_type' in valid_df.columns:\n        valid_df.boxplot(column='similarity_semantic_similarity', by='bias_type', ax=axes[1])\n        axes[1].set_xlabel('Cognitive Bias Type')\n        axes[1].set_ylabel('Semantic Similarity')\n        axes[1].set_title('Semantic Similarity by Bias Type', fontweight='bold')\n        axes[1].get_figure().suptitle('')\n        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')\n        axes[1].grid(alpha=0.3)\n    \n    plt.tight_layout()\n    plt.savefig(RESULTS_DIR / 'semantic_similarity_analysis.png', dpi=300, bbox_inches='tight')\n    plt.show()\n    \n    print(\"[SUCCESS] Saved: semantic_similarity_analysis.png\")\nelse:\n    print(\"[WARNING] Semantic similarity data not available\")"
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## Visualization 10: Stereotype Reinforcement Score\n\nComposite score combining trait mention and semantic similarity."
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "# Stereotype reinforcement score analysis\nif 'similarity_stereotype_reinforcement_score' in valid_df.columns:\n    fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n    \n    # 1. Distribution of reinforcement scores\n    axes[0, 0].hist(valid_df['similarity_stereotype_reinforcement_score'].dropna(), \n                   bins=40, edgecolor='black', alpha=0.7, color='tomato')\n    axes[0, 0].set_xlabel('Stereotype Reinforcement Score')\n    axes[0, 0].set_ylabel('Count')\n    axes[0, 0].set_title('Distribution of Stereotype Reinforcement Scores', fontweight='bold')\n    axes[0, 0].grid(alpha=0.3)\n    \n    # Add statistics\n    mean_reinf = valid_df['similarity_stereotype_reinforcement_score'].mean()\n    median_reinf = valid_df['similarity_stereotype_reinforcement_score'].median()\n    stats_text = f\"Mean: {mean_reinf:.3f}\\\\nMedian: {median_reinf:.3f}\"\n    axes[0, 0].text(0.98, 0.97, stats_text, transform=axes[0, 0].transAxes,\n                   verticalalignment='top', horizontalalignment='right',\n                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\n    \n    # 2. Reinforcement score by model\n    if 'model_id_short' in valid_df.columns:\n        model_reinf = valid_df.groupby('model_id_short')['similarity_stereotype_reinforcement_score'].mean().sort_values(ascending=False)\n        model_reinf.plot(kind='barh', ax=axes[0, 1], color='tomato', edgecolor='black')\n        axes[0, 1].set_xlabel('Mean Reinforcement Score')\n        axes[0, 1].set_ylabel('Model')\n        axes[0, 1].set_title('Mean Stereotype Reinforcement by Model', fontweight='bold')\n        axes[0, 1].grid(alpha=0.3, axis='x')\n    \n    # 3. Reinforcement score by stereotype type\n    if 'emgsd_stereotype_type' in valid_df.columns:\n        valid_df.boxplot(column='similarity_stereotype_reinforcement_score', \n                        by='emgsd_stereotype_type', ax=axes[1, 0])\n        axes[1, 0].set_xlabel('Stereotype Type')\n        axes[1, 0].set_ylabel('Reinforcement Score')\n        axes[1, 0].set_title('Reinforcement Score by Stereotype Type', fontweight='bold')\n        axes[1, 0].get_figure().suptitle('')\n        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')\n        axes[1, 0].grid(alpha=0.3)\n    \n    # 4. Reinforcement score vs drift score\n    if 'drift_drift_score' in valid_df.columns:\n        axes[1, 1].scatter(valid_df['drift_drift_score'], \n                          valid_df['similarity_stereotype_reinforcement_score'],\n                          alpha=0.4, s=20, color='tomato')\n        axes[1, 1].set_xlabel('HEARTS Drift Score')\n        axes[1, 1].set_ylabel('Stereotype Reinforcement Score')\n        axes[1, 1].set_title('Drift vs Stereotype Reinforcement', fontweight='bold')\n        axes[1, 1].grid(alpha=0.3)\n        \n        # Calculate correlation\n        corr = valid_df[['drift_drift_score', 'similarity_stereotype_reinforcement_score']].corr().iloc[0, 1]\n        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}', \n                       transform=axes[1, 1].transAxes,\n                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))\n    \n    plt.tight_layout()\n    plt.savefig(RESULTS_DIR / 'stereotype_reinforcement_analysis.png', dpi=300, bbox_inches='tight')\n    plt.show()\n    \n    print(\"[SUCCESS] Saved: stereotype_reinforcement_analysis.png\")\nelse:\n    print(\"[WARNING] Stereotype reinforcement data not available\")"
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": "## Similarity Score Summary Statistics"
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "# Calculate comprehensive similarity statistics\nprint(\"\\\\n\" + \"=\"*70)\nprint(\"SIMILARITY SCORE SUMMARY STATISTICS\")\nprint(\"=\"*70)\n\nsimilarity_cols = [\n    'similarity_stereotype_score',\n    'similarity_semantic_similarity', \n    'similarity_stereotype_reinforcement_score',\n    'similarity_trait_mentioned',\n    'similarity_is_stereotype'\n]\n\navailable_cols = [col for col in similarity_cols if col in valid_df.columns]\n\nif available_cols:\n    similarity_stats = valid_df[available_cols].describe()\n    print(\"\\\\nOverall Statistics:\")\n    display(similarity_stats.round(4))\n    \n    # By model\n    if 'model_id_short' in valid_df.columns:\n        print(\"\\\\nBy Model:\")\n        model_sim_stats = valid_df.groupby('model_id_short')[available_cols].mean()\n        display(model_sim_stats.round(4))\n    \n    # By bias type\n    if 'bias_type' in valid_df.columns:\n        print(\"\\\\nBy Bias Type:\")\n        bias_sim_stats = valid_df.groupby('bias_type')[available_cols].mean()\n        display(bias_sim_stats.round(4))\nelse:\n    print(\"[WARNING] No similarity score columns available\")"
    }
]

# Load existing notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the position to insert (after ANOVA cell, before "Top Examples")
insert_position = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'Top Examples: Highest and Lowest Drift' in source:
            insert_position = i
            break

if insert_position is None:
    # If not found, append at the end
    insert_position = len(notebook['cells'])

# Insert new cells
for i, new_cell in enumerate(new_cells):
    notebook['cells'].insert(insert_position + i, new_cell)

# Save updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"[SUCCESS] Added {len(new_cells)} similarity analysis cells to the notebook")
print(f"[SUCCESS] Updated: {notebook_path}")
