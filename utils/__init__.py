"""Public API του utils package.

Re-exports από το `utils.helpers` ώστε στα notebooks να μπορούμε να γράφουμε
`from utils import load_clean_cached, RANDOM_STATE` αντί για το πιο μακρύ
`from utils.helpers import ...`.
"""

from utils.helpers import (
    RANDOM_STATE,
    DATA_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    CLEAN_PARQUET,
    load_cic_ids_2017,
    load_clean_cached,
    save_clean_cached,
    handle_infinities,
    drop_duplicates_report,
    subsample_to_target,
    kmeans_result_is_pathological,
    save_figure,
    save_results,
    top_correlated_pairs,
    find_low_variance_features,
    find_highly_correlated_features,
    # Q2 classification helpers
    evaluate_classifier,
    manual_grid_search,
    plot_confusion,
    plot_feature_importance,
    plot_lr_coefficients,
    plot_decision_tree,
    plot_one_forest_tree,
    # Q3 clustering helpers
    evaluate_clustering,
    plot_elbow_silhouette,
    plot_k_distance,
    plot_pca_scatter_comparison,
    plot_cluster_label_heatmap,
    plot_dendrogram,
)

__all__ = [
    "RANDOM_STATE",
    "DATA_DIR",
    "FIGURES_DIR",
    "RESULTS_DIR",
    "CLEAN_PARQUET",
    "load_cic_ids_2017",
    "load_clean_cached",
    "save_clean_cached",
    "handle_infinities",
    "drop_duplicates_report",
    "subsample_to_target",
    "kmeans_result_is_pathological",
    "save_figure",
    "save_results",
    "top_correlated_pairs",
    "find_low_variance_features",
    "find_highly_correlated_features",
    "evaluate_classifier",
    "manual_grid_search",
    "plot_confusion",
    "plot_feature_importance",
    "plot_lr_coefficients",
    "plot_decision_tree",
    "plot_one_forest_tree",
    "evaluate_clustering",
    "plot_elbow_silhouette",
    "plot_k_distance",
    "plot_pca_scatter_comparison",
    "plot_cluster_label_heatmap",
    "plot_dendrogram",
]
