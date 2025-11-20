"""pXRF Unsupervised Machine Learning Pipeline with Labeled Data

This module implements a complete unsupervised machine learning pipeline for analyzing
portable X-Ray Fluorescence (pXRF) geochemical data. The pipeline combines:
  1. Deep learning autoencoder for dimensionality reduction
  2. HDBSCAN clustering for pattern discovery
  3. Supervised classification using Random Forest

The pipeline is designed for labeled datasets where material types are known,
enabling evaluation of clustering quality and training of classifiers.

Typical Workflow:
    1. Load pXRF data from CSV with element concentrations and material labels
    2. Standardize features and split into train/test sets
    3. Train autoencoder to learn compact latent representations
    4. Apply HDBSCAN clustering in latent space
    5. Evaluate clustering against true labels
    6. Train binary and multiclass classifiers
    7. Compare with baseline methods (PCA+HDBSCAN, KMeans)
    8. Generate comprehensive visualizations and reports

Example:
    $ python "Autoencoder HDBscan Labeled.py" --csv data/AllPXRF_FINAL_14Oct.csv \
        --out "Dr Sharp Results" --latent 8 --epochs 100

Author: CS472 Final Project
Date: November 2025
"""

import argparse
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, 
                             silhouette_score, classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import hdbscan
import seaborn as sns

# configuration
default_elements = ['Ag','Al','As','Au','Ca','Cr','Cu','Fe','K','Mg','Mn','Ni','P','Pb','Sr','Ti','V','Zn']
seed = 42

# ============================================================================
# Helper Functions
# ============================================================================

def ensure_dir(p):
    """Create a directory if it doesn't exist.
    
    Args:
        p (str or Path): Path to the directory to create.
        
    Returns:
        None
        
    Example:
        >>> ensure_dir('results/experiment1')
        # Creates results/experiment1/ directory if it doesn't exist
    """
    Path(p).mkdir(parents=True, exist_ok=True)

def load_and_prepare(csv_path, element_cols=None, fillna_zero=True):
    """Load and preprocess pXRF data from CSV file.
    
    This function loads geochemical element concentration data, cleans it,
    and extracts material labels and spatial coordinates if available.
    
    Args:
        csv_path (str): Path to input CSV file containing pXRF measurements.
        element_cols (list, optional): List of element column names to use as features.
            If None, uses default_elements list. Defaults to None.
        fillna_zero (bool, optional): Whether to fill missing values with 0. Defaults to True.
        
    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): Feature matrix of shape (n_samples, n_features)
            - y (np.ndarray): Material type labels (e.g., 'soil', 'ceramic', 'obsidian')
            - y_binary (np.ndarray): Binary labels ('soil' vs 'non-soil')
            - element_cols (list): List of element column names used
            - coords (pd.DataFrame or None): X_Coord and Y_Coord if available
            
    Data Cleaning:
        - Missing values are filled with 0 (if fillna_zero=True)
        - Negative values are set to 0
        - Unknown/blank material labels are replaced with 'soil'
        
    Example:
        >>> X, y, y_bin, cols, coords = load_and_prepare('data.csv')
        >>> print(X.shape)  # (1000, 18) - 1000 samples, 18 elements
        >>> print(cols)     # ['Ag', 'Al', 'As', ...]
    """
    print('step 1: load csv')
    df = pd.read_csv(csv_path)
    if element_cols is None:
        element_cols = [c for c in default_elements if c in df.columns]
    print(f'using element columns: {element_cols}')
    X = df[element_cols].copy()
    if fillna_zero:
        X = X.fillna(0)
    X[X < 0] = 0
    y = df.get('material')
    if y is None:
        y = pd.Series(['unknown'] * len(X))
    else:
        y = y.fillna('soil')
        y = y.replace(['blank','unknown',''], 'soil')
    
    y_binary = pd.Series(['soil' if mat == 'soil' else 'non-soil' for mat in y])
    
    coords = None
    if 'X_Coord' in df.columns and 'Y_Coord' in df.columns:
        coords = df[['X_Coord','Y_Coord']]
    return X.values.astype(float), y.values, y_binary.values, element_cols, coords

def build_autoencoder(input_dim, latent_dim, hidden_dims=(64,32), lr=1e-3):
    """Build a deep autoencoder neural network for dimensionality reduction.
    
    Creates a symmetric encoder-decoder architecture that learns to compress
    high-dimensional geochemical data into a lower-dimensional latent space
    while preserving important patterns.
    
    Architecture:
        Encoder: input -> hidden_dims[0] -> hidden_dims[1] -> latent_dim
        Decoder: latent_dim -> hidden_dims[1] -> hidden_dims[0] -> input_dim
        
    Args:
        input_dim (int): Number of input features (e.g., 18 for 18 elements).
        latent_dim (int): Dimensionality of latent space (compressed representation).
        hidden_dims (tuple, optional): Sizes of hidden layers. Defaults to (64, 32).
        lr (float, optional): Learning rate for Adam optimizer. Defaults to 1e-3.
        
    Returns:
        tuple: A tuple containing:
            - ae (keras.Model): Complete autoencoder model for training
            - encoder (keras.Model): Encoder portion for generating latent representations
            
    Example:
        >>> ae, encoder = build_autoencoder(input_dim=18, latent_dim=8)
        >>> ae.summary()  # Shows architecture
        >>> # Train: ae.fit(X_train, X_train, epochs=100)
        >>> # Encode: Z = encoder.predict(X_test)
    """
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for h in hidden_dims:
        x = layers.Dense(h, activation='relu')(x)
    z = layers.Dense(latent_dim, activation=None, name='latent')(x)
    x = z
    for h in reversed(hidden_dims):
        x = layers.Dense(h, activation='relu')(x)
    out = layers.Dense(input_dim, activation='linear')(x)
    ae = models.Model(inp, out, name='autoencoder')
    encoder = models.Model(inp, z, name='encoder')
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return ae, encoder

# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(csv_path, out_dir='Dr Sharp Results', latent_dim=8, epochs=100, batch_size=32, run_baselines=True, use_pca_for_viz=True, test_size=0.2):
    """Execute the complete unsupervised learning pipeline for pXRF data analysis.
    
    This is the main function that orchestrates the entire workflow:
      1. Data loading and preprocessing
      2. Train/test split with stratification
      3. Feature standardization
      4. Autoencoder training for dimensionality reduction
      5. HDBSCAN clustering in latent space
      6. Cluster evaluation (ARI, NMI, Silhouette)
      7. Binary and multiclass classification
      8. Baseline comparisons (PCA+HDBSCAN, KMeans)
      9. Visualization generation
      10. Results export
      
    Args:
        csv_path (str): Path to input CSV file with pXRF element data and labels.
        out_dir (str, optional): Output directory for results. Defaults to 'Dr Sharp Results'.
        latent_dim (int, optional): Dimensionality of latent space. Defaults to 8.
        epochs (int, optional): Number of training epochs for autoencoder. Defaults to 100.
        batch_size (int, optional): Batch size for autoencoder training. Defaults to 32.
        run_baselines (bool, optional): Whether to run baseline methods. Defaults to True.
        use_pca_for_viz (bool, optional): Use PCA for 2D visualization. Defaults to True.
        test_size (float, optional): Fraction of data for testing (0.0-1.0). Defaults to 0.2.
        
    Returns:
        dict: Dictionary containing evaluation metrics:
            - silhouette_latent_train/test: Silhouette scores
            - ARI_train/test: Adjusted Rand Index
            - NMI_train/test: Normalized Mutual Information
            - cluster_counts_train/test: Number of samples per cluster
            - noisy_fraction_train/test: Fraction of noise points
            - majority_vote_accuracy_train/test: Accuracy of cluster-to-material mapping
            - binary_classification: Binary classifier metrics
            - multiclass_classification: Multiclass classifier metrics
            
    Output Files Created:
        Models:
            - autoencoder.keras: Trained autoencoder model
            - encoder.keras: Encoder portion only
            - scaler.pkl: StandardScaler for feature normalization
            - hdbscan_model.pkl: Fitted HDBSCAN clusterer
            - binary_classifier.pkl: Binary Random Forest classifier
            - multiclass_classifier.pkl: Multiclass Random Forest classifier
            
        Data:
            - latent_train.npy, latent_test.npy: Latent representations
            - hdbscan_labels_train.csv, hdbscan_labels_test.csv: Cluster assignments
            - cluster_assignments_train.csv, cluster_assignments_test.csv: Full results
            - cluster_to_material.json: Cluster ID to material type mapping
            
        Visualizations:
            - training_loss.png: Autoencoder training curves
            - latent_pca_hdbscan_train.png: 2D PCA visualization of clusters (train)
            - latent_pca_hdbscan_test.png: 2D PCA visualization of clusters (test)
            - confusion_matrix_binary.png: Binary classification confusion matrix
            - confusion_matrix_multiclass.png: Multiclass confusion matrix
            
        Metrics:
            - hdbscan_results.json: Comprehensive clustering and classification metrics
            - baselines.json: Baseline method comparison results
            
    Example:
        >>> results = run_pipeline(
        ...     csv_path='data/pxrf_data.csv',
        ...     out_dir='experiment_results',
        ...     latent_dim=8,
        ...     epochs=100,
        ...     test_size=0.2
        ... )
        >>> print(f"Test ARI: {results['ARI_test']:.3f}")
        >>> print(f"Binary Accuracy: {results['binary_classification']['accuracy']:.3f}")
        
    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If CSV doesn't contain required element columns.
    """
    ensure_dir(out_dir)
    X_raw, y_raw, y_binary, element_cols, coords = load_and_prepare(csv_path)
    n_samples, n_features = X_raw.shape
    print(f'step 2: preprocess - {n_samples} samples, {n_features} features')
    
    print(f'step 2.5: train-test split ({int((1-test_size)*100)}-{int(test_size*100)})')
    X_train, X_test, y_train, y_test, y_binary_train, y_binary_test, idx_train, idx_test = train_test_split(
        X_raw, y_raw, y_binary, np.arange(len(X_raw)), 
        test_size=test_size, random_state=seed, stratify=y_binary
    )
    
    coords_train = coords.iloc[idx_train] if coords is not None else None
    coords_test = coords.iloc[idx_test] if coords is not None else None
    
    print(f'train: {len(X_train)} samples, test: {len(X_test)} samples')
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    with open(Path(out_dir)/'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print('step 3: build and train autoencoder')
    ae, encoder = build_autoencoder(n_features, latent_dim)
    history = ae.fit(X_train_scaled, X_train_scaled, validation_split=0.1, epochs=epochs, batch_size=batch_size,
                     callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=2)
    ae.save(Path(out_dir)/'autoencoder.keras')
    encoder.save(Path(out_dir)/'encoder.keras')

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='training loss', linewidth=2)
    plt.plot(history.history.get('val_loss', []), label='validation loss', linewidth=2)
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('mse loss', fontsize=12)
    plt.legend(fontsize=11, loc='upper right')
    plt.title('autoencoder training progress', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/'training_loss.png', dpi=300)
    plt.close()

    print('step 4: encode to latent space')
    Z_train = encoder.predict(X_train_scaled)
    Z_test = encoder.predict(X_test_scaled)
    np.save(Path(out_dir)/'latent_train.npy', Z_train)
    np.save(Path(out_dir)/'latent_test.npy', Z_test)

    print('step 5: run hdbscan on latent space (train)')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, int(0.01*len(Z_train))), prediction_data=True)
    cluster_labels_train = clusterer.fit_predict(Z_train)
    cluster_labels_test, strengths = hdbscan.approximate_predict(clusterer, Z_test)
    
    with open(Path(out_dir)/'hdbscan_model.pkl', 'wb') as f:
        pickle.dump(clusterer, f)
    
    pd.DataFrame({'cluster': cluster_labels_train}).to_csv(Path(out_dir)/'hdbscan_labels_train.csv', index=False)
    pd.DataFrame({'cluster': cluster_labels_test}).to_csv(Path(out_dir)/'hdbscan_labels_test.csv', index=False)

    print('step 6: evaluation')
    results = {}

    true_labels_train = y_train
    true_labels_test = y_test

    valid_idx_train = cluster_labels_train != -1
    if valid_idx_train.sum() >= 2:
        sil_train = silhouette_score(Z_train[valid_idx_train], cluster_labels_train[valid_idx_train])
    else:
        sil_train = float('nan')
    results['silhouette_latent_train'] = float(np.nan_to_num(sil_train, nan=-1.0))
    
    valid_idx_test = cluster_labels_test != -1
    if valid_idx_test.sum() >= 2:
        sil_test = silhouette_score(Z_test[valid_idx_test], cluster_labels_test[valid_idx_test])
    else:
        sil_test = float('nan')
    results['silhouette_latent_test'] = float(np.nan_to_num(sil_test, nan=-1.0))

    try:
        ari_train = adjusted_rand_score(true_labels_train, cluster_labels_train)
        nmi_train = normalized_mutual_info_score(true_labels_train, cluster_labels_train)
    except Exception:
        ari_train, nmi_train = float('nan'), float('nan')
    results['ARI_train'] = float(np.nan_to_num(ari_train, nan=-1.0))
    results['NMI_train'] = float(np.nan_to_num(nmi_train, nan=-1.0))
    
    try:
        ari_test = adjusted_rand_score(true_labels_test, cluster_labels_test)
        nmi_test = normalized_mutual_info_score(true_labels_test, cluster_labels_test)
    except Exception:
        ari_test, nmi_test = float('nan'), float('nan')
    results['ARI_test'] = float(np.nan_to_num(ari_test, nan=-1.0))
    results['NMI_test'] = float(np.nan_to_num(nmi_test, nan=-1.0))

    unique_train, counts_train = np.unique(cluster_labels_train, return_counts=True)
    cluster_counts_train = dict(zip(map(str, unique_train), counts_train.tolist()))
    results['cluster_counts_train'] = cluster_counts_train
    
    unique_test, counts_test = np.unique(cluster_labels_test, return_counts=True)
    cluster_counts_test = dict(zip(map(str, unique_test), counts_test.tolist()))
    results['cluster_counts_test'] = cluster_counts_test

    noisy_frac_train = np.mean(cluster_labels_train == -1)
    noisy_frac_test = np.mean(cluster_labels_test == -1)
    results['noisy_fraction_train'] = float(noisy_frac_train)
    results['noisy_fraction_test'] = float(noisy_frac_test)

    from collections import Counter, defaultdict
    cluster_to_materials = defaultdict(list)
    for cl, mat in zip(cluster_labels_train, y_train):
        if cl != -1:
            cluster_to_materials[cl].append(mat)
    cluster_map = {}
    for cl, mats in cluster_to_materials.items():
        cluster_map[cl] = Counter(mats).most_common(1)[0][0]
    pred_material_train = []
    for cl in cluster_labels_train:
        if cl == -1:
            pred_material_train.append('noise')
        else:
            pred_material_train.append(cluster_map.get(cl, 'unknown'))
    pred_material_train = np.array(pred_material_train)
    mask_train = cluster_labels_train != -1
    if mask_train.sum() > 0:
        acc_train = float((pred_material_train[mask_train] == y_train[mask_train]).sum() / mask_train.sum())
    else:
        acc_train = float('nan')
    results['majority_vote_accuracy_train'] = acc_train
    
    pred_material_test = []
    for cl in cluster_labels_test:
        if cl == -1:
            pred_material_test.append('noise')
        else:
            pred_material_test.append(cluster_map.get(cl, 'unknown'))
    pred_material_test = np.array(pred_material_test)
    mask_test = cluster_labels_test != -1
    if mask_test.sum() > 0:
        acc_test = float((pred_material_test[mask_test] == y_test[mask_test]).sum() / mask_test.sum())
    else:
        acc_test = float('nan')
    results['majority_vote_accuracy_test'] = acc_test

    with open(Path(out_dir)/'cluster_to_material.json','w') as f:
        safe_map = {str(int(k)): v for k, v in cluster_map.items()}
        json.dump(safe_map, f, indent=2)
    
    # classification tasks
    print('step 6.5: train classifiers on latent representations')
    
    print('  - binary classification (soil vs non-soil)')
    clf_binary = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=10)
    clf_binary.fit(Z_train, y_binary_train)
    y_binary_pred = clf_binary.predict(Z_test)
    
    results['binary_classification'] = {
        'accuracy': float(accuracy_score(y_binary_test, y_binary_pred)),
        'f1_score': float(f1_score(y_binary_test, y_binary_pred, pos_label='non-soil')),
        'classification_report': classification_report(y_binary_test, y_binary_pred, output_dict=True)
    }
    
    with open(Path(out_dir)/'binary_classifier.pkl', 'wb') as f:
        pickle.dump(clf_binary, f)
    
    cm_binary = confusion_matrix(y_binary_test, y_binary_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'number of samples'},
                xticklabels=['non-soil', 'soil'], yticklabels=['non-soil', 'soil'], 
                annot_kws={'size': 14})
    plt.title('confusion matrix: binary classification', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('true material type', fontsize=12, fontweight='bold')
    plt.xlabel('predicted material type', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(out_dir)/'confusion_matrix_binary.png', dpi=300)
    plt.close()
    
    print('  - multi-class classification (all material types)')
    clf_multi = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=10)
    clf_multi.fit(Z_train, y_train)
    y_multi_pred = clf_multi.predict(Z_test)
    
    results['multiclass_classification'] = {
        'accuracy': float(accuracy_score(y_test, y_multi_pred)),
        'f1_score_weighted': float(f1_score(y_test, y_multi_pred, average='weighted')),
        'classification_report': classification_report(y_test, y_multi_pred, output_dict=True)
    }
    
    with open(Path(out_dir)/'multiclass_classifier.pkl', 'wb') as f:
        pickle.dump(clf_multi, f)
    
    cm_multi = confusion_matrix(y_test, y_multi_pred)
    unique_classes = sorted(np.unique(np.concatenate([y_test, y_multi_pred])))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'number of samples'},
                xticklabels=unique_classes, yticklabels=unique_classes, annot_kws={'size': 10})
    plt.title('confusion matrix: multi-class classification', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('true material type', fontsize=12, fontweight='bold')
    plt.xlabel('predicted material type', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/'confusion_matrix_multiclass.png', dpi=300)
    plt.close()
    
    print('\n=== binary classification report ===')
    print(classification_report(y_binary_test, y_binary_pred))
    print('\n=== multi-class classification report ===')
    print(classification_report(y_test, y_multi_pred))

    with open(Path(out_dir)/'hdbscan_results.json','w') as f:
        json.dump(results, f, indent=2)

    # visualization
    if use_pca_for_viz:
        print('step 7: visualization - pca to 2d for plotting')
        pca2 = PCA(n_components=2, random_state=seed)
        Z2_train = pca2.fit_transform(Z_train)
        Z2_test = pca2.transform(Z_test)
        
        df_viz_train = pd.DataFrame({'x': Z2_train[:,0], 'y': Z2_train[:,1], 
                                      'cluster': cluster_labels_train, 'material': y_train})
        plt.figure(figsize=(12,8))
        for cl in np.unique(cluster_labels_train):
            sel = df_viz_train['cluster'] == cl
            label = f'cluster {cl}' if cl!=-1 else 'noise/outliers'
            plt.scatter(df_viz_train.loc[sel,'x'], df_viz_train.loc[sel,'y'], 
                       label=label, s=35, alpha=0.6, edgecolors='black', linewidths=0.5)
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=10, framealpha=0.9)
        plt.title('hdbscan cluster assignments (training set)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('first principal component (pc1)', fontsize=12)
        plt.ylabel('second principal component (pc2)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(out_dir)/'latent_pca_hdbscan_train.png', dpi=300)
        plt.close()
        
        df_viz_test = pd.DataFrame({'x': Z2_test[:,0], 'y': Z2_test[:,1], 
                                     'cluster': cluster_labels_test, 'material': y_test})
        plt.figure(figsize=(12,8))
        for cl in np.unique(cluster_labels_test):
            sel = df_viz_test['cluster'] == cl
            label = f'cluster {cl}' if cl!=-1 else 'noise/outliers'
            plt.scatter(df_viz_test.loc[sel,'x'], df_viz_test.loc[sel,'y'], 
                       label=label, s=35, alpha=0.6, edgecolors='black', linewidths=0.5)
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=10, framealpha=0.9)
        plt.title('hdbscan cluster assignments (test set)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('first principal component (pc1)', fontsize=12)
        plt.ylabel('second principal component (pc2)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(out_dir)/'latent_pca_hdbscan_test.png', dpi=300)
        plt.close()

    # baseline comparisons
    if run_baselines:
        print('step 8: baselines - pca+hdbscan and kmeans')
        baselines = {}
        pca = PCA(n_components=min(20, n_features), random_state=seed)
        X_pca_train = pca.fit_transform(X_train_scaled)
        X_pca_test = pca.transform(X_test_scaled)
        clusterer_b = hdbscan.HDBSCAN(min_cluster_size=max(5, int(0.01*len(X_pca_train))), prediction_data=True)
        labels_pca_hdb_train = clusterer_b.fit_predict(X_pca_train)
        labels_pca_hdb_test, _ = hdbscan.approximate_predict(clusterer_b, X_pca_test)
        baselines['pca_hdbscan_ARI_train'] = float(np.nan_to_num(adjusted_rand_score(y_train, labels_pca_hdb_train), nan=-1.0))
        baselines['pca_hdbscan_NMI_train'] = float(np.nan_to_num(normalized_mutual_info_score(y_train, labels_pca_hdb_train), nan=-1.0))
        baselines['pca_hdbscan_ARI_test'] = float(np.nan_to_num(adjusted_rand_score(y_test, labels_pca_hdb_test), nan=-1.0))
        baselines['pca_hdbscan_NMI_test'] = float(np.nan_to_num(normalized_mutual_info_score(y_test, labels_pca_hdb_test), nan=-1.0))
        
        with open(Path(out_dir)/'pca_hdbscan_model.pkl', 'wb') as f:
            pickle.dump({'pca': pca, 'hdbscan': clusterer_b}, f)

        k = len(np.unique(y_train)) if len(np.unique(y_train))>1 else 5
        kmeans = KMeans(n_clusters=min(k, max(2, len(X_train_scaled)//10)), random_state=seed)
        km_labels_train = kmeans.fit_predict(Z_train)
        km_labels_test = kmeans.predict(Z_test)
        baselines['kmeans_latent_ARI_train'] = float(np.nan_to_num(adjusted_rand_score(y_train, km_labels_train), nan=-1.0))
        baselines['kmeans_latent_NMI_train'] = float(np.nan_to_num(normalized_mutual_info_score(y_train, km_labels_train), nan=-1.0))
        baselines['kmeans_latent_ARI_test'] = float(np.nan_to_num(adjusted_rand_score(y_test, km_labels_test), nan=-1.0))
        baselines['kmeans_latent_NMI_test'] = float(np.nan_to_num(normalized_mutual_info_score(y_test, km_labels_test), nan=-1.0))
        
        with open(Path(out_dir)/'kmeans_model.pkl', 'wb') as f:
            pickle.dump(kmeans, f)

        with open(Path(out_dir)/'baselines.json','w') as f:
            json.dump(baselines, f, indent=2)
            
    print('step 9: write cluster assignment csv')
    out_df_train = pd.DataFrame({'cluster': cluster_labels_train, 'material': y_train, 'material_binary': y_binary_train})
    if coords_train is not None:
        out_df_train = pd.concat([out_df_train.reset_index(drop=True), coords_train.reset_index(drop=True)], axis=1)
    out_df_train.to_csv(Path(out_dir)/'cluster_assignments_train.csv', index=False)
    
    out_df_test = pd.DataFrame({'cluster': cluster_labels_test, 'material': y_test, 'material_binary': y_binary_test,
                                 'binary_pred': y_binary_pred, 'multiclass_pred': y_multi_pred})
    if coords_test is not None:
        out_df_test = pd.concat([out_df_test.reset_index(drop=True), coords_test.reset_index(drop=True)], axis=1)
    out_df_test.to_csv(Path(out_dir)/'cluster_assignments_test.csv', index=False)

    print('pipeline complete. results saved to', out_dir)
    return results

# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pxrf unsupervised pipeline: autoencoder + hdbscan + classification')
    parser.add_argument('--csv', type=str, required=True, help='path to input csv file with pxrf element data')
    parser.add_argument('--out', type=str, default='Dr Sharp Results', help='output directory for results and figures')
    parser.add_argument('--latent', type=int, default=8, help='dimensionality of latent space')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs for autoencoder')
    parser.add_argument('--batch', type=int, default=32, help='batch size for autoencoder training')
    parser.add_argument('--no-baselines', action='store_true', help='skip baseline comparisons')
    parser.add_argument('--test-size', type=float, default=0.2, help='test set fraction')
    args = parser.parse_args()
    run_pipeline(args.csv, out_dir=args.out, latent_dim=args.latent, epochs=args.epochs, 
                 batch_size=args.batch, run_baselines=not args.no_baselines, test_size=args.test_size)