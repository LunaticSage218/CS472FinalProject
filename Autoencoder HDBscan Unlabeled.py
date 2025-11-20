# pxrf unsupervised pipeline: autoencoder + hdbscan + optional classification
import argparse
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, classification_report,
                             confusion_matrix, accuracy_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import hdbscan
import seaborn as sns
from collections import Counter, defaultdict

# configuration
default_elements = [
    'Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Mn','Fe','Co','Ni','Cu','Zn',
    'Ga','Ge','As','Se','Br','Rb','Sr','Y','Zr','Mo','Ba','Pr','Eu','Ta','W','Bi','Po'
]
seed = 42

# helper functions
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_and_prepare(csv_path, feature_cols=None, target_col=None, fillna_zero=True):
    print('step 1: load csv')
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        print("Parser error detected. Skipping malformed lines...")
        df = pd.read_csv(csv_path, on_bad_lines='skip')

    if feature_cols is None:
        feature_cols = default_elements

    # Only keep columns that exist in the CSV
    used_feature_cols = [c for c in feature_cols if c in df.columns]
    print(f'Using feature columns: {used_feature_cols}')

    # Keep only numeric columns
    X = df[used_feature_cols].apply(pd.to_numeric, errors='coerce')

    if fillna_zero:
        X = X.fillna(0)

    # Clip negative values safely
    X = X.clip(lower=0)

    # Extract target labels
    if target_col is not None and target_col in df.columns:
        y = df[target_col].fillna('unknown')
    else:
        y = pd.Series(['unknown'] * len(X))

    y_binary = pd.Series(['soil' if mat == 'soil' else 'non-soil' for mat in y])

    coords = None
    if 'X_Coord' in df.columns and 'Y_Coord' in df.columns:
        coords = df[['X_Coord','Y_Coord']]

    return X.values.astype(float), y.values, y_binary.values, used_feature_cols, coords

def build_autoencoder(input_dim, latent_dim, hidden_dims=(64,32), lr=1e-3):
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

# training + clustering pipeline
def run_pipeline(csv_path, out_dir='results', latent_dim=8, epochs=100, batch_size=32,
                 run_baselines=True, use_pca_for_viz=True, test_size=0.2,
                 feature_cols=None, target_col=None):

    ensure_dir(out_dir)

    # Step 1: load and prepare
    X_raw, y_raw, y_binary, element_cols, coords = load_and_prepare(
        csv_path, feature_cols=feature_cols, target_col=target_col
    )
    has_labels = y_raw is not None
    n_samples, n_features = X_raw.shape
    print(f'step 2: preprocess - {n_samples} samples, {n_features} features')

    # train-test split or full data if no labels
    if has_labels:
        X_train, X_test, y_train, y_test, y_binary_train, y_binary_test, idx_train, idx_test = train_test_split(
            X_raw, y_raw, y_binary, np.arange(len(X_raw)),
            test_size=test_size, random_state=seed, stratify=y_binary
        )
    else:
        X_train = X_test = X_raw
        y_train = y_test = np.array(['unknown']*len(X_raw))
        y_binary_train = y_binary_test = None
        idx_train = idx_test = np.arange(len(X_raw))

    coords_train = coords.iloc[idx_train] if coords is not None else None
    coords_test = coords.iloc[idx_test] if coords is not None else None

    print(f'train: {len(X_train)} samples, test: {len(X_test)} samples')

    # Step 2: scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    with open(Path(out_dir)/'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Step 3: build and train autoencoder
    print('step 3: build and train autoencoder')
    ae, encoder = build_autoencoder(n_features, latent_dim)
    history = ae.fit(
        X_train_scaled, X_train_scaled,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=2
    )
    ae.save(Path(out_dir)/'autoencoder.keras')
    encoder.save(Path(out_dir)/'encoder.keras')

    # Training loss plot
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

        # Step 4: encode to latent space
    print('step 4: encode to latent space')
    Z_train = encoder.predict(X_train_scaled)
    Z_test = encoder.predict(X_test_scaled)
    np.save(Path(out_dir)/'latent_train.npy', Z_train)
    np.save(Path(out_dir)/'latent_test.npy', Z_test)

    # ---------- NEW: PCA on latent space (Path 1) ----------
    print('step 4.5: PCA on latent space for clustering')
    latent_pca_dim = 5  # you can try 3–5 here
    pca_latent = PCA(n_components=latent_pca_dim, random_state=seed)
    Z_train_pca = pca_latent.fit_transform(Z_train)
    Z_test_pca = pca_latent.transform(Z_test)

    # save PCA model for later use if needed
    with open(Path(out_dir)/'latent_pca_model.pkl', 'wb') as f:
        pickle.dump(pca_latent, f)

    # Step 5: HDBSCAN clustering on PCA(latent)
    print('step 5: run hdbscan on PCA(latent) (train)')
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=20,   # Path 1: smaller, fixed cluster size
        min_samples=10,
        prediction_data=True
    )
    cluster_labels_train = clusterer.fit_predict(Z_train_pca)

    # check if we actually got any clusters (labels != -1)
    unique_labels = np.unique(cluster_labels_train)
    print("HDBSCAN unique train labels:", unique_labels)

    if len([l for l in unique_labels if l != -1]) == 0:
        print("WARNING: HDBSCAN found no clusters on PCA(latent); marking all as noise.")
        cluster_labels_test = np.full(len(Z_test_pca), -1, dtype=int)
        strengths = np.zeros(len(Z_test_pca))
    else:
        cluster_labels_test, strengths = hdbscan.approximate_predict(clusterer, Z_test_pca)

    with open(Path(out_dir)/'hdbscan_model.pkl', 'wb') as f:
        pickle.dump(clusterer, f)

    pd.DataFrame({'cluster': cluster_labels_train}).to_csv(
        Path(out_dir)/'hdbscan_labels_train.csv', index=False
    )
    pd.DataFrame({'cluster': cluster_labels_test}).to_csv(
        Path(out_dir)/'hdbscan_labels_test.csv', index=False
    )

    # Step 6: Evaluation (unchanged, but now using cluster_labels_* from PCA(latent))
    results = {}
    if has_labels and y_train[0] != 'unknown':
        print('step 6: evaluation')
        # Silhouette score on PCA(latent) space
        valid_idx_train = cluster_labels_train != -1
        results['silhouette_latent_train'] = float(np.nan_to_num(
            silhouette_score(Z_train_pca[valid_idx_train], cluster_labels_train[valid_idx_train])
            if valid_idx_train.sum() >= 2 else -1,
            nan=-1.0
        ))
        valid_idx_test = cluster_labels_test != -1
        results['silhouette_latent_test'] = float(np.nan_to_num(
            silhouette_score(Z_test_pca[valid_idx_test], cluster_labels_test[valid_idx_test])
            if valid_idx_test.sum() >= 2 else -1,
            nan=-1.0
        ))

        # Majority vote mapping (same logic as before)
        cluster_to_material = defaultdict(list)
        for cl, mat in zip(cluster_labels_train, y_train):
            if cl != -1:
                cluster_to_material[cl].append(mat)
        cluster_map = {cl: Counter(mats).most_common(1)[0][0]
                       for cl, mats in cluster_to_material.items()}

        pred_material_train = np.array(
            ['noise' if cl == -1 else cluster_map.get(cl, 'unknown')
             for cl in cluster_labels_train]
        )
        pred_material_test = np.array(
            ['noise' if cl == -1 else cluster_map.get(cl, 'unknown')
             for cl in cluster_labels_test]
        )
        mask_train = cluster_labels_train != -1
        mask_test = cluster_labels_test != -1
        results['majority_vote_accuracy_train'] = float(
            (pred_material_train[mask_train] == y_train[mask_train]).sum() / mask_train.sum()
        ) if mask_train.sum() > 0 else -1
        results['majority_vote_accuracy_test'] = float(
            (pred_material_test[mask_test] == y_test[mask_test]).sum() / mask_test.sum()
        ) if mask_test.sum() > 0 else -1

        with open(Path(out_dir)/'cluster_to_material.json', 'w') as f:
            safe_map = {str(int(k)): v for k, v in cluster_map.items()}
            json.dump(safe_map, f, indent=2)

        # classifiers still trained on original latent Z_train / Z_test
        print('step 6.5: train classifiers on latent representations')
        if y_binary_train is not None:
            clf_binary = RandomForestClassifier(
                n_estimators=100, random_state=seed, max_depth=10
            )
            clf_binary.fit(Z_train, y_binary_train)
            y_binary_pred = clf_binary.predict(Z_test)
            results['binary_classification'] = {
                'accuracy': float(accuracy_score(y_binary_test, y_binary_pred)),
                'f1_score': float(f1_score(y_binary_test, y_binary_pred, pos_label='other')),
                'classification_report': classification_report(
                    y_binary_test, y_binary_pred, output_dict=True
                )
            }
            with open(Path(out_dir)/'binary_classifier.pkl', 'wb') as f:
                pickle.dump(clf_binary, f)

        if y_train is not None:
            clf_multi = RandomForestClassifier(
                n_estimators=100, random_state=seed, max_depth=10
            )
            clf_multi.fit(Z_train, y_train)
            y_multi_pred = clf_multi.predict(Z_test)
            results['multiclass_classification'] = {
                'accuracy': float(accuracy_score(y_test, y_multi_pred)),
                'f1_score_weighted': float(f1_score(y_test, y_multi_pred, average='weighted')),
                'classification_report': classification_report(
                    y_test, y_multi_pred, output_dict=True
                )
            }
            with open(Path(out_dir)/'multiclass_classifier.pkl', 'wb') as f:
                pickle.dump(clf_multi, f)
    else:
        print("No labels detected: skipping evaluation and classification steps.")

    # Step 7: PCA visualization (now from PCA(latent) 2D)
    if use_pca_for_viz:
        print('step 7: visualization - PCA(latent) to 2d for plotting')
        pca2 = PCA(n_components=2, random_state=seed)
        Z2_train = pca2.fit_transform(Z_train)
        # we only need train here for the plot; you can also transform test if you like

        df_viz_train = pd.DataFrame({
            'x': Z2_train[:, 0],
            'y': Z2_train[:, 1],
            'cluster': cluster_labels_train
        })
        if has_labels and y_train[0] != 'unknown':
            df_viz_train['material'] = y_train

        plt.figure(figsize=(12, 8))
        for cl in np.unique(cluster_labels_train):
            sel = df_viz_train['cluster'] == cl
            label = f'cluster {cl}' if cl != -1 else 'noise/outliers'
            plt.scatter(
                df_viz_train.loc[sel, 'x'],
                df_viz_train.loc[sel, 'y'],
                label=label,
                s=35, alpha=0.6, edgecolors='black', linewidths=0.5
            )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, framealpha=0.9)
        plt.title('hdbscan cluster assignments on PCA(latent) (training set)',
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('first principal component (pc1)', fontsize=12)
        plt.ylabel('second principal component (pc2)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(out_dir)/'latent_pca_hdbscan_train.png', dpi=300)
        plt.close()


    # Step 8: Baselines (robust to unlabeled)
    if run_baselines:
        print('step 8: baselines - PCA(latent-independent) + HDBSCAN and KMeans')
        baselines = {}

        # ---------- NEW PCA (4–6 dims) ----------
        pca_baseline_dim = 6   # you can try 4–10
        pca = PCA(n_components=min(pca_baseline_dim, n_features), random_state=seed)
        X_pca_train = pca.fit_transform(X_train_scaled)
        X_pca_test = pca.transform(X_test_scaled)

        # ---------- FIXED HDBSCAN BASELINE PARAMETERS ----------
        clusterer_b = hdbscan.HDBSCAN(
            min_cluster_size=20,   # smaller, like Step 5
            min_samples=5,
            prediction_data=True
        )
        labels_pca_hdb_train = clusterer_b.fit_predict(X_pca_train)

        # Check cluster existence
        unique_b = np.unique(labels_pca_hdb_train)
        print("Baseline HDBSCAN unique labels:", unique_b)

        if len([x for x in unique_b if x != -1]) == 0:
            print("Baseline HDBSCAN found no clusters; marking all as noise.")
            labels_pca_hdb_test = np.full(len(X_pca_test), -1, dtype=int)
        else:
            labels_pca_hdb_test, _ = hdbscan.approximate_predict(clusterer_b, X_pca_test)

        with open(Path(out_dir)/'pca_hdbscan_model.pkl', 'wb') as f:
            pickle.dump({'pca': pca, 'hdbscan': clusterer_b}, f)

        # ---------- KMeans baseline ----------
        k = 5  # KMeans does not need labels, use fixed or elbow method
        kmeans = KMeans(n_clusters=k, random_state=seed)
        km_labels_train = kmeans.fit_predict(X_pca_train)
        km_labels_test = kmeans.predict(X_pca_test)

        with open(Path(out_dir)/'kmeans_model.pkl', 'wb') as f:
            pickle.dump(kmeans, f)

        # save baseline metrics for unlabeled data
        baselines['baseline_cluster_counts'] = {
            'hdbscan_clusters': int(len(unique_b) - (1 if -1 in unique_b else 0)),
            'kmeans_clusters': int(k)
        }

        with open(Path(out_dir)/'baselines.json','w') as f:
            json.dump(baselines, f, indent=2)


    # Step 10: Save cluster assignments
    out_df_train = pd.DataFrame({'cluster': cluster_labels_train})
    out_df_test = pd.DataFrame({'cluster': cluster_labels_test})
    if has_labels and y_train[0] != 'unknown':
        out_df_train['material'] = y_train
        out_df_train['material_binary'] = y_binary_train
        out_df_test['material'] = y_test
        out_df_test['material_binary'] = y_binary_test
    if coords_train is not None:
        out_df_train = pd.concat([out_df_train.reset_index(drop=True), coords_train.reset_index(drop=True)], axis=1)
    if coords_test is not None:
        out_df_test = pd.concat([out_df_test.reset_index(drop=True), coords_test.reset_index(drop=True)], axis=1)
    out_df_test.to_csv(Path(out_dir)/'cluster_assignments_test.csv', index=False)
    out_df_train.to_csv(Path(out_dir)/'cluster_assignments_train.csv', index=False)

    print('pipeline complete. results saved to', out_dir)
    return results


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pxrf unsupervised pipeline: autoencoder + hdbscan + optional classification')
    parser.add_argument('--csv', type=str, required=True, help='path to input csv file')
    parser.add_argument('--out', type=str, default='results', help='output directory')
    parser.add_argument('--latent', type=int, default=8, help='latent space dimensionality')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for autoencoder')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--no-baselines', action='store_true', help='skip baseline comparisons')
    parser.add_argument('--test-size', type=float, default=0.2, help='test set fraction')
    parser.add_argument('--features', type=str, default=None, help='comma-separated list of feature columns')
    parser.add_argument('--target', type=str, default=None, help='target column (optional)')
    args = parser.parse_args()
    
    feature_cols = args.features.split(',') if args.features else None
    run_pipeline(args.csv, out_dir=args.out, latent_dim=args.latent, epochs=args.epochs, 
                 batch_size=args.batch, run_baselines=not args.no_baselines, test_size=args.test_size,
                 feature_cols=feature_cols, target_col=args.target)
