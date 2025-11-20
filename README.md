# Using Unsupervised Methods to Classify X-ray Fluorescence Data  
**By Larry Griffith, Matthew Hashim, Jared Kagie**

---

## How to Run the Code

### 1. Create a Virtual Environment  
Use a virtual environment to avoid package conflicts:

```
python -m venv <venv_name>
```

### 2. Activate the Environment  
On Windows:

```
<venv_name>\Scripts\activate
```

### 3. Install Required Packages

```
pip install -r requirements.txt
```

Deactivate anytime with:

```
deactivate
```

---

## Running the Pipeline

Once your environment is ready, run **one** of the following:

### **Labeled Version**
```
python "Autoencoder HDBscan Labeled.py" --csv=path\to\AllPXRF_FINAL_14Oct.csv --out=path\to\output
```

### **Unlabeled Version**
```
python "Autoencoder HDBscan Unlabeled.py" --csv=path\to\MV0811-14JC_merged_sec1-4.csv --out=path\to\output
```

(Or use `MV0811-14JC_merged_sec5-8.csv` depending on which file you want to test.)

---

## Optional Flags

### **Common Flags (Both Versions)**  
| Flag | Description |
|------|-------------|
| `--csv` | Path to input CSV file (**required**) |
| `--out` | Output directory |
| `--latent` | Size of latent representation |
| `--epochs` | Autoencoder training iterations |
| `--batch` | Training batch size |
| `--test-size` | Fraction of data for testing (0.0–1.0) |
| `--no-baselines` | Disable baseline comparisons |

### **Unlabeled Version Only**
| Flag | Description |
|------|-------------|
| `--features` | Comma-separated list of element columns |
| `--target` | Optional column for evaluation |

---

## Expected Output Structure (Labeled Version)

```
Results/
├── Models/
│   ├── autoencoder.keras          # Trained autoencoder
│   ├── encoder.keras              # Encoder (feature extractor)
│   ├── hdbscan_model.pkl          # Cluster model
│   ├── binary_classifier.pkl      # Soil vs non-soil classifier
│   └── multiclass_classifier.pkl  # Material classifier
│
├── Data/
│   ├── latent_train.npy           # Latent features
│   ├── cluster_assignments.csv    # Cluster IDs
│   └── cluster_to_material.json   # Cluster → label mapping
│
├── Visualizations/
│   ├── training_loss.png          # Autoencoder loss curve
│   ├── latent_pca_hdbscan.png     # Cluster visualization
│   └── confusion_matrix.png       # Classification results
│
└── Metrics/
    ├── hdbscan_results.json       # Cluster evaluation
    └── baselines.json             # Baseline comparisons
```
