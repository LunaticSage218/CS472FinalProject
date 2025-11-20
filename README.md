# Using Unsupervised Methods to Classify X-ray Fluorescence Data

## By Larry Griffith, Matthew Hashim, Jared Kagie

# To Run The Code

First, create a virtual envornment, so there's no conflicts with python package versions

In a directory of your choosing, run: python -m venv <virtual enviornment name>

Then activate the venv by running <virtual enviornment name>\Scripts\activate, this will create a virtual envornment for your python code, making sure no conflicts occur

Then run pip install -r requirements.txt, this will install the packages, only within the enviornment, so you can't run this code while the enviornment is deactivated (type deactivate in the terminal to turn off the virtual envornment)

To run the code once the virtual enviornment is set up, run one of the two commands:  
python "Autoencoder HDBscan Labeled.py" --csv=path\to\AllPXRF_FINAL_14Oct.csv --out=whatever\output\dir  
or  
python "Autoencoder HDBscan Unlabeled.py" --csv=path\to\MV0811-14JC_merged_sec1-4.csv (or path\to\data\MV0811-14JC_merged_sec5-8.csv, depending on what file you want to test) --out=whatever\output\dir

You can also add some additional flags:

**Common Flags (Both Versions)**  
--csv - Path to input CSV file (required)  
--out - Output directory for results  
--latent - Dimensionality of latent space (compression level)  
--epochs - Number of training iterations for autoencoder  
--batch - Number of samples processed per training batch  
--test-size - Fraction of data reserved for testing (0.0-1.0)  
--no-baselines - Skip baseline method comparisons  

**Unlabeled Version Only**  
--features - Comma-separated list of element columns to use  
--target - Optional target column name for evaluation  

The output for the labeled version will look like this:

Results/
├── Models/
│   ├── autoencoder.keras          # Trained autoencoder
│   ├── encoder.keras              # Encoder for feature extraction
│   ├── hdbscan_model.pkl          # Fitted clusterer
│   ├── binary_classifier.pkl      # Soil vs non-soil classifier
│   └── multiclass_classifier.pkl  # Material type classifier
├── Data/
│   ├── latent_train.npy           # Latent representations
│   ├── cluster_assignments.csv    # Final cluster labels
│   └── cluster_to_material.json   # Cluster mapping (labeled)
├── Visualizations/
│   ├── training_loss.png          # Autoencoder training curve
│   ├── latent_pca_hdbscan.png     # 2D cluster visualization
│   └── confusion_matrix.png       # Classification performance
└── Metrics/
    ├── hdbscan_results.json       # Comprehensive evaluation
    └── baselines.json             # Method comparisons
