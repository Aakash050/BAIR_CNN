# Smaller BAIR download: Parquet route

**A converted subset is already available:** `data/parquet/bair_hf_subset/bair_subset.parquet` contains 1,000 training clips and 256 test clips (8.73 MiB). Install `pyarrow av` in your notebook kernel and start at notebook 01; `SOURCE="parquet"` and `ALLOW_DOWNLOAD=False` are already set.

To recreate the subset, start with [notebook 00](notebooks/00_huggingface_to_parquet.ipynb). It streams 1,000 training clips and 256 test clips from selected community Hugging Face tar shards into `data/parquet/bair_hf_subset/bair_subset.parquet`. The Parquet preserves MP4 video bytes and paired action arrays. Run notebooks 01–03 afterward; notebook 01 defaults to `SOURCE="parquet"` with TFDS downloads disabled.

For this route install `numpy pandas matplotlib torch scikit-image pyarrow av jupyterlab`; TensorFlow and TensorFlow Datasets are optional unless using the original raw/TFDS route. The converter is also callable as `python scripts/convert_hf_bair.py` from the project root.

The selected Hugging Face revision is `776ad3b`. Videos are community-reencoded MP4s, which may differ from original pixels. The training subset is selected in source order. Report these limits. Notebook 01 decodes Parquet into a NumPy training cache, so keep disk space for both. The original official test clips are kept separate; the community index calls them validation, but we reserve them as test data.

An incomplete or interrupted conversion leaves a `.partial` file and raises an error. Use a new output directory for retries or a different subset. Completed datasets are never overwritten.

# BAIR video world model: first draft

Three main Jupyter notebooks implement the supplied proposal using real BAIR data:

1. [Data preparation](notebooks/01_bair_data_preparation.ipynb): raw BAIR TFRecords or prepared TensorFlow Datasets, clip splits, normalization, and exploration.
2. [Model training](notebooks/02_bair_model_training.ipynb): action-conditioned encoder-decoder, one-step Model A, multi-step Model B, and validation-selected checkpoints.
3. [Evaluation and report](notebooks/03_bair_evaluation_and_report.ipynb): persistence comparison, MSE/PSNR/SSIM, motion-region diagnostics, paired bootstrap comparisons, visualizations, and report prose.

Install `requirements.txt` in a Python 3.11 or 3.12 environment and open the notebooks in numerical order. Install a PyTorch build suited to your compute system. TensorFlow is used only to import data; training and evaluation use PyTorch.

For the original raw/TFDS route, set `SOURCE="raw"` or `SOURCE="tfds"` in notebook 01, then set `RAW_ROOT` to the extracted BAIR `softmotion30_44k` directory containing `train/` and `test/`, or select `SOURCE="tfds"` and set `TFDS_ROOT`. No download runs by default. The [official dataset catalog](https://www.tensorflow.org/datasets/catalog/bair_robot_pushing_small) documents download and storage sizes. Run from this project root or its `notebooks/` directory.

Run notebook 02, then paste its printed run directory into notebook 03. Diagnose on validation first; switch `SPLIT` to `test` once choices are frozen. Processed clips are under `data/processed/`; checkpoints and reports are under `artifacts/`.

These notebooks contain no synthetic data, trained checkpoints, or invented results. Full execution requires the BAIR dataset. The default training configuration is a starting experiment, and conclusions remain pending. Review the stated action/frame indexing convention against your data release before interpreting results.
