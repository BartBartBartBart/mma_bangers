# mma_bangers
Repository for the Multimedia Analytics assignment.

## Basic Setup
To develop on your machine, here are some tips.

First, please download the data from [here](https://www.kaggle.com/datasets/stackoverflow/stacksample).
Put it in a directory `data` in the repository.

create the conda environment and activate it:
   ```bash
   conda env create -f environment.yaml
   conda activate mma
   ```

Then:

1. Go into the challenge-icml-2024 git submodule:

   ```bash
   cd challenge-icml-2024
   ```

2. Install tmx in editable mode:

   ```bash
   pip install -e '.[all]'
   ```
   **Notes:**
   - Requires pip >= 21.3. Refer: [PEP 660](https://peps.python.org/pep-0660/).
   - On Windows, use `pip install -e .[all]` instead (without quotes around `[all]`).

4. Install torch, torch-scatter, torch-sparse with or without CUDA depending on your needs.

    ```bash
    pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
    ```

    where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, or `cu115` depending on your PyTorch installation (`torch.version.cuda`).

## Run interactive app
When you have installed the data and activated the environment you can run the app by executing:

```bash
python app.py
```
