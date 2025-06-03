# Temporal Betweenness Centrality Prediction

## Table of Contents
- [Requirements](#requirements)
- [Running the Script](#running-the-script)
  - [Example: Run in Training Mode (default)](#example-run-in-training-mode-default)
  - [Example: Run in Test Mode](#example-run-in-test-mode)
  - [Command-Line Argument Reference](#command-line-argument-reference)
- [Saving the Model](#saving-the-model)
- [Loading a Saved Model](#loading-a-saved-model)
- [Evaluation: Comparing TSBM/TSFMBM with MANTRA](#evaluation-comparing-tsbmtsfmbm-with-mantra)
  - [Example Script: test.py](#example-script-testpy)
  - [Expected Structure](#expected-structure)
  - [Run the Evaluation](#run-the-evaluation)
  - [Example Output](#example-output)
- [Batch Evaluation: test_multiple.py](#batch-evaluation-test_multiplepy)


This project provides a framework for approximating **Temporal Shortest Betweenness Centrality (TSBM)** and **Temporal Shortest-Foremost Betweenness Centrality (TSFMBM)** using graph neural networks.

## Requirements

- Python 3.8+
- PyTorch
- NetworkX
- pandas

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Script

You can run the model with various command-line options.

### Example: Run in Training Mode (default)

```bash
python -u main.py -d edit-wamazon --bet sh
```

This configuration:
- Enables training mode
- Uses **shortest** betweenness (`--bet sh`)
- Trains and evaluates the dataset `edit-wamazon` (`-d edit-wamazon`)

### Example: Run in Test Mode

```bash
python -u main.py -d edit-wamazon --bet sfm --test
```

This configuration:
- Enables test mode (`--test`)
- Uses **shortest-foremost** betweenness (`--bet sfm`)
- Evaluates the pre-trained model on `edit-wamazon`

### Command-Line Argument Reference

| Flag             | Description                                                      | Default              |
|------------------|------------------------------------------------------------------|----------------------|
| `--test`         | Runs the model in test mode (loads saved models)                | `False`              |
| `--bet`          | Betweenness type: `sh` for shortest, `sfm` for shortest-foremost | `sh`                 |
| `-d`, `--data`   | Dataset name to use for training/testing                         | `edit-tgwiktionary`  |

## Saving the Model

To save a trained model:

```python
torch.save(MLP_model.state_dict(), './saved_models/model_MLP_1.pth')
torch.save(tatkc_tgat_model.state_dict(), './saved_models/model_TGAT_1.pth')
```

## Loading a Saved Model

To load previously saved models for testing or evaluation:

```python
tatkc_tgat_model.load_state_dict(torch.load('./saved_models/model_TGAT_1.pth'))
MLP_model.load_state_dict(torch.load('./saved_models/model_MLP_1.pth'))
```

## Evaluation: Comparing TSBM/TSFMBM with MANTRA

You can compare the performance of the TSBM or TSFMBM models against existing approximation baselines (e.g., MANTRA) using the `test.py` script.

### Example Script: `test.py`

This script loads ground truth values and predictions, then computes:

- Top-k Accuracy (Top@1%, Top@5%, Top@10%, Top@20%)
- Jaccard Index between top-k sets
- Weighted Kendall Tau (for ranking agreement)

#### Expected Structure

```python
# Ground truth (from TSFMBM)
with open('data/test/Real/shf-bc_scores/graph_edit-facebook_wall_shf_bet.txt', 'r') as f2:

# Approximation (from MANTRA)
with open('data/apx/facebook_wall_sfm_apx.txt', 'r') as f1:
```

#### Run the Evaluation

Ensure the predictions and true values are aligned, then run:

```bash
python test.py
```

#### Example Output

```
Top@1%: 0.3850 | Top@5%: 0.6120 | Top@10%: 0.3510 | Top@20%: 0.2890 | Jaccard: 0.2980
Kendall Tau (all nodes):      0.2614
Kendall Tau (non-zero only):  0.2987
```

This allows direct performance comparison of MANTRA and learned models.

## Batch Evaluation: `test_multiple.py`

To evaluate performance across multiple datasets in one run, use the `test_multiple.py` script provided in the repository.

By default, this script runs the model on each dataset using the following parameters:
- `--test`: Enables test mode (uses saved models)
- `--bet sfm`: Uses **shortest-foremost** betweenness centrality
- `-d <dataset>`: Specifies the dataset to evaluate (looped over a predefined list)

### Run it with:

```bash
python test_multiple.py
```

This will sequentially evaluate all listed datasets using the **TSFMBM** model in test mode with `--bet sfm`.

>  **Note:**  
> If you want to change the betweenness type (e.g., from **shortest-foremost** to **shortest**), open `test_multiple.py` and modify the `--bet` argument inside the command construction line:
>
> ```python
> command = ["python", "-u", "main.py", "-d", d, "--bet", "sfm", "--test"]
> ```
>
> Change `"sfm"` to `"sh"` to evaluate using the **shortest** betweenness model (TSBM).
