# Temporal Betweenness Centrality Prediction

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
python -u main.py
python -u main.py -d edit-wamazon --bet sh
```

This configuration:
- Enables training mode
- Uses **shortest** betweenness (`--bet sh`)
- After training, it evaluates the testing dataset `edit-wamazon` (`-d edit-wamazon`)



### Example: Run in Test Mode
```bash
python -u main.py -d edit-wamazon --bet sfm --test
```

This configuration:
- Enables test mode (`--test`)
- Uses **shortest-foremost** betweenness (`--bet sfm`)
- Evaluates the testing dataset `edit-wamazon` (`-d edit-wamazon`)



### Command-Line Argument Reference

| Flag             | Description                                                      | Default              |
|------------------|------------------------------------------------------------------|----------------------|
| `--test`         | Runs the model in test mode (loads saved models)                | `False`              |
| `--bet`          | Betweenness mode: `sh` for shortest, `sfm` for shortest-foremost | `sh`                 |
| `-d`, `--data`   | Dataset name to use for training/testing                         | `edit-tgwiktioanry`  |

## Saving the Model

To save a trained model, the following lines are used in the code:
```python
torch.save(MLP_model.state_dict(), './saved_models/model_MLP_1.pth')
torch.save(tatkc_tgat_model.state_dict(), './saved_models/model_TGAT_1.pth')
```

## Loading a Saved Model

To load the models for testing or evaluation, use:
```python
tatkc_tgat_model.load_state_dict(torch.load('./saved_models/model_TGAT_1.pth'))
MLP_model.load_state_dict(torch.load('./saved_models/model_MLP_1.pth'))
```

## Evaluation: Comparing TSBM/TSFMBM with MANTRA

You can evaluate the performance of the TSBM or TSFMBM model against existing approximation baselines (e.g., MANTRA) using the `test.py` script.

### Example Script: `test.py`

This script loads the ground truth values (e.g., from TSBM) and the predicted values (e.g., from MANTRA), computes:

- Top-k accuracy (Top@1%, Top@5%, Top@10%, Top@20%)
- Jaccard index between top-k sets
- Weighted Kendall Tau (for ranking agreement)

### Expected Structure:
```python
# Ground truth scores (e.g., from TSFMBM model)
with open('data/test/Real/shf-bc_scores/graph_edit-facebook_wall_shf_bet.txt', 'r') as f2:

# Approximate values (e.g., from MANTRA)
with open('data/apx/facebook_wall_sfm_apx.txt', 'r') as f1:
```

### Run the evaluation:
Make sure the predicted and true values are aligned in length, then execute:

```bash
python test.py
```

### Output example:
```
Top@1%: 0.3850 | Top@5%: 0.6120 | Top@10%: 0.3510 | Top@20%: 0.2890 | Jaccard: 0.2980
Kendall Tau (all nodes):      0.2614
Kendall Tau (non-zero only):  0.2987
```

This allows for direct comparison of MANTRA and learned models in terms of ranking quality and top-k overlap.

