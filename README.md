# Temporal Betweenness Centrality Prediction

This project provides a framework for approximating **Temporal Shortest Betweenness Centrality (TSBM)** and **Temporal Shortest-Foremost Betweenness Centrality (TSFMBM)** using graph neural networks.

## Running the Script

You can run the model with various command-line options.

### Example: Run in Test Mode
```bash
python -u main.py -d edit-wamazon --bet sfm --test
```

This configuration:
- Enables test mode (`--test`)
- Uses **shortest-foremost** betweenness (`--bet sfm`)
- Uses the dataset `edit-wamazon` (`-d edit-wamazon`)

### Example: Run in Training Mode (default)
```bash
python -u main.py
```

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

## Directory Structure

```
.
├── main.py
├── nx2graphs.py
├── saved_models/
│   ├── model_TGAT_1.pth
│   └── model_MLP_1.pth
├── data/
│   └── test/
│       └── Real/
├── ...
```

## Requirements

- Python 3.8+
- PyTorch
- NetworkX
- pandas

Install dependencies with:
```bash
pip install -r requirements.txt
```

## License

This project is released under the MIT License.
