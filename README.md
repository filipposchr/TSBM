# TSB

- Running the code:
`python -u main.py -d edit-tgwiktionary --lr 0.01`

- For training the model set `testing = False`

- Models are saved to:
  - `torch.save(MLP_model.state_dict(), './saved_models/model_MLP_1.pth')`
  - `torch.save(tatkc_tgat_model.state_dict(), './saved_models/model_TGAT_1.pth')`
