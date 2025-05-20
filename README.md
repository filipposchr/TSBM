# TSB

- Running the code:
`python -u main.py -d edit-tgwiktionary`

- Training the model:
  - Set `testing = False`
  - Saving the model:
    - `torch.save(MLP_model.state_dict(), './saved_models/model_MLP_1.pth')`
    - `torch.save(tatkc_tgat_model.state_dict(), './saved_models/model_TGAT_1.pth')`

- Testing the model:
  - Set `testing = True`
  - Loading saved model:
    - `tatkc_tgat_model.load_state_dict(torch.load('./saved_models/model_TGAT_1.pth'))`
    - `MLP_model.load_state_dict(torch.load('./saved_models/model_MLP_1.pth'))`
