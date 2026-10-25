# Trained checkpoints

The notebooks copy the best validation checkpoints here:

- `cnn_best.pt`: M1, the supplied four-block CNN trained from scratch.
- `transfer_best.pt`: M2, adapted and fine-tuned ImageNet ResNet-18.

After a Kaggle run, download the result ZIP, then copy the two files from its
`weights/` folder into this directory. No random or untrained weights are shipped
as trained models. Checkpoints contain class order and preprocessing metadata.

Large binary checkpoints are ignored by Git. Include them in the submission ZIP,
or upload them separately and put their real download links in this file.
