# Historical checkpoint notes

This document is retained as a historical reference. The old weights/ folder has
been replaced by models/. Current selected files are models/cnn_best.pt and
models/transfer_best.pt; see the root README for current commands.

# Trained checkpoints

The notebooks copy the best validation checkpoints here:

- `cnn_best.pt`: M1, the supplied four-block CNN trained from scratch.
- `transfer_best.pt`: M2, DDAMFN initialized from official DDAMFN++ AffectNet-7 and fine-tuned on FER-2013.

After a Kaggle run, download the result ZIP, then copy the two files from its
`weights/` folder into this directory. No random or untrained weights are shipped
as trained models. Checkpoints contain class order and preprocessing metadata.

Large binary checkpoints are ignored by Git. Include them in the submission ZIP,
or upload them separately and put their real download links in this file.

Older ResNet-18 checkpoints remain loadable; new training uses DDAMFN.
The model architecture and its input transform are stored in each checkpoint.
