# Dataset

This folder is gitignored — download the dataset yourself before running anything.

## UCI HAR Dataset

Source: [UCI Machine Learning Repository, dataset 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
(Anguita et al., 2013).

Direct download:

```text
https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
```

Use the **full UCI zip**, not a Kaggle CSV mirror — the Kaggle mirrors typically
only include the 561 pre-computed feature vectors and omit the raw
`Inertial Signals/` folder that the CNN and BiLSTM need. Note the downloaded
zip contains a *nested* `UCI HAR Dataset.zip` — extract that one too.

Extract it so the layout looks like:

```text
data/
  UCI HAR Dataset/
    train/
      Inertial Signals/
      X_train.txt
      y_train.txt
      subject_train.txt
    test/
      Inertial Signals/
      X_test.txt
      y_test.txt
      subject_test.txt
    activity_labels.txt
    features.txt
```

Then verify the pipeline from the project root:

```text
python src/har_data.py --root "data/UCI HAR Dataset"
```

This checks subject independence between train/test and reports the
train/val/test shapes under the shared protocol (`har_data.load_bundle`).
