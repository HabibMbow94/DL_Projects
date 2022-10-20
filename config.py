import argparse
from importlib.metadata import metadata


args = argparse.Namespace(
    lr=1e-4,
    bs = 8,
    train_size=0.8,
    path= "data/images",
    metadata= "data/metadata_csv/metadata_ok.csv",
    wd = 1.0
)