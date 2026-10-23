"""Retired legacy split generator; importing this module does not write files.

The historical script randomly overwrote root train.txt/test.txt from one local
sequence. Current official TRAIN isolation is fixed under
configs/official_split_20261023/ and enforced by train_official.py.
"""

def main():
    raise SystemExit("Retired: use configs/official_split_20261023/train.txt and val.txt; no split files were changed.")

if __name__ == "__main__":
    main()
