from streaming.base.util import merge_index
import argparse

# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_root', type=str, default=None)
    args = parser.parse_args()
    out_root = args.out_root
    merge_index(out_root, keep_local=True)

