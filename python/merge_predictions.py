from prepost_dismec.postprocess import merge_pred
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default=None, type=str, required=True,
                    help="The directory of the prediction files of DiSMEC")
    args = parser.parse_args()

    merge_pred(args.pred_dir)

if __name__ == '__main__':
    main()

    