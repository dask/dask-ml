import argparse


def make_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--scheduler-address', default=None,
                        help="Address for the scheduler node. Runs locally "
                             "(threaded) by default")
    parser.add_argument('-l', '--local', action="store_true",
                        help="Use the local file system.")
    parser.add_argument("--scikit-learn", action="store_true",
                        help="Use scikit-learn for comparison")
