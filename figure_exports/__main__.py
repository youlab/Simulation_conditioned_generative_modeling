"""Shared export runners: python -m figure_exports preprocessing."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure', choices=['preprocessing'])
    parser.parse_args()
    from .export_preprocessing import export_preprocessing
    export_preprocessing()


if __name__ == '__main__':
    main()
