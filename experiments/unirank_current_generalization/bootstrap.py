"""Stage the tested current validation-macro best for native-dataset transfer."""
import argparse
import importlib.util
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def bootstrap(previous, report, root):
    helper_path = PROJECT / 'experiments/unirank_click_generalization/bootstrap.py'
    spec = importlib.util.spec_from_file_location('fixed_transfer_stage_helper', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    result = helper.bootstrap(previous, root, source_report=report, adapter_folder=HERE)
    shutil.copy2(helper_path, result / 'staging-helper-at-run.py')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(bootstrap(args.previous, args.source_report, args.root))
