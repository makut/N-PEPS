import argparse
import multiprocessing.dummy
import os
import subprocess

# import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-path', type=str)
    parser.add_argument('--result-path', type=str)
    parser.add_argument('--agg-type', type=str)
    parser.add_argument('--alpha', type=str)
    parser.add_argument('--gps-timeout', type=str)
    parser.add_argument('--peps-timeout', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--agg-mode', type=str)
    parser.add_argument('--model-base-path', type=str)

    args = parser.parse_args()

    def run_one_split(split):
        subprocess.check_call([
            'python', '-m', 'scripts.solve_problems',
            '--test_path', os.path.join(args.test_path, split),
            '--result_path', args.result_path,
            '--agg_type', args.agg_type,
            '--alpha', args.alpha,
            '--gps_timeout', args.gps_timeout,
            '--peps_timeout', args.peps_timeout,
            '--method', args.method,
            '--agg_mode', args.agg_mode,
            '--model_base_path', args.model_base_path,
        ])

    # for split in os.listdir(args.test_path):
    #     run_one_split(split)

    with multiprocessing.dummy.Pool(processes=5) as pool:
        pool.map(run_one_split, os.listdir(args.test_path))
    # joblib.Parallel(n_jobs=5)(joblib.delayed(run_one_split)(split) for split in os.listdir(args.test_path))
