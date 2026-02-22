from experiments.auto_run_tests import build_configs, run_configs_and_collect

if __name__ == '__main__':
    cfgs = build_configs()
    # run only first 2 configs as a smoke validation
    df = run_configs_and_collect(cfgs[:2], out_prefix='auto_test_smoke')
    print(df)
