from inspect import signature
from experiments.auto_run_tests import run_configs_and_collect
print('signature:', signature(run_configs_and_collect))
print('ok')
