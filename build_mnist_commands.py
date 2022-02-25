import os
import sys



METHODS = ['test_finetune_moe', 'test_usage_gmoe', 'kensemble', 'ensemble', 'umix']
METHODS = [METHODS[int(sys.argv[1])]]

MBS = [1,10,20,50][::-1]
SIZE = [4, 16, 64]
K = [2,4]

commands = []

for run in [2]:

    for method in METHODS:

        is_growing = method in ['ensemble', 'test_usage_gmoe']
        for mb in MBS:

            # no need to run growing methods if they don't grow
            if mb == 1 and is_growing:
                continue

            # no need to run huge model for growing methods
            for size in ([4, 16] if is_growing else SIZE):

                command = f'PYTHONPATH=./ python configs/mnist/run.py -cn {method}.yaml n_megabatches={mb} clmodel.max_epochs=100 clmodel.patience=100 clmodel.patience_delta=0 clmodel.model.n_layers=2 clmodel.model.size_layers={size} +run={run}'

                if method in ['kensemble', 'umix']:
                    for k in K:
                        command += f' clmodel.k={k}'
                        commands += [command]
                else:
                    commands += [command]

print('total of  ', len(commands))

for i, cmd in enumerate(commands):
    print(f'launching {i} / {len(commands)} of method {METHODS[0]}')
    print(cmd)
    os.system(cmd)
