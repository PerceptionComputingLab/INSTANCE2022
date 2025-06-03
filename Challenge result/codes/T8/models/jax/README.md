
Dependancies
- jax
- https://github.com/e3nn/e3nn-jax (home made)
- haiku
- optax
- ml_collections
- absl
- wandb
- nibabel

To train the network, we use the script `script_train.py`. The training loop is defined in `functions.py`. The configuration files are in the folder `configs/`. We used `submit2.py` for the last submission. The files for the submission are in the directory `submit_docker/`.
`script_evaluate.py` and `script_ensemble_average.py` were only use during the seach for hyper parameters phase.

Command used to train the networks for the submission:

```
python script_train.py --name mynetwork0 --config configs/submit2.py --config.seed_init=0 --config.seed_train=0
python script_train.py --name mynetwork1 --config configs/submit2.py --config.seed_init=1 --config.seed_train=1
python script_train.py --name mynetwork2 --config configs/submit2.py --config.seed_init=2 --config.seed_train=2
python script_train.py --name mynetwork3 --config configs/submit2.py --config.seed_init=3 --config.seed_train=3
python script_train.py --name mynetwork4 --config configs/submit2.py --config.seed_init=4 --config.seed_train=4
python script_train.py --name mynetwork5 --config configs/submit2.py --config.seed_init=5 --config.seed_train=5
python script_train.py --name mynetwork6 --config configs/submit2.py --config.seed_init=6 --config.seed_train=6
```

Command used to evaluate? see the file `submit_docker/Dockerfile`.

Version of libraries:

```
chex                   0.1.4
dm-haiku               0.0.7
e3nn-jax               0.8.0
jax                    0.3.15
jaxlib                 0.3.15+cuda11.cudnn82
ml-collections         0.1.1
nibabel                4.0.1
numpy                  1.23.1
optax                  0.1.3
pip                    22.2.2
scipy                  1.8.1
wandb                  0.13.1
```
