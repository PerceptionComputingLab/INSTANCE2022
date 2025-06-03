from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.seed_init = 1
    config.seed_train = 1
    config.seed_shuffle_data = 1
    config.testset = 20
    config.trainset = 80

    config.train_steps = 200_000
    config.weight_avg = 1 / 20_000

    config.model = config_dict.ConfigDict()
    config.model.name = "v1"
    config.model.path_normalization = "element"
    config.model.gradient_normalization = "element"
    config.model.dummy = False
    config.model.equivariance = "E3"
    config.model.width = 5
    config.model.num_radial_basis_sh = (
        2,  # L=0
        2,  # L=1
        2,  # L=2
        0,  # L=3
        0,  # L=4
    )
    config.model.relative_start_sh = (
        0.0,  # L=0
        0.0,  # L=1
        0.0,  # L=2
        0.0,  # L=3
        0.0,  # L=4
    )
    config.model.min_zoom = 0.36
    config.model.downsampling = 2.0
    config.model.conv_diameter = 5.0
    config.model.instance_norm_eps = 0.6

    config.optimizer = config_dict.ConfigDict()
    config.optimizer.lr = 1e-3
    config.optimizer.algorithm = "adam"
    config.optimizer.lr_div_step = 99_999_999
    config.optimizer.lr_div_factor = 0.1
    config.optimizer.lr_div_factor_min = 1.0

    config.augmentation = config_dict.ConfigDict()
    config.augmentation.noise = 0.0
    config.augmentation.deformation = 1.0
    config.augmentation.deformation_temperature = 5e-4
    return config
