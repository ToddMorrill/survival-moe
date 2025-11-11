from survkit.configs import ArgParser, WandbConfig, TrainConfig

def get_config(command_str):
    parser = ArgParser([WandbConfig, TrainConfig])

    wandb_config, train_config, unknown_args = parser.parse_args_into_dataclasses(
        args=command_str.split(),
        return_remaining_strings=True,
        args_file_flag='--experiment_args',
    )
    return wandb_config, train_config