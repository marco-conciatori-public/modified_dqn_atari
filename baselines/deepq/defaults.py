# def atari():
    # return dict(
    #     network='conv_only',
    #     lr=1e-4,
    #     buffer_size=10000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.01,
    #     train_freq=4,
    #     learning_starts=10000,
    #     target_network_update_freq=1000,
    #     gamma=0.99,
    #     checkpoint_freq=10000,
    #     checkpoint_path=None,
    #     dueling=True
    # )

def atari():
    # network_kwargs = {}

    return dict(
        # network='cnn',
        lr=1e-4,
        replay_batch_size=32,
        lb_batch_size=16,

        # network_kwargs=network_kwargs,
        # total_timesteps=100000,

        # print_freq=100,

        # learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        train_freq=4,
        buffer_size=10000,
        # exploration_final_eps=100,
        load_path=None,
        checkpoint_freq=10000,
        checkpoint_path=None,
        # dueling=False,
        param_noise=False,
        # seed=None,
        callback=None,
    )

def retro():
    return atari()

