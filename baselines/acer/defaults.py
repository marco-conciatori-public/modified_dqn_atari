def atari():
    return dict(
        lrschedule='constant',
        # network='cnn',
        q_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=10,
        lr=7e-4,
        rprop_epsilon=1e-5,
        rprop_alpha=0.99,
        gamma=0.99,
        buffer_size=50000,
        replay_ratio=4,
        replay_start=10000,
        c=10.0,
        trust_region=True,
        alpha=0.99,
        delta=1,
        load_path=None
    )
