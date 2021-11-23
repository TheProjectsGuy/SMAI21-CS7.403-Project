Latest (and best) checkpoint: cp_5.ckpt
Model: ./model

Training details:
    num_batches = 16   # Number of batches (128)
    ns_pbat = 500  # Number of samples per batch (3000)

    model.compile(optimizer=optimizers.Adam(),
    loss=losses.MeanSquaredError(),
    metrics=[metrics.MeanSquaredError()])
