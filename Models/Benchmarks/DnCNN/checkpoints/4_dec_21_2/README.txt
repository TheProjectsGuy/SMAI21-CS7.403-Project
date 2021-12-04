Checkpoint: cp_50.ckpt
Model: ./../24_nov_21/model

Training details
    num_batches = 16   # Number of batches (128)
    ns_pbat = 500  # Number of samples per batch (3000)

    model.compile(optimizer=optimizers.Adam(),
    loss=losses.MeanSquaredError(),
    metrics=[metrics.MeanSquaredError()])

    Final step:
        500/500 - 30s - loss: 366.8364 - mean_squared_error: 366.8366 - 30s/epoch - 61ms/step
