Checkpoint: cp_10.ckpt
Model: ./../24_nov_21/model

Training details
    num_batches = 16   # Number of batches (128)
    ns_pbat = 500  # Number of samples per batch (3000)

    model.compile(optimizer=optimizers.Adam(),
    loss=losses.MeanSquaredError(),
    metrics=[metrics.MeanSquaredError()])

    Final step:
        500/500 - 29s - loss: 444.7917 - mean_squared_error: 444.7916 - 29s/epoch - 58ms/step
