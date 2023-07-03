CREATE TABLE gbm_metrics (
    testing_start       DATE,
    testing_end         DATE,

    model_id            BIGINT,
    val_criterion       VARCHAR(20),

    rtn_bottom          DECIMAL(18,4),
    rtn_weighted        DECIMAL(18,4),

    mse                 DECIMAL(18,6),
    rmse                DECIMAL(18,6),
    mae                 DECIMAL(18,6),
    mape                DECIMAL(18,6),
    dir_acc             DECIMAL(18,6),

    training_start      DATE,
    training_end        DATE,
    validation_start    DATE,
    validation_end      DATE,

    PRIMARY KEY (testing_start, testing_end, model_id, val_criterion)
);