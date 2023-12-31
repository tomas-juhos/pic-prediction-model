CREATE TABLE gbm_metrics (
    universe_constr     VARCHAR(50),
    testing_start       DATE,
    testing_end         DATE,

    model_id            BIGINT,
    val_criterion       VARCHAR(20),

    rtn_bottom          DECIMAL(18,4),
    rtn_weighted        DECIMAL(18,4),
    rtn_random          DECIMAL(18,4),
    rtn_benchmark       DECIMAL(18,4),

    mse                 DECIMAL(18,4),
    rmse                DECIMAL(18,4),
    mae                 DECIMAL(18,4),
    mape                DECIMAL(18,4),
    dir_acc             DECIMAL(5,2),

    training_start      DATE,
    training_end        DATE,
    validation_start    DATE,
    validation_end      DATE,

    PRIMARY KEY (universe_constr, testing_start, testing_end, model_id, val_criterion)
);