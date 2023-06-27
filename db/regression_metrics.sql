CREATE TABLE regression_metrics (
    testing_start       TIMESTAMP,
    testing_end         TIMESTAMP,

    model               VARCHAR(20),
    train_criterion     VARCHAR(20),
    val_criterion       VARCHAR(20),

    mse                 DECIMAL(18,4),
    rtn_bottom          DECIMAL(18,4),
    rtn_weighted        DECIMAL(18,4),

    training_start      TIMESTAMP,
    training_end        TIMESTAMP,
    validation_start    TIMESTAMP,
    validation_end      TIMESTAMP,

    PRIMARY KEY (testing_start, testing_end, model, train_criterion, val_criterion)
);