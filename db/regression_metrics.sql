CREATE TABLE regression_metrics (
    testing_start       DATE,
    testing_end         DATE,

    model               VARCHAR(20),
    train_criterion     VARCHAR(20),
    val_criterion       VARCHAR(20),

    rtn_bottom          DECIMAL(18,4),
    rtn_weighted        DECIMAL(18,4),

    mse                 DECIMAL(18,6),
    f_pvalue            DECIMAL(18,6),
    r_sqr               DECIMAL(18,6),

    training_start      DATE,
    training_end        DATE,
    validation_start    DATE,
    validation_end      DATE,

    PRIMARY KEY (testing_start, testing_end, model, train_criterion, val_criterion)
);