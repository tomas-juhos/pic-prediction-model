CREATE TABLE gbm_parameters (
    testing_start               DATE,
    testing_end                 DATE,

    model_id                    BIGINT,
    val_criterion               VARCHAR(20),

    max_depth                   INTEGER,
    num_leaves                  INTEGER,
    min_data_in_leaf            INTEGER,
    seed                        INTEGER,
    -- time_steps                  INTEGER,
    -- "verbose"                   INTEGER,


    PRIMARY KEY (testing_start, testing_end, model_id, val_criterion)
);