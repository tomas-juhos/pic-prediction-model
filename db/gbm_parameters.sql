CREATE TABLE gbm_parameters (
    universe_constr             VARCHAR(50),
    testing_start               DATE,
    testing_end                 DATE,

    model_id                    BIGINT,
    val_criterion               VARCHAR(20),

    max_depth                   INTEGER,
    num_leaves                  INTEGER,
    min_data_in_leaf            INTEGER,
    seed                        INTEGER,

    PRIMARY KEY (universe_constr, testing_start, testing_end, model_id, val_criterion)
);