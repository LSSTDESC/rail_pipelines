

PZ_ALGORITHMS = dict(
    train_z=dict(
        Inform='TrainZInformer',
        Estimate='TrainZEstimator',
        Module='rail.estimation.algos.train_z',
    ),
    simplenn=dict(
        Inform='SklNeurNetInformer',
        Estimate='SklNeurNetEstimator',
        Module='rail.estimation.algos.sklearn_neurnet',
    ),
    knn=dict(
        Inform='KNearNeighInformer',
        Estimate='KNearNeighEstimator',
        Module='rail.estimation.algos.k_nearneigh',
    ),
    bpz=dict(
        Inform='BPZliteInformer',
        Estimate='BPZliteEstimator',
        Module='rail.estimation.algos.bpz_lite',
    ),
    fzboost=dict(
        Inform='FlexZBoostInformer',
        Estimate='FlexZBoostEstimator',
        Module='rail.estimation.algos.flexzboost',
    ),
    gpz=dict(
        Inform='GPzInformer',
        Estimate='GPzEstimator',
        Module='rail.estimation.algos.gpz',
    ),
    tpz=dict(
        Inform='TPZliteInformer',
        Estimate='TPZliteEstimator',
        Module='rail.estimation.algos.tpz_lite',
    ),
    #lephare=dict(
    #    Inform='LephareInformer',
    #    Estimate='LephareEstimator',
    #    Module='rail.estimation.algos.knn',
    #),
)
