

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
    #tpz=dict(
    #    Inform='TPZliteInformer',
    #    Estimate='TPZliteEstimator',
    #    Module='rail.estimation.algos.tpz_lite',
    #),
    #lephare=dict(
    #    Inform='LephareInformer',
    #    Estimate='LephareEstimator',
    #    Module='rail.estimation.algos.knn',
    #),
)

CLASSIFIERS = dict(
    equal_count = dict(
        Classify="EqualCountClassifier",
        Module="rail.estimation.algos.equal_count",
    ),
    uniform_binning = dict(
        Classify="UniformBinningClassifier",
        Module="rail.estimation.algos.uniform_binning",
    ),
)


SUMMARIZERS = dict(
    naive_stack = dict(
        Summarize="NaiveStackMaskedSummarizer",
        Module="rail.estimation.algos.naive_stack",
    ),
    point_est_hist = dict(
        Summarize="PointEstHistMaskedSummarizer",
        Module="rail.estimation.algos.point_est_hist",
    ),
#    var_inf = dict(
#        Summarize=VarInfStackMaskedSummarizer,
#        Module=rail.estimation.algos.var_inf,
#    ),
)

SPEC_SELECTIONS = dict(
    GAMA=dict(
        Select="SpecSelection_GAMA",
        Module="rail.creation.degraders.spectroscopic_selections",
    ),
    BOSS=dict(
        Select="SpecSelection_BOSS",
        Module="rail.creation.degraders.spectroscopic_selections",
    ),
    VVDSf02=dict(
        Select="SpecSelection_VVDSf02",
        Module="rail.creation.degraders.spectroscopic_selections",
    ),
    zCOSMOS=dict(
        Select="SpecSelection_zCOSMOS",
        Module="rail.creation.degraders.spectroscopic_selections",
    ),
    HSC=dict(
        Select="SpecSelection_HSC",
        Module="rail.creation.degraders.spectroscopic_selections",
    ),
)
