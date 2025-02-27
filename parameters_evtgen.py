
model_base_dir = "/data/ana/PointSource/DNNCascade/utils/exported_models/version-0.0"

default_dnn_cascade_selection = {

    # 2-Cascade model
    'EGen_2Cascade_Reco_config': {
        'seed_base': 'HESEMonopodFit',
        'seed_settings': {
            'seed_distances': [5, 50, 300],
            'additional_seeds': ['event_selection_cascade'],
            'add_reverse': False,
            'nside': 1,
            'min_energy': 1,
            'cluster_settings': {
                'pulse_key': 'SplitInIceDSTPulses',
                'n_clusters': 5,
                'min_dist': 200,
                'min_cluster_charge': 3,
                'min_hit_doms': 3,
            },
        },
        'add_circular_err': False,
        'add_covariances': True,
        'parameter_boundaries': {
            'cascade_x': [-750, 750],
            'cascade_y': [-750, 750],
            'cascade_z': [-800, 750],
            'cascade_energy': [0, 1e8],
            'cascade_cascade_00001_distance': [-2000, 2000],
            'cascade_cascade_00001_energy': [0, 1e8],
        },
        'scipy_optimizer_settings': {'options': {'gtol': 1}},

        'pulse_key': 'SplitInIceDSTPulses',
        'partial_exclusion': True,
        'exclude_bright_doms': True,
        'dom_and_tw_exclusions': [
            'BadDomsList', 'CalibrationErrata', 'SaturationWindows'],
        'merge_pulses_time_threshold': None,
        'models_dir': model_base_dir + '/egenerator',
    },

}