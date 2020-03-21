best_params_grid_search = {'bootstrap': True,
                           'max_depth': 7,
                           'max_features': 'sqrt',
                           'min_samples_leaf': 2,
                           'min_samples_split': 2,
                           'n_estimators': 10}

grid_search_5_split = {'bootstrap': True,
                       'max_depth': 7,
                       'max_features': 'auto',
                       'min_samples_leaf': 1,
                       'min_samples_split': 10,
                       'n_estimators': 31}

best_params_randomize_search = {'bootstrap': True,
                                'max_depth': 8,
                                'max_features': 'auto',
                                'min_samples_leaf': 1,
                                'min_samples_split': 10,
                                'n_estimators': 157}

randomize_search_5_split = {'bootstrap': True,
                            'max_depth': 8,
                            'max_features': 'sqrt',
                            'min_samples_leaf': 2,
                            'min_samples_split': 5,
                            'n_estimators': 94}