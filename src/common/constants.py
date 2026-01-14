channel_params = {
        "tv_spend": {"decay": 0.6, "gamma": 0.5},
        "digital_spend": {"decay": 0.4, "gamma": 0.6},
        "search_spend": {"decay": 0.3, "gamma": 0.5},
        "social_spend": {"decay": 0.5, "gamma": 0.4},
    }

features_mmm = [
        "tv_spend_adstock",
        "digital_spend_adstock",
        "search_spend_adstock",
        "social_spend_adstock",
        "promo_flag",
        "holiday_flag",
        "price_index",
    ]

baseline_features = [
        "price_index",
        "promo_flag",
        "holiday_flag",
        "weekofyear",
    ]