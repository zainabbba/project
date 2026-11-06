def predict_rul(models, features):
    """Return one array of RUL predictions per fitted model."""
    return {
        name: model.predict(features)
        for name, model in models.items()
    }