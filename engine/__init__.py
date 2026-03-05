# Engine layer: models, trainer, predictor, fick
__version__ = "0.1.0"

from engine.fick import fick_analytical, generate_mock_data  # noqa: I001

__all__ = ["__version__", "fick_analytical", "generate_mock_data"]
