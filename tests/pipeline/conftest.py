# Import xgboost before torch to avoid a segfault on macOS.
# PyTorch initialises its OpenMP thread pool at import time; if xgboost is
# imported afterwards it races with that pool when building a DMatrix.
# Importing xgboost first registers its own thread backend safely.
import xgboost  # noqa: F401
