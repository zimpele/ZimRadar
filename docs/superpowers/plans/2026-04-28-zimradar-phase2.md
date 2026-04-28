# ZimRadar Phase 2 — Intelligence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ZoeDepth terrain analysis, Chronos time-series forecasting, XGBoost tabular risk classification, and a pgvector + cross-encoder RAG retriever, with CI-gated evals for all four models.

**Architecture:** Four independent modules in `src/pipeline/` and `src/rag/retriever.py`. All blocking model calls wrapped with `asyncio.to_thread`. Depth inference is Redis-cached by `(s3_path, model_version)`. XGBoost model artifact stored in S3 and loaded at inference time. Each module has its own eval under `tests/evals/`, gated in CI.

**Tech Stack:** ZoeDepth (`Intel/zoedepth-nyu`, `transformers`), Chronos (`amazon/chronos-t5-small`, `chronos-forecasting` package), XGBoost 2.0, CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `sentence-transformers`), pgvector raw SQL, fakeredis (tests).

---

## File Map

**Create:**
- `src/storage/cache.py` — Redis inference cache (get/set by `(s3_path, model_version)`)
- `src/storage/migrations/002_phase2.sql` — `depth_results` table
- `src/pipeline/depth.py` — ZoeDepth pipeline + `run_depth_for_tile(tile_id, s3_path)`
- `src/pipeline/forecasting.py` — Chronos pipeline + `run_forecast_for_region(region_id)`
- `src/pipeline/classifier.py` — XGBoost train/infer + `run_classification_for_region(region_id)`
- `src/rag/retriever.py` — pgvector similarity + metadata filters + CrossEncoder reranking
- `tests/storage/test_cache.py`
- `tests/pipeline/test_depth.py`
- `tests/pipeline/test_forecasting.py`
- `tests/pipeline/test_classifier.py`
- `tests/rag/test_retriever.py`
- `tests/fixtures/noaa_holdout.csv` — 500-row synthetic NOAA daily series (generated in Task 9)
- `tests/fixtures/xgboost_labels.json` — 120 synthetic labelled feature records (generated in Task 10)
- `tests/evals/test_forecast_eval.py` — CRPS regression threshold < 0.90
- `tests/evals/test_tabular_eval.py` — AUC-ROC regression threshold ≥ 0.80

**Modify:**
- `pyproject.toml` — add `xgboost>=2.0`, `chronos-forecasting>=1.0`
- `src/storage/models.py` — append `DepthResult` ORM class
- `src/storage/s3.py` — add `upload_model(local_path, s3_key)` and `download_model(s3_key, dest_path)`
- `.github/workflows/ci.yml` — add `forecast-eval` and `tabular-eval` jobs

---

### Task 1: Add Phase 2 dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add xgboost and chronos-forecasting to dependencies**

In `pyproject.toml`, add to the `dependencies` list after `"pillow>=10.0"`:

```toml
  "xgboost>=2.0",
  "chronos-forecasting>=1.0",
```

The full `dependencies` list ends with:
```toml
  "pillow>=10.0",
  "xgboost>=2.0",
  "chronos-forecasting>=1.0",
]
```

- [ ] **Step 2: Reinstall and verify imports**

Run: `pip install uv && uv pip install --system -e .`

Then verify:
```bash
python -c "import xgboost; import chronos; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add xgboost and chronos-forecasting dependencies"
```

---

### Task 2: Redis inference cache

**Files:**
- Create: `src/storage/cache.py`
- Create: `tests/storage/test_cache.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/storage/test_cache.py
import fakeredis
import pytest
from unittest.mock import patch


def test_cache_roundtrip():
    fake = fakeredis.FakeRedis(decode_responses=True)
    with patch("src.storage.cache._client", fake):
        from src.storage import cache
        cache.set_cached("key1", {"result": 42})
        assert cache.get_cached("key1") == {"result": 42}


def test_cache_miss_returns_none():
    fake = fakeredis.FakeRedis(decode_responses=True)
    with patch("src.storage.cache._client", fake):
        from src.storage import cache
        assert cache.get_cached("nonexistent") is None


def test_make_cache_key_format():
    from src.storage.cache import make_cache_key
    assert make_cache_key("s3://b/tile.tif", "model-v1") == "inference:model-v1:s3://b/tile.tif"


def test_ttl_default_is_7_days():
    from src.storage.cache import DEFAULT_TTL
    assert DEFAULT_TTL == 7 * 24 * 3600
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/storage/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.storage.cache'`

- [ ] **Step 3: Implement the cache module**

```python
# src/storage/cache.py
import json
import redis as redis_lib
from src.config import get_settings

DEFAULT_TTL = 7 * 24 * 3600  # 7 days

_client: redis_lib.Redis | None = None


def _get_client() -> redis_lib.Redis:
    global _client
    if _client is None:
        _client = redis_lib.from_url(get_settings().redis_url, decode_responses=True)
    return _client


def make_cache_key(s3_path: str, model_version: str) -> str:
    return f"inference:{model_version}:{s3_path}"


def get_cached(key: str) -> dict | None:
    raw = _get_client().get(key)
    return json.loads(raw) if raw else None


def set_cached(key: str, value: dict, ttl: int = DEFAULT_TTL) -> None:
    _get_client().setex(key, ttl, json.dumps(value))
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/storage/test_cache.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/storage/cache.py tests/storage/test_cache.py
git commit -m "feat: add Redis inference cache module"
```

---

### Task 3: Depth results DB schema

**Files:**
- Create: `src/storage/migrations/002_phase2.sql`
- Modify: `src/storage/models.py`

- [ ] **Step 1: Write the migration SQL**

```sql
-- src/storage/migrations/002_phase2.sql
CREATE TABLE IF NOT EXISTS depth_results (
    id SERIAL PRIMARY KEY,
    tile_id INTEGER NOT NULL REFERENCES sentinel2_tiles(id) ON DELETE CASCADE,
    flood_zone_geojson JSONB,
    depth_map_s3_path TEXT,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tile_id, model_version)
);
```

- [ ] **Step 2: Add the ORM class to `src/storage/models.py`**

Append after the `FailedIngestion` class (at the end of the file):

```python
class DepthResult(Base):
    __tablename__ = "depth_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tile_id: Mapped[int] = mapped_column(ForeignKey("sentinel2_tiles.id", ondelete="CASCADE"))
    flood_zone_geojson: Mapped[dict | None] = mapped_column(JSONB)
    depth_map_s3_path: Mapped[str | None] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("tile_id", "model_version"),)
```

- [ ] **Step 3: Verify the model imports cleanly**

Run: `python -c "from src.storage.models import DepthResult; print(DepthResult.__tablename__)"`
Expected: `depth_results`

- [ ] **Step 4: Commit**

```bash
git add src/storage/migrations/002_phase2.sql src/storage/models.py
git commit -m "feat: add depth_results table and ORM model"
```

---

### Task 4: S3 model upload/download methods

**Files:**
- Modify: `src/storage/s3.py`
- Create: `tests/storage/test_s3.py` (already exists — add to it)

- [ ] **Step 1: Write failing tests for the new S3 methods**

Add these two tests to `tests/storage/test_s3.py` (open the file first, append before the final newline):

```python
def test_upload_model_uses_tiles_bucket(mock_boto_client):
    from src.storage.s3 import S3Client
    client = S3Client()
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(b'{}')
        tmp = f.name
    try:
        key = client.upload_model(tmp, "models/xgboost.json")
        assert key == "models/xgboost.json"
        mock_boto_client.upload_file.assert_called_with(tmp, "zimradar-tiles", "models/xgboost.json")
    finally:
        os.unlink(tmp)


def test_download_model_creates_parent_dirs(mock_boto_client, tmp_path):
    from src.storage.s3 import S3Client
    client = S3Client()
    dest = str(tmp_path / "subdir" / "model.json")
    client.download_model("models/xgboost.json", dest)
    mock_boto_client.download_file.assert_called_with("zimradar-tiles", "models/xgboost.json", dest)
```

- [ ] **Step 2: Run to confirm they fail**

Run: `pytest tests/storage/test_s3.py::test_upload_model_uses_tiles_bucket -v`
Expected: FAIL with `AttributeError: 'S3Client' object has no attribute 'upload_model'`

- [ ] **Step 3: Add methods to `src/storage/s3.py`**

Append after the `upload_pdf` method:

```python
    def upload_model(self, local_path: str, s3_key: str) -> str:
        settings = get_settings()
        self._client.upload_file(local_path, settings.s3_bucket_tiles, s3_key)
        return s3_key

    def download_model(self, s3_key: str, dest_path: str) -> None:
        settings = get_settings()
        if dirname := os.path.dirname(dest_path):
            os.makedirs(dirname, exist_ok=True)
        self._client.download_file(settings.s3_bucket_tiles, s3_key, dest_path)
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/storage/test_s3.py -v`
Expected: All PASSED (including pre-existing tests)

- [ ] **Step 5: Commit**

```bash
git add src/storage/s3.py tests/storage/test_s3.py
git commit -m "feat: add S3 model upload/download methods"
```

---

### Task 5: ZoeDepth depth estimation pipeline

**Files:**
- Create: `src/pipeline/depth.py`
- Create: `tests/pipeline/test_depth.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_depth.py
import json
import numpy as np
import os
import pytest
import rasterio
import tempfile
import torch
from rasterio.transform import from_bounds
from unittest.mock import AsyncMock, MagicMock, patch


def _make_test_tile(path: str) -> None:
    data = np.random.default_rng(0).random((3, 64, 64)).astype(np.float32)
    transform = from_bounds(0, 0, 1, 1, 64, 64)
    with rasterio.open(
        path, "w", driver="GTiff", height=64, width=64, count=3,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)


def test_depth_pipeline_estimate_returns_flood_zone_geojson():
    from src.pipeline.depth import DepthPipeline, MODEL_VERSION

    fake_depth = torch.tensor(np.random.default_rng(1).random((64, 64)).astype(np.float32))
    mock_pipe = MagicMock(return_value={"predicted_depth": fake_depth})

    with tempfile.TemporaryDirectory() as tmpdir:
        tile_path = os.path.join(tmpdir, "tile.tif")
        _make_test_tile(tile_path)

        dp = DepthPipeline.__new__(DepthPipeline)
        dp._pipe = mock_pipe
        result = dp.estimate(tile_path)

    assert result["model_version"] == MODEL_VERSION
    assert result["flood_zone_geojson"]["type"] == "FeatureCollection"
    assert isinstance(result["flood_zone_geojson"]["features"], list)


def test_depth_pipeline_flood_mask_covers_top_10_percent():
    from src.pipeline.depth import DepthPipeline

    # Depth map: values 1..64*64, top 10% should be the highest values
    depth_vals = np.arange(1, 64 * 64 + 1, dtype=np.float32).reshape(64, 64)
    fake_depth = torch.from_numpy(depth_vals)
    mock_pipe = MagicMock(return_value={"predicted_depth": fake_depth})

    with tempfile.TemporaryDirectory() as tmpdir:
        tile_path = os.path.join(tmpdir, "tile.tif")
        _make_test_tile(tile_path)

        dp = DepthPipeline.__new__(DepthPipeline)
        dp._pipe = mock_pipe
        result = dp.estimate(tile_path)

    # Flood zone features are present for the highest-depth pixels
    assert len(result["flood_zone_geojson"]["features"]) > 0


@pytest.mark.asyncio
async def test_run_depth_for_tile_skips_download_when_cached():
    from src.pipeline.depth import run_depth_for_tile, MODEL_VERSION

    cached_result = {
        "flood_zone_geojson": {"type": "FeatureCollection", "features": []},
        "model_version": MODEL_VERSION,
    }

    mock_session = MagicMock()
    mock_session.execute = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("src.pipeline.depth.get_cached", return_value=cached_result),
        patch("src.pipeline.depth.S3Client") as mock_s3,
        patch("src.pipeline.depth.DepthPipeline") as mock_dp,
        patch("src.pipeline.depth.get_async_session", return_value=mock_session),
    ):
        await run_depth_for_tile(tile_id=1, processed_s3_path="s3/path.tif")

    mock_s3.assert_not_called()
    mock_dp.assert_not_called()
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_run_depth_for_tile_caches_result_on_miss():
    from src.pipeline.depth import run_depth_for_tile, MODEL_VERSION

    fresh_result = {
        "flood_zone_geojson": {"type": "FeatureCollection", "features": []},
        "model_version": MODEL_VERSION,
    }

    mock_session = MagicMock()
    mock_session.execute = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    set_cached_calls = []

    with (
        patch("src.pipeline.depth.get_cached", return_value=None),
        patch("src.pipeline.depth.set_cached", side_effect=lambda k, v: set_cached_calls.append(v)),
        patch("src.pipeline.depth.S3Client"),
        patch("src.pipeline.depth.DepthPipeline") as mock_dp_cls,
        patch("src.pipeline.depth.get_async_session", return_value=mock_session),
    ):
        mock_dp_cls.return_value.estimate.return_value = fresh_result
        await run_depth_for_tile(tile_id=2, processed_s3_path="s3/other.tif")

    assert len(set_cached_calls) == 1
    assert set_cached_calls[0]["model_version"] == MODEL_VERSION
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/pipeline/test_depth.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline.depth'`

- [ ] **Step 3: Implement the depth pipeline**

```python
# src/pipeline/depth.py
import asyncio
import json
import logging
import numpy as np
import torch
import rasterio
from datetime import datetime, timezone
from PIL import Image
from sqlalchemy import text
from src.storage.db import get_async_session
from src.storage.cache import make_cache_key, get_cached, set_cached

logger = logging.getLogger(__name__)

MODEL_ID = "Intel/zoedepth-nyu"
MODEL_VERSION = "zoedepth-nyu-v1"


class DepthPipeline:
    def __init__(self):
        from transformers import pipeline as hf_pipeline
        self._pipe = hf_pipeline("depth-estimation", model=MODEL_ID, device="cpu")

    def estimate(self, tile_path: str) -> dict:
        with rasterio.open(tile_path) as src:
            data = src.read([1, 2, 3])  # (3, H, W) float32 [0, 1]
            transform = src.transform

        rgb = (data.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(rgb)

        result = self._pipe(image)
        depth_map = result["predicted_depth"].squeeze().numpy()  # (H, W)

        # Lowest 10% elevation = largest depth values (overhead view: far = low elevation)
        threshold = np.percentile(depth_map, 90)
        flood_mask = (depth_map >= threshold).astype(np.uint8)

        from rasterio import features
        shapes = list(features.shapes(flood_mask, transform=transform))
        flood_features = [
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {"label": "flood_accumulation_zone"},
            }
            for geom, val in shapes
            if val == 1
        ]

        return {
            "flood_zone_geojson": {"type": "FeatureCollection", "features": flood_features},
            "model_version": MODEL_VERSION,
        }


async def run_depth_for_tile(tile_id: int, processed_s3_path: str) -> None:
    import tempfile
    import os
    from src.storage.s3 import S3Client

    cache_key = make_cache_key(processed_s3_path, MODEL_VERSION)
    result = get_cached(cache_key)

    if result is None:
        s3 = S3Client()
        pipeline = await asyncio.to_thread(DepthPipeline)
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "tile.tif")
            await asyncio.to_thread(s3.download_tile, processed_s3_path, local_path)
            result = await asyncio.to_thread(pipeline.estimate, local_path)
        set_cached(cache_key, result)

    async with get_async_session() as session:
        await session.execute(
            text("""
                INSERT INTO depth_results
                    (tile_id, flood_zone_geojson, model_version, created_at)
                VALUES
                    (:tile_id, :flood_zone_geojson::jsonb, :model_version, :created_at)
                ON CONFLICT (tile_id, model_version) DO NOTHING
            """),
            {
                "tile_id": tile_id,
                "flood_zone_geojson": json.dumps(result["flood_zone_geojson"]),
                "model_version": result["model_version"],
                "created_at": datetime.now(timezone.utc),
            },
        )
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/pipeline/test_depth.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/depth.py tests/pipeline/test_depth.py
git commit -m "feat: add ZoeDepth depth estimation pipeline with Redis caching"
```

---

### Task 6: Chronos forecasting pipeline

**Files:**
- Create: `src/pipeline/forecasting.py`
- Create: `tests/pipeline/test_forecasting.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_forecasting.py
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_compute_flood_risk_flag_triggers_when_threshold_exceeded():
    from src.pipeline.forecasting import _compute_flood_risk_flag

    historical = list(range(1, 101))  # p95 ≈ 95
    high = [96.0, 97.0, 98.0] + [1.0] * 27   # has 3 consecutive > p95
    low = [1.0] * 30
    # 12/20 = 0.6 > 0.3 → True
    samples = np.array([high] * 12 + [low] * 8)
    assert _compute_flood_risk_flag(samples, historical) is True


def test_compute_flood_risk_flag_does_not_trigger_below_threshold():
    from src.pipeline.forecasting import _compute_flood_risk_flag

    historical = list(range(1, 101))
    high = [96.0, 97.0, 98.0] + [1.0] * 27
    low = [1.0] * 30
    # 5/20 = 0.25 < 0.3 → False
    samples = np.array([high] * 5 + [low] * 15)
    assert _compute_flood_risk_flag(samples, historical) is False


def test_compute_fire_risk_flag_triggers_when_hot_and_vegetation_declining():
    from src.pipeline.forecasting import _compute_fire_risk_flag

    hot = [41.0] * 30   # all 30 days > 40°C → any 7 consecutive are > 40°C
    samples = np.array([hot] * 10)  # P = 1.0 > 0.3
    assert _compute_fire_risk_flag(samples, vegetation_trend=-0.01) is True


def test_compute_fire_risk_flag_no_trigger_when_vegetation_stable():
    from src.pipeline.forecasting import _compute_fire_risk_flag

    hot = [41.0] * 30
    samples = np.array([hot] * 10)
    assert _compute_fire_risk_flag(samples, vegetation_trend=0.0) is False


def test_compute_fire_risk_flag_no_trigger_when_cool():
    from src.pipeline.forecasting import _compute_fire_risk_flag

    cool = [20.0] * 30  # never exceeds 40°C
    samples = np.array([cool] * 10)
    assert _compute_fire_risk_flag(samples, vegetation_trend=-0.05) is False


@pytest.mark.asyncio
async def test_run_forecast_for_region_skips_when_insufficient_data():
    from src.pipeline.forecasting import run_forecast_for_region

    mock_rows = MagicMock()
    mock_rows.fetchall.return_value = [
        MagicMock(precipitation_mm=1.0, temp_max_c=20.0)
    ] * 10  # only 10 rows, < 30 minimum

    mock_session = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_rows)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("src.pipeline.forecasting.get_async_session", return_value=mock_session):
        await run_forecast_for_region(region_id=1)

    # Only the initial SELECT is called; no INSERT occurs
    assert mock_session.execute.call_count == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/pipeline/test_forecasting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline.forecasting'`

- [ ] **Step 3: Implement the forecasting pipeline**

```python
# src/pipeline/forecasting.py
import asyncio
import json
import logging
import numpy as np
import torch
from datetime import datetime, timezone
from sqlalchemy import text
from src.storage.db import get_async_session

logger = logging.getLogger(__name__)

MODEL_ID = "amazon/chronos-t5-small"
MODEL_VERSION = "chronos-t5-small-v1"
FORECAST_HORIZONS = [30, 60, 90]
NUM_SAMPLES = 20
MIN_HISTORY_ROWS = 30


class _ChronosPipeline:
    def __init__(self):
        from chronos import ChronosPipeline as _CP
        self._pipe = _CP.from_pretrained(
            MODEL_ID, device_map="cpu", torch_dtype=torch.float32
        )

    def forecast(self, series: list[float], horizon: int) -> dict:
        context = torch.tensor(series, dtype=torch.float32).unsqueeze(0)
        samples = self._pipe.predict(context, prediction_length=horizon, num_samples=NUM_SAMPLES)
        arr = samples.squeeze(0).numpy()  # (NUM_SAMPLES, horizon)
        return {
            "mean": arr.mean(axis=0).tolist(),
            "q10": np.percentile(arr, 10, axis=0).tolist(),
            "q90": np.percentile(arr, 90, axis=0).tolist(),
            "samples": arr.tolist(),
        }


def _compute_flood_risk_flag(samples_30d: np.ndarray, historical_precip: list[float]) -> bool:
    """P(any 3 consecutive days exceed the 95th historical percentile) > 0.3"""
    if not historical_precip:
        return False
    p95 = float(np.percentile(historical_precip, 95))
    count = 0
    for sample in samples_30d:
        for i in range(len(sample) - 2):
            if sample[i] > p95 and sample[i + 1] > p95 and sample[i + 2] > p95:
                count += 1
                break
    return (count / len(samples_30d)) > 0.3


def _compute_fire_risk_flag(temp_samples_30d: np.ndarray, vegetation_trend: float) -> bool:
    """P(any 7 consecutive days > 40°C AND vegetation declining) > 0.3"""
    if vegetation_trend >= 0:
        return False
    count = 0
    for sample in temp_samples_30d:
        for i in range(len(sample) - 6):
            if all(sample[i + j] > 40.0 for j in range(7)):
                count += 1
                break
    return (count / len(temp_samples_30d)) > 0.3


async def run_forecast_for_region(region_id: int) -> None:
    async with get_async_session() as session:
        rows_result = await session.execute(
            text("""
                SELECT date, precipitation_mm, temp_max_c
                FROM noaa_observations
                WHERE region_id = :rid
                ORDER BY date
            """),
            {"rid": region_id},
        )
        obs = rows_result.fetchall()

    if len(obs) < MIN_HISTORY_ROWS:
        logger.warning(
            "Insufficient NOAA data for region %d (%d rows, need %d)",
            region_id,
            len(obs),
            MIN_HISTORY_ROWS,
        )
        return

    precip_series = [float(r.precipitation_mm or 0.0) for r in obs]
    temp_series = [float(r.temp_max_c or 20.0) for r in obs]

    chronos = await asyncio.to_thread(_ChronosPipeline)

    forecasts: dict = {}
    for horizon in FORECAST_HORIZONS:
        fc = await asyncio.to_thread(chronos.forecast, precip_series, horizon)
        forecasts[f"forecast_{horizon}d"] = {k: v for k, v in fc.items() if k != "samples"}

    precip_fc_full = await asyncio.to_thread(chronos.forecast, precip_series, 30)
    temp_fc_full = await asyncio.to_thread(chronos.forecast, temp_series, 30)

    precip_samples_30d = np.array(precip_fc_full["samples"])
    temp_samples_30d = np.array(temp_fc_full["samples"])

    # Vegetation trend from segmentation history
    async with get_async_session() as session:
        veg_result = await session.execute(
            text("""
                SELECT (sr.area_stats->>'vegetation')::float AS veg_pct
                FROM segmentation_results sr
                JOIN sentinel2_tiles t ON t.id = sr.tile_id
                WHERE t.region_id = :rid
                ORDER BY t.date DESC
                LIMIT 10
            """),
            {"rid": region_id},
        )
        veg_data = [float(r.veg_pct) for r in veg_result if r.veg_pct is not None]

    vegetation_trend = (
        float(np.polyfit(range(len(veg_data)), veg_data, 1)[0])
        if len(veg_data) >= 2
        else 0.0
    )

    flood_risk_flag = _compute_flood_risk_flag(precip_samples_30d, precip_series)
    fire_risk_flag = _compute_fire_risk_flag(temp_samples_30d, vegetation_trend)

    async with get_async_session() as session:
        await session.execute(
            text("""
                INSERT INTO forecasts
                    (region_id, forecast_30d, forecast_60d, forecast_90d,
                     flood_risk_flag, fire_risk_flag, model_version, created_at)
                VALUES
                    (:region_id, :forecast_30d::jsonb, :forecast_60d::jsonb, :forecast_90d::jsonb,
                     :flood_risk_flag, :fire_risk_flag, :model_version, :created_at)
            """),
            {
                "region_id": region_id,
                "forecast_30d": json.dumps(forecasts["forecast_30d"]),
                "forecast_60d": json.dumps(forecasts["forecast_60d"]),
                "forecast_90d": json.dumps(forecasts["forecast_90d"]),
                "flood_risk_flag": flood_risk_flag,
                "fire_risk_flag": fire_risk_flag,
                "model_version": MODEL_VERSION,
                "created_at": datetime.now(timezone.utc),
            },
        )
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/pipeline/test_forecasting.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/forecasting.py tests/pipeline/test_forecasting.py
git commit -m "feat: add Chronos time-series forecasting pipeline"
```

---

### Task 7: XGBoost tabular classifier

**Files:**
- Create: `src/pipeline/classifier.py`
- Create: `tests/pipeline/test_classifier.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_classifier.py
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_train_classifier_produces_valid_probabilities():
    from src.pipeline.classifier import train_classifier, FEATURE_NAMES, RISK_TIERS

    rng = np.random.default_rng(42)
    X = rng.random((80, len(FEATURE_NAMES))).astype(np.float32)
    y = rng.integers(0, len(RISK_TIERS), size=80)

    model = train_classifier(X, y)

    proba = model.predict_proba(X[:1])
    assert proba.shape == (1, len(RISK_TIERS))
    assert abs(proba[0].sum() - 1.0) < 1e-5


def test_classify_region_features_returns_valid_tier():
    from src.pipeline.classifier import (
        classify_region_features,
        train_classifier,
        FEATURE_NAMES,
        RISK_TIERS,
    )

    rng = np.random.default_rng(0)
    X = rng.random((80, len(FEATURE_NAMES))).astype(np.float32)
    y = rng.integers(0, len(RISK_TIERS), size=80)
    model = train_classifier(X, y)

    with patch("src.pipeline.classifier.load_classifier_from_s3", return_value=model):
        features = {name: 0.5 for name in FEATURE_NAMES}
        tier, confidence = classify_region_features(features)

    assert tier in RISK_TIERS
    assert 0.0 <= confidence <= 1.0


@pytest.mark.asyncio
async def test_run_classification_for_region_writes_risk_assessment():
    from src.pipeline.classifier import run_classification_for_region

    mock_features = {
        "flood_events_5yr": 2.0,
        "avg_precipitation_trend": 0.1,
        "vegetation_loss_pct": 0.05,
        "urban_density": 0.3,
        "elevation_variance": 20.0,
        "infrastructure_age_proxy": 0.5,
    }

    mock_forecast_row = MagicMock(flood_risk_flag=True, fire_risk_flag=False)
    mock_fc_result = MagicMock(fetchone=MagicMock(return_value=mock_forecast_row))

    insert_calls = []
    mock_session = MagicMock()
    mock_session.execute = AsyncMock(
        side_effect=[mock_fc_result, AsyncMock(side_effect=lambda *a, **kw: insert_calls.append(kw))]
    )
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("src.pipeline.classifier.build_features_for_region", return_value=mock_features),
        patch("src.pipeline.classifier.classify_region_features", return_value=("moderate", 0.7)),
        patch("src.pipeline.classifier.get_async_session", return_value=mock_session),
    ):
        await run_classification_for_region(region_id=1)

    # Both SELECT (forecast) and INSERT (risk_assessment) were executed
    assert mock_session.execute.call_count == 2


def test_composite_score_formula():
    from src.pipeline.classifier import _composite_score
    from src.config import get_settings

    w1, w2, w3 = get_settings().risk_weights
    score = _composite_score(confidence=0.8, flood_flag=True, fire_flag=False)
    expected = w1 * 0.8 + w2 * 1.0 + w3 * 0.0
    assert abs(score - expected) < 1e-6
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/pipeline/test_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline.classifier'`

- [ ] **Step 3: Implement the classifier module**

```python
# src/pipeline/classifier.py
import asyncio
import logging
import numpy as np
import xgboost as xgb
from datetime import datetime, timezone
from sqlalchemy import text
from src.config import get_settings
from src.storage.db import get_async_session

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "flood_events_5yr",
    "avg_precipitation_trend",
    "vegetation_loss_pct",
    "urban_density",
    "elevation_variance",
    "infrastructure_age_proxy",
]
RISK_TIERS = ["low", "moderate", "high", "critical"]
MODEL_S3_KEY = "models/xgboost_risk_classifier.json"


def train_classifier(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X, y)
    return model


def save_classifier_to_s3(model: xgb.XGBClassifier) -> None:
    import tempfile
    import os
    from src.storage.s3 import S3Client

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name
    try:
        model.save_model(temp_path)
        S3Client().upload_model(temp_path, MODEL_S3_KEY)
    finally:
        os.unlink(temp_path)


def load_classifier_from_s3() -> xgb.XGBClassifier:
    import tempfile
    import os
    from src.storage.s3 import S3Client

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name
    try:
        S3Client().download_model(MODEL_S3_KEY, temp_path)
        model = xgb.XGBClassifier()
        model.load_model(temp_path)
        return model
    finally:
        os.unlink(temp_path)


def classify_region_features(features: dict[str, float]) -> tuple[str, float]:
    model = load_classifier_from_s3()
    X = np.array([[features[k] for k in FEATURE_NAMES]])
    proba = model.predict_proba(X)[0]
    tier_idx = int(np.argmax(proba))
    return RISK_TIERS[tier_idx], float(proba[tier_idx])


def _composite_score(confidence: float, flood_flag: bool, fire_flag: bool) -> float:
    w1, w2, w3 = get_settings().risk_weights
    return w1 * confidence + w2 * float(flood_flag) + w3 * float(fire_flag)


async def build_features_for_region(region_id: int) -> dict[str, float]:
    async with get_async_session() as session:
        flood_result = await session.execute(
            text("""
                SELECT COUNT(*) FROM fema_declarations
                WHERE disaster_type ILIKE '%flood%'
                AND declaration_date >= NOW() - INTERVAL '5 years'
            """),
        )
        flood_events = int(flood_result.scalar() or 0)

        precip_result = await session.execute(
            text("""
                SELECT precipitation_mm FROM noaa_observations
                WHERE region_id = :rid AND date >= NOW() - INTERVAL '2 years'
                ORDER BY date
            """),
            {"rid": region_id},
        )
        precip_rows = precip_result.fetchall()

    precip_values = [float(r.precipitation_mm or 0.0) for r in precip_rows]
    precip_trend = (
        float(np.polyfit(range(len(precip_values)), precip_values, 1)[0])
        if len(precip_values) >= 2
        else 0.0
    )

    async with get_async_session() as session:
        seg_result = await session.execute(
            text("""
                SELECT sr.area_stats
                FROM segmentation_results sr
                JOIN sentinel2_tiles t ON t.id = sr.tile_id
                WHERE t.region_id = :rid
                ORDER BY t.date DESC
                LIMIT 5
            """),
            {"rid": region_id},
        )
        seg_rows = seg_result.fetchall()

    veg_fracs = [float((r.area_stats or {}).get("vegetation", 0.3)) for r in seg_rows]
    urban_fracs = [float((r.area_stats or {}).get("urban", 0.1)) for r in seg_rows]

    # seg_rows ordered DESC, so index 0 is newest, -1 is oldest
    vegetation_loss_pct = max(0.0, veg_fracs[-1] - veg_fracs[0]) if len(veg_fracs) >= 2 else 0.0
    urban_density = float(np.mean(urban_fracs)) if urban_fracs else 0.1

    async with get_async_session() as session:
        depth_result = await session.execute(
            text("""
                SELECT COUNT(*) FROM depth_results dr
                JOIN sentinel2_tiles t ON t.id = dr.tile_id
                WHERE t.region_id = :rid
            """),
            {"rid": region_id},
        )
        depth_count = int(depth_result.scalar() or 0)

    elevation_variance = min(100.0, float(depth_count) * 10.0)
    infrastructure_age_proxy = 0.5  # static until OSM feature extraction is added in Phase 3

    return {
        "flood_events_5yr": float(flood_events),
        "avg_precipitation_trend": precip_trend,
        "vegetation_loss_pct": vegetation_loss_pct,
        "urban_density": urban_density,
        "elevation_variance": elevation_variance,
        "infrastructure_age_proxy": infrastructure_age_proxy,
    }


async def run_classification_for_region(region_id: int) -> None:
    features = await build_features_for_region(region_id)
    tier, confidence = await asyncio.to_thread(classify_region_features, features)

    async with get_async_session() as session:
        forecast_result = await session.execute(
            text("""
                SELECT flood_risk_flag, fire_risk_flag FROM forecasts
                WHERE region_id = :rid ORDER BY created_at DESC LIMIT 1
            """),
            {"rid": region_id},
        )
        forecast_row = forecast_result.fetchone()

    flood_flag = bool(forecast_row.flood_risk_flag) if forecast_row else False
    fire_flag = bool(forecast_row.fire_risk_flag) if forecast_row else False
    composite = _composite_score(confidence, flood_flag, fire_flag)

    async with get_async_session() as session:
        await session.execute(
            text("""
                INSERT INTO risk_assessments
                    (region_id, risk_tier, confidence, composite_score, assessed_at)
                VALUES
                    (:region_id, :risk_tier, :confidence, :composite_score, :assessed_at)
            """),
            {
                "region_id": region_id,
                "risk_tier": tier,
                "confidence": confidence,
                "composite_score": composite,
                "assessed_at": datetime.now(timezone.utc),
            },
        )
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/pipeline/test_classifier.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/classifier.py tests/pipeline/test_classifier.py
git commit -m "feat: add XGBoost tabular risk classifier"
```

---

### Task 8: RAG retriever (pgvector + cross-encoder reranking)

**Files:**
- Create: `src/rag/retriever.py`
- Create: `tests/rag/test_retriever.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/rag/test_retriever.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_candidate(text: str, similarity: float = 0.5) -> dict:
    return {
        "id": 1,
        "text": text,
        "source_type": "fema",
        "source_id": "DR-1234",
        "chunk_index": 0,
        "metadata": {"county_fips": "06037"},
        "similarity": similarity,
    }


def test_vec_to_pg_formats_correctly():
    from src.rag.retriever import _vec_to_pg
    result = _vec_to_pg([0.1, -0.2, 0.3])
    assert result.startswith("[")
    assert result.endswith("]")
    parts = result[1:-1].split(",")
    assert len(parts) == 3
    assert abs(float(parts[0]) - 0.1) < 1e-6


@pytest.mark.asyncio
async def test_retrieve_returns_empty_list_when_no_candidates():
    from src.rag.retriever import retrieve

    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([]))

    mock_session = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("src.rag.retriever.get_async_session", return_value=mock_session),
        patch("src.rag.retriever.TextEmbedder") as mock_emb_cls,
    ):
        mock_emb_cls.return_value.embed.return_value = [0.1] * 384
        result = await retrieve("flood risk in Los Angeles")

    assert result == []


@pytest.mark.asyncio
async def test_retrieve_reranks_and_returns_top_k():
    from src.rag.retriever import retrieve

    rows = [
        MagicMock(id=i, chunk_text=f"text {i}", source_type="fema",
                  source_id="DR-100", chunk_index=i, metadata={}, similarity=0.5)
        for i in range(10)
    ]

    mock_session = MagicMock()
    mock_session.execute = AsyncMock(return_value=iter(rows))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    # Cross-encoder gives highest score to item 7
    ce_scores = [float(i) for i in range(10)]
    ce_scores[7] = 99.0

    with (
        patch("src.rag.retriever.get_async_session", return_value=mock_session),
        patch("src.rag.retriever.TextEmbedder") as mock_emb_cls,
        patch("src.rag.retriever._get_cross_encoder") as mock_ce_factory,
    ):
        mock_emb_cls.return_value.embed.return_value = [0.1] * 384
        mock_ce_factory.return_value.predict.return_value = ce_scores
        result = await retrieve("query", top_k=3)

    assert len(result) == 3
    assert result[0]["text"] == "text 7"  # highest rerank score
    assert "rerank_score" in result[0]


@pytest.mark.asyncio
async def test_retrieve_builds_metadata_filter_clause():
    from src.rag.retriever import retrieve

    executed_sqls = []
    mock_session = MagicMock()
    mock_session.execute = AsyncMock(
        side_effect=lambda sql, params=None: (
            executed_sqls.append(str(sql)) or iter([])
        )
    )
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("src.rag.retriever.get_async_session", return_value=mock_session),
        patch("src.rag.retriever.TextEmbedder") as mock_emb_cls,
    ):
        mock_emb_cls.return_value.embed.return_value = [0.1] * 384
        await retrieve("query", county_fips="06037", disaster_type="flood")

    assert len(executed_sqls) == 1
    assert "county_fips" in executed_sqls[0]
    assert "disaster_type" in executed_sqls[0]
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/rag/test_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.retriever'`

- [ ] **Step 3: Implement the retriever**

```python
# src/rag/retriever.py
import asyncio
from typing import Any
from sentence_transformers import CrossEncoder
from sqlalchemy import text
from src.storage.db import get_async_session
from src.rag.embed import TextEmbedder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_INITIAL = 20
TOP_K_FINAL = 5

_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def _vec_to_pg(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


async def retrieve(
    query: str,
    *,
    county_fips: str | None = None,
    disaster_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top_k: int = TOP_K_FINAL,
) -> list[dict[str, Any]]:
    embedder = TextEmbedder()
    query_vec = await asyncio.to_thread(embedder.embed, query)
    vec_str = _vec_to_pg(query_vec)

    filter_clauses: list[str] = []
    params: dict[str, Any] = {"embedding": vec_str, "limit": TOP_K_INITIAL}

    if county_fips:
        filter_clauses.append("(metadata->>'county_fips') = :county_fips")
        params["county_fips"] = county_fips
    if disaster_type:
        filter_clauses.append("(metadata->>'disaster_type') = :disaster_type")
        params["disaster_type"] = disaster_type
    if date_from:
        filter_clauses.append("(metadata->>'date')::date >= :date_from::date")
        params["date_from"] = date_from
    if date_to:
        filter_clauses.append("(metadata->>'date')::date <= :date_to::date")
        params["date_to"] = date_to

    where = "WHERE " + " AND ".join(filter_clauses) if filter_clauses else ""

    sql = text(f"""
        SELECT id, chunk_text, source_type, source_id, chunk_index, metadata,
               1 - (embedding <=> :embedding::vector) AS similarity
        FROM text_embeddings
        {where}
        ORDER BY embedding <=> :embedding::vector
        LIMIT :limit
    """)

    async with get_async_session() as session:
        result = await session.execute(sql, params)
        candidates = [
            {
                "id": row.id,
                "text": row.chunk_text,
                "source_type": row.source_type,
                "source_id": row.source_id,
                "chunk_index": row.chunk_index,
                "metadata": row.metadata or {},
                "similarity": float(row.similarity),
            }
            for row in result
        ]

    if not candidates:
        return []

    cross_encoder = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in candidates]
    scores = await asyncio.to_thread(cross_encoder.predict, pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/rag/test_retriever.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Run full unit test suite to catch regressions**

Run: `pytest tests/ -v --ignore=tests/evals/ -x`
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add src/rag/retriever.py tests/rag/test_retriever.py
git commit -m "feat: add pgvector + cross-encoder RAG retriever"
```

---

### Task 9: Generate NOAA holdout fixture and Chronos CRPS eval

**Files:**
- Create: `tests/fixtures/noaa_holdout.csv`
- Create: `tests/evals/test_forecast_eval.py`

- [ ] **Step 1: Generate the NOAA holdout fixture**

Run this Python snippet once from the repo root to create the fixture:

```python
import csv, math, random
from datetime import date, timedelta

random.seed(42)
start = date(2022, 1, 1)
rows = []
for i in range(500):
    d = start + timedelta(days=i)
    # Seasonal precipitation: peaks in winter months
    season = math.sin(2 * math.pi * i / 365)
    precip = max(0.0, 3.0 + 2.0 * season + random.gauss(0, 1.5))
    # Seasonal temperature: peaks in summer
    temp = 18.0 - 10.0 * season + random.gauss(0, 2.0)
    rows.append([d.isoformat(), round(precip, 2), round(temp, 2)])

import os
os.makedirs("tests/fixtures", exist_ok=True)
with open("tests/fixtures/noaa_holdout.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "precipitation_mm", "temp_max_c"])
    writer.writerows(rows)
print(f"Generated {len(rows)} rows")
```

Run: `python -c "exec(open('generate_fixture.py').read())"` or paste directly into a Python shell.
Expected output: `Generated 500 rows`

Verify: `wc -l tests/fixtures/noaa_holdout.csv` → `501` (500 rows + header)

- [ ] **Step 2: Write the CRPS eval test**

```python
# tests/evals/test_forecast_eval.py
import csv
import numpy as np
import pytest
import torch
from pathlib import Path

CRPS_THRESHOLD = 0.90
HOLDOUT_PATH = Path(__file__).parent.parent / "fixtures" / "noaa_holdout.csv"
TRAIN_SIZE = 410   # ~82% of 500
FORECAST_HORIZON = 90


def _crps_ensemble(samples: np.ndarray, observation: float) -> float:
    """CRPS for a single timestep: E|X-y| - 0.5*E|X-X'|"""
    n = len(samples)
    term1 = np.mean(np.abs(samples - observation))
    sorted_s = np.sort(samples)
    k = np.arange(1, n + 1)
    term2 = np.sum((2 * k - n - 1) * sorted_s) / (n * n)
    return float(term1 - term2)


def _mean_crps(forecast_samples: np.ndarray, actuals: np.ndarray) -> float:
    """
    forecast_samples: (n_samples, horizon)
    actuals: (horizon,)
    """
    return float(np.mean([
        _crps_ensemble(forecast_samples[:, t], actuals[t])
        for t in range(len(actuals))
    ]))


@pytest.mark.slow
def test_chronos_crps_below_threshold():
    from chronos import ChronosPipeline

    rows = []
    with open(HOLDOUT_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(float(row["precipitation_mm"]))

    assert len(rows) == 500, f"Expected 500 rows, got {len(rows)}"

    train_series = rows[:TRAIN_SIZE]
    actuals = np.array(rows[TRAIN_SIZE : TRAIN_SIZE + FORECAST_HORIZON])

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    context = torch.tensor(train_series, dtype=torch.float32).unsqueeze(0)
    samples = pipeline.predict(context, prediction_length=FORECAST_HORIZON, num_samples=100)
    forecast_samples = samples.squeeze(0).numpy()  # (100, FORECAST_HORIZON)

    crps = _mean_crps(forecast_samples, actuals)
    print(f"\nChronos CRPS on 90-day NOAA holdout: {crps:.4f} (threshold: {CRPS_THRESHOLD})")

    assert crps < CRPS_THRESHOLD, (
        f"Chronos CRPS {crps:.4f} exceeds regression threshold {CRPS_THRESHOLD}. "
        "This blocks merge — check model or data."
    )
```

- [ ] **Step 3: Run the eval to baseline it (requires GPU or is slow on CPU)**

Run: `pytest tests/evals/test_forecast_eval.py -v -m slow`
Expected: PASSED with CRPS value printed (e.g., `Chronos CRPS on 90-day NOAA holdout: 0.71`)

If CRPS ≥ 0.90 on first run: check that `amazon/chronos-t5-small` downloaded correctly. The model needs an internet connection on first run.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/noaa_holdout.csv tests/evals/test_forecast_eval.py
git commit -m "feat: add NOAA holdout fixture and Chronos CRPS eval"
```

---

### Task 10: Generate XGBoost labels fixture and AUC-ROC eval

**Files:**
- Create: `tests/fixtures/xgboost_labels.json`
- Create: `tests/evals/test_tabular_eval.py`

- [ ] **Step 1: Generate the XGBoost labels fixture**

Run this Python snippet once from the repo root:

```python
import json, math, os
import numpy as np

rng = np.random.default_rng(42)
records = []
for _ in range(120):
    flood_events = int(rng.poisson(1.5))
    precip_trend = float(rng.uniform(-0.5, 0.5))
    veg_loss = float(rng.uniform(0, 0.3))
    urban = float(rng.uniform(0, 1))
    elev_var = float(rng.uniform(0, 100))
    infra_age = float(rng.uniform(0, 1))

    # Deterministic label derived from features (no noise, so AUC can be high)
    score = (
        min(flood_events / 5.0, 1.0) * 0.35
        + max(precip_trend, 0) * 0.25
        + veg_loss * 0.20
        + urban * 0.15
        + (1 - elev_var / 100) * 0.05
    )
    if score < 0.20:
        label = 0   # low
    elif score < 0.40:
        label = 1   # moderate
    elif score < 0.60:
        label = 2   # high
    else:
        label = 3   # critical

    records.append({
        "flood_events_5yr": flood_events,
        "avg_precipitation_trend": round(precip_trend, 4),
        "vegetation_loss_pct": round(veg_loss, 4),
        "urban_density": round(urban, 4),
        "elevation_variance": round(elev_var, 4),
        "infrastructure_age_proxy": round(infra_age, 4),
        "label": label,
    })

os.makedirs("tests/fixtures", exist_ok=True)
with open("tests/fixtures/xgboost_labels.json", "w") as f:
    json.dump(records, f, indent=2)
print(f"Generated {len(records)} labelled records")
```

Expected output: `Generated 120 labelled records`

- [ ] **Step 2: Write the AUC-ROC eval test**

```python
# tests/evals/test_tabular_eval.py
import json
import numpy as np
import pytest
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

AUC_THRESHOLD = 0.80
LABELS_PATH = Path(__file__).parent.parent / "fixtures" / "xgboost_labels.json"


@pytest.mark.slow
def test_xgboost_auc_roc_above_threshold():
    from src.pipeline.classifier import train_classifier, FEATURE_NAMES, RISK_TIERS

    with open(LABELS_PATH) as f:
        records = json.load(f)

    X = np.array([[r[k] for k in FEATURE_NAMES] for r in records], dtype=np.float32)
    y = np.array([r["label"] for r in records])
    classes = list(range(len(RISK_TIERS)))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []

    for train_idx, val_idx in skf.split(X, y):
        model = train_classifier(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[val_idx])
        y_bin = label_binarize(y[val_idx], classes=classes)
        auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
        fold_aucs.append(auc)

    mean_auc = float(np.mean(fold_aucs))
    print(f"\nXGBoost 5-fold mean AUC-ROC: {mean_auc:.4f} (threshold: {AUC_THRESHOLD})")

    assert mean_auc >= AUC_THRESHOLD, (
        f"XGBoost AUC-ROC {mean_auc:.4f} is below regression threshold {AUC_THRESHOLD}. "
        "This blocks merge — check features or model config."
    )
```

- [ ] **Step 3: Run the eval to baseline it**

Run: `pytest tests/evals/test_tabular_eval.py -v -m slow`
Expected: PASSED with AUC-ROC value printed (e.g., `XGBoost 5-fold mean AUC-ROC: 0.91`)

If AUC < 0.80: check the label generation script — the deterministic scoring function should produce labels that XGBoost can learn cleanly. Re-run the fixture generation with more records (increase to 200).

- [ ] **Step 4: Add `scikit-learn` to eval extras in `pyproject.toml`**

In `pyproject.toml`, the `eval` extras already have `datasets>=2.0` and `torchgeo>=0.5`. Add `scikit-learn`:
```toml
eval = [
  "datasets>=2.0",
  "torchgeo>=0.5",
  "scikit-learn>=1.4",
]
```

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/xgboost_labels.json tests/evals/test_tabular_eval.py pyproject.toml
git commit -m "feat: add XGBoost labels fixture and AUC-ROC eval"
```

---

### Task 11: Update CI for new evals

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add forecast-eval and tabular-eval jobs**

In `.github/workflows/ci.yml`, after the existing `eval` job (which runs segmentation eval), add two new jobs:

```yaml
  forecast-eval:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'pull_request'

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install uv && uv pip install --system -e ".[dev,eval]"

      - name: Run Chronos CRPS eval
        run: pytest tests/evals/test_forecast_eval.py -v -m slow

  tabular-eval:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'pull_request'

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install uv && uv pip install --system -e ".[dev,eval]"

      - name: Run XGBoost AUC-ROC eval
        run: pytest tests/evals/test_tabular_eval.py -v -m slow
```

- [ ] **Step 2: Verify the YAML is valid**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "YAML valid"`
Expected: `YAML valid`

- [ ] **Step 3: Run the full unit test suite one final time**

Run: `pytest tests/ -v --ignore=tests/evals/ -x`
Expected: All PASSED

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add forecast and tabular eval jobs to CI pipeline"
```
