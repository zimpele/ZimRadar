from datetime import datetime, date
from typing import Any
from uuid import UUID
from sqlalchemy import Integer, Float, Boolean, Text, Date, DateTime, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Region(Base):
    __tablename__ = "regions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    bbox: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Sentinel2Tile(Base):
    __tablename__ = "sentinel2_tiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[int] = mapped_column(ForeignKey("regions.id", ondelete="CASCADE"))
    s3_path: Mapped[str] = mapped_column(Text, nullable=False)
    processed_s3_path: Mapped[str | None] = mapped_column(Text)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("region_id", "s3_path"),)


class NOAAObservation(Base):
    __tablename__ = "noaa_observations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    station_id: Mapped[str] = mapped_column(Text, nullable=False)
    region_id: Mapped[int] = mapped_column(ForeignKey("regions.id", ondelete="CASCADE"))
    date: Mapped[date] = mapped_column(Date, nullable=False)
    precipitation_mm: Mapped[float | None] = mapped_column(Float)
    temp_max_c: Mapped[float | None] = mapped_column(Float)
    temp_min_c: Mapped[float | None] = mapped_column(Float)
    soil_moisture: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (UniqueConstraint("station_id", "date"),)


class FEMADeclaration(Base):
    __tablename__ = "fema_declarations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    disaster_number: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    state: Mapped[str | None] = mapped_column(Text)
    county_fips: Mapped[str | None] = mapped_column(Text)
    disaster_type: Mapped[str | None] = mapped_column(Text)
    declaration_date: Mapped[date | None] = mapped_column(Date)
    incident_begin: Mapped[date | None] = mapped_column(Date)
    incident_end: Mapped[date | None] = mapped_column(Date)
    declaration_title: Mapped[str | None] = mapped_column(Text)


class SegmentationResult(Base):
    __tablename__ = "segmentation_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tile_id: Mapped[int] = mapped_column(ForeignKey("sentinel2_tiles.id", ondelete="CASCADE"))
    geojson: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    area_stats: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    flood_zone_geojson: Mapped[dict | None] = mapped_column(JSONB)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[int] = mapped_column(ForeignKey("regions.id", ondelete="CASCADE"))
    risk_tier: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    assessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint("risk_tier IN ('low', 'moderate', 'high', 'critical')", name="ck_risk_tier"),
    )


class Forecast(Base):
    __tablename__ = "forecasts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[int] = mapped_column(ForeignKey("regions.id", ondelete="CASCADE"))
    forecast_30d: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    forecast_60d: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    forecast_90d: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    flood_risk_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    fire_risk_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    region_id: Mapped[int] = mapped_column(ForeignKey("regions.id", ondelete="CASCADE"))
    narrative: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    factuality_score: Mapped[float | None] = mapped_column(Float)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    low_confidence: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tile_id: Mapped[int] = mapped_column(ForeignKey("sentinel2_tiles.id", ondelete="CASCADE"))
    embedding: Mapped[list[float]] = mapped_column(Vector(512))
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class TextEmbedding(Base):
    __tablename__ = "text_embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    __table_args__ = (UniqueConstraint("source_type", "source_id", "chunk_index"),)


class FailedIngestion(Base):
    __tablename__ = "failed_ingestion"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[int | None] = mapped_column(Integer)
    flow_name: Mapped[str] = mapped_column(Text, nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    failed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
