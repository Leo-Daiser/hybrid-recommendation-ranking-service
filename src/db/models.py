from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from src.db.base import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    item_id = Column(Integer, index=True)
    rating = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), index=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RecommendationRequest(Base):
    __tablename__ = "recommendation_requests"
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, index=True, nullable=False)
    k = Column(Integer, nullable=False)
    model_version = Column(String, nullable=True)
    scoring_mode = Column(String, nullable=True)
    fallback_used = Column(Boolean, nullable=True)
    warnings_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CandidateCache(Base):
    __tablename__ = "candidate_cache"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    candidates = Column(JSON)
    model_version = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RankedRecommendation(Base):
    __tablename__ = "ranked_recommendations"
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, index=True, nullable=False)
    user_id = Column(Integer, index=True, nullable=False)
    item_id = Column(Integer, index=True, nullable=False)
    rank = Column(Integer, nullable=False)
    score = Column(Float, nullable=True)
    retrieval_score = Column(Float, nullable=True)
    explanation_json = Column(JSON, nullable=True)
    model_version = Column(String, nullable=True)
    scoring_mode = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FeedbackLog(Base):
    __tablename__ = "feedback_logs"
    id = Column(Integer, primary_key=True, index=True)
    feedback_id = Column(String, unique=True, index=True, nullable=False)
    request_id = Column(String, index=True, nullable=True)
    user_id = Column(Integer, index=True, nullable=False)
    item_id = Column(Integer, index=True, nullable=False)
    event_type = Column(String, nullable=False)
    event_value = Column(Float, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
