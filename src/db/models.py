from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.sql import func
from src.db.base import Base

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Item(Base):
    __tablename__ = 'items'
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Interaction(Base):
    __tablename__ = 'interactions'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    item_id = Column(Integer, ForeignKey('items.id'), index=True)
    rating = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), index=True)

class ModelVersion(Base):
    __tablename__ = 'model_versions'
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class RecommendationRequest(Base):
    __tablename__ = 'recommendation_requests'
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    model_version = Column(String, ForeignKey('model_versions.version'), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class CandidateCache(Base):
    __tablename__ = 'candidate_cache'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    candidates = Column(JSON)
    model_version = Column(String, ForeignKey('model_versions.version'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class RankedRecommendation(Base):
    __tablename__ = 'ranked_recommendations'
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, ForeignKey('recommendation_requests.request_id'), index=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    item_id = Column(Integer, ForeignKey('items.id'), index=True)
    rank = Column(Integer)
    score = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class FeedbackLog(Base):
    __tablename__ = 'feedback_logs'
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    item_id = Column(Integer, ForeignKey('items.id'), index=True)
    event_type = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
