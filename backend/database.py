import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'backend', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, 'history.db')

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    duration_seconds = Column(Float, default=0.0)
    predicted_genre = Column(String(50), nullable=False)
    predicted_genre_vi = Column(String(100), nullable=False)
    confidence_percentage = Column(Float, default=0.0)
    tempo_bpm = Column(Float, default=0.0)
    rms_energy = Column(Float, default=0.0)
    spectral_centroid_hz = Column(Float, default=0.0)
    processing_time_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

class HistoryRepository:
    @staticmethod
    def add_entry(
        filename: str,
        file_size_bytes: int,
        duration_seconds: float,
        predicted_genre: str,
        predicted_genre_vi: str,
        confidence_percentage: float,
        tempo_bpm: float,
        rms_energy: float,
        spectral_centroid_hz: float,
        processing_time_ms: float
    ) -> Dict[str, Any]:
        session = SessionLocal()
        try:
            entry = PredictionHistory(
                filename=filename,
                file_size_bytes=file_size_bytes,
                duration_seconds=duration_seconds,
                predicted_genre=predicted_genre,
                predicted_genre_vi=predicted_genre_vi,
                confidence_percentage=confidence_percentage,
                tempo_bpm=tempo_bpm,
                rms_energy=rms_energy,
                spectral_centroid_hz=spectral_centroid_hz,
                processing_time_ms=processing_time_ms,
                created_at=datetime.datetime.utcnow()
            )
            session.add(entry)
            session.commit()
            session.refresh(entry)
            return {
                "id": entry.id,
                "filename": entry.filename,
                "predicted_genre": entry.predicted_genre,
                "predicted_genre_vi": entry.predicted_genre_vi,
                "confidence_percentage": entry.confidence_percentage,
                "tempo_bpm": entry.tempo_bpm,
                "created_at": entry.created_at.strftime("%Y-%m-%d %H:%M:%S")
            }
        finally:
            session.close()

    @staticmethod
    def get_all(limit: int = 50) -> List[Dict[str, Any]]:
        session = SessionLocal()
        try:
            entries = session.query(PredictionHistory).order_by(PredictionHistory.created_at.desc()).limit(limit).all()
            return [
                {
                    "id": e.id,
                    "filename": e.filename,
                    "file_size_kb": round(e.file_size_bytes / 1024, 1),
                    "duration_seconds": e.duration_seconds,
                    "predicted_genre": e.predicted_genre,
                    "predicted_genre_vi": e.predicted_genre_vi,
                    "confidence_percentage": e.confidence_percentage,
                    "tempo_bpm": e.tempo_bpm,
                    "rms_energy": e.rms_energy,
                    "spectral_centroid_hz": e.spectral_centroid_hz,
                    "processing_time_ms": round(e.processing_time_ms, 1),
                    "created_at": e.created_at.strftime("%d/%m/%Y %H:%M:%S")
                }
                for e in entries
            ]
        finally:
            session.close()

    @staticmethod
    def delete_entry(entry_id: int) -> bool:
        session = SessionLocal()
        try:
            entry = session.query(PredictionHistory).filter(PredictionHistory.id == entry_id).first()
            if entry:
                session.delete(entry)
                session.commit()
                return True
            return False
        finally:
            session.close()

    @staticmethod
    def get_stats() -> Dict[str, Any]:
        session = SessionLocal()
        try:
            entries = session.query(PredictionHistory).all()
            total = len(entries)
            if total == 0:
                return {
                    "total_predictions": 0,
                    "avg_confidence": 0.0,
                    "avg_bpm": 0.0,
                    "genre_distribution": {}
                }

            genre_counts: Dict[str, int] = {}
            conf_sum = 0.0
            bpm_sum = 0.0

            for e in entries:
                genre_counts[e.predicted_genre] = genre_counts.get(e.predicted_genre, 0) + 1
                conf_sum += e.confidence_percentage
                bpm_sum += e.tempo_bpm

            return {
                "total_predictions": total,
                "avg_confidence": round(conf_sum / total, 1),
                "avg_bpm": round(bpm_sum / total, 1),
                "genre_distribution": genre_counts
            }
        finally:
            session.close()
