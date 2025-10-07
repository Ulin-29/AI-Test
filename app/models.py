from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, func, JSON, Text, Enum as PyEnum
from sqlalchemy.orm import sessionmaker, relationship, validates
from sqlalchemy.ext.declarative import declarative_base
from .database import Base, engine
import datetime

class StatusEnum(PyEnum):
    DITERIMA = "DITERIMA"
    DITOLAK = "DITOLAK"
    PROSES = "DALAM PROSES"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    phone = Column(String, unique=True, index=True)
    password = Column(String)
    name = Column(String, default="")
    photo = Column(String, default="default.png")

    # kolom baru untuk menyimpan OTP
    otp_code = Column(String, nullable=True)
    otp_expiry = Column(DateTime, nullable=True)

    dokumen = relationship("Dokumen", back_populates="pemilik")
    
    @validates("email")
    def convert_lower(self, key, value):
        return value.strip().lower()

class Dokumen(Base):
    __tablename__ = "dokumen"
    id = Column(Integer, primary_key=True, index=True)
    nama_dokumen = Column(String, index=True)
    nama_file_unik = Column(String, index=True)
    tipe_dokumen = Column(String, nullable=True)
    tanggal = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="DALAM PROSES")
    skor = Column(Integer, default=0)
    hasil_verifikasi = Column(JSON, nullable=True)
    ringkasan = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(Integer, ForeignKey("users.id"))
    pemilik = relationship("User", back_populates="dokumen")

def init_db():
    Base.metadata.create_all(bind=engine)