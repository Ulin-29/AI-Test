from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. ALAMAT DATABASE
# Ini adalah "alamat" tempat aplikasi akan menyimpan semua datanya.
# Kita menggunakan SQLite, yang akan membuat satu file database bernama `database.db`.
# Ini adalah cara termudah untuk memulai.
SQLALCHEMY_DATABASE_URL = "sqlite:///./database.db"


# 2. MESIN PENGHUBUNG
# 'engine' ini adalah mesin yang bertugas menghubungkan aplikasi Anda
# dengan file database sesuai alamat di atas.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)


# 3. PINTU SESI
# Setiap kali aplikasi perlu bicara dengan database (misal menyimpan user baru),
# ia akan membuka "pintu" sementara yang disebut sesi.
# 'SessionLocal' inilah yang membuat pintu-pintu tersebut.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# 4. CETAK BIRU MODEL
# 'Base' ini adalah cetak biru dasar yang akan digunakan oleh file `models.py`
# untuk memberitahu SQLAlchemy cara membuat tabel-tabel di dalam database.
Base = declarative_base()