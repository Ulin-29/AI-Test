from fastapi.templating import Jinja2Templates
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from datetime import datetime
import smtplib
import pytz
import os
import traceback

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")     

print(f"🔍 TEMPLATE PATH: {TEMPLATE_DIR}")  
templates = Jinja2Templates(directory=TEMPLATE_DIR)

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "youremail@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "your-app-password")

# ======================================================================
# 1. Email Notifikasi Login
# ======================================================================
def send_notification_email(
    to_email, subject, title, message,
    device_name=None, waktu_login=None, lokasi=None, ip_address=None, koordinat=None,
    footer="Jika ini bukan kamu, segera ubah password akun kamu."
):
    try:
        html_content = templates.get_template("login_email.html").render(
            subject=subject,
            title=title,
            message=message,
            device_name=device_name,
            waktu_login=waktu_login,
            lokasi=lokasi,
            ip_address=ip_address,
            koordinat=koordinat,
            footer=footer
        )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "[Website] Login Berhasil"
        msg["From"] = f"Keamanan Akun <{EMAIL_SENDER}>"
        msg["To"] = to_email
        msg.attach(MIMEText(html_content, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, to_email, msg.as_string())

        print(f"✅ Email notifikasi login terkirim ke {to_email}")

    except Exception as e:
        print(f"❌ Gagal kirim email notifikasi login: {e}")
        traceback.print_exc()

# ======================================================================
# 2. Email Notifikasi Registrasi
# ======================================================================
def send_register_email(to_email, username):
    try:
        subject = "Registrasi Berhasil"
        title = "Selamat Bergabung!"
        message = f"Halo {username}, akun kamu berhasil dibuat."

        html_content = templates.get_template("register_email.html").render(
            subject=subject,
            title=title,
            message=message,
            username=username
        )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[Website] {subject}"
        msg["From"] = f"Keamanan Akun <{EMAIL_SENDER}>"
        msg["To"] = to_email
        msg.attach(MIMEText(html_content, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, to_email, msg.as_string())

        print(f"✅ Email registrasi terkirim ke {to_email}")

    except Exception as e:
        print(f"❌ Gagal kirim email registrasi: {e}")
        traceback.print_exc()

# ======================================================================
# 3. Email Password Berhasil Diubah
# ======================================================================
def send_password_changed_email(to_email, username):
    try:
        # ambil waktu saat ini dalam WIB
        wib = pytz.timezone("Asia/Jakarta")
        change_time = datetime.now(wib).strftime("%d %B %Y, %H:%M WIB")

        html_content = templates.get_template("pwchanged_email.html").render(
            subject="Password Berhasil Diubah",
            username=username,
            change_time=change_time
        )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "[Website] Password Berhasil Diubah"
        msg["From"] = f"Keamanan Akun <{EMAIL_SENDER}>"
        msg["To"] = to_email
        msg.attach(MIMEText(html_content, "html")) 

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, to_email, msg.as_string())
 
        print(f"✅ Email ubah password terkirim ke {to_email}")

    except Exception as e:
        print(f"❌ Gagal kirim email ubah password: {e}")
        traceback.print_exc()

# ======================================================================
# 4. Email OTP Reset Password
# ======================================================================
def send_email_otp(to_email: str, otp_code: str):
    subject = "[Website] Kode OTP Reset Password"
    try:
        # Render template HTML untuk email
        html_content = templates.get_template("otp_email.html").render(
            subject=subject,
            title="Reset Password",
            message="Kode OTP untuk reset password kamu adalah:",
            otp_code=otp_code
        )

        # Buat email MIME
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"Verifikasi <{EMAIL_SENDER}>"
        msg["To"] = to_email
        msg.attach(MIMEText(html_content, "html"))

        # Kirim email via SMTP
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, to_email, msg.as_string())

        print(f"[INFO] OTP terkirim ke {to_email}")

    except Exception as e:
        print(f"[ERROR] Gagal kirim OTP ke {to_email}: {e}")