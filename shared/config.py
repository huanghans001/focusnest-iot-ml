"""Pengaturan bersama untuk HP dan laptop (dibaca dari file .env)."""
import os  # membaca environment variable
import re  # membersihkan PROJECT_ID
import socket  # mencari IP laptop di jaringan
from dotenv import load_dotenv  # membaca file .env

load_dotenv()  # muat isi file .env ke environment

PROJECT_ID = re.sub(r"[^A-Za-z0-9_-]", "_", os.getenv("PROJECT_ID", "focusnest_demo"))  # nama unik kelompok
MQTT_HOST = os.getenv("MQTT_HOST", "broker.hivemq.com")  # alamat broker
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))  # 8883 = MQTT over TLS
MQTT_TLS = os.getenv("MQTT_TLS", "true").lower() == "true"  # pakai TLS atau tidak
MQTT_USER = os.getenv("MQTT_USER", "")  # username (kosong untuk broker publik)
MQTT_PASS = os.getenv("MQTT_PASS", "")  # password (kosong untuk broker publik)
WINDOW_SEC = float(os.getenv("WINDOW_SEC", "2"))  # panjang jendela pengolahan (detik)
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8000"))  # port dashboard web
MOVE_RMS_THRESHOLD = float(os.getenv("MOVE_RMS_THRESHOLD", "0.7"))  # ambang HP dianggap bergerak (m/s2)
STALE_SEC = 10.0  # data lebih tua dari ini dianggap hilang
ALERT_COOLDOWN_SEC = 15.0  # jeda minimal antar alert sejenis
USE_ML = os.getenv("USE_ML", "true").lower() == "true"  # false = paksa algoritma rule-based
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "focus_rf.joblib"))  # lokasi model
SCORE_SMOOTH_ALPHA = float(os.getenv("SCORE_SMOOTH_ALPHA", "0.5"))  # bobot data baru pada penghalusan EWMA (1 = tanpa penghalusan)
EVENT_MIN_PROBA = float(os.getenv("EVENT_MIN_PROBA", "0.5"))  # probabilitas minimal kelas gangguan agar jadi event
COLLECT_SKIP_SEC = 3.0  # masa transisi mode collect: data awal diabaikan (detik)


def topic(suffix):
    """Bentuk topic MQTT: focusnest/<PROJECT_ID>/<suffix>."""
    return f"focusnest/{PROJECT_ID}/{suffix}"  # contoh: focusnest/kel_07/edge/edge1/telemetry


def lan_ip():
    """Cari IP laptop di jaringan WiFi/hotspot (untuk membuka dashboard dari HP)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket UDP (tidak benar-benar mengirim)
    try:
        s.connect(("8.8.8.8", 80))  # pilih antarmuka jaringan yang aktif
        return s.getsockname()[0]  # ambil alamat IP lokal
    except OSError:
        return "127.0.0.1"  # tidak ada jaringan
    finally:
        s.close()  # tutup socket
