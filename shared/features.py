"""Ekstraksi fitur SATU-SATUNYA untuk logging data dan inferensi (mencegah training-serving skew).

PRIVASI: semua fitur hanya angka agregat (getaran, lux, dBFS, jumlah ketukan, jumlah gerakan mouse).
Murni Python (tanpa numpy).
"""
import math  # cek nilai tak hingga / NaN
from shared import config  # ambang gerak HP
from shared import processing as p  # EwmaStats untuk z-score suara

CLASSES = ["FOCUS", "PHONE_DISTRACTION", "NOISE_DISTRACTION", "DARK_ROOM"]  # 4 kelas keluaran model

FEATURE_NAMES = [
    "motion_rms",  # getaran HP (m/s2)
    "orientation",  # -1 telungkup, 0 miring, 1 telentang
    "lux",  # cahaya (lux); LUX_SENTINEL bila sensor tertutup
    "lux_valid",  # 1 bila lux valid, 0 bila tidak
    "noise_dbfs",  # level suara laptop (dBFS)
    "noise_delta",  # kenaikan suara dari baseline ruangan (dB)
    "noise_zscore",  # z-score lonjakan suara
    "keys_per_min",  # ketukan per menit (jumlah saja)
    "mouse_moves",  # jumlah gerakan mouse
    "move_streak",  # jendela berturut-turut HP bergerak
    "laptop_idle",  # 1 bila tidak mengetik dan mouse diam
]  # URUTAN INI DIPAKAI CSV, MODEL, DAN INFERENSI

CSV_COLUMNS = FEATURE_NAMES + ["label", "session_id", "timestamp"]  # kolom data/sessions.csv (hanya angka agregat + label)

LUX_SENTINEL = -1.0  # pengganti lux yang tidak valid (bukan NaN)
ORIENTATION_CODE = {"face_down": -1.0, "tilted": 0.0, "face_up": 1.0}  # encode orientasi HP


class FeatureTracker:
    """Menyimpan keadaan antar jendela: baseline suara, z-score, dan move_streak (dulu ada di FocusEngine)."""

    def __init__(self):
        self.noise_base = None  # baseline suara ruangan (dBFS)
        self.stats = p.EwmaStats()  # statistik untuk deteksi lonjakan suara
        self.move_streak = 0  # berapa jendela berturut-turut HP bergerak
        self.last_stamp = None  # penanda telemetri HP terakhir

    def reset_noise(self):
        """Kalibrasi ulang suara ruangan (dipanggil saat sesi baru)."""
        self.noise_base = None  # kalibrasi ulang saat ruangan tenang
        self.stats = p.EwmaStats()  # reset statistik suara

    def observe(self, hp, pc):
        """Perbarui keadaan dari telemetri terbaru; return ctx = noise_delta, noise_zscore, move_streak."""
        level = pc["noise_dbfs"]  # level suara dari laptop
        if self.noise_base is None:
            self.noise_base = level  # data pertama = baseline
        delta = level - self.noise_base  # kenaikan suara (dB)
        z = self.stats.zscore(delta)  # z-score lonjakan suara
        if delta < 6.0:
            self.noise_base += 0.05 * delta  # baseline ikut bergeser pelan bila ruangan berubah sedikit

        moving = hp["motion_rms"] >= config.MOVE_RMS_THRESHOLD  # HP bergerak?
        stamp = hp.get("sent_at_ms")  # penanda paket telemetri
        if stamp != self.last_stamp:
            self.last_stamp = stamp  # hanya hitung sekali per paket baru
            self.move_streak = self.move_streak + 1 if moving else 0  # naik bila terus bergerak
        return {"noise_delta": delta, "noise_zscore": z, "move_streak": self.move_streak}


def _num(value, default=0.0):
    """Ubah ke float; None / NaN / tak hingga menjadi default (model tidak boleh menerima NaN)."""
    try:
        x = float(value)  # coba ubah ke angka
    except (TypeError, ValueError):
        return default  # bukan angka
    return x if math.isfinite(x) else default  # buang NaN dan inf


def build_features(hp, pc, ctx):
    """hp/pc = telemetri edge1/edge2, ctx = hasil FeatureTracker.observe(). Return dict berurutan FEATURE_NAMES."""
    orient = hp.get("orientation")  # posisi HP
    lux = None if orient == "face_down" else hp.get("lux")  # sensor cahaya tertutup bila telungkup
    lux_valid = lux is not None and math.isfinite(_num(lux, float("nan")))  # lux ada dan bukan NaN
    keys = _num(pc.get("keys_per_min"))  # ketukan per menit
    moves = _num(pc.get("mouse_moves"))  # gerakan mouse
    values = {
        "motion_rms": _num(hp.get("motion_rms")),
        "orientation": ORIENTATION_CODE.get(orient, 0.0),  # tak dikenal dianggap miring
        "lux": _num(lux, LUX_SENTINEL) if lux_valid else LUX_SENTINEL,
        "lux_valid": 1.0 if lux_valid else 0.0,
        "noise_dbfs": _num(pc.get("noise_dbfs")),
        "noise_delta": _num(ctx.get("noise_delta")),
        "noise_zscore": _num(ctx.get("noise_zscore")),
        "keys_per_min": keys,
        "mouse_moves": moves,
        "move_streak": _num(ctx.get("move_streak")),
        "laptop_idle": 1.0 if keys < 5 and moves < 5 else 0.0,  # tidak mengetik dan mouse diam
    }
    return {name: values[name] for name in FEATURE_NAMES}  # urutan konsisten


def to_vector(features):
    """Dict fitur -> list angka berurutan FEATURE_NAMES (siap masuk model)."""
    return [features[name] for name in FEATURE_NAMES]
