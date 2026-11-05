"""Penulis dataset untuk mode collect. PRIVASI: hanya angka agregat + label + session_id + timestamp."""
import csv  # tulis CSV
import datetime  # cap waktu baris
import os  # folder dan cek file
from shared import config  # WINDOW_SEC dan masa transisi
from shared.features import CLASSES, CSV_COLUMNS  # kolom dan kelas yang sah


class DataLogger:
    """Menulis satu baris fitur per jendela ke CSV (header dan folder dibuat otomatis)."""

    def __init__(self, path, label, session_id, now, skip_sec=None):
        if label not in CLASSES:
            raise ValueError(f"label harus salah satu dari {CLASSES}")  # label tidak sah
        if not str(session_id).strip():
            raise ValueError("session_id tidak boleh kosong")
        self.path, self.label, self.session_id = path, label, str(session_id).strip()  # tujuan dan label
        self.start = now  # waktu (monoton) proses dimulai
        self.skip_sec = config.COLLECT_SKIP_SEC if skip_sec is None else skip_sec  # masa transisi
        self.last_write = None  # waktu baris terakhir
        self.last_stamp = None  # paket HP terakhir yang ditulis
        self.count = 0  # jumlah baris yang ditulis proses ini
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)  # buat folder data/
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh), [])  # baca header lama
            if header != CSV_COLUMNS:
                raise ValueError(f"{path} punya kolom berbeda dari versi kode ini; pindahkan/hapus file lama")
        else:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(CSV_COLUMNS)  # tulis header

    def maybe_write(self, features, now, stamp=None):
        """Tulis satu baris bila masa transisi lewat, paket HP baru, dan sudah ~1 jendela. Return True bila menulis."""
        if features is None or now - self.start < self.skip_sec:
            return False  # belum ada data / masih masa transisi
        if stamp is not None and stamp == self.last_stamp:
            return False  # paket yang sama sudah ditulis
        if self.last_write is not None and now - self.last_write < 0.9 * config.WINDOW_SEC:
            return False  # terlalu cepat
        row = dict(features)  # hanya fitur numerik
        row.update(label=self.label, session_id=self.session_id, timestamp=datetime.datetime.now().isoformat(timespec="seconds"))
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writerow({k: row[k] for k in CSV_COLUMNS})
        self.last_write, self.last_stamp = now, stamp  # catat
        self.count += 1
        return True
