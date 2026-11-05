"""Mesin sensor fusion FocusNest: menggabungkan data HP (edge1) dan laptop (edge2).

Dua mode keputusan:
  - "ml"   : klasifikasi Random Forest (shared/model.py) -> probabilitas 4 kelas.
  - "rule" : algoritma rule-based lama (bobot WEIGHTS + aturan if-else). Dipakai sebagai fallback dan baseline.
"""
import time  # cap waktu untuk alert
from shared import config  # ambang batas dan cooldown
from shared import processing as p  # fungsi skor
from shared.features import CLASSES, FeatureTracker, build_features  # fitur bersama

EVENT_TYPE = {  # kelas gangguan -> jenis alert (format alert lama tidak berubah)
    "PHONE_DISTRACTION": "PHONE_DISTRACTION",
    "NOISE_DISTRACTION": "NOISE_SPIKE",
    "DARK_ROOM": "TOO_DARK",
}


def rule_class(f):
    """Keputusan rule-based lama dari fitur -> salah satu CLASSES (prioritas dari atas). Juga dipakai sebagai baseline."""
    if f["move_streak"] >= 2 and f["laptop_idle"]:
        return "PHONE_DISTRACTION"  # HP terangkat saat laptop sepi
    if f["noise_zscore"] > 3.0 and f["noise_delta"] >= 10.0:
        return "NOISE_DISTRACTION"  # lonjakan suara
    if f["lux_valid"] and f["lux"] < 100:
        return "DARK_ROOM"  # terlalu gelap
    return "FOCUS"  # tidak ada gangguan


def event_message(cls, f):
    """Pesan alert untuk kelas gangguan."""
    if cls == "PHONE_DISTRACTION":
        return "HP terangkat saat laptop sepi. Letakkan HP, kembali fokus!"
    if cls == "NOISE_DISTRACTION":
        if f["noise_delta"] >= 3.0:
            return f"Tiba-tiba berisik (+{f['noise_delta']:.0f} dB dari suasana tadi)."
        return "Suasana terlalu berisik."
    if f["lux_valid"]:
        return f"Ruangan terlalu gelap ({f['lux']:.0f} lux). Nyalakan lampu."
    return "Ruangan terlalu gelap. Nyalakan lampu."


class FocusEngine:
    """Menghitung skor fokus, status, alert, dan statistik sesi belajar."""

    def __init__(self, model=None):
        """model = objek dengan predict_proba(fitur) (FocusModel); None = algoritma rule-based."""
        self.model = model  # None -> mode rule
        self.active = False  # apakah sesi belajar sedang berjalan
        self.session_start = 0.0  # waktu mulai sesi
        self.focus_s = 0.0  # total detik dengan skor >= 70
        self.distractions = 0  # jumlah gangguan (alert) pada sesi
        self.tracker = FeatureTracker()  # baseline suara, z-score, move_streak
        self.last_alert = {}  # waktu alert terakhir per jenis
        self.proba = None  # probabilitas kelas yang sudah dihaluskan (EWMA)
        self.last_features = None  # fitur jendela terakhir (untuk mode collect)
        self.last_t = None  # waktu update terakhir

    @classmethod
    def auto(cls, log=print):
        """Pakai model bila ada dan valid; kalau tidak, fallback ke rule-based. Cetak satu baris log mode aktif."""
        if not config.USE_ML:
            log("[MODEL] mode ATURAN (rule-based): dimatikan lewat USE_ML=false / --no-ml")
            return cls()
        from shared.model import FocusModel, ModelError  # import malas: sklearn tidak wajib
        model = FocusModel(config.MODEL_PATH)
        try:
            model.load()  # muat dan validasi model
        except ModelError as err:
            log(f"[MODEL] fallback ke ATURAN (rule-based): {err}")
            return cls()
        log(f"[MODEL] aktif: Random Forest | {model.describe()}")
        return cls(model)

    @property
    def mode(self):
        """'ml' atau 'rule'."""
        return "ml" if self.model is not None else "rule"

    def start_session(self, now):
        """Mulai sesi baru: reset statistik dan kalibrasi ulang suara ruangan."""
        self.active = True  # sesi aktif
        self.session_start = now  # catat waktu mulai
        self.focus_s = 0.0  # reset menit fokus
        self.distractions = 0  # reset gangguan
        self.tracker.reset_noise()  # kalibrasi ulang saat ruangan tenang
        self.proba = None  # reset penghalusan
        self.last_alert = {}  # reset cooldown

    def stop_session(self):
        """Akhiri sesi (statistik terakhir tetap tampil di dashboard)."""
        self.active = False  # sesi berhenti

    def _model_info(self):
        """Info model untuk dashboard: sumber algoritma dan probabilitas tiap kelas."""
        info = {"source": self.mode, "proba": {}}  # proba kosong bila mode rule / belum ada data
        if self.model is not None:
            if self.proba:
                info["proba"] = {c: round(v, 3) for c, v in self.proba.items()}
            info["data_source"] = self.model.meta.get("data_source")  # synthetic / synthetic+real / real
            info["trained_at"] = self.model.meta.get("trained_at")  # tanggal latih
        return info

    def _status(self, state, reason, score, scores, now):
        """Susun dict status yang dikirim ke dashboard dan HP."""
        elapsed = now - self.session_start if self.active else 0.0  # lama sesi
        return {
            "state": state,  # IDLE / WAITING / FOCUS / FAIR / DISTRACTED / PHONE_DISTRACTION / ...
            "reason": reason,  # penjelasan singkat
            "focus_score": None if score is None else round(score, 1),  # skor 0-100
            "scores": {k: None if v is None else round(v, 1) for k, v in scores.items()},  # skor per komponen
            "model": self._model_info(),  # sumber algoritma + probabilitas kelas
            "session": {
                "active": self.active,  # sesi berjalan?
                "elapsed_s": round(elapsed),  # lama sesi (detik)
                "focus_s": round(self.focus_s),  # detik fokus
                "distractions": self.distractions,  # jumlah gangguan
            },
            "sent_at_ms": int(time.time() * 1000),  # cap waktu
        }

    def _infer(self, feats):
        """Probabilitas kelas dari model + penghalusan EWMA. None bila model gagal (otomatis fallback ke rule)."""
        try:
            raw = self.model.predict_proba(feats)  # dict kelas -> probabilitas
        except Exception as err:  # model rusak saat jalan: jangan hentikan detector
            print(f"[MODEL] gagal memprediksi ({err}). Fallback ke ATURAN (rule-based).")
            self.model, self.proba = None, None
            return None
        a = config.SCORE_SMOOTH_ALPHA  # bobot data baru
        if self.proba is None:
            self.proba = dict(raw)  # data pertama
        else:
            self.proba = {c: a * raw[c] + (1 - a) * self.proba[c] for c in CLASSES}  # campur baru dan lama
        return self.proba

    def update(self, hp, pc, now):
        """hp/pc = telemetri terbaru edge1/edge2 (None bila basi). Return (status, daftar_alert)."""
        dt = 0.0 if self.last_t is None else min(now - self.last_t, 5.0)  # selang waktu sejak update lalu
        self.last_t = now  # simpan waktu
        alerts = []  # alert yang akan dikirim balik ke HP
        self.last_features = None  # tidak ada fitur bila data tidak lengkap
        if hp is None or pc is None:
            return self._status("WAITING", "Menunggu data HP (edge1) dan laptop (edge2)", None, {}, now), alerts

        # --- fitur (SAMA untuk logging, rule, dan model) ---
        ctx = self.tracker.observe(hp, pc)  # baseline suara, z-score, move_streak
        f = build_features(hp, pc, ctx)  # fitur numerik satu jendela
        self.last_features = f  # dipakai mode collect
        lux = f["lux"] if f["lux_valid"] else None  # None = sensor cahaya tertutup

        # --- skor komponen (ditampilkan di dashboard; dipakai penuh oleh mode rule) ---
        scores = {
            "noise": p.noise_score(f["noise_delta"]),  # skor suara
            "light": None if lux is None else p.light_score(lux),  # skor cahaya (None = tidak valid)
            "phone": p.phone_score(hp["motion_rms"], hp.get("orientation"), config.MOVE_RMS_THRESHOLD),  # skor HP
            "activity": p.activity_score(pc["keys_per_min"], pc["mouse_moves"]),  # skor aktivitas laptop
        }

        # --- keputusan: model ML bila ada, kalau tidak aturan lama ---
        event = None  # (jenis alert, pesan)
        proba = self._infer(f) if self.model is not None else None
        if proba is not None:
            score = proba["FOCUS"] * 100.0  # skor fokus = probabilitas FOCUS x 100
            top = max((c for c in CLASSES if c != "FOCUS"), key=lambda c: proba[c])  # gangguan paling mungkin
            if proba[top] >= config.EVENT_MIN_PROBA and proba[top] > proba["FOCUS"]:
                event = (EVENT_TYPE[top], event_message(top, f))
        else:
            score = p.focus_score(scores)  # skor fokus berbobot (rule-based)
            cls = rule_class(f)
            if cls != "FOCUS":
                event = (EVENT_TYPE[cls], event_message(cls, f))

        if event:
            state, reason = event  # gangguan jelas mengalahkan skor biasa
        else:
            state = "FOCUS" if score >= 70 else "FAIR" if score >= 40 else "DISTRACTED"  # status dari skor
            reason = f"Skor fokus {score:.0f} dari 100"
        if not self.active:
            state, reason = "IDLE", "Sesi belum dimulai. Tekan Mulai sesi di dashboard."  # hanya memantau

        # --- alert dan statistik hanya saat sesi aktif ---
        if self.active:
            if score >= 70:
                self.focus_s += dt  # tambah detik fokus
            if event and now - self.last_alert.get(event[0], -1e9) >= config.ALERT_COOLDOWN_SEC:
                self.last_alert[event[0]] = now  # catat waktu alert
                if event[0] != "TOO_DARK":
                    self.distractions += 1  # gelap bukan gangguan perilaku
                alerts.append({"type": event[0], "message": event[1], "sent_at_ms": int(time.time() * 1000)})
        return self._status(state, reason, score, scores, now), alerts
