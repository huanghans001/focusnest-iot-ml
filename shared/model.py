"""Pembungkus model Random Forest FocusNest: muat, validasi, dan prediksi probabilitas 4 kelas.

scikit-learn diimpor malas (hanya saat load) supaya modul lain tetap jalan tanpa scikit-learn (fallback rule-based).
"""
import math  # cek NaN
import os  # cek file
import pickle  # model dibungkus pickle di dalam file joblib agar metadata bisa diperiksa lebih dulu
from shared.features import CLASSES, FEATURE_NAMES, to_vector  # urutan fitur bersama

SOURCE_LABEL = {"synthetic": "sintetis", "synthetic+real": "sintetis+asli", "real": "asli"}  # sumber data latih


class ModelError(Exception):
    """Model tidak bisa dipakai (hilang, rusak, tidak cocok); engine akan fallback ke rule-based."""


def save_bundle(model, meta, path):
    """Simpan model + metadata ke satu file joblib. Dipakai train_model.py."""
    import joblib  # hanya untuk training
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump({"meta": meta, "model_pkl": pickle.dumps(model)}, path, compress=3)  # metadata terbaca tanpa membuka model


class FocusModel:
    """Model klasifikasi fokus. Pakai: m = FocusModel(path); m.load(); m.predict_proba(fitur)."""

    def __init__(self, path):
        self.path = path  # lokasi file .joblib
        self.model = None  # RandomForestClassifier
        self.meta = {}  # metadata latih

    def load(self):
        """Muat dan validasi model. Raise ModelError dengan pesan jelas bila tidak bisa dipakai."""
        if not os.path.isfile(self.path):
            raise ModelError(f"file model tidak ditemukan ({self.path})")
        try:
            import joblib  # library penyimpan model
            import sklearn  # untuk cek versi
        except ImportError as err:
            raise ModelError(f"scikit-learn/joblib belum terpasang ({err}). Jalankan: pip install -r requirements.txt")
        try:
            bundle = joblib.load(self.path)  # hanya dict berisi metadata + bytes
            meta, blob = bundle["meta"], bundle["model_pkl"]
        except Exception as err:
            raise ModelError(f"file model rusak atau bukan model FocusNest ({type(err).__name__}: {err})")
        if meta.get("feature_names") != FEATURE_NAMES:
            raise ModelError("fitur model tidak cocok dengan kode saat ini (FEATURE_NAMES berbeda); latih ulang dengan train_model.py")
        if list(meta.get("classes", [])) != CLASSES:
            raise ModelError("daftar kelas model tidak cocok dengan kode saat ini; latih ulang dengan train_model.py")
        if meta.get("sklearn_version") != sklearn.__version__:
            raise ModelError(f"versi scikit-learn tidak cocok (model dilatih dengan {meta.get('sklearn_version')}, terpasang {sklearn.__version__}); "
                             f"pasang versi yang sama (pip install -r requirements.txt) atau latih ulang")
        try:
            model = pickle.loads(blob)  # buka model setelah versi terbukti cocok
        except Exception as err:
            raise ModelError(f"model tidak bisa dibuka ({type(err).__name__}: {err})")
        if list(model.classes_) != sorted(CLASSES):
            raise ModelError("kelas di dalam model tidak cocok; latih ulang dengan train_model.py")
        model.n_jobs = 1  # satu jendela per prediksi: tanpa thread pool supaya cepat dan tidak memblokir
        self.model, self.meta = model, meta
        return self

    def predict_proba(self, features):
        """features = dict dari build_features(). Return dict kelas -> probabilitas (jumlah ~1.0)."""
        vec = to_vector(features)  # urutan FEATURE_NAMES
        if any(not math.isfinite(v) for v in vec):
            raise ValueError("fitur mengandung NaN/inf")  # model tidak boleh menerima NaN
        probs = self.model.predict_proba([vec])[0]  # satu baris
        out = {c: 0.0 for c in CLASSES}
        out.update({c: float(p) for c, p in zip(self.model.classes_, probs)})
        return out

    def describe(self):
        """Satu baris ringkasan: sumber data latih dan tanggal latih."""
        src = SOURCE_LABEL.get(self.meta.get("data_source"), "tidak diketahui")
        return f"data latih: {src} | dilatih: {self.meta.get('trained_at', '?')}"
