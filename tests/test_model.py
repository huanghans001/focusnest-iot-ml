"""Uji model Random Forest: muat, prediksi, fallback, dan penolakan model yang tidak cocok."""
import io, os, sys, tempfile, unittest  # modul standar
from unittest import mock  # tiruan modul / konfigurasi
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import config  # MODEL_PATH, USE_ML
from shared.engine import FocusEngine  # mesin fusion
from shared.features import CLASSES, FEATURE_NAMES, build_features  # fitur bersama
from shared.model import FocusModel, ModelError, save_bundle  # pembungkus model
from tools import generate_synthetic_data as gen  # data sintetis

try:
    import sklearn  # noqa: F401
    from sklearn.ensemble import RandomForestClassifier
    HAS_SK = True
except ImportError:
    HAS_SK = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLED = os.path.join(ROOT, "model", "focus_rf.joblib")
CTX = {"noise_delta": 0.0, "noise_zscore": 0.0, "move_streak": 0}
FITUR = build_features({"motion_rms": 0.04, "orientation": "face_up", "lux": 400.0}, {"noise_dbfs": -95.0, "keys_per_min": 30.0, "mouse_moves": 3}, CTX)


def latih_kecil(path, feature_names=None, sklearn_version=None):
    """Latih model singkat dari data sintetis lalu simpan ke `path` (meta bisa dipalsukan untuk uji penolakan)."""
    import sklearn as sk
    rows = gen.generate(1200, 12, seed=11)
    X = [[r[n] for n in FEATURE_NAMES] for r in rows]
    y = [r["label"] for r in rows]
    rf = RandomForestClassifier(n_estimators=30, max_depth=8, class_weight="balanced", random_state=0).fit(X, y)
    meta = {"feature_names": feature_names or FEATURE_NAMES, "classes": CLASSES, "sklearn_version": sklearn_version or sk.__version__,
            "trained_at": "2026-01-01", "data_source": "synthetic"}
    save_bundle(rf, meta, path)


@unittest.skipUnless(HAS_SK, "scikit-learn belum terpasang")
class TestBundled(unittest.TestCase):
    def test_model_bawaan_ada_dan_cocok(self):
        self.assertTrue(os.path.isfile(BUNDLED), "model/focus_rf.joblib harus ikut di repo")
        m = FocusModel(BUNDLED).load()
        self.assertEqual(m.meta["feature_names"], FEATURE_NAMES)
        self.assertEqual(m.meta["classes"], CLASSES)
        self.assertEqual(m.meta["sklearn_version"], sklearn.__version__)
        self.assertIn(m.meta["data_source"], ("synthetic", "synthetic+real", "real"))
        self.assertIn("dilatih", m.describe())

    def test_prediksi_model_bawaan(self):
        p = FocusModel(BUNDLED).load().predict_proba(FITUR)
        self.assertAlmostEqual(sum(p.values()), 1.0, places=6)
        self.assertEqual(set(p), set(CLASSES))


@unittest.skipUnless(HAS_SK, "scikit-learn belum terpasang")
class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "kecil.joblib")
        latih_kecil(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_proba_berjumlah_satu_dan_skor_0_100(self):
        m = FocusModel(self.path).load()
        for hp in ({"motion_rms": 2.0, "orientation": "tilted", "lux": 300.0}, {"motion_rms": 0.03, "orientation": "face_down", "lux": None}):
            p = m.predict_proba(build_features(hp, {"noise_dbfs": -60.0, "keys_per_min": 0, "mouse_moves": 0}, {"noise_delta": 30.0, "noise_zscore": 12.0, "move_streak": 3}))
            self.assertAlmostEqual(sum(p.values()), 1.0, places=6)
            self.assertTrue(0.0 <= p["FOCUS"] * 100 <= 100.0)

    def test_nan_ditolak(self):
        m = FocusModel(self.path).load()
        with self.assertRaises(ValueError):
            m.predict_proba(dict(FITUR, lux=float("nan")))

    def test_fallback_bila_file_tidak_ada(self):
        log = []
        with mock.patch.object(config, "MODEL_PATH", os.path.join(self.tmp.name, "tidak_ada.joblib")), mock.patch.object(config, "USE_ML", True):
            e = FocusEngine.auto(log=log.append)
        self.assertEqual(e.mode, "rule")
        self.assertIn("fallback", log[0])
        self.assertEqual(e.update(None, None, 0.0)[0]["model"]["source"], "rule")

    def test_model_aktif_dan_log(self):
        log = []
        with mock.patch.object(config, "MODEL_PATH", self.path), mock.patch.object(config, "USE_ML", True):
            e = FocusEngine.auto(log=log.append)
        self.assertEqual(e.mode, "ml")
        self.assertIn("Random Forest", log[0])
        self.assertIn("sintetis", log[0])

    def test_use_ml_false_pakai_aturan(self):
        with mock.patch.object(config, "USE_ML", False):
            self.assertEqual(FocusEngine.auto(log=lambda s: None).mode, "rule")

    def test_fitur_berbeda_ditolak(self):
        path = os.path.join(self.tmp.name, "fitur_beda.joblib")
        latih_kecil(path, feature_names=list(reversed(FEATURE_NAMES)))
        with self.assertRaisesRegex(ModelError, "fitur"):
            FocusModel(path).load()
        with mock.patch.object(config, "MODEL_PATH", path):
            self.assertEqual(FocusEngine.auto(log=lambda s: None).mode, "rule")  # engine fallback

    def test_versi_sklearn_beda_ditolak(self):
        path = os.path.join(self.tmp.name, "versi_beda.joblib")
        latih_kecil(path, sklearn_version="0.0.1")
        with self.assertRaisesRegex(ModelError, "versi scikit-learn"):
            FocusModel(path).load()

    def test_file_rusak_ditolak(self):
        path = os.path.join(self.tmp.name, "rusak.joblib")
        with open(path, "wb") as fh:
            fh.write(b"bukan model")
        with self.assertRaisesRegex(ModelError, "rusak"):
            FocusModel(path).load()

    def test_sklearn_tidak_terpasang_fallback(self):
        with mock.patch.dict(sys.modules, {"sklearn": None}):  # import sklearn -> ImportError
            with self.assertRaisesRegex(ModelError, "belum terpasang"):
                FocusModel(self.path).load()


if __name__ == "__main__":
    unittest.main()
