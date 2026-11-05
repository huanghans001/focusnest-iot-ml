"""Uji FocusEngine mode ML: skenario fokus / HP diangkat / berisik / gelap, cooldown, dan fallback tengah jalan.
Model dilatih singkat dari data sintetis (tanpa jaringan, mikrofon, atau sensor sungguhan)."""
import os, sys, tempfile, unittest  # modul standar
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import config  # ALERT_COOLDOWN_SEC
from shared.engine import FocusEngine  # mesin fusion
from shared.features import CLASSES, FEATURE_NAMES  # fitur bersama
from shared.model import FocusModel, save_bundle  # pembungkus model
from tools import generate_synthetic_data as gen  # data sintetis

try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    HAS_SK = True
except ImportError:
    HAS_SK = False


def hp(rms=0.04, orient="face_up", lux=400.0, stamp=0):
    """Telemetri HP palsu (default: diam, layar terlihat, terang)."""
    return {"motion_rms": rms, "orientation": orient, "lux": lux, "sent_at_ms": stamp}


def pc(noise=-95.0, kpm=30.0, moves=3):
    """Telemetri laptop palsu (default: tenang, mengetik)."""
    return {"noise_dbfs": noise, "keys_per_min": kpm, "mouse_moves": moves}


@unittest.skipUnless(HAS_SK, "scikit-learn belum terpasang")
class TestEngineML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rows = gen.generate(3600, 30, seed=5)
        rf = RandomForestClassifier(n_estimators=40, max_depth=8, class_weight="balanced", random_state=0).fit(
            [[r[n] for n in FEATURE_NAMES] for r in rows], [r["label"] for r in rows])
        cls.path = os.path.join(cls.tmp.name, "uji.joblib")
        save_bundle(rf, {"feature_names": FEATURE_NAMES, "classes": CLASSES, "sklearn_version": sklearn.__version__,
                         "trained_at": "2026-01-01", "data_source": "synthetic"}, cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def engine(self, start=True):
        e = FocusEngine(FocusModel(self.path).load())
        if start:
            e.start_session(0.0)
        return e

    def jalan(self, e, bikin, detik):
        """Jalankan `detik` update (1 per detik); return (status terakhir, semua alert)."""
        alerts, status = [], None
        for i in range(detik):
            h, c = bikin(i)
            status, a = e.update(h, c, float(i))
            alerts += a
        return status, alerts

    def test_fokus(self):
        s, alerts = self.jalan(self.engine(), lambda i: (hp(stamp=i), pc()), 10)
        self.assertEqual(s["state"], "FOCUS")
        self.assertGreaterEqual(s["focus_score"], 70)
        self.assertEqual(alerts, [])
        self.assertEqual(s["model"]["source"], "ml")
        self.assertAlmostEqual(sum(s["model"]["proba"].values()), 1.0, places=1)  # dibulatkan 3 desimal per kelas
        self.assertAlmostEqual(s["focus_score"], s["model"]["proba"]["FOCUS"] * 100, delta=0.1)  # skor = proba FOCUS x 100

    def test_hp_diangkat(self):
        s, alerts = self.jalan(self.engine(), lambda i: (hp(rms=2.0, orient="tilted", lux=300.0, stamp=i), pc(kpm=0, moves=0)), 5)
        self.assertEqual(s["state"], "PHONE_DISTRACTION")
        self.assertLess(s["focus_score"], 40)
        self.assertEqual(alerts[0]["type"], "PHONE_DISTRACTION")
        self.assertEqual(set(alerts[0]), {"type", "message", "sent_at_ms"})  # format alert lama

    def test_berisik(self):
        s, alerts = self.jalan(self.engine(), lambda i: (hp(stamp=i), pc(noise=-95.0 if i < 12 else -60.0)), 18)
        self.assertEqual(s["state"], "NOISE_SPIKE")
        self.assertEqual([a["type"] for a in alerts], ["NOISE_SPIKE"])

    def test_gelap_bukan_gangguan_perilaku(self):
        e = self.engine()
        s, alerts = self.jalan(e, lambda i: (hp(lux=20.0, stamp=i), pc()), 5)
        self.assertEqual(s["state"], "TOO_DARK")
        self.assertEqual([a["type"] for a in alerts], ["TOO_DARK"])
        self.assertEqual(e.distractions, 0)  # gelap tidak dihitung gangguan

    def test_cooldown_alert(self):
        e = self.engine()
        _, alerts = self.jalan(e, lambda i: (hp(rms=2.0, orient="tilted", lux=300.0, stamp=i), pc(kpm=0, moves=0)), int(config.ALERT_COOLDOWN_SEC) - 1)
        self.assertEqual(len(alerts), 1)  # hanya 1 alert dalam masa cooldown
        self.assertEqual(e.distractions, 1)
        s, a = e.update(hp(rms=2.0, orient="tilted", lux=300.0, stamp=99), pc(kpm=0, moves=0), config.ALERT_COOLDOWN_SEC + 1.0)
        self.assertEqual(len(a), 1)  # setelah cooldown alert boleh muncul lagi
        self.assertEqual(e.distractions, 2)

    def test_penghalusan_tidak_loncat(self):
        e = self.engine()
        self.jalan(e, lambda i: (hp(stamp=i), pc()), 6)  # fokus stabil
        s, _ = e.update(hp(rms=2.0, orient="tilted", lux=300.0, stamp=50), pc(kpm=0, moves=0), 7.0)  # satu jendela mengganggu
        self.assertGreater(s["focus_score"], 25)  # EWMA: skor tidak langsung jatuh ke ~0

    def test_waktu_fokus_dan_sesi_idle(self):
        e = self.engine()
        s, _ = self.jalan(e, lambda i: (hp(stamp=i), pc()), 11)
        self.assertGreater(s["session"]["focus_s"], 5)  # detik fokus bertambah saat skor >= 70
        e.stop_session()
        s, a = e.update(hp(stamp=200), pc(), 12.0)
        self.assertEqual((s["state"], a), ("IDLE", []))

    def test_menunggu_data(self):
        s, _ = self.engine().update(None, pc(), 0.0)
        self.assertEqual((s["state"], s["model"]["proba"]), ("WAITING", {}))

    def test_fallback_bila_model_error_saat_jalan(self):
        e = self.engine()
        e.model.predict_proba = lambda f: (_ for _ in ()).throw(RuntimeError("rusak"))  # model gagal
        s, _ = self.jalan(e, lambda i: (hp(stamp=i), pc(kpm=100)), 2)
        self.assertEqual(s["model"]["source"], "rule")  # otomatis pindah ke aturan
        self.assertEqual(e.mode, "rule")
        self.assertEqual(s["state"], "FOCUS")

    def test_status_kompatibel(self):
        s, _ = self.jalan(self.engine(), lambda i: (hp(stamp=i), pc()), 2)
        for key in ("focus_score", "state", "reason", "scores", "session", "model", "sent_at_ms"):
            self.assertIn(key, s)
        self.assertEqual(set(s["scores"]), {"noise", "light", "phone", "activity"})


class TestEngineRuleMode(unittest.TestCase):
    def test_mode_rule_punya_field_model(self):
        e = FocusEngine()
        e.start_session(0.0)
        s, _ = e.update(hp(), pc(kpm=100), 1.0)
        self.assertEqual(s["model"], {"source": "rule", "proba": {}})
        self.assertEqual(e.mode, "rule")


if __name__ == "__main__":
    unittest.main()
