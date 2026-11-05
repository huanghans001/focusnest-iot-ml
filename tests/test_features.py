"""Uji ekstraksi fitur. Jalankan: python -m unittest discover -s tests -v"""
import math, os, sys, unittest  # modul standar
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # agar folder shared terbaca
from shared.features import FEATURE_NAMES, LUX_SENTINEL, FeatureTracker, build_features, to_vector  # fitur bersama

CTX = {"noise_delta": 1.0, "noise_zscore": 0.5, "move_streak": 0}


def hp(rms=0.04, orient="face_up", lux=300.0, stamp=1):
    """Telemetri HP palsu."""
    return {"motion_rms": rms, "orientation": orient, "lux": lux, "sent_at_ms": stamp}


def pc(noise=-95.0, kpm=30.0, moves=3):
    """Telemetri laptop palsu."""
    return {"noise_dbfs": noise, "keys_per_min": kpm, "mouse_moves": moves}


class TestFeatures(unittest.TestCase):
    def test_urutan_fitur_konsisten(self):
        f = build_features(hp(), pc(), CTX)
        self.assertEqual(list(f.keys()), FEATURE_NAMES)  # urutan dict = FEATURE_NAMES
        self.assertEqual(len(to_vector(f)), len(FEATURE_NAMES))
        self.assertEqual(to_vector(f), [f[n] for n in FEATURE_NAMES])
        self.assertEqual(len(set(FEATURE_NAMES)), len(FEATURE_NAMES))  # tidak ada nama ganda

    def test_fitur_minimal_ada(self):
        for name in ["motion_rms", "orientation", "lux", "lux_valid", "noise_dbfs", "noise_delta", "noise_zscore",
                     "keys_per_min", "mouse_moves", "move_streak", "laptop_idle"]:
            self.assertIn(name, FEATURE_NAMES)

    def test_lux_none(self):
        f = build_features(hp(lux=None), pc(), CTX)
        self.assertEqual(f["lux"], LUX_SENTINEL)  # sentinel, bukan NaN
        self.assertEqual(f["lux_valid"], 0.0)

    def test_lux_telungkup_dianggap_tertutup(self):
        f = build_features(hp(orient="face_down", lux=250.0), pc(), CTX)  # sensor tertutup walau ada angka
        self.assertEqual((f["lux"], f["lux_valid"]), (LUX_SENTINEL, 0.0))

    def test_lux_valid(self):
        f = build_features(hp(lux=312.5), pc(), CTX)
        self.assertEqual((f["lux"], f["lux_valid"]), (312.5, 1.0))

    def test_orientasi_numerik(self):
        codes = [build_features(hp(orient=o), pc(), CTX)["orientation"] for o in ("face_down", "tilted", "face_up", "aneh")]
        self.assertEqual(codes, [-1.0, 0.0, 1.0, 0.0])

    def test_tanpa_nan(self):
        kotor = [hp(rms=float("nan"), lux=float("nan")), hp(lux=float("inf")), hp(lux=None)]
        for h in kotor:
            f = build_features(h, pc(noise=float("-inf"), kpm=float("nan")), {"noise_delta": float("nan"), "noise_zscore": None, "move_streak": 0})
            self.assertTrue(all(isinstance(v, float) and math.isfinite(v) for v in f.values()), f)

    def test_laptop_idle(self):
        self.assertEqual(build_features(hp(), pc(kpm=0, moves=0), CTX)["laptop_idle"], 1.0)
        self.assertEqual(build_features(hp(), pc(kpm=60, moves=0), CTX)["laptop_idle"], 0.0)
        self.assertEqual(build_features(hp(), pc(kpm=0, moves=9), CTX)["laptop_idle"], 0.0)


class TestTracker(unittest.TestCase):
    def test_streak_dihitung_sekali_per_paket(self):
        t = FeatureTracker()
        self.assertEqual(t.observe(hp(rms=2.0, stamp=1), pc())["move_streak"], 1)
        self.assertEqual(t.observe(hp(rms=2.0, stamp=1), pc())["move_streak"], 1)  # paket sama
        self.assertEqual(t.observe(hp(rms=2.0, stamp=2), pc())["move_streak"], 2)
        self.assertEqual(t.observe(hp(rms=0.01, stamp=3), pc())["move_streak"], 0)  # diam -> reset

    def test_baseline_dan_zscore(self):
        t = FeatureTracker()
        for i in range(15):
            ctx = t.observe(hp(stamp=i), pc(noise=-95.0))
        self.assertAlmostEqual(ctx["noise_delta"], 0.0, places=3)
        lonjakan = t.observe(hp(stamp=99), pc(noise=-65.0))
        self.assertAlmostEqual(lonjakan["noise_delta"], 30.0, places=3)
        self.assertGreater(lonjakan["noise_zscore"], 3.0)


if __name__ == "__main__":
    unittest.main()
