"""Uji penulis dataset mode collect dan generator sintetis."""
import csv, math, os, sys, tempfile, unittest  # modul standar
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.datalog import DataLogger  # penulis dataset
from shared.features import CLASSES, CSV_COLUMNS, FEATURE_NAMES  # kolom dan kelas
from tools import generate_synthetic_data as gen  # generator sintetis

FITUR = {n: 0.0 for n in FEATURE_NAMES}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDataLogger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "baru", "sessions.csv")  # folder belum ada
    def tearDown(self):
        self.tmp.cleanup()

    def baca(self):
        with open(self.path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def test_header_dan_folder_otomatis(self):
        DataLogger(self.path, "FOCUS", "s1", 0.0)
        with open(self.path, encoding="utf-8") as fh:
            self.assertEqual(fh.readline().strip().split(","), CSV_COLUMNS)

    def test_abaikan_3_detik_pertama(self):
        lg = DataLogger(self.path, "FOCUS", "s1", 100.0)
        self.assertFalse(lg.maybe_write(FITUR, 102.9, 1))  # masih masa transisi
        self.assertTrue(lg.maybe_write(FITUR, 103.0, 2))
        self.assertEqual(len(self.baca()), 1)

    def test_satu_baris_per_jendela(self):
        lg = DataLogger(self.path, "NOISE_DISTRACTION", "s2", 0.0)
        self.assertTrue(lg.maybe_write(FITUR, 10.0, 1))
        self.assertFalse(lg.maybe_write(FITUR, 11.0, 1))  # paket sama
        self.assertFalse(lg.maybe_write(FITUR, 10.5, 2))  # terlalu cepat
        self.assertTrue(lg.maybe_write(FITUR, 12.0, 2))
        rows = self.baca()
        self.assertEqual([r["label"] for r in rows], ["NOISE_DISTRACTION"] * 2)
        self.assertEqual(rows[0]["session_id"], "s2")
        self.assertEqual(list(rows[0].keys()), CSV_COLUMNS)  # hanya angka agregat + label + id + waktu

    def test_tanpa_fitur_tidak_menulis(self):
        self.assertFalse(DataLogger(self.path, "FOCUS", "s1", 0.0).maybe_write(None, 50.0, 1))

    def test_label_salah_ditolak(self):
        with self.assertRaises(ValueError):
            DataLogger(self.path, "TIDUR", "s1", 0.0)

    def test_header_beda_ditolak(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w") as fh:
            fh.write("a,b,c\n")
        with self.assertRaises(ValueError):
            DataLogger(self.path, "FOCUS", "s1", 0.0)

    def test_append_tidak_menimpa(self):
        DataLogger(self.path, "FOCUS", "s1", 0.0).maybe_write(FITUR, 10.0, 1)
        DataLogger(self.path, "DARK_ROOM", "s2", 0.0).maybe_write(FITUR, 10.0, 1)
        self.assertEqual([r["session_id"] for r in self.baca()], ["s1", "s2"])


class TestSintetis(unittest.TestCase):
    def test_seed_sama_data_sama(self):
        self.assertEqual(gen.generate(300, 6, seed=1), gen.generate(300, 6, seed=1))
        self.assertNotEqual(gen.generate(300, 6, seed=1), gen.generate(300, 6, seed=2))

    def test_empat_kelas_dan_tanpa_nan(self):
        rows = gen.generate(1200, 12, seed=3)
        self.assertEqual({r["label"] for r in rows}, set(CLASSES))
        for r in rows:
            self.assertEqual(list(r.keys()), CSV_COLUMNS)  # kolom identik dengan data asli
            self.assertTrue(all(math.isfinite(r[n]) for n in FEATURE_NAMES))
        self.assertEqual(len({r["session_id"] for r in rows}), 12)
        self.assertTrue(any(r["lux_valid"] == 0.0 for r in rows) and any(r["lux_valid"] == 1.0 for r in rows))  # ada lux None

    def test_file_contoh_ada(self):
        with open(os.path.join(ROOT, "data", "sessions.sample.csv"), newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 20)
        self.assertEqual(list(rows[0].keys()), CSV_COLUMNS)


if __name__ == "__main__":
    unittest.main()
