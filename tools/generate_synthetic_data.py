"""Pembangkit data SINTETIS untuk melatih model awal FocusNest (bukan data sensor asli).

Jalankan:  python tools/generate_synthetic_data.py                       (6000 baris -> data/synthetic.csv)
           python tools/generate_synthetic_data.py --rows 20 --sessions 4 --prefix sample_ --out data/sessions.sample.csv
Fitur dibangun lewat shared.features.build_features() yang sama dengan inferensi, jadi kolomnya identik dengan data asli.
"""
import argparse  # argumen perintah
import csv  # tulis CSV
import datetime  # cap waktu sintetis
import os  # path dan folder
import random  # pembangkit acak dengan seed
import sys  # menambah path proyek
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # agar folder shared terbaca
from shared import config  # WINDOW_SEC dan ambang
from shared.features import CLASSES, CSV_COLUMNS, build_features  # fitur yang sama dengan inferensi

LABEL_NOISE = 0.04  # 4% label diacak (di rentang 3-5%) supaya data tidak terlalu mudah
MODE_WEIGHTS = {  # pola aktivitas laptop per kelas: (mengetik, menjelajah, membaca/diam)
    "FOCUS": (0.45, 0.25, 0.30), "PHONE_DISTRACTION": (0.20, 0.25, 0.55),
    "NOISE_DISTRACTION": (0.40, 0.25, 0.35), "DARK_ROOM": (0.45, 0.25, 0.30),
}


def _clip(x, lo, hi):
    """Batasi x ke [lo, hi]."""
    return max(lo, min(hi, x))


def _session_params(rng):
    """Ciri satu sesi: tiap sesi punya baseline suara, cahaya, dan gaya kerja sendiri (agar GroupKFold bermakna)."""
    quiet = -96.0 + rng.gauss(0, 5)  # ruangan hening (dBFS)
    if rng.random() < 0.15:
        quiet = rng.uniform(-62, -45)  # mikrofon/ruangan yang lebih ramai
    return {"base": quiet, "lux": rng.uniform(220, 600), "std": rng.uniform(1.0, 2.2), "typing": rng.uniform(0.8, 1.2)}


def _laptop(rng, cls, sp):
    """Aktivitas laptop: keys_per_min 0-70, mouse_moves 0-30 (jumlah saja, bukan isi)."""
    mode = rng.choices(["type", "browse", "read"], weights=MODE_WEIGHTS[cls])[0]
    if mode == "type":
        return _clip(rng.uniform(20, 70) * sp["typing"], 0, 70), rng.randint(0, 15)
    if mode == "browse":
        return rng.uniform(0, 12), rng.randint(5, 30)
    return rng.choice([0.0, 0.0, rng.uniform(0, 4.5)]), rng.randint(0, 4)  # membaca: nyaris tidak ada input


def _hp_still(rng):
    """HP diam: motion_rms ~ N(0.045, 0.02)."""
    orient = rng.choices(["face_down", "face_up", "tilted"], weights=[0.55, 0.35, 0.10])[0]
    return max(0.003, rng.gauss(0.045, 0.02)), orient


def _hp_moving(rng):
    """HP diangkat/dipegang: motion_rms 0.7 s.d. 3.0."""
    orient = rng.choices(["tilted", "face_up", "face_down"], weights=[0.65, 0.25, 0.10])[0]
    return rng.uniform(0.7, 3.0), orient


def _streak(rng):
    """Jumlah jendela berturut-turut HP bergerak (>=1)."""
    return min(1 + int(rng.expovariate(0.45)), 12)


def _noise_delta_quiet(rng, sp):
    """Kenaikan suara saat ruangan tenang (sekitar 0 dB, kadang batuk/ketikan keras)."""
    d = rng.gauss(0.3, sp["std"] * 0.8)
    return d + (rng.uniform(3, 8) if rng.random() < 0.05 else 0.0)


def _row(rng, cls, sp):
    """Satu jendela 2 detik untuk kelas `cls` -> (hp, pc, ctx) mentah seperti telemetri asli."""
    motion, orient = _hp_still(rng)  # default: HP diam
    streak = 1 if rng.random() < 0.03 else 0  # senggolan kecil sesekali
    delta = _noise_delta_quiet(rng, sp)  # default: ruangan tenang
    lux = _clip(rng.gauss(sp["lux"], 110), 150, 800)  # cahaya normal
    keys, moves = _laptop(rng, cls, sp)

    if cls == "FOCUS":
        r = rng.random()
        if r < 0.08:  # kasus campuran: HP diambil sebentar sambil laptop aktif
            motion, orient = _hp_moving(rng); streak = rng.randint(1, 2)
            keys, moves = rng.uniform(15, 70), rng.randint(3, 20)
        elif r < 0.14:
            delta = rng.uniform(4, 12)  # suara sedang (batuk, kursi digeser)
        elif r < 0.19:
            lux = rng.uniform(60, 135)  # sore agak redup (tumpang tindih dengan gelap)
    elif cls == "PHONE_DISTRACTION":
        if rng.random() < 0.75:
            motion, orient = _hp_moving(rng); streak = _streak(rng)  # HP jelas diangkat
        else:
            motion, orient = rng.uniform(0.15, 0.7), rng.choice(["tilted", "face_up"]); streak = rng.randint(0, 1)  # awal mengangkat
        if rng.random() < 0.15:
            delta = rng.uniform(8, 30)  # kasus campuran: HP bergerak sambil berisik
    elif cls == "NOISE_DISTRACTION":
        r = rng.random()
        delta = rng.uniform(10, 40) if r < 0.70 else rng.uniform(5, 10) if r < 0.85 else rng.uniform(40, 55)
        if rng.random() < 0.15:  # kasus campuran: HP bergerak sambil berisik
            motion, orient = _hp_moving(rng); streak = rng.randint(1, 3)
    else:  # DARK_ROOM
        lux = rng.uniform(1, 100) if rng.random() < 0.85 else rng.uniform(100, 140)
        orient = rng.choices(["face_up", "tilted", "face_down"], weights=[0.50, 0.45, 0.05])[0]  # telungkup -> lux None (jarang: sensor harus terbuka agar gelap terdeteksi)
        if rng.random() < 0.06:
            delta = rng.uniform(4, 10)

    if rng.random() < 0.03:
        lux = None  # pembacaan lux gagal
    z = (delta - rng.uniform(-0.5, 0.5)) / sp["std"] + rng.gauss(0, 0.3)  # z-score lonjakan suara
    hp = {"motion_rms": motion, "orientation": orient, "lux": lux}
    pc = {"noise_dbfs": max(-120.0, sp["base"] + delta + rng.gauss(0, 0.5)), "keys_per_min": keys, "mouse_moves": moves}
    ctx = {"noise_delta": delta, "noise_zscore": z, "move_streak": streak}
    return hp, pc, ctx


def generate(rows=6000, sessions=30, seed=42, prefix="syn_"):
    """Bangkitkan `rows` baris dari `sessions` sesi (kelas bergilir). Hasil sama untuk seed sama."""
    rng = random.Random(seed)  # seed tetap = hasil bisa direproduksi
    start = datetime.datetime(2026, 1, 1, 8, 0, 0)  # cap waktu sintetis
    per, extra = divmod(rows, sessions)  # baris per sesi
    out, i = [], 0
    for s in range(sessions):
        cls = CLASSES[s % len(CLASSES)]  # kelas sesi (bergilir)
        sp = _session_params(rng)
        for _ in range(per + (1 if s < extra else 0)):
            hp, pc, ctx = _row(rng, cls, sp)
            row = build_features(hp, pc, ctx)  # SAMA dengan inferensi
            label = cls
            if rng.random() < LABEL_NOISE:
                label = rng.choice([c for c in CLASSES if c != cls])  # label diacak
            row = {k: round(v, 4) for k, v in row.items()}
            row.update(label=label, session_id=f"{prefix}{s + 1:03d}",
                       timestamp=(start + datetime.timedelta(seconds=config.WINDOW_SEC * i)).isoformat(timespec="seconds"))
            out.append(row)
            i += 1
    return out


def write_csv(rows, path):
    """Tulis baris ke CSV (folder dibuat otomatis)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Bangkitkan data sintetis FocusNest")
    ap.add_argument("--rows", type=int, default=6000, help="jumlah baris (default 6000)")
    ap.add_argument("--sessions", type=int, default=30, help="jumlah sesi sintetis (default 30)")
    ap.add_argument("--seed", type=int, default=42, help="seed acak (default 42)")
    ap.add_argument("--prefix", default="syn_", help="awalan session_id (default syn_)")
    ap.add_argument("--out", default="data/synthetic.csv", help="file keluaran")
    args = ap.parse_args()
    rows = generate(args.rows, args.sessions, args.seed, args.prefix)
    write_csv(rows, args.out)
    print(f"[SINTETIS] {len(rows)} baris, {args.sessions} sesi -> {args.out} (seed={args.seed})")
    for c in CLASSES:
        print(f"  {c:<18} {sum(r['label'] == c for r in rows)} baris")


if __name__ == "__main__":
    main()
