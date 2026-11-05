"""Latih model Random Forest FocusNest, evaluasi, dan simpan ke model/focus_rf.joblib.

Contoh:
  python tools/generate_synthetic_data.py
  python train_model.py --data data/synthetic.csv                                   (model awal, sintetis saja)
  python train_model.py --data data/synthetic.csv data/sessions.csv --test-data data/uji.csv
PRIVASI: CSV hanya berisi angka agregat + label + session_id + timestamp.
"""
import argparse  # argumen perintah
import datetime  # tanggal latih
import os  # path
import sys  # keluar dengan pesan
import warnings  # peringatan
import joblib  # versi joblib
import matplotlib  # grafik
matplotlib.use("Agg")  # tanpa layar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from shared import model as model_io  # format file model
from shared.engine import rule_class  # algoritma rule-based lama (baseline)
from shared.features import CLASSES, FEATURE_NAMES  # fitur dan kelas bersama

SEED = 42  # seed tetap
MIN_PER_CLASS = 50  # di bawah ini = peringatan "terlalu sedikit"
IMBALANCE_RATIO = 3.0  # kelas terbesar / terkecil di atas ini = peringatan "tidak seimbang"
GRID = [(n, d) for n in (100, 200) for d in (8, 12, None)]  # pencarian hyperparameter kecil (--search)
TITLE_SYN = "Validasi pada data sintetis (hanya sanity check, bukan bukti performa nyata)"
TITLE_REAL = "Pengujian pada data sensor asli (--test-data)"


def source_of(path):
    """Nama file mengandung 'synthetic' atau 'sample' = sintetis; selain itu data asli."""
    name = os.path.basename(path).lower()
    return "synthetic" if ("synthetic" in name or "sample" in name) else "real"


def load_csvs(paths, what):
    """Baca dan validasi CSV, gabungkan, tambah kolom 'source'."""
    frames = []
    for path in paths:
        if not os.path.isfile(path):
            sys.exit(f"ERROR: file {what} tidak ditemukan: {path}")
        df = pd.read_csv(path)
        missing = [c for c in FEATURE_NAMES + ["label", "session_id"] if c not in df.columns]
        if missing:
            sys.exit(f"ERROR: {path} kehilangan kolom: {missing}")
        if df[FEATURE_NAMES].isna().any().any() or not np.isfinite(df[FEATURE_NAMES].to_numpy(dtype=float)).all():
            sys.exit(f"ERROR: {path} berisi NaN/inf di kolom fitur")
        bad = sorted(set(df["label"]) - set(CLASSES))
        if bad:
            sys.exit(f"ERROR: {path} berisi label tidak dikenal: {bad}. Yang sah: {CLASSES}")
        df = df[FEATURE_NAMES + ["label", "session_id"]].copy()
        df["session_id"] = df["session_id"].astype(str)
        df["source"] = source_of(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def check_counts(df, what, require_all):
    """Tampilkan jumlah baris per kelas; beri peringatan bila sedikit atau tidak seimbang."""
    counts = df["label"].value_counts().reindex(CLASSES, fill_value=0)
    print(f"\nJumlah baris per kelas ({what}):")
    for c in CLASSES:
        print(f"  {c:<18} {counts[c]:>6}")
    print(f"  total {len(df)} baris, {df['session_id'].nunique()} sesi")
    if require_all and (counts == 0).any():
        sys.exit(f"ERROR: kelas tanpa data: {[c for c in CLASSES if counts[c] == 0]}. Kumpulkan data untuk semua 4 kelas.")
    few = [c for c in CLASSES if 0 < counts[c] < MIN_PER_CLASS]
    if few:
        print(f"PERINGATAN: kelas terlalu sedikit data (<{MIN_PER_CLASS} baris): {few}. Hasil bisa tidak stabil.")
    nz = counts[counts > 0]
    if len(nz) > 1 and nz.max() / nz.min() > IMBALANCE_RATIO:
        print(f"PERINGATAN: data tidak seimbang (terbesar/terkecil = {nz.max() / nz.min():.1f}x). class_weight='balanced' membantu, tapi kumpulkan lebih banyak data kelas kecil.")
    return counts


def make_rf(n_estimators, max_depth):
    """Random Forest dengan class_weight balanced dan seed tetap."""
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=3,
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)


def metrics(y_true, y_pred):
    """Accuracy, macro-F1, dan precision/recall/F1 per kelas + confusion matrix (urutan CLASSES)."""
    rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
            "report": rep, "cm": confusion_matrix(y_true, y_pred, labels=CLASSES)}


def print_metrics(title, m):
    """Cetak metrik ke terminal."""
    print(f"\n=== {title} ===\naccuracy = {m['accuracy']:.3f} | macro-F1 = {m['macro_f1']:.3f}")
    print(f"{'kelas':<18}{'precision':>10}{'recall':>8}{'f1':>8}{'n':>7}")
    for c in CLASSES:
        r = m["report"][c]
        print(f"{c:<18}{r['precision']:>10.3f}{r['recall']:>8.3f}{r['f1-score']:>8.3f}{int(r['support']):>7}")
    print("confusion matrix (baris = benar, kolom = prediksi):")
    for c, row in zip(CLASSES, m["cm"]):
        print(f"  {c:<18}" + "".join(f"{v:>7}" for v in row))


def plot_cm(cm, title, path):
    """Simpan confusion matrix sebagai PNG."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Label benar"); ax.set_title(title, fontsize=10)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_importance(names, values, path):
    """Simpan feature importance (urut terbesar) sebagai PNG."""
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh([names[i] for i in order], [values[i] for i in order], color="#17806d")
    ax.set_xlabel("Importance (Gini)"); ax.set_title("Feature importance Random Forest")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def md_metrics(m):
    """Tabel markdown precision/recall/F1 per kelas."""
    lines = ["| Kelas | Precision | Recall | F1 | Jumlah |", "|---|---|---|---|---|"]
    for c in CLASSES:
        r = m["report"][c]
        lines.append(f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    return "\n".join(lines)


def md_cm(cm):
    """Confusion matrix sebagai tabel markdown."""
    lines = ["| Benar \\ Prediksi | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for c, row in zip(CLASSES, cm):
        lines.append(f"| **{c}** | " + " | ".join(str(int(v)) for v in row) + " |")
    return "\n".join(lines)


def write_report(path, ctx):
    """Tulis docs/EVALUASI.md (Bahasa Indonesia)."""
    meta, cv, imp = ctx["meta"], ctx["cv"], ctx["importance"]
    L = ["# Evaluasi Model Random Forest FocusNest", "",
         f"Dibuat otomatis oleh `train_model.py` pada {meta['trained_at']} (scikit-learn {meta['sklearn_version']}).", ""]
    if meta["data_source"] == "synthetic":
        L += ["> **Peringatan:** model ini dilatih HANYA dari data sintetis (simulasi). Angka di bawah belum membuktikan performa pada sensor sungguhan.", ""]
    L += ["## 1. Data latih", "", "| Sumber | " + " | ".join(CLASSES) + " | Total |", "|---|" + "---|" * (len(CLASSES) + 1)]
    for src, cnt in ctx["train_counts"].items():
        L.append(f"| {src} | " + " | ".join(str(int(cnt[c])) for c in CLASSES) + f" | {int(cnt.sum())} |")
    L += ["", f"Hyperparameter: `n_estimators={meta['hyperparams']['n_estimators']}`, `max_depth={meta['hyperparams']['max_depth']}`, "
          f"`class_weight=balanced`, `random_state={SEED}`." + (" Dipilih lewat pencarian kecil (`--search`) pada validasi silang yang sama dengan bagian 2, sehingga angka bagian 2 sedikit optimistis." if ctx["searched"] else ""), "",
          f"## 2. {TITLE_SYN}", "",
          f"GroupKFold {cv['folds']} lipatan berdasarkan `session_id` (bukan split acak per baris). Data sintetis dibangkitkan dengan asumsi dari penulis kode, "
          "sehingga skor tinggi di sini hanya menunjukkan bahwa pipeline berjalan dan model dapat mempelajari pola yang sengaja dimasukkan ke generator.", "",
          f"**Accuracy: {cv['m']['accuracy']:.3f} | Macro-F1: {cv['m']['macro_f1']:.3f}**", "", md_metrics(cv["m"]), "", "Confusion matrix (gabungan semua lipatan):", "", md_cm(cv["m"]["cm"]), "",
          "![Confusion matrix validasi sintetis](confusion_matrix_sintetis.png)", ""]
    real = ctx.get("real")
    if real:
        rf, rule = real["rf"], real["rule"]
        d = rf["macro_f1"] - rule["macro_f1"]
        verdict = ("selisih macro-F1 kecil (<0,02), kedua algoritma setara pada data ini" if abs(d) < 0.02
                   else f"Random Forest lebih baik (macro-F1 +{d:.3f})" if d > 0 else f"Algoritma rule-based lebih baik (macro-F1 {-d:.3f} lebih tinggi)")
        L += [f"## 3. {TITLE_REAL}", "",
              f"Data uji: {', '.join(real['files'])} ({real['n']} baris, {real['sessions']} sesi). Data ini TIDAK dipakai untuk melatih model.", "",
              "| Metrik | Rule-based (lama) | Random Forest |", "|---|---|---|",
              f"| Accuracy | {rule['accuracy']:.3f} | {rf['accuracy']:.3f} |", f"| Macro-F1 | {rule['macro_f1']:.3f} | {rf['macro_f1']:.3f} |"]
        for c in CLASSES:
            L.append(f"| F1 {c} | {rule['report'][c]['f1-score']:.3f} | {rf['report'][c]['f1-score']:.3f} |")
        L += ["", f"**Ringkasan:** {verdict}.", "", "### Random Forest", "", md_metrics(rf), "", md_cm(rf["cm"]), "",
              "![Confusion matrix RF data asli](confusion_matrix_asli_rf.png)", "", "### Rule-based (lama)", "", md_metrics(rule), "", md_cm(rule["cm"]), "",
              "![Confusion matrix rule-based data asli](confusion_matrix_asli_rule.png)", ""]
        if real["missing"]:
            L += [f"Catatan: data uji tidak memiliki kelas {real['missing']}, sehingga metrik kelas itu bernilai 0 dan tidak bermakna.", ""]
    L += ["## Feature importance", "", "| Fitur | Importance |", "|---|---|"] + [f"| {n} | {v:.3f} |" for n, v in imp] + ["",
          "![Feature importance](feature_importance.png)", "",
          "## Batasan data sintetis", "",
          "- Distribusi sintetis ditentukan oleh penulis generator (`tools/generate_synthetic_data.py`) dari nilai tipikal proyek; bukan hasil pengukuran. Pola yang \"dipelajari\" model adalah pola yang kita tulis sendiri.",
          "- Kelas dibuat saling tumpang tindih, ada kasus campuran, dan 4% label diacak, tetapi dunia nyata lebih berantakan (mikrofon, hotspot, kebiasaan orang berbeda).",
          "- Sensor cahaya HP tertutup saat telungkup, sehingga ruangan gelap sulit dikenali pada kondisi itu (`lux_valid = 0`); generator hanya meniru sebagian kasus ini.",
          "- Karena itu model bawaan adalah **model awal**. Untuk performa yang bisa dipertanggungjawabkan, rekam data asli (`detector.py --collect`), latih ulang, dan uji dengan `--test-data` pada sesi yang berbeda.", "",
          "## Cara mereproduksi", "", "```bash", "python tools/generate_synthetic_data.py", ctx["cmd"], "```", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description="Latih model Random Forest FocusNest")
    ap.add_argument("--data", nargs="+", default=["data/synthetic.csv"], help="CSV latih (boleh lebih dari satu)")
    ap.add_argument("--test-data", nargs="*", default=[], help="CSV data sensor ASLI yang hanya dipakai untuk uji")
    ap.add_argument("--out", default="model/focus_rf.joblib", help="lokasi model keluaran")
    ap.add_argument("--docs-dir", default="docs", help="folder laporan dan gambar")
    ap.add_argument("--folds", type=int, default=5, help="jumlah lipatan GroupKFold")
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=12, help="0 = tanpa batas")
    ap.add_argument("--search", action="store_true", help="pencarian hyperparameter kecil (n_estimators x max_depth)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    if set(map(os.path.abspath, args.data)) & set(map(os.path.abspath, args.test_data)):
        sys.exit("ERROR: file yang sama ada di --data dan --test-data (kebocoran data uji).")
    os.makedirs(args.docs_dir, exist_ok=True)

    train = load_csvs(args.data, "latih")
    counts = check_counts(train, "data latih", require_all=True)
    by_src = {s: train[train.source == s]["label"].value_counts().reindex(CLASSES, fill_value=0) for s in train.source.unique()}
    for s, c in by_src.items():
        print(f"  sumber {s}: {int(c.sum())} baris")
    X, y, g = train[FEATURE_NAMES].to_numpy(dtype=float), train["label"].to_numpy(), train["session_id"].to_numpy()
    folds = min(args.folds, train["session_id"].nunique())
    if folds < 2:
        sys.exit("ERROR: butuh minimal 2 session_id berbeda untuk GroupKFold.")
    if folds < args.folds:
        print(f"PERINGATAN: hanya {folds} sesi, lipatan diturunkan dari {args.folds} ke {folds}.")
    cv = GroupKFold(n_splits=folds)

    n_est, depth = args.n_estimators, (args.max_depth or None)
    if args.search:
        print("\nPencarian hyperparameter (GroupKFold, macro-F1):")
        best = None
        for n, d in GRID:
            pred = cross_val_predict(make_rf(n, d), X, y, groups=g, cv=cv)
            f1 = f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)
            print(f"  n_estimators={n:<4} max_depth={str(d):<5} macro-F1={f1:.3f}")
            if best is None or f1 > best[0]:
                best = (f1, n, d)
        _, n_est, depth = best
        print(f"  dipilih: n_estimators={n_est}, max_depth={depth}")

    # --- Bagian 1: validasi silang pada data latih (sanity check bila datanya sintetis) ---
    pred_cv = cross_val_predict(make_rf(n_est, depth), X, y, groups=g, cv=cv)
    m_cv = metrics(y, pred_cv)
    only_syn = set(train.source) == {"synthetic"}
    print_metrics(f"{TITLE_SYN if only_syn else 'Validasi silang GroupKFold pada data latih'} [{folds} lipatan]", m_cv)
    plot_cm(m_cv["cm"], f"Validasi GroupKFold ({'sintetis' if only_syn else 'data latih'})", os.path.join(args.docs_dir, "confusion_matrix_sintetis.png"))

    # --- Model akhir pada seluruh data latih ---
    rf = make_rf(n_est, depth).fit(X, y)
    imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda t: -t[1])
    print("\nFeature importance (terbesar dulu):")
    for n, v in imp:
        print(f"  {n:<14} {v:.3f}")
    plot_importance([n for n, _ in imp], [v for _, v in imp], os.path.join(args.docs_dir, "feature_importance.png"))

    # --- Bagian 2: uji pada data sensor asli (hanya bila --test-data) ---
    ctx = {"cv": {"m": m_cv, "folds": folds}, "importance": imp, "train_counts": by_src, "searched": args.search}
    if args.test_data:
        test = load_csvs(args.test_data, "uji")
        tc = check_counts(test, "data uji", require_all=False)
        if (test.source == "synthetic").any():
            print("PERINGATAN: --test-data berisi file bernama sintetis/sample; seharusnya data sensor asli.")
        Xt, yt = test[FEATURE_NAMES].to_numpy(dtype=float), test["label"].to_numpy()
        rf_m = metrics(yt, rf.predict(Xt))
        rule_m = metrics(yt, [rule_class(r) for r in test[FEATURE_NAMES].to_dict("records")])
        print_metrics(f"{TITLE_REAL}: Random Forest", rf_m)
        print_metrics(f"{TITLE_REAL}: Rule-based (lama)", rule_m)
        print(f"\nPerbandingan macro-F1: rule-based {rule_m['macro_f1']:.3f} vs Random Forest {rf_m['macro_f1']:.3f}")
        plot_cm(rf_m["cm"], "Data asli: Random Forest", os.path.join(args.docs_dir, "confusion_matrix_asli_rf.png"))
        plot_cm(rule_m["cm"], "Data asli: Rule-based", os.path.join(args.docs_dir, "confusion_matrix_asli_rule.png"))
        ctx["real"] = {"rf": rf_m, "rule": rule_m, "n": len(test), "sessions": test["session_id"].nunique(),
                       "files": [os.path.basename(p) for p in args.test_data], "missing": [c for c in CLASSES if tc[c] == 0]}

    # --- Simpan model + metadata ---
    n_syn, n_real = int((train.source == "synthetic").sum()), int((train.source == "real").sum())
    meta = {"feature_names": FEATURE_NAMES, "classes": CLASSES, "sklearn_version": sklearn.__version__, "joblib_version": joblib.__version__,
            "trained_at": datetime.date.today().isoformat(), "n_rows_synthetic": n_syn, "n_rows_real": n_real,
            "data_source": "synthetic" if n_real == 0 else "real" if n_syn == 0 else "synthetic+real",
            "hyperparams": {"n_estimators": n_est, "max_depth": depth}, "random_state": SEED, "train_files": [os.path.basename(p) for p in args.data]}
    model_io.save_bundle(rf, meta, args.out)
    ctx["meta"] = ctx_meta = meta
    ctx["cmd"] = "python train_model.py --data " + " ".join(args.data) + (" --test-data " + " ".join(args.test_data) if args.test_data else "") + (" --search" if args.search else "")
    write_report(os.path.join(args.docs_dir, "EVALUASI.md"), ctx)
    print(f"\nModel disimpan: {args.out} ({n_syn} baris sintetis, {n_real} baris asli)\nLaporan: {os.path.join(args.docs_dir, 'EVALUASI.md')}")
    if n_real == 0:
        print("PERINGATAN: model dilatih HANYA dari data sintetis. Latih ulang dengan data asli: --data data/synthetic.csv data/sessions.csv --test-data <csv uji>.")


if __name__ == "__main__":
    main()
