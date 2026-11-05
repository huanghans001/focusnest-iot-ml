"""NODE 3 (laptop): menerima telemetri dari edge1 & edge2, sensor fusion, kirim status + alert, host dashboard.

Jalankan:  python detector.py                      (normal: model Random Forest, fallback ke aturan)
           python detector.py --no-ml              (paksa algoritma rule-based)
           python detector.py --collect --label FOCUS --session-id fokus_01   (rekam data latih ke data/sessions.csv)
"""
import argparse  # argumen perintah
import functools  # membuat handler HTTP dengan folder tertentu
import http.server  # server web sederhana untuk dashboard
import json  # parsing JSON
import pathlib  # path folder dashboard
import threading  # lock dan thread server web
import time  # jeda dan waktu
from shared import config, mqtt_client  # pengaturan dan MQTT
from shared.datalog import DataLogger  # penulis dataset (mode collect)
from shared.engine import FocusEngine  # mesin sensor fusion
from shared.features import CLASSES  # daftar label yang sah

DASH_DIR = pathlib.Path(__file__).parent / "dashboard"  # folder dashboard
DATA_FILE = pathlib.Path(__file__).parent / "data" / "sessions.csv"  # tujuan mode collect
LOCK = threading.Lock()  # lindungi data bersama antar thread
LATEST = {}  # device_id -> (telemetri, waktu terima)
ENGINE = FocusEngine()  # mesin fusion (diganti di main() sesuai mode)


def on_message(client, userdata, msg):
    """Terima telemetri edge dan perintah dari dashboard."""
    try:
        data = json.loads(msg.payload)  # baca JSON
    except ValueError:
        return  # abaikan pesan rusak
    if msg.topic.endswith("/telemetry"):
        device = msg.topic.split("/")[-2]  # edge1 / edge2
        with LOCK:
            LATEST[device] = (data, time.monotonic())  # simpan paket terbaru
    elif msg.topic == config.topic("cmd"):
        action = data.get("action")  # start / stop
        with LOCK:
            if action == "start":
                ENGINE.start_session(time.monotonic())  # mulai sesi
                print("[CMD] sesi dimulai")
            elif action == "stop":
                ENGINE.stop_session()  # akhiri sesi
                print("[CMD] sesi dihentikan")


def fresh(device, now):
    """Telemetri device bila belum basi, kalau tidak None."""
    item = LATEST.get(device)  # (data, waktu)
    if item and now - item[1] <= config.STALE_SEC:
        return item[0]  # masih segar
    return None  # basi atau belum ada


def serve_dashboard():
    """Jalankan web server statis untuk folder dashboard (bisa dibuka dari HP di jaringan yang sama)."""
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass  # jangan cetak log tiap request
    handler = functools.partial(Quiet, directory=str(DASH_DIR))  # layani folder dashboard
    server = http.server.ThreadingHTTPServer(("0.0.0.0", config.DASHBOARD_PORT), handler)  # semua antarmuka
    threading.Thread(target=server.serve_forever, daemon=True).start()  # jalankan di latar belakang


def parse_args(argv=None):
    """Baca argumen perintah; --collect mewajibkan --label dan --session-id."""
    ap = argparse.ArgumentParser(description="FocusNest detector (NODE 3)")
    ap.add_argument("--collect", action="store_true", help="mode pengumpulan data: tulis fitur tiap jendela ke data/sessions.csv")
    ap.add_argument("--label", choices=CLASSES, help="label kelas untuk semua baris sesi ini (wajib bersama --collect)")
    ap.add_argument("--session-id", help="nama sesi rekaman, mis. fokus_01 (wajib bersama --collect)")
    ap.add_argument("--no-ml", action="store_true", help="paksa algoritma rule-based (tanpa model)")
    args = ap.parse_args(argv)
    if args.collect and not (args.label and args.session_id):
        ap.error("--collect membutuhkan --label dan --session-id")
    if not args.collect and (args.label or args.session_id):
        ap.error("--label/--session-id hanya dipakai bersama --collect")
    return args


def main():
    global ENGINE
    args = parse_args()
    print("FocusNest - NODE 3 / Laptop (processing server + dashboard)")
    logger = None  # penulis data (hanya mode collect)
    if args.collect:
        logger = DataLogger(DATA_FILE, args.label, args.session_id, time.monotonic())  # abaikan 3 detik pertama
        ENGINE = FocusEngine()  # collect memakai fitur yang sama; keputusan tidak dipakai
        print(f"[COLLECT] label={args.label} sesi={args.session_id} -> {DATA_FILE} (3 detik pertama diabaikan, Ctrl+C untuk berhenti)")
    elif args.no_ml:
        ENGINE = FocusEngine()  # rule-based
        print("[MODEL] mode ATURAN (rule-based): dipaksa lewat --no-ml")
    else:
        ENGINE = FocusEngine.auto()  # model bila ada dan valid, kalau tidak fallback
    try:
        serve_dashboard()  # nyalakan dashboard
    except OSError as err:
        print(f"[DASHBOARD] gagal membuka port {config.DASHBOARD_PORT}: {err}")
    print(f"[DASHBOARD] di laptop : http://localhost:{config.DASHBOARD_PORT}/?project={config.PROJECT_ID}")
    print(f"[DASHBOARD] dari HP   : http://{config.lan_ip()}:{config.DASHBOARD_PORT}/?project={config.PROJECT_ID}")
    print(f"[CONFIG] PROJECT_ID: {config.PROJECT_ID}")
    try:
        client = mqtt_client.make_client("detector", subscribe=[config.topic("edge/+/telemetry"), config.topic("cmd")], on_message=on_message)  # sambung MQTT
    except OSError as err:
        print(f"[DETECTOR] tidak bisa terhubung ke broker: {err}. Cek internet/hotspot.")
        return
    print("[DETECTOR] menunggu edge1 (HP) dan edge2 (laptop)...")
    last_state = None  # untuk mencetak saat status berubah
    try:
        while True:
            time.sleep(1)  # evaluasi tiap detik
            now = time.monotonic()  # waktu monoton
            with LOCK:
                hp, pc = fresh("edge1", now), fresh("edge2", now)  # data segar
                status, alerts = ENGINE.update(hp, pc, now)  # sensor fusion
                if logger and logger.maybe_write(ENGINE.last_features, now, hp.get("sent_at_ms") if hp else None):
                    print(f"[COLLECT] baris #{logger.count} ({args.label})")  # satu baris per jendela
            status["edges"] = {"edge1": hp is not None, "edge2": pc is not None}  # edge mana yang aktif
            mqtt_client.publish_json(client, "status", status, qos=0, retain=True)  # kirim ke dashboard & HP
            for alert in alerts:
                mqtt_client.publish_json(client, "alerts", alert, qos=1)  # kirim balik ke HP (arah 2)
                print(f"[ALERT] {alert['type']}: {alert['message']}")
            if status["state"] != last_state:
                last_state = status["state"]  # simpan
                print(f"[STATUS] {status['state']} | skor={status['focus_score']} | {status['reason']}")
    except KeyboardInterrupt:
        print("\n[DETECTOR] berhenti." + (f" {logger.count} baris data ditulis ke {DATA_FILE}." if logger else ""))
    finally:
        client.loop_stop()  # hentikan loop MQTT
        client.disconnect()  # putus koneksi


if __name__ == "__main__":
    main()
