import pandas as pd
import os
import json
from pytorch_tabnet.tab_model import TabNetClassifier
import psutil
import torch

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
PARAMS_PATH = os.path.join(MODEL_DIR, "model_params.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "network.pt")

def load_model():
    with open(PARAMS_PATH, "r") as f:
        params = json.load(f)
    model = TabNetClassifier(**params)
    model.load_model(WEIGHTS_PATH)
    return model

def extract_features_from_process(proc):
    try:
        mem_info = proc.memory_info()
        io = proc.io_counters()
        return {
            "VirtualSize": mem_info.vms,
            "PrivateBytes": mem_info.rss,
            "Handles": proc.num_fds() if hasattr(proc, "num_fds") else 0,
            "Threads": proc.num_threads(),
            "ReadOperationCount": io.read_count,
            "WriteOperationCount": io.write_count,
            "ReadTransferCount": io.read_bytes,
            "WriteTransferCount": io.write_bytes,
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None

def run_ram_analysis():
    model = load_model()

    processes = []
    features = []

    for proc in psutil.process_iter(['pid', 'name']):
        feats = extract_features_from_process(proc)
        if feats:
            processes.append((proc.pid, proc.info['name']))
            features.append(feats)

    if not features:
        print("No accessible processes found.")
        return

    df = pd.DataFrame(features)
    preds = model.predict(df.values)

    print(f"{'PID':<7}{'Process':<30}{'Verdict'}")
    print("-" * 50)
    for (pid, name), pred in zip(processes, preds):
        label = "⚠️ MALWARE" if pred == 1 else "✅ BENIGN"
        print(f"{pid:<7}{name:<30}{label}")
