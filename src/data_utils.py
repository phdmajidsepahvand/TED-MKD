!pip install wfdb

import os
import os
import wfdb
import zipfile
from src.datatsets import BaselineDataset,CrossWindowECGDataset

def download_data(records_url,records_local,download_dir):
    
    os.system(f"wget -q -O {records_30_local} {records_30_url}")

    with open(records_30_local, "r") as f:
        patient_dirs = sorted(set(line.strip().split("/")[0] for line in f if line.strip()))

    print(f" Total patient folders found under /30/: {len(patient_dirs)}")

    N = 100
    selected_dirs = patient_dirs[:N]
    print(f" Checking first {N} patient folders...")

    total_records = 0
    all_records_by_folder = {}

    for pid in selected_dirs:
        url = f"https://physionet.org/files/mimic3wdb/1.0/30/{pid}/RECORDS"
        local_filename = f"RECORDS_{pid}.txt"

        result = os.system(f"wget -q -O {local_filename} {url}")

        if os.path.exists(local_filename):
            with open(local_filename, "r") as f:
                records = [line.strip() for line in f if line.strip()]
            all_records_by_folder[pid] = records
            total_records += len(records)
        else:
            print(f" RECORDS file missing for {pid}")

    print(f"\n Total number of records found across {len(selected_dirs)} folders: {total_records}")

    usable_records = []
    
    zip_filename = "usable_records.zip"
    os.makedirs(download_dir, exist_ok=True)

    total_checked = 0
    total_usable = 0

    for pid, records in all_records_by_folder.items():
        for r in records:
            total_checked += 1
            hea_url = f"https://physionet.org/files/mimic3wdb/1.0/30/{pid}/{r}.hea"
            dat_url = f"https://physionet.org/files/mimic3wdb/1.0/30/{pid}/{r}.dat"

            hea_file = os.path.join(download_dir, f"{r}.hea")
            dat_file = os.path.join(download_dir, f"{r}.dat")

            os.system(f"wget -q -O {hea_file} {hea_url}")
            os.system(f"wget -q -O {dat_file} {dat_url}")

            try:
                rec = wfdb.rdrecord(os.path.join(download_dir, r))
                channels = rec.sig_name
                has_ecg = any("ECG" in ch or "II" in ch for ch in channels)
                has_abp = any("ABP" in ch for ch in channels)

                if has_ecg and has_abp:
                    usable_records.append((pid, r))
                    total_usable += 1
                    print(f"✅ {r} - valid")
                else:
                    os.remove(hea_file)
                    os.remove(dat_file)
            except:
                if os.path.exists(hea_file): os.remove(hea_file)
                if os.path.exists(dat_file): os.remove(dat_file)

    print(f"\n🔍 Checked {total_checked} records.")
    print(f" {total_usable} usable records downloaded to: {download_dir}")

    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in os.listdir(download_dir):
            zipf.write(os.path.join(download_dir, file), arcname=file)

    print(f"\n All usable files zipped into: {zip_filename}")
    
def load_Data(usable_dir):
    window_size = 3000
    stride = 1000

    data_windows = []
    labels = []
    record_names = []
    
    files = sorted([f[:-4] for f in os.listdir(usable_dir) if f.endswith(".hea")])

    for rec_name in files:
        try:
            record = wfdb.rdrecord(os.path.join(usable_dir, rec_name))
            signals = record.p_signal
            channels = record.sig_name

            ecg_idx = next(i for i, ch in enumerate(channels) if "ECG" in ch or "II" in ch)
            abp_idx = next(i for i, ch in enumerate(channels) if "ABP" in ch)

            ecg = signals[:, ecg_idx]
            abp = signals[:, abp_idx]

            for start in range(0, len(ecg) - window_size + 1, stride):
                ecg_win = ecg[start:start+window_size]
                abp_win = abp[start:start+window_size]

                sbp = np.max(abp_win)
                dbp = np.min(abp_win)

                label = 1 if sbp > 140 or dbp > 90 else 0

                data_windows.append(ecg_win)
                labels.append(label)
                record_names.append(f"{rec_name}_{start}")

            print(f"✅ {rec_name} processed")

        except Exception as e:
            print(f"❌ {rec_name} skipped: {e}")

    X = np.array(data_windows)
    y = np.array(labels)

    print(f"\n🧪 Total samples: {X.shape[0]}")
    print(f"🔖 Label distribution: 0 → {(y == 0).sum()} | 1 → {(y == 1).sum()}")

    X = np.where(np.isnan(X), np.mean(X[~np.isnan(X)]), X)
    return X,y

def create_Dataloader(X, y, batch_size):
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = CrossWindowECGDataset(X_train_raw, y_train_raw, windows_per_sample=3)
    test_dataset  = CrossWindowECGDataset(X_test_raw,  y_test_raw,  windows_per_sample=3)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size)

    pos_weight = torch.tensor([len(y_train_raw) / sum(y_train_raw)], dtype=torch.float32).to(device)

    train_dataset = CrossWindowECGDataset(X_train_raw, y_train_raw)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    return train_loader, test_loader