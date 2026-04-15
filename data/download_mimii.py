#!/usr/bin/env python3
"""
download_mimii.py — Download and extract the MIMII dataset (6 dB SNR, all 4 machine types).

Downloads from Zenodo record 3384388. Each machine type is a separate zip (~6–10 GB).
Total download: ~30 GB.

Usage:
    python data/download_mimii.py

Requirements:
    unzip must be installed: sudo apt install unzip
"""
import argparse 
import hashlib
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MIMII_ROOT, MACHINE_TYPES, MACHINE_IDS


# ── SNR configuration ────────────────────────────────────────────────────────         
# All three SNR variants are hosted in the same Zenodo record.
# Files are named {snr_prefix}_{machine_type}.zip where snr_prefix                                         
# is "6_dB", "0_dB", or "-6_dB".                                                                           
MIMII_ZENODO_RECORD = "3384388"                                                                            
                                                                                                            
VALID_SNRS = ("6dB", "0dB", "-6dB")                                                                        
                                                                                                            

def build_files_for_snr(snr: str) -> list[dict]:                                                           
    """Build the per-machine-type download manifest for one SNR."""                                        
    if snr not in VALID_SNRS:                                                                              
        raise ValueError(f"snr must be one of {VALID_SNRS}, got {snr!r}")                                  
                                                                                                            
                                                
    snr_prefix = snr.replace("dB", "_dB")                                                                  
                                                                                                            
    sizes = {                                                                                            
        "fan":    "~9.5 GB",                                                                               
        "pump":   "~7.1 GB",                                                                               
        "slider": "~6.6 GB",
        "valve":  "~6.4 GB",                                                                               
    }                                                                                                      

    files = []                                                                                             
    for mtype in ("fan", "pump", "slider", "valve"):
        name = f"{snr_prefix}_{mtype}.zip"                                                                 
        files.append({
            "name":         name,                                                                          
            "url":          f"https://zenodo.org/api/records/{MIMII_ZENODO_RECORD}/files/{name}/content",
            "md5":          None,                                                                          
            "size":         sizes[mtype],                                                                  
            "machine_type": mtype,                                                                         
        })                                                                                                 
    return files
                         

# ── helpers ─────────────────────────────────────────────────────────────────

def check_unzip():
    result = subprocess.run(["which", "unzip"], capture_output=True)
    if result.returncode != 0:
        print("ERROR: unzip not found. Install it with: sudo apt install unzip")
        sys.exit(1)


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str):
    """Download with wget — resumes partial downloads automatically."""
    print(f"  Downloading to {dest} …")
    subprocess.run(
        ["wget", "--continue", "--progress=bar:force", "-O", dest, url],
        check=True,
    )


def verify(path: str, expected_md5: str | None) -> bool:
    if expected_md5 is None:
        return True   # skip verification if checksum not provided
    print(f"  Verifying {os.path.basename(path)} …", end=" ", flush=True)
    actual = md5_file(path)
    if actual == expected_md5:
        print("OK")
        return True
    print(f"FAILED\n  Expected: {expected_md5}\n  Got:      {actual}")
    return False


def extract_and_flatten(zip_path: str, machine_type: str, output_dir: str):                                                                              
    """         
    Extract zip to a temp dir then move files into the expected layout:
    <output_dir>/{machine_type}/{machine_id}/{normal,abnormal}/*.wav                                
                                                                                                    
    The MIMII zips contain an internal top-level folder (e.g. dev_data_fan_6dB/)                      
    which is stripped during extraction.                                                              
    """                                                                                               
    import shutil                                                                                     
    import tempfile                                                                                   
                
    tmp = tempfile.mkdtemp(dir=output_dir)                                          
    try:                                                                                              
        print(f"  Extracting {os.path.basename(zip_path)} …")
        subprocess.run(["unzip", "-q", zip_path, "-d", tmp], check=True)                              

        top_dirs = [d for d in os.listdir(tmp) if os.path.isdir(os.path.join(tmp, d))]                
        if len(top_dirs) != 1:
            raise RuntimeError(f"Expected one top-level dir in zip, found: {top_dirs}")               
        src_root = os.path.join(tmp, top_dirs[0])
                                                                                                    
        for mid in os.listdir(src_root):
            src = os.path.join(src_root, mid)                                                         
            if not os.path.isdir(src):                                                                
                continue
            dst = os.path.join(output_dir, machine_type, mid)                                                                                                            
            if os.path.exists(dst):
                print(f"    {machine_type}/{mid} already exists — skipping move.")                    
                continue                                                                              
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)                                                                     
            n_wav = sum(1 for _, _, fs in os.walk(dst) for f in fs if f.endswith(".wav"))
            print(f"    → {machine_type}/{mid}  ({n_wav} WAVs)")                                      
    finally:                                                                                          
        shutil.rmtree(tmp, ignore_errors=True)  


def migrate_legacy_flat_layout(mimii_root: Path) -> None:                            
    """One-shot: move flat data/mimii/<machine_type>/ into data/mimii/6dB/.                           
                                                                                                    
    The old downloader wrote directly under data/mimii/fan/... . The new
    layout nests every SNR under its own subdir. If we detect the legacy                              
    flat layout (any machine_type dir sitting directly under mimii_root),                             
    move it into data/mimii/6dB/ so the existing data is preserved.                                   
                                                                                                    
    Idempotent: does nothing if there is no legacy data to migrate.                                   
    """                                                                                               
    import shutil
                                                                                                    
    legacy_dirs = [                                                                                   
        mimii_root / mt
        for mt in MACHINE_TYPES                                                                       
        if (mimii_root / mt).is_dir()
    ]
    if not legacy_dirs:
        return  # nothing flat to migrate                                                             

    target_root = mimii_root / "6dB"                                                                  
    already_nested = target_root.is_dir() and any(
        (target_root / mt).is_dir() for mt in MACHINE_TYPES                                           
    )
    if already_nested:                                                                                
        print(  
            f"  Migration skipped: {target_root} already has nested "                                 
            f"machine dirs but flat dirs also exist — please resolve by hand."                        
        )                                                                                             
        return                                                                                        
                                                                                                    
    print(f"  Migrating legacy flat layout → {target_root}")                                          
    target_root.mkdir(parents=True, exist_ok=True)
    for d in legacy_dirs:                                                                             
        dst = target_root / d.name
        print(f"    mv {d}  →  {dst}")                                                                
        shutil.move(str(d), str(dst))

# ── main ────────────────────────────────────────────────────────────────────
                
def main():                                                                         
    parser = argparse.ArgumentParser(                                                       
        description="Download and extract MIMII for one SNR.",                    
    )                                                                             
    parser.add_argument(                                                          
        "--snr", default="6dB", choices=VALID_SNRS,                                            
        help="Which MIMII SNR variant to download (default: 6dB).",                            
    )                                                                             
    args = parser.parse_args()                                                    
    snr = args.snr                                                                
                
    check_unzip()                                                                                     
                
    # Step 0: migrate any legacy flat 6dB layout before touching new SNR dirs    
    os.makedirs(str(MIMII_ROOT), exist_ok=True)                                   
    migrate_legacy_flat_layout(MIMII_ROOT)                                        
                                                                                                    
    output_dir = str(MIMII_ROOT / snr)                                            
                                                                                   
    os.makedirs(output_dir, exist_ok=True)                                        
                                                                                                    
    files = build_files_for_snr(snr)                                              
                                                                                                    
    print(f"\n{'─' * 60}")
    print(f"  MIMII dataset — {snr}, 4 machine types")                            
                                                                                         
    print(f"  Destination: {output_dir}/")
    print(f"{'─' * 60}\n")                                                                            
                
    for f in files:                                                               
 
        mtype = f["machine_type"]
        dest  = os.path.join(output_dir, f["name"])                               
                                                                                      
                                                                                                    
        # Per-SNR skip: is this machine type already fully extracted for THIS snr?                    
        extracted_ids = [
            mid for mid in MACHINE_IDS                                                                
            if os.path.isdir(os.path.join(output_dir, mtype, mid))              
                                                                                           
        ]                                                                                             
        if len(extracted_ids) == len(MACHINE_IDS):                                                    
            print(f"  {mtype}: all {len(MACHINE_IDS)} machine IDs already present — skipping.")
            continue                                                                                  

        print(f"\n  {f['name']} ({f['size']})")                                                       
                
        if os.path.exists(dest):                                                                      
            print(f"  Zip already present — skipping download.")
        else:
            download(f["url"], dest)                                                                  

        if not verify(dest, f["md5"]):                                                                
            print(f"  ERROR: {f['name']} checksum mismatch. Delete and retry.")
            continue                                                                                  

        extract_and_flatten(dest, mtype, output_dir)                 
                                                                                                    
        os.remove(dest)
        print(f"  Removed {f['name']} (extraction complete)")
                                                                                                    
    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")                                                                            
    print(f"  Summary: {output_dir}/")                                                                
    total_wavs = 0                                                                                    
    for mtype in MACHINE_TYPES:                                                                       
        for mid in MACHINE_IDS:
            path = os.path.join(output_dir, mtype, mid)   
            if os.path.isdir(path):                                                                   
                n = sum(1 for _, _, fs in os.walk(path) for f in fs if f.endswith(".wav"))            
                total_wavs += n
                print(f"    {mtype}/{mid}: {n} WAVs")                                                 
            else:                                                                                     
                print(f"    {mtype}/{mid}: MISSING")
    print(f"  Total: {total_wavs} WAV files")                                                         
    print(f"{'─' * 60}\n")    


if __name__ == "__main__":
    main()
