#!/usr/bin/env python3
"""
Simple data sync tool for uploading data to remote server
Supports: compression, upload, and remote extraction
"""

import os
import zipfile
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

try:
    import paramiko
    from scp import SCPClient
    from tqdm import tqdm
except ImportError:
    print("Please install: pip install paramiko scp tqdm")
    exit(1)

# ========== Configuration ==========
REMOTE_HOST = "ctrl.zyh111.icu"
REMOTE_PORT = 233
REMOTE_USER = "zyh"
REMOTE_PASSWORD = "1"
REMOTE_PATH = "/home/zyh/Fruit-Classifier"

LOCAL_DATA = Path("data/train_augment")
ZIP_NAME = "data_upload.zip"
CHUNK_SIZE = 10 * 1024 * 1024  # 10MB per chunk for parallel upload
MAX_WORKERS = 4  # Number of parallel upload threads

# ========== Functions ==========

def compress_data():
    """Compress local data directory with progress bar"""
    print("[1/4] Compressing data...")
    
    # Count total files first
    files = [f for f in LOCAL_DATA.rglob("*") if f.is_file()]
    total = len(files)
    
    with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        with tqdm(total=total, desc="  Compressing", unit="file") as pbar:
            for file in files:
                zipf.write(file, file.relative_to(LOCAL_DATA.parent))
                pbar.update(1)
    
    size = Path(ZIP_NAME).stat().st_size / 1024 / 1024
    print(f"  Compressed {total} files, {size:.2f} MB")
    return size


def split_file(filename, chunk_size):
    """将文件分割成多个块"""
    file_size = os.path.getsize(filename)
    num_chunks = math.ceil(file_size / chunk_size)
    
    chunks = []
    with open(filename, 'rb') as f:
        for i in range(num_chunks):
            chunk_name = f"{filename}.part{i:03d}"
            chunk_data = f.read(chunk_size)
            with open(chunk_name, 'wb') as chunk_file:
                chunk_file.write(chunk_data)
            chunks.append(chunk_name)
    
    return chunks


def upload_chunk(chunk_path, chunk_index, total_chunks):
    """上传单个文件块"""
    try:
        # 为每个线程创建独立的SSH连接
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(REMOTE_HOST, REMOTE_PORT, REMOTE_USER, REMOTE_PASSWORD)
        
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(chunk_path, f"{REMOTE_PATH}/{os.path.basename(chunk_path)}")
        
        ssh.close()
        return True, None
    except Exception as e:
        return False, str(e)


def upload_file(ssh):
    """Multi-threaded upload with progress bar"""
    print("[2/4] Uploading to remote server...")
    
    file_size = Path(ZIP_NAME).stat().st_size
    size_mb = file_size / 1024 / 1024
    
    # 如果文件小于30MB,使用单线程上传
    if size_mb < 30:
        print(f"  File size: {size_mb:.1f} MB (single-threaded upload)")
        
        with tqdm(total=file_size, desc="  Uploading", unit="B", unit_scale=True) as pbar:
            def progress(filename, size, sent):
                pbar.update(sent - pbar.n)
            
            with SCPClient(ssh.get_transport(), progress=progress) as scp:
                scp.put(ZIP_NAME, f"{REMOTE_PATH}/{ZIP_NAME}")
        
        print(f"  Uploaded to {REMOTE_HOST}")
        return
    
    # 大文件使用多线程上传
    print(f"  File size: {size_mb:.1f} MB (multi-threaded upload)")
    print(f"  Splitting into {CHUNK_SIZE/1024/1024:.0f}MB chunks...")
    
    chunks = split_file(ZIP_NAME, CHUNK_SIZE)
    total_chunks = len(chunks)
    print(f"  Split into {total_chunks} chunks, using {MAX_WORKERS} threads")
    
    # 并行上传所有块
    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {
            executor.submit(upload_chunk, chunk, i, total_chunks): (chunk, i)
            for i, chunk in enumerate(chunks)
        }
        
        with tqdm(total=total_chunks, desc="  Uploading", unit="chunk") as pbar:
            for future in as_completed(future_to_chunk):
                chunk, idx = future_to_chunk[future]
                success, error = future.result()
                
                if not success:
                    raise Exception(f"Chunk {idx} upload failed: {error}")
                
                success_count += 1
                pbar.update(1)
    
    # 清理本地块文件
    for chunk in chunks:
        os.remove(chunk)
    
    # 在远程服务器上合并文件
    print("  Merging chunks on remote server...")
    merge_cmd = f"cat {REMOTE_PATH}/{ZIP_NAME}.part* > {REMOTE_PATH}/{ZIP_NAME} && rm -f {REMOTE_PATH}/{ZIP_NAME}.part*"
    stdin, stdout, stderr = ssh.exec_command(merge_cmd)
    exit_status = stdout.channel.recv_exit_status()
    
    if exit_status != 0:
        raise Exception(f"Failed to merge chunks on remote server")
    
    print(f"  Uploaded {total_chunks} chunks successfully")


def extract_remote(ssh):
    """Extract zip file on remote server with progress"""
    print("[3/4] Extracting on remote server...")
    
    # Create directory if not exists
    ssh.exec_command(f'mkdir -p {REMOTE_PATH}')
    
    # Extract and remove zip
    cmd = f'cd {REMOTE_PATH} && unzip -o {ZIP_NAME} && rm {ZIP_NAME}'
    stdin, stdout, stderr = ssh.exec_command(cmd)
    
    # Show extraction progress
    with tqdm(desc="  Extracting", bar_format='{desc}: {elapsed}') as pbar:
        stdout.channel.recv_exit_status()  # Wait for completion
        pbar.update(1)
    
    print("  Extracted successfully")


def cleanup():
    """Remove local zip file"""
    print("[4/4] Cleaning up...")
    
    if Path(ZIP_NAME).exists():
        Path(ZIP_NAME).unlink()
        print("  Cleanup complete")
    else:
        print("  Nothing to clean")


def main():
    """Main sync process"""
    print("=" * 60)
    print("Data Sync Tool")
    print("=" * 60)
    print(f"Target: {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_PORT}")
    print(f"Local:  {LOCAL_DATA}")
    print(f"Remote: {REMOTE_PATH}/data\n")
    
    # Check local data exists
    if not LOCAL_DATA.exists():
        print(f"Error: {LOCAL_DATA} not found")
        return
    
    try:
        # Step 1: Compress
        compress_data()
        
        # Step 2-3: Connect and upload
        print("\nConnecting to server...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(REMOTE_HOST, REMOTE_PORT, REMOTE_USER, REMOTE_PASSWORD)
        print("  Connected\n")
        
        upload_file(ssh)
        extract_remote(ssh)
        
        ssh.close()
        
        # Step 4: Cleanup
        cleanup()
        
        print("\n" + "=" * 60)
        print("Sync completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        # Cleanup on error
        if Path(ZIP_NAME).exists():
            Path(ZIP_NAME).unlink()


if __name__ == "__main__":
    main()
