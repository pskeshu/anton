#!/usr/bin/env python3
"""
Extract All BBBC021 ZIP Files

Extract all downloaded ZIP files and organize the images.
"""

import zipfile
import os
from pathlib import Path
import time

def extract_all_zips():
    """Extract all ZIP files in the images directory."""
    
    print("🗂️  BBBC021 ZIP File Extraction")
    print("=" * 40)
    
    images_dir = Path("data/bbbc021/images")
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    # Find all ZIP files
    zip_files = list(images_dir.glob("*.zip"))
    
    if not zip_files:
        print("❌ No ZIP files found")
        return
    
    print(f"📦 Found {len(zip_files)} ZIP files")
    
    total_extracted = 0
    total_size = 0
    
    for i, zip_path in enumerate(zip_files):
        zip_name = zip_path.name
        extract_dir = images_dir / zip_path.stem
        
        print(f"\n📂 Extracting {i+1}/{len(zip_files)}: {zip_name}")
        print(f"   Size: {zip_path.stat().st_size // 1024 // 1024}MB")
        
        # Check if already extracted
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"   ✅ Already extracted to {extract_dir}")
            # Count existing files
            existing_files = list(extract_dir.rglob("*.tif")) + list(extract_dir.rglob("*.png"))
            total_extracted += len(existing_files)
            continue
        
        # Create extraction directory
        extract_dir.mkdir(exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files to extract
                file_list = zip_ref.namelist()
                image_files = [f for f in file_list if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.bmp'))]
                
                print(f"   🔄 Extracting {len(image_files)} images...")
                
                extracted_count = 0
                for file_name in image_files:
                    try:
                        # Extract to the directory
                        zip_ref.extract(file_name, extract_dir)
                        extracted_count += 1
                        total_extracted += 1
                        
                        # Progress indicator every 100 files
                        if extracted_count % 100 == 0:
                            print(f"      Progress: {extracted_count}/{len(image_files)} files")
                            
                    except Exception as e:
                        print(f"      ⚠️  Failed to extract {file_name}: {e}")
                
                print(f"   ✅ Extracted {extracted_count} images to {extract_dir}")
                
                # Calculate total size
                dir_size = sum(f.stat().st_size for f in extract_dir.rglob("*") if f.is_file())
                total_size += dir_size
                print(f"   📊 Directory size: {dir_size // 1024 // 1024}MB")
                
        except Exception as e:
            print(f"   ❌ Failed to extract {zip_name}: {e}")
        
        # Brief pause between extractions
        time.sleep(1)
    
    print(f"\n🎉 Extraction Complete!")
    print(f"📊 Total images extracted: {total_extracted}")
    print(f"📊 Total extracted size: {total_size // 1024 // 1024}MB")
    
    # Show directory structure
    print(f"\n📁 Extracted Directories:")
    for extract_dir in sorted(images_dir.iterdir()):
        if extract_dir.is_dir() and not extract_dir.name.endswith('.zip'):
            image_count = len(list(extract_dir.rglob("*.tif"))) + len(list(extract_dir.rglob("*.png")))
            if image_count > 0:
                print(f"   {extract_dir.name}: {image_count} images")

if __name__ == "__main__":
    extract_all_zips()