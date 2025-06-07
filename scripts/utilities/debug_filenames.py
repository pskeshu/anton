#!/usr/bin/env python3
"""Debug filename parsing for BBBC021."""

import re
from pathlib import Path

def debug_filenames():
    """Debug filename parsing."""
    
    base_dir = Path("data/bbbc021/images")
    
    print("🔍 Debugging filename structure...")
    
    sample_files = []
    
    for week_dir in base_dir.iterdir():
        if not week_dir.is_dir() or week_dir.name.endswith('.zip'):
            continue
            
        print(f"\n📂 Week directory: {week_dir.name}")
        
        # Look for nested directory structure
        nested_dirs = list(week_dir.glob("Week*"))
        if nested_dirs:
            actual_dir = nested_dirs[0]
            print(f"   Nested dir: {actual_dir.name}")
        else:
            actual_dir = week_dir
            print(f"   Direct dir: {actual_dir.name}")
        
        # Get first few files
        tif_files = list(actual_dir.glob("*.tif"))[:3]
        print(f"   TIF files found: {len(list(actual_dir.glob('*.tif')))}")
        
        for tif_file in tif_files:
            print(f"   Sample file: {tif_file.name}")
            
            # Parse filename
            parts = tif_file.stem.split('_')
            print(f"      Parts: {parts}")
            
            if len(parts) >= 4:
                week = parts[0]
                date = parts[1] 
                well = parts[2]
                site_info = parts[3]
                
                print(f"      Week: {week}, Date: {date}, Well: {well}, Site info: {site_info}")
                
                # Extract site number and channel
                site_match = re.search(r's(\d+)', site_info)
                channel_match = re.search(r'w(\d)', site_info)
                
                if site_match and channel_match:
                    site = site_match.group(1)
                    channel = channel_match.group(1)
                    print(f"      Site: s{site}, Channel: w{channel}")
                    
                    key = f"{week}_{date}_{well}_s{site}"
                    print(f"      Key: {key}")
                else:
                    print(f"      ❌ Could not parse site/channel from: {site_info}")
            
            sample_files.append(tif_file.name)
        
        if len(sample_files) >= 10:
            break
    
    print(f"\n📊 Sample files analyzed: {len(sample_files)}")

if __name__ == "__main__":
    debug_filenames()