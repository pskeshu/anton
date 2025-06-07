#!/usr/bin/env python3
"""
Create Multi-Channel Composites for BBBC021 Analysis

Combine DAPI, Tubulin, and Actin channels into RGB composites for VLM analysis.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("❌ PIL not available - cannot create composites")

class BBBC021CompositeCreator:
    """Create multi-channel composites from BBBC021 images."""
    
    def __init__(self):
        self.base_dir = Path("data/bbbc021/images")
        self.composites_dir = Path("data/bbbc021/composites")
        self.composites_dir.mkdir(exist_ok=True)
        
        # Load metadata
        self.image_metadata = self._load_image_metadata()
        self.moa_metadata = self._load_moa_metadata()
        
    def _load_image_metadata(self) -> Optional[pd.DataFrame]:
        """Load image metadata."""
        metadata_path = "data/bbbc021/image_metadata.csv"
        if Path(metadata_path).exists():
            return pd.read_csv(metadata_path)
        return None
    
    def _load_moa_metadata(self) -> Optional[pd.DataFrame]:
        """Load MOA metadata."""
        metadata_path = "data/bbbc021/moa_metadata.csv"
        if Path(metadata_path).exists():
            return pd.read_csv(metadata_path)
        return None
    
    def find_channel_triplets(self, max_samples: int = 50) -> List[Dict]:
        """Find sets of 3 channels (DAPI, Tubulin, Actin) for the same well/site."""
        
        print(f"🔍 Finding channel triplets (max {max_samples} samples)...")
        
        # Parse filenames to group by well/site
        triplets = {}
        
        for week_dir in self.base_dir.iterdir():
            if not week_dir.is_dir() or week_dir.name.endswith('.zip'):
                continue
                
            print(f"   📂 Scanning {week_dir.name}...")
            
            # Look for nested directory structure
            nested_dirs = list(week_dir.glob("Week*"))
            if nested_dirs:
                actual_dir = nested_dirs[0]  # Use first nested directory
            else:
                actual_dir = week_dir
            
            for image_file in actual_dir.glob("*.tif"):
                # Parse filename: Week1_150607_B02_s2_w23ECDA1C9...
                parts = image_file.stem.split('_')
                if len(parts) >= 5:
                    week = parts[0]
                    date = parts[1] 
                    well = parts[2]
                    site_info = parts[3]  # s2, s1, etc.
                    channel_info = parts[4]  # w23ECDA1C9..., w14CB040E2..., etc.
                    
                    # Extract site number
                    site_match = re.search(r's(\d+)', site_info)
                    # Extract channel number from beginning of channel_info
                    channel_match = re.search(r'^w(\d)', channel_info)
                    
                    if site_match and channel_match:
                        site = site_match.group(1)
                        channel = channel_match.group(1)
                        
                        # Create unique key for this well/site
                        key = f"{week}_{date}_{well}_s{site}"
                        
                        if key not in triplets:
                            triplets[key] = {}
                        
                        triplets[key][f"w{channel}"] = str(image_file)
        
        # Find complete triplets (w1, w2, w4)
        complete_triplets = []
        for key, channels in triplets.items():
            if 'w1' in channels and 'w2' in channels and 'w4' in channels:
                complete_triplets.append({
                    'id': key,
                    'dapi_path': channels['w1'],      # w1 = DAPI
                    'tubulin_path': channels['w2'],   # w2 = Tubulin  
                    'actin_path': channels['w4'],     # w4 = Actin
                    'well': key.split('_')[2],
                    'site': key.split('_')[3]
                })
        
        print(f"   ✅ Found {len(complete_triplets)} complete channel triplets")
        
        # Return sample for testing
        return complete_triplets[:max_samples]
    
    def create_composite(self, triplet: Dict) -> Optional[str]:
        """Create RGB composite from 3-channel triplet."""
        
        if not PIL_AVAILABLE:
            return None
            
        try:
            # Load the three channels
            dapi_img = Image.open(triplet['dapi_path'])
            tubulin_img = Image.open(triplet['tubulin_path'])
            actin_img = Image.open(triplet['actin_path'])
            
            # Convert to numpy arrays
            dapi = np.array(dapi_img).astype(float)
            tubulin = np.array(tubulin_img).astype(float)
            actin = np.array(actin_img).astype(float)
            
            # Normalize each channel to 0-255
            def normalize_channel(img):
                img_norm = (img - img.min()) / (img.max() - img.min()) * 255
                return img_norm.astype(np.uint8)
            
            dapi_norm = normalize_channel(dapi)
            tubulin_norm = normalize_channel(tubulin)
            actin_norm = normalize_channel(actin)
            
            # Create RGB composite
            height, width = dapi_norm.shape
            composite = np.zeros((height, width, 3), dtype=np.uint8)
            
            composite[:, :, 0] = actin_norm      # Red channel = Actin
            composite[:, :, 1] = tubulin_norm    # Green channel = Tubulin
            composite[:, :, 2] = dapi_norm       # Blue channel = DAPI
            
            # Save composite
            composite_filename = f"{triplet['id']}_composite.png"
            composite_path = self.composites_dir / composite_filename
            
            composite_img = Image.fromarray(composite, 'RGB')
            composite_img.save(composite_path)
            
            return str(composite_path)
            
        except Exception as e:
            print(f"      ❌ Failed to create composite for {triplet['id']}: {e}")
            return None
    
    def get_compound_info(self, well: str) -> Dict:
        """Get compound information for a well."""
        
        # Try to match with metadata
        if self.image_metadata is not None:
            # Find matching row in image metadata
            matching_rows = self.image_metadata[
                self.image_metadata['Image_Metadata_Well_DAPI'] == well
            ]
            
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                compound = row.get('Image_Metadata_Compound', 'unknown')
                concentration = row.get('Image_Metadata_Concentration', 0)
                
                # Get MOA from MOA metadata
                if self.moa_metadata is not None:
                    moa_rows = self.moa_metadata[
                        self.moa_metadata['compound'] == compound
                    ]
                    
                    if not moa_rows.empty:
                        moa = moa_rows.iloc[0]['moa']
                    else:
                        moa = 'unknown'
                else:
                    moa = 'unknown'
                
                return {
                    'compound': compound,
                    'moa': moa,
                    'concentration': concentration
                }
        
        # Default compound info
        return {
            'compound': 'unknown',
            'moa': 'unknown',
            'concentration': 0
        }
    
    def create_analysis_dataset(self, max_composites: int = 20) -> pd.DataFrame:
        """Create a dataset of composites ready for analysis."""
        
        print(f"🧬 Creating Analysis Dataset ({max_composites} composites)")
        print("=" * 50)
        
        # Find channel triplets
        triplets = self.find_channel_triplets(max_samples=max_composites)
        
        if not triplets:
            print("❌ No channel triplets found")
            return pd.DataFrame()
        
        # Create composites
        analysis_data = []
        successful_composites = 0
        
        for i, triplet in enumerate(triplets):
            print(f"\n🔬 Creating composite {i+1}/{len(triplets)}: {triplet['id']}")
            
            # Create composite image
            composite_path = self.create_composite(triplet)
            
            if composite_path:
                # Get compound information
                compound_info = self.get_compound_info(triplet['well'])
                
                analysis_data.append({
                    'composite_id': triplet['id'],
                    'composite_path': composite_path,
                    'well': triplet['well'],
                    'site': triplet['site'],
                    'compound': compound_info['compound'],
                    'moa': compound_info['moa'],
                    'concentration': compound_info['concentration'],
                    'dapi_path': triplet['dapi_path'],
                    'tubulin_path': triplet['tubulin_path'],
                    'actin_path': triplet['actin_path']
                })
                
                successful_composites += 1
                print(f"      ✅ Composite created: {compound_info['compound']} ({compound_info['moa']})")
            else:
                print(f"      ❌ Failed to create composite")
        
        # Create DataFrame
        df = pd.DataFrame(analysis_data)
        
        if not df.empty:
            # Save dataset
            dataset_path = "data/bbbc021/multichannel_analysis_dataset.csv"
            df.to_csv(dataset_path, index=False)
            
            print(f"\n🎉 Analysis Dataset Created!")
            print(f"   ✅ {successful_composites} composites created")
            print(f"   📊 Compounds: {df['compound'].nunique()}")
            print(f"   📊 MOAs: {df['moa'].nunique()}")
            print(f"   💾 Dataset saved: {dataset_path}")
            
            # Show compound distribution
            compound_counts = df['compound'].value_counts()
            print(f"\n📈 Compound Distribution:")
            for compound, count in compound_counts.items():
                print(f"   {compound}: {count} images")
        
        return df

def main():
    """Create multi-channel composites for analysis."""
    
    creator = BBBC021CompositeCreator()
    
    # Create analysis dataset
    dataset = creator.create_analysis_dataset(max_composites=20)
    
    if not dataset.empty:
        print(f"\n🚀 Ready for connection discovery analysis!")
        print(f"   Use: python run_multichannel_discovery.py")
    else:
        print("❌ No dataset created")

if __name__ == "__main__":
    main()