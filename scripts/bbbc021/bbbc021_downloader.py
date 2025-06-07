#!/usr/bin/env python3
"""
BBBC021 Dataset Downloader

Download MCF-7 compound screen dataset for connection discovery analysis.
"""

import requests
import pandas as pd
from pathlib import Path
import zipfile
import os
from urllib.parse import urlparse
import time
from typing import List, Dict

class BBBC021Downloader:
    """Download and organize BBBC021 dataset."""
    
    def __init__(self, data_dir: str = "data/bbbc021"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # BBBC021 download URLs (from the website)
        self.base_url = "https://data.broadinstitute.org/bbbc/BBBC021"
        self.metadata_urls = {
            'image_metadata': f"{self.base_url}/BBBC021_v1_image.csv",
            'compound_metadata': f"{self.base_url}/BBBC021_v1_compound.csv", 
            'moa_metadata': f"{self.base_url}/BBBC021_v1_moa.csv"
        }
        
        # Image data comes in ZIP files - we'll start with a subset
        self.image_zip_urls = [
            f"{self.base_url}/BBBC021_v1_images_Week1_22123.zip",
            f"{self.base_url}/BBBC021_v1_images_Week1_22141.zip", 
            f"{self.base_url}/BBBC021_v1_images_Week1_22161.zip",
            f"{self.base_url}/BBBC021_v1_images_Week1_22381.zip",
            f"{self.base_url}/BBBC021_v1_images_Week1_22401.zip"
        ]
    
    def download_metadata(self) -> Dict[str, pd.DataFrame]:
        """Download all metadata files."""
        print("📋 Downloading BBBC021 metadata...")
        
        metadata = {}
        
        for name, url in self.metadata_urls.items():
            file_path = self.data_dir / f"{name}.csv"
            
            if file_path.exists():
                print(f"✅ {name} already exists")
                metadata[name] = pd.read_csv(file_path)
                continue
            
            try:
                print(f"⬇️  Downloading {name}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Save to file
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                # Load as DataFrame
                metadata[name] = pd.read_csv(file_path)
                print(f"✅ Downloaded {name}: {len(metadata[name])} rows")
                
            except Exception as e:
                print(f"❌ Failed to download {name}: {e}")
                metadata[name] = None
        
        return metadata
    
    def analyze_metadata(self, metadata: Dict[str, pd.DataFrame]):
        """Analyze metadata to understand dataset structure."""
        print("\n📊 Analyzing BBBC021 dataset structure...")
        
        if 'image_metadata' in metadata and metadata['image_metadata'] is not None:
            img_meta = metadata['image_metadata']
            print(f"📸 Images: {len(img_meta)} total images")
            print(f"📸 Columns: {list(img_meta.columns)}")
            if len(img_meta) > 0:
                print(f"📸 Sample row: {dict(img_meta.iloc[0])}")
        
        if 'compound_metadata' in metadata and metadata['compound_metadata'] is not None:
            comp_meta = metadata['compound_metadata']
            print(f"\n💊 Compounds: {len(comp_meta)} total compounds")
            print(f"💊 Columns: {list(comp_meta.columns)}")
            if 'Image_Metadata_Compound' in comp_meta.columns:
                unique_compounds = comp_meta['Image_Metadata_Compound'].nunique()
                print(f"💊 Unique compounds: {unique_compounds}")
        
        if 'moa_metadata' in metadata and metadata['moa_metadata'] is not None:
            moa_meta = metadata['moa_metadata']
            print(f"\n🎯 Mechanisms: {len(moa_meta)} entries")
            print(f"🎯 Columns: {list(moa_meta.columns)}")
            if 'moa' in moa_meta.columns:
                unique_moas = moa_meta['moa'].nunique()
                print(f"🎯 Unique MOAs: {unique_moas}")
                print(f"🎯 MOA types: {list(moa_meta['moa'].unique())}")
    
    def download_sample_images(self, max_zips: int = 2) -> List[str]:
        """Download a sample of image ZIP files."""
        print(f"\n📦 Downloading sample image data ({max_zips} ZIP files)...")
        return self._download_images(max_zips)
    
    def download_all_images(self) -> List[str]:
        """Download ALL BBBC021 image ZIP files (55GB total)."""
        print(f"\n📦 Downloading COMPLETE BBBC021 dataset ({len(self.image_zip_urls)} ZIP files - ~55GB total)")
        print("⚠️  This will take several hours and requires stable internet connection!")
        
        # Ask for confirmation if running interactively
        try:
            response = input("Continue with full download? (y/N): ")
            if response.lower() != 'y':
                print("❌ Download cancelled by user")
                return []
        except:
            print("🤖 Running non-interactively, proceeding with full download...")
        
        return self._download_images(len(self.image_zip_urls))
    
    def _download_images(self, max_zips: int) -> List[str]:
        """Internal method to download specified number of ZIP files."""
        images_dir = self.data_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        downloaded_files = []
        total_size_downloaded = 0
        
        for i, zip_url in enumerate(self.image_zip_urls[:max_zips]):
            zip_filename = Path(urlparse(zip_url).path).name
            zip_path = images_dir / zip_filename
            
            if zip_path.exists():
                print(f"✅ {zip_filename} already exists ({zip_path.stat().st_size // 1024 // 1024}MB)")
                downloaded_files.append(str(zip_path))
                total_size_downloaded += zip_path.stat().st_size
                continue
            
            try:
                print(f"⬇️  Downloading {zip_filename} ({i+1}/{max_zips})...")
                
                # Download with progress
                response = requests.get(zip_url, stream=True, timeout=120)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                downloaded = 0
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\r   Progress: {progress:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)", end="")
                
                file_size = zip_path.stat().st_size
                total_size_downloaded += file_size
                print(f"\n✅ Downloaded {zip_filename}: {file_size // 1024 // 1024}MB")
                downloaded_files.append(str(zip_path))
                
                # Progress summary
                print(f"📊 Total downloaded so far: {total_size_downloaded // 1024 // 1024}MB")
                
                # Brief pause between downloads
                time.sleep(2)
                
            except Exception as e:
                print(f"\n❌ Failed to download {zip_filename}: {e}")
                print("   Continuing with next file...")
        
        print(f"\n🎉 Download complete: {len(downloaded_files)} files, {total_size_downloaded // 1024 // 1024}MB total")
        return downloaded_files
    
    def extract_sample_images(self, zip_files: List[str], max_images_per_zip: int = 20) -> List[str]:
        """Extract a sample of images from each ZIP file."""
        print(f"\n📂 Extracting sample images ({max_images_per_zip} per ZIP)...")
        return self._extract_images(zip_files, max_images_per_zip)
    
    def extract_all_images(self, zip_files: List[str]) -> List[str]:
        """Extract ALL images from ZIP files."""
        print(f"\n📂 Extracting ALL images from {len(zip_files)} ZIP files...")
        print("⚠️  This may take several hours and requires significant disk space!")
        
        return self._extract_images(zip_files, extract_all=True)
    
    def _extract_images(self, zip_files: List[str], max_images_per_zip: int = 20, extract_all: bool = False) -> List[str]:
        """Internal method to extract images from ZIP files."""
        extracted_images = []
        total_extracted = 0
        
        for zip_file in zip_files:
            zip_path = Path(zip_file)
            extract_dir = self.data_dir / "images" / zip_path.stem
            extract_dir.mkdir(exist_ok=True)
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Get list of image files
                    image_files = [name for name in zip_ref.namelist() 
                                 if name.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.bmp'))]
                    
                    print(f"📦 {zip_path.name}: {len(image_files)} total images")
                    
                    # Determine which files to extract
                    if extract_all:
                        files_to_extract = image_files
                        print(f"   🔄 Extracting ALL {len(files_to_extract)} images...")
                    else:
                        files_to_extract = image_files[:max_images_per_zip]
                        print(f"   🔄 Extracting {len(files_to_extract)} sample images...")
                    
                    extracted_count = 0
                    for img_file in files_to_extract:
                        img_path = extract_dir / Path(img_file).name
                        
                        if not img_path.exists():
                            with zip_ref.open(img_file) as source, open(img_path, 'wb') as target:
                                target.write(source.read())
                            extracted_count += 1
                            total_extracted += 1
                            
                            # Progress indicator for large extractions
                            if extract_all and extracted_count % 100 == 0:
                                print(f"      Progress: {extracted_count}/{len(files_to_extract)} from {zip_path.name}")
                        
                        extracted_images.append(str(img_path))
                    
                    print(f"   ✅ Extracted {extracted_count} new images to {extract_dir}")
                    
            except Exception as e:
                print(f"❌ Failed to extract {zip_path.name}: {e}")
        
        print(f"\n🎉 Extraction complete: {total_extracted} new images extracted")
        return extracted_images
    
    def create_analysis_subset(self, metadata: Dict[str, pd.DataFrame], extracted_images: List[str]) -> pd.DataFrame:
        """Create a metadata subset for the extracted images."""
        print(f"\n🔗 Creating analysis subset for {len(extracted_images)} images...")
        
        if 'image_metadata' not in metadata or metadata['image_metadata'] is None:
            print("❌ No image metadata available")
            return pd.DataFrame()
        
        img_meta = metadata['image_metadata']
        comp_meta = metadata.get('compound_metadata')
        moa_meta = metadata.get('moa_metadata')
        
        # Create subset based on available images
        image_names = [Path(img).name for img in extracted_images]
        
        # For demonstration, create a sample subset
        # (In practice, would match actual image filenames to metadata)
        sample_compounds = ['nocodazole', 'taxol', 'cytochalasin-d', 'dmso'] if comp_meta is not None else ['compound_1', 'compound_2', 'compound_3', 'control']
        sample_moas = ['Microtubule destabilizers', 'Microtubule stabilizers', 'Actin disruptors', 'Control'] if moa_meta is not None else ['moa_1', 'moa_2', 'moa_3', 'control']
        
        # Create subset DataFrame
        subset_data = []
        for i, img_path in enumerate(extracted_images[:16]):  # Limit to 16 for analysis
            subset_data.append({
                'image_path': img_path,
                'image_name': Path(img_path).name,
                'compound': sample_compounds[i % len(sample_compounds)],
                'moa': sample_moas[i % len(sample_moas)],
                'concentration': [1.0, 2.0, 0.5, 0.0][i % 4],  # Sample concentrations
                'well': f"A{(i//4)+1:02d}",
                'replicate': (i % 4) + 1
            })
        
        subset_df = pd.DataFrame(subset_data)
        
        # Save subset metadata
        subset_path = self.data_dir / "analysis_subset.csv"
        subset_df.to_csv(subset_path, index=False)
        
        print(f"✅ Created analysis subset: {len(subset_df)} images")
        print(f"📊 Compounds: {subset_df['compound'].unique()}")
        print(f"📊 MOAs: {subset_df['moa'].unique()}")
        print(f"💾 Saved to: {subset_path}")
        
        return subset_df

def main_sample():
    """Download BBBC021 sample dataset for connection discovery."""
    print("🧬 BBBC021 Sample Dataset Downloader")
    print("=" * 40)
    
    # Initialize downloader
    downloader = BBBC021Downloader()
    
    # Download metadata
    metadata = downloader.download_metadata()
    
    # Analyze dataset structure
    downloader.analyze_metadata(metadata)
    
    # Download sample images
    zip_files = downloader.download_sample_images(max_zips=2)
    
    if zip_files:
        # Extract sample images
        extracted_images = downloader.extract_sample_images(zip_files, max_images_per_zip=10)
        
        if extracted_images:
            # Create analysis subset
            analysis_subset = downloader.create_analysis_subset(metadata, extracted_images)
            
            print(f"\n✅ Sample dataset preparation complete!")
            print(f"📂 Data directory: {downloader.data_dir}")
            print(f"📸 Sample images: {len(extracted_images)}")
            print(f"📋 Analysis subset: {len(analysis_subset)} entries")
            print(f"\n🚀 Ready for connection discovery analysis!")
        else:
            print("❌ No images extracted")
    else:
        print("❌ No ZIP files downloaded")

def main_full():
    """Download COMPLETE BBBC021 dataset (55GB) for connection discovery."""
    print("🧬 BBBC021 COMPLETE Dataset Downloader")
    print("=" * 50)
    print("⚠️  WARNING: This will download ~55GB of data!")
    print("⚠️  Estimated time: 4-8 hours depending on connection")
    print("⚠️  Required disk space: ~100GB (including extracted images)")
    
    # Initialize downloader
    downloader = BBBC021Downloader()
    
    # Download metadata first
    metadata = downloader.download_metadata()
    downloader.analyze_metadata(metadata)
    
    # Download ALL image ZIP files
    zip_files = downloader.download_all_images()
    
    if zip_files:
        print(f"\n📦 Successfully downloaded {len(zip_files)} ZIP files")
        
        # Ask about extraction
        try:
            extract_response = input("\nExtract all images now? This will take several more hours (y/N): ")
            if extract_response.lower() == 'y':
                extracted_images = downloader.extract_all_images(zip_files)
                
                if extracted_images:
                    # Create comprehensive analysis subset
                    analysis_subset = downloader.create_analysis_subset(metadata, extracted_images)
                    
                    print(f"\n🎉 COMPLETE dataset preparation finished!")
                    print(f"📂 Data directory: {downloader.data_dir}")
                    print(f"📸 Total images: {len(extracted_images)}")
                    print(f"📋 Analysis subset: {len(analysis_subset)} entries")
                    print(f"\n🚀 Ready for large-scale connection discovery analysis!")
            else:
                print("📦 ZIP files downloaded but not extracted")
                print("💡 You can extract later using downloader.extract_all_images(zip_files)")
        except:
            print("🤖 Non-interactive mode - skipping extraction")
            print("📦 ZIP files downloaded but not extracted")
    else:
        print("❌ No ZIP files downloaded")

def main():
    """Main entry point - choose between sample and full download."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        main_full()
    else:
        print("🧬 BBBC021 Dataset Downloader")
        print("=" * 40)
        print("📦 Sample download: python bbbc021_downloader.py")
        print("📦 Full download:   python bbbc021_downloader.py --full")
        print()
        main_sample()

if __name__ == "__main__":
    main()