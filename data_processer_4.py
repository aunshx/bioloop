import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
from pyproj import Transformer
import json
import shutil
import argparse

CA_BOUNDS = {
    'west': -124.482003,
    'east': -114.131211,
    'north': 42.009517,
    'south': 32.534156
}

CDL_CROP_MAPPING = {
    0: "Background",
    1: "Corn",
    2: "Cotton",
    3: "Rice",
    4: "Sorghum",
    5: "Soybeans",
    6: "Sunflower",
    10: "Peanuts",
    11: "Tobacco",
    12: "Sweet Corn",
    13: "Pop or Orn Corn",
    14: "Mint",
    21: "Barley",
    22: "Durum Wheat",
    23: "Spring Wheat",
    24: "Winter Wheat",
    25: "Other Small Grains",
    26: "Dbl Crop WinWht/Soybeans",
    27: "Rye",
    28: "Oats",
    29: "Millet",
    30: "Speltz",
    31: "Canola",
    32: "Flaxseed",
    33: "Safflower",
    34: "Rape Seed",
    35: "Mustard",
    36: "Alfalfa",
    37: "Other Hay/Non Alfalfa",
    38: "Camelina",
    39: "Buckwheat",
    41: "Sugarbeets",
    42: "Dry Beans",
    43: "Potatoes",
    44: "Other Crops",
    45: "Sugarcane",
    46: "Sweet Potatoes",
    47: "Misc Vegs & Fruits",
    48: "Watermelons",
    49: "Onions",
    50: "Cucumbers",
    51: "Chick Peas",
    52: "Lentils",
    53: "Peas",
    54: "Tomatoes",
    55: "Caneberries",
    56: "Hops",
    57: "Herbs",
    58: "Clover/Wildflowers",
    59: "Sod/Grass Seed",
    60: "Switchgrass",
    61: "Fallow/Idle Cropland",
    62: "Pasture/Grass",
    63: "Forest",
    64: "Shrubland",
    65: "Barren",
    66: "Cherries",
    67: "Peaches",
    68: "Apples",
    69: "Grapes",
    70: "Christmas Trees",
    71: "Other Tree Crops",
    72: "Citrus",
    74: "Pecans",
    75: "Almonds",
    76: "Walnuts",
    77: "Pears",
    81: "Clouds/No Data",
    82: "Developed",
    83: "Water",
    87: "Wetlands",
    88: "Nonag/Undefined",
    92: "Aquaculture",
    111: "Open Water",
    112: "Perennial Ice/Snow",
    121: "Developed/Open Space",
    122: "Developed/Low Intensity",
    123: "Developed/Med Intensity",
    124: "Developed/High Intensity",
    131: "Barren",
    141: "Deciduous Forest",
    142: "Evergreen Forest",
    143: "Mixed Forest",
    152: "Shrubland",
    176: "Grassland/Pasture",
    190: "Woody Wetlands",
    195: "Herbaceous Wetlands",
    204: "Pistachios",
    205: "Triticale",
    206: "Carrots",
    207: "Asparagus",
    208: "Garlic",
    209: "Cantaloupes",
    210: "Prunes",
    211: "Olives",
    212: "Oranges",
    213: "Honeydew Melons",
    214: "Broccoli",
    215: "Avocados",
    216: "Peppers",
    217: "Pomegranates",
    218: "Nectarines",
    219: "Greens",
    220: "Plums",
    221: "Strawberries",
    222: "Squash",
    223: "Apricots",
    224: "Vetch",
    225: "Dbl Crop WinWht/Corn",
    226: "Dbl Crop Oats/Corn",
    227: "Lettuce",
    228: "Dbl Crop Triticale/Corn",
    229: "Pumpkins",
    230: "Dbl Crop Lettuce/Durum Wht",
    231: "Dbl Crop Lettuce/Cantaloupe",
    232: "Dbl Crop Lettuce/Cotton",
    233: "Dbl Crop Lettuce/Barley",
    234: "Dbl Crop Durum Wht/Sorghum",
    235: "Dbl Crop Barley/Sorghum",
    236: "Dbl Crop WinWht/Sorghum",
    237: "Dbl Crop Barley/Corn",
    238: "Dbl Crop WinWht/Cotton",
    239: "Dbl Crop Soybeans/Cotton",
    240: "Dbl Crop Soybeans/Oats",
    241: "Dbl Crop Corn/Soybeans",
    242: "Blueberries",
    243: "Cabbage",
    244: "Cauliflower",
    245: "Celery",
    246: "Radishes",
    247: "Turnips",
    248: "Eggplants",
    249: "Gourds",
    250: "Cranberries",
    254: "Dbl Crop Barley/Soybeans"
}

def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

class SingleYearCDLProcessor:
    def __init__(self, year, base_dir="data", output_dir="/Volumes/External/cdl_processed", chunk_size=1000):
        self.year = year
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        
        # Create directory structure
        self.chunks_dir = self.output_dir / "chunks" / str(year)
        self.final_dir = self.output_dir / "final"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        
        # Create progress file
        self.progress_file = self.chunks_dir / "progress.json"
        self.load_progress()
        
        # Setup coordinate transformers
        self.transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
        self.transformer_inverse = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
        
        # Get California bounds in EPSG:5070
        self.ca_bounds_5070 = self.get_ca_bounds_5070()
    
    def get_ca_bounds_5070(self):
        """Convert CA bounds to EPSG:5070"""
        west, north = self.transformer_inverse.transform(CA_BOUNDS['west'], CA_BOUNDS['north'])
        east, south = self.transformer_inverse.transform(CA_BOUNDS['east'], CA_BOUNDS['south'])
        return {'west': west, 'east': east, 'north': north, 'south': south}
    
    def load_progress(self):
        """Load processing progress"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {'last_chunk': 0, 'status': 'processing'}
    
    def save_progress(self, chunk):
        """Save processing progress"""
        self.progress['last_chunk'] = chunk
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f)
    
    def is_in_california(self, x, y):
        """Check if coordinates are within California bounds"""
        return (self.ca_bounds_5070['west'] <= x <= self.ca_bounds_5070['east'] and 
                self.ca_bounds_5070['south'] <= y <= self.ca_bounds_5070['north'])
    
    def save_chunk(self, df, chunk_num):
        """Save chunk with compression"""
        if df is None or len(df) == 0:
            return
            
        output_file = self.chunks_dir / f"chunk_{chunk_num:05d}.csv.gz"
        df.to_csv(output_file, index=False, compression='gzip')
        log(f"Saved chunk {chunk_num} with {len(df):,} records")
    
    def process_chunk(self, src, window):
        """Process a single chunk"""
        try:
            chunk_data = src.read(1, window=window)
            
            rows, cols = np.meshgrid(
                range(window.row_off, window.row_off + window.height),
                range(window.col_off, window.col_off + window.width),
                indexing='ij'
            )
            
            x_coords, y_coords = rasterio.transform.xy(
                src.transform,
                rows.ravel(),
                cols.ravel()
            )
            
            ca_mask = np.array([
                self.is_in_california(x, y) 
                for x, y in zip(x_coords, y_coords)
            ]).reshape(chunk_data.shape)
            
            valid_mask = (chunk_data > 0) & ca_mask
            if not np.any(valid_mask):
                return None
            
            rows, cols = np.nonzero(valid_mask)
            x_coords, y_coords = rasterio.transform.xy(
                src.transform,
                rows + window.row_off,
                cols + window.col_off
            )
            
            lons, lats = self.transformer.transform(x_coords, y_coords)
            
            df = pd.DataFrame({
                'year': self.year,
                'crop_code': chunk_data[valid_mask],
                'longitude': lons,
                'latitude': lats
            })
            
            df['crop_name'] = df['crop_code'].map(CDL_CROP_MAPPING)
            
            return df
            
        except Exception as e:
            log(f"Error processing chunk: {str(e)}")
            return None
    
    def merge_chunks(self):
        """Merge all chunks into final file"""
        try:
            log("Merging chunks...")
            
            chunks = sorted(self.chunks_dir.glob("chunk_*.csv.gz"))
            if not chunks:
                log("No chunks found to merge")
                return
            
            all_data = []
            total_records = 0
            
            for chunk_file in chunks:
                df = pd.read_csv(chunk_file, compression='gzip')
                all_data.append(df)
                total_records += len(df)
                log(f"Read {chunk_file.name} - Running total: {total_records:,} records")
            
            final_df = pd.concat(all_data, ignore_index=True)
            final_file = self.final_dir / f"cdl_california_{self.year}.csv.gz"
            final_df.to_csv(final_file, index=False, compression='gzip')
            
            log(f"Saved merged file with {total_records:,} records")
            
            # Clean up
            log("Cleaning up chunks directory")
            shutil.rmtree(self.chunks_dir)
            
            log("Processing complete")
            
        except Exception as e:
            log(f"Error merging chunks: {str(e)}")
    
    def process(self):
        """Process the specified year"""
        # Check if final file already exists
        final_file = self.final_dir / f"cdl_california_{self.year}.csv.gz"
        if final_file.exists():
            log(f"Year {self.year} already processed. Final file exists.")
            return
        
        file_path = self.base_dir / f"{self.year}_30m_cdls" / f"{self.year}_30m_cdls.tif"
        if not file_path.exists():
            log(f"Error: File not found: {file_path}")
            return
        
        start_chunk = self.progress['last_chunk']
        log(f"Processing year {self.year} starting from chunk {start_chunk}")
        
        try:
            with rasterio.open(file_path) as src:
                height = src.height
                width = src.width
                
                total_chunks = ((height + self.chunk_size - 1) // self.chunk_size) * \
                             ((width + self.chunk_size - 1) // self.chunk_size)
                
                chunk_num = 0
                
                for y in range(0, height, self.chunk_size):
                    chunk_height = min(self.chunk_size, height - y)
                    for x in range(0, width, self.chunk_size):
                        chunk_num += 1
                        
                        if chunk_num <= start_chunk:
                            continue
                        
                        if chunk_num % 100 == 0:
                            progress = (chunk_num / total_chunks) * 100
                            log(f"Progress: {progress:.1f}% ({chunk_num}/{total_chunks} chunks)")
                        
                        chunk_width = min(self.chunk_size, width - x)
                        window = Window(x, y, chunk_width, chunk_height)
                        
                        chunk_df = self.process_chunk(src, window)
                        if chunk_df is not None:
                            self.save_chunk(chunk_df, chunk_num)
                        
                        self.save_progress(chunk_num)
                        
                        # Clear memory
                        del chunk_df
            
            # Merge chunks after processing
            # self.merge_chunks()
            
        except Exception as e:
            log(f"Error processing year {self.year}: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Process CDL data for a specific year")
    parser.add_argument("year", type=int, help="Year to process (e.g., 2008)")
    parser.add_argument("--output", type=str, default="/Volumes/aunsh_hd/cdl_processed",
                      help="Output directory path")
    parser.add_argument("--chunk-size", type=int, default=2000,
                      help="Chunk size for processing")
    
    args = parser.parse_args()
    
    processor = SingleYearCDLProcessor(
        year=args.year,
        output_dir=args.output,
        chunk_size=args.chunk_size
    )
    
    processor.process()

if __name__ == "__main__":
    main()
