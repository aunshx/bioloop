import rasterio
from rasterio import features
from rasterio.windows import Window
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
from pyproj import Transformer

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

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

class CaliforniaCDLProcessor:
    def __init__(self, base_dir="data", chunk_size=1000):
        self.base_dir = Path(base_dir)
        self.chunk_size = chunk_size
        self.output_dir = Path("processed_data")
        self.output_dir.mkdir(exist_ok=True)
        
        self.transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    
    def save_chunk(self, df, year, chunk_num):
        if df is None or len(df) == 0:
            return
            
        output_file = self.output_dir / f"ca_cdl_data_{year}_chunk_{chunk_num:05d}.csv"
        df.to_csv(output_file, index=False)
        log(f"Saved chunk {chunk_num} with {len(df):,} records")
        
    def process_chunk(self, src, window, year, transform):
        try:
            chunk_data = src.read(1, window=window)
            
            valid_mask = chunk_data > 0
            if not np.any(valid_mask):
                return None
                
            rows, cols = np.nonzero(valid_mask)
            
            rows += window.row_off
            cols += window.col_off
            
            x_coords, y_coords = rasterio.transform.xy(
                src.transform,
                rows,
                cols
            )
            
            lons, lats = self.transformer.transform(x_coords, y_coords)
            
            df = pd.DataFrame({
                'year': year,
                'crop_code': chunk_data[valid_mask],
                'longitude': lons,
                'latitude': lats
            })
            
            df['crop_name'] = df['crop_code'].map(CDL_CROP_MAPPING)
            
            return df
            
        except Exception as e:
            log(f"Error processing chunk: {str(e)}")
            return None

    def merge_year_chunks(self, year):
        log(f"Merging chunks for year {year}")
        chunks = sorted(self.output_dir.glob(f"ca_cdl_data_{year}_chunk_*.csv"))
        
        if not chunks:
            return
            
        dfs = []
        for chunk_file in chunks:
            df = pd.read_csv(chunk_file)
            dfs.append(df)
            chunk_file.unlink()
        
        final_df = pd.concat(dfs, ignore_index=True)
        
        output_file = self.output_dir / f"ca_cdl_data_{year}.csv"
        final_df.to_csv(output_file, index=False)
        log(f"Saved merged file for {year} with {len(final_df):,} records")

    def process_single_year(self, year):
        try:
            if (self.output_dir / f"ca_cdl_data_{year}.csv").exists():
                log(f"Year {year} already processed, skipping...")
                return None
                
            file_path = self.base_dir / f"{year}_30m_cdls" / f"{year}_30m_cdls.tif"
            if not file_path.exists():
                log(f"Warning: File not found for year {year}: {file_path}")
                return None
                
            log(f"Processing {year} from {file_path}")
            
            with rasterio.open(file_path) as src:
                height = src.height
                width = src.width
                
                total_chunks = ((height + self.chunk_size - 1) // self.chunk_size) * \
                             ((width + self.chunk_size - 1) // self.chunk_size)
                
                chunk_num = 0
                
                for y in range(0, height, self.chunk_size):
                    chunk_height = min(self.chunk_size, height - y)
                    for x in range(0, width, self.chunk_size):
                        chunk_width = min(self.chunk_size, width - x)
                        chunk_num += 1
                        
                        if chunk_num % 100 == 0:
                            progress = (chunk_num / total_chunks) * 100
                            log(f"Progress: {progress:.1f}% ({chunk_num}/{total_chunks} chunks)")
                        
                        window = Window(x, y, chunk_width, chunk_height)
                        transform = src.window_transform(window)
                        
                        chunk_df = self.process_chunk(src, window, year, transform)
                        if chunk_df is not None:
                            self.save_chunk(chunk_df, year, chunk_num)
                        
                        del chunk_df
                        
            self.merge_year_chunks(year)
            
            log(f"Year {year} processing complete")
            return True
                
        except Exception as e:
            log(f"Error processing year {year}: {str(e)}")
            import traceback
            log(f"Traceback: {traceback.format_exc()}")
            return None

    def process_all_years(self, start_year=2008, end_year=2022):
        log(f"Starting processing for years {start_year}-{end_year}")
        
        for year in range(start_year, end_year + 1):
            if (self.output_dir / f"ca_cdl_data_{year}.csv").exists():
                log(f"Skipping year {year} (already processed)")
                continue
                
            self.process_single_year(year)
            
if __name__ == "__main__":
    processor = CaliforniaCDLProcessor(chunk_size=1000)
    processor.process_all_years()
