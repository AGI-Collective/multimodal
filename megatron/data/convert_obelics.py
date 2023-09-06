import os
import pandas as pd
from tqdm import tqdm

obelics_filepath = "/p/fastdata/mmlaion/obelics/OBELICS/data"
obelics_img_to_dataset_filepath = "/p/scratch/ccstdl/mhatre1/obelics_converted"

samples = 2

for filename in tqdm(os.listdir(obelics_filepath)[:samples]):
    df = pd.read_parquet(os.path.join(obelics_filepath, filename))
    images_df = df.explode('images')
    images_df = images_df[images_df['images'].notna()].reset_index(drop=True)                        
    images_df.to_parquet(os.path.join(obelics_img_to_dataset_filepath, filename))