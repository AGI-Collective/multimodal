import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import IterableDataset

class ObelicsData(IterableDataset):
	def __init__(self, filepath, converted_filepath, images_filepath):
		self.filepath = filepath
		self.converted_filepath = converted_filepath
		self.images_filepath = images_filepath
		self.filenames = [os.path.join(filepath, filename) for filename in os.listdir(filepath)]
		self.converted_filenames = [os.path.join(converted_filepath, filename) for filename in os.listdir(converted_filepath)]
		self.prev_shard_size = 0

	def __iter__(self):

		# TODO - remove manual indexing
		for filename, converted_filename in zip(self.filenames[:2][-1::-1], self.converted_filenames[-1::-1]):
			df = pd.read_parquet(filename)
			converted_df = pd.read_parquet(converted_filename)

			# Set the "images" column in converted_df to be the key
			converted_df = converted_df.set_index('images')

			images_array = []

			print (df.shape)
			for idx, row in tqdm(df.iterrows()):
				images = row['images']

				for image in images:
					# Find the row number of the image url in converted_df
					if image is None:
						continue
					row_number = converted_df.index.get_loc(image)

					shard_id = (self.prev_shard_size + row_number) // 10000
					idx_in_shard = (self.prev_shard_size + row_number) % 10000

					# Read in images shard converting shard_id to 5 digit string
					shard_df = pd.read_parquet(os.path.join(self.images_filepath, f'{shard_id:05d}.parquet'))

					# Convert shard_id to 5 digit string and idx_in_shard to 4 digit string and concatenate them
					image_id = f'{shard_id:05d}{idx_in_shard:04d}'
					shard_df = shard_df.set_index('key')

					# Return row with image_id as key
					image_url = shard_df.iloc[shard_df.index.get_loc(image_id)]['url']

					# assert url
					assert(image_url == image)

				texts = row['texts']
				#yield {'images': images, 'texts': texts}

			self.prev_shard_size += len(converted_df)

if __name__ == "__main__":
	filepath = "/p/fastdata/mmlaion/obelics/OBELICS/data"
	converted_filepath = "/p/scratch/ccstdl/mhatre1/obelics_converted"
	images_filepath = "/p/scratch/ccstdl/mhatre1/multimodal/megatron/data/images"
	dataset = ObelicsData(filepath, converted_filepath, images_filepath)
	for i in dataset:
		print(i)