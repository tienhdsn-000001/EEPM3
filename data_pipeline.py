import os
import numpy as np
import pandas as pd
import pyBigWig
import jax.numpy as jnp

class GTExBigWigParser:
    """
    Parser to extract and bin coverage data from GTEx BigWig files
    across a defined 1Mb interval at 128-bp resolution.
    """
    def __init__(self, data_dir="data/gtex", metadata_path="data/metadata.csv", seq_len=1048576, bin_size=128, target_bins=7812):
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.seq_len = seq_len
        self.bin_size = bin_size
        self.num_bins = self.seq_len // self.bin_size  # 8192
        self.target_bins = target_bins # 7812
        
        # Calculate cropping to match AlphaGenome's target shape
        # AlphaGenome standard crop to 7812 bins
        crop_total = self.num_bins - self.target_bins
        self.crop_start = crop_total // 2
        self.crop_end = crop_total - self.crop_start
        
        self.track_metadata = self._load_metadata()
        
    def _load_metadata(self):
        """
        Loads metadata.csv which maps GTEx files to AlphaGenome 5930 track indices.
        Expected columns: 'filepath', 'tissue', 'age_bracket', 'track_index'
        """
        if not os.path.exists(self.metadata_path):
            return pd.DataFrame(columns=['filepath', 'tissue', 'age_bracket', 'track_index'])
        return pd.read_csv(self.metadata_path)

    def extract_interval(self, filepath: str, chrom: str, start: int, end: int) -> np.ndarray:
        """
        Extracts data from a BigWig file and bins it identically to the required resolution.
        """
        if not os.path.exists(filepath):
            return np.zeros(self.target_bins, dtype=np.float32)
            
        try:
            bw = pyBigWig.open(filepath)
            if not bw.isBigWig():
                bw.close()
                return np.zeros(self.target_bins, dtype=np.float32)
                
            # Use pyBigWig's extremely efficient stats function for binning
            # 'mean' over exactly num_bins
            vals = bw.stats(chrom, start, end, type="mean", nBins=self.num_bins)
            bw.close()
            
            # Replace Nones (no coverage) with 0.0
            vals = np.array([v if v is not None else 0.0 for v in vals], dtype=np.float32)
            
            # Apply padding/cropping logic to reach 7812 matching AlphaGenome standards
            if self.crop_start > 0 or self.crop_end > 0:
                vals = vals[self.crop_start : self.num_bins - self.crop_end]
                
            return vals
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return np.zeros(self.target_bins, dtype=np.float32)

class GTExDataLoader:
    """
    Constructs the target tensors (T) and mask tensors (M) required for Masked Modality Loss function.
    """
    def __init__(self, parser: GTExBigWigParser, num_tracks: int = 5930):
        self.parser = parser
        self.num_tracks = num_tracks
        self.target_bins = parser.target_bins
        
    def generate_batch(self, batch_metadata: list, chrom: str, start: int) -> dict:
        """
        batch_metadata: list of Pandas DataFrames mapping files for each sample in the batch.
        Returns a Flambax/JAX PyTree.
        """
        batch_size = len(batch_metadata)
        end = start + self.parser.seq_len
        
        # Pre-allocate numpy arrays for efficiency, convert to JAX PyTree at the end
        # T: (Batch, Sequence_Length/128, 5930)
        # M: (Batch, Sequence_Length/128, 5930)
        T = np.zeros((batch_size, self.target_bins, self.num_tracks), dtype=np.float32)
        M = np.zeros((batch_size, self.target_bins, self.num_tracks), dtype=np.float32)
        
        for b, sample_meta in enumerate(batch_metadata):
            # Each item in batch_metadata contains rows of available files
            # and their designated AlphaGenome track indices
            for _, row in sample_meta.iterrows():
                track_idx = int(row['track_index'])
                filepath = row['filepath']
                
                # Verify index bounds
                if 0 <= track_idx < self.num_tracks:
                    # Parse BigWig directly into binned slice
                    track_data = self.parser.extract_interval(filepath, chrom, start, end)
                    
                    T[b, :, track_idx] = track_data
                    M[b, :, track_idx] = 1.0  # 1 indicating data is available
                    
        # Structure as Flax/JAX PyTrees
        batch_dict = {
            'targets': jnp.array(T),
            'masks': jnp.array(M)
        }
        
        return batch_dict
