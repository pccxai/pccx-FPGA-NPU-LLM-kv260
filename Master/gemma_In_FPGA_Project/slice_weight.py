import numpy as np
import os

def slice_weight_for_npu(weight_name, weight_matrix, block_size=64, save_dir="./npu_bram_tiles"):
    """
    Tiling the huge LLM weight matrix to fit the NPU Systolic Array (64x64) standard.
    A function that pads with 0 when necessary and saves it as a Bin file that can be uploaded to BRAM.
    """
    out_features, in_features = weight_matrix.shape
    print(f" Target Layer: [{weight_name}] | Original Shape: {out_features}x{in_features}")

    # 1. Calculate padding (Memory Alignment) exactly in 64 units.
    # Same as the concept of pitch memory allocation in CUDA.
    pad_out = (block_size - (out_features % block_size)) % block_size
    pad_in  = (block_size - (in_features % block_size)) % block_size

    if pad_out > 0 or pad_in > 0:
        print(f"Apply Memory Alignment: Add Zero-padding (+{pad_out}, +{pad_in})")
        # Filling the right and bottom with zeros using numpy.pad
        aligned_matrix = np.pad(weight_matrix, ((0, pad_out), (0, pad_in)), mode='constant', constant_values=0)
    else:
        aligned_matrix = weight_matrix

    aligned_out, aligned_in = aligned_matrix.shape

    # Calculate Grid size (how many 64x64 blocks will come out)
    grid_y = aligned_out // block_size
    grid_x = aligned_in // block_size
    print(f"Grid configuration: {grid_y} x {grid_x} Blocks (Total: {grid_y * grid_x} tiles)")

    # 2. Slicing into 64x64 blocks (Tiling)
    # Twists the shape so that it continues beautifully into a 64x64 chunk in memory (Numpy magic!)
    tiles = aligned_matrix.reshape(grid_y, block_size, grid_x, block_size)
    tiles = tiles.swapaxes(1, 2)  # shape: (grid_y, grid_x, block_size, block_size)

    # 3. Save as a file for easy upload to hardware (BRAM)
    os.makedirs(save_dir, exist_ok=True)

    for y in range(grid_y):
        for x in range(grid_x):
            tile_data = tiles[y, x]

            # On the actual board, it is fastest to transfer to .bin (Raw Binary),
            # Since we are currently in Phase 2 Python verification stage, will save it as .npy.
            file_name = f"{weight_name}_block_Y{y}_X{x}.npy"
            np.save(os.path.join(save_dir, file_name), tile_data)

    print(f"Slicing completed! Save location: {save_dir}/\n")
    return tiles

if __name__ == "__main__":
    # --- [Test Scenario] ---
    # Proceed to assume that it is the Attention Query Projection weight of Gemma 3N.
    # Example: in_features = 2048, out_features = 2048
    # But for edge case testing, will put in a strange number (2000x1500) that doesn't fall to 64.

    dummy_q_proj = np.random.randn(2000, 1500).astype(np.float32)

    # NPU slicer in action.
    slice_weight_for_npu(
        weight_name="layers.0.self_attn.q_proj",
        weight_matrix=dummy_q_proj,
        block_size=64
    )