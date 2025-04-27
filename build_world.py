# %%
import time
from world import Region
import random

# %%
WORLD_DIM = (10, 10)
REGION_DIM = (3, 3)
WORD_MAP = {}
ALL_COORDS = set((x, y) for x in range(WORLD_DIM[0]) for y in range(WORLD_DIM[1]))
start_time = time.time()
seed = 42
random.seed(seed)


# %%
def get_random_empty_coord():
    """Generates a random coordinate pair (x, y) within WORLD_DIM bounds
    that does not already exist as a key in the MAP dictionary.

    Returns None if no empty coordinates are available.
    """
    # consider making these global vars for effeciency
    occupied_coords = set(WORD_MAP.keys())
    available_coords = list(ALL_COORDS - occupied_coords)

    if not available_coords:
        return None  # Or raise an error, depending on desired behavior

    return random.choice(available_coords)


def generate_region():
    """Generates a region of the world."""
    region_coord = get_random_empty_coord()
    if region_coord is None:
        raise ValueError("No empty coordinates available.")

    region = Region(region_coord)
    WORD_MAP[region_coord] = region


generate_region()

# %%
print(WORD_MAP)
# %%
