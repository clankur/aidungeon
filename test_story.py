# %%
from importlib import reload
import character

reload(character)

import storyteller

reload(storyteller)

import world

reload(world)

# %%
from storyteller import Storyteller
from world import World, Tile
from character import Character

# %%
world = World(
    world_dims=(3, 3),
    region_dims=(4, 4),
    province_dims=(10, 10),
)
region = world.create_region((0, 0))
province = region.subregions[0][0]
house = province.create_building("House", (0, 0))

storyteller = Storyteller(world)
# %%
characters_story = storyteller.generate_character_story()
# %%
print(characters_story)
# %%
function_calls = storyteller.init_story(characters_story)
function_calls
# %%
world.graph.nodes

# %%
location = house.tiles[(0, 0)]
# %%
location

# %%
