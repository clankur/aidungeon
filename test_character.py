# %%
from character import Character
from graph import KnowledgeGraph
from relationships import RelationshipType
from world import Province, Region, TerrainType, World

# %%
world = World(
    world_dims=(3, 3),
    region_dims=(4, 4),
    province_dims=(10, 10),
)
# World of 9 Region, 16 Provinces of size 10 x 10
region = world.create_region((0, 0))
region.name = "Misthalin"
subregion = region.subregions[0][0]
subregion.name = "Lumbridge"
location = subregion.create_building("House", [(0, 0), (0, 1), (1, 0), (1, 1)])
# %%
world.get_entity("House")
# %%
character = Character("John Smith", "male", "human", location, world)
father = Character("Jacob Smith", "male", "human", location, world)

# %%
character.add_familial_edge(world, character, RelationshipType.IS_CHILD_OF, father)
# %%
query = "John Smith"
query_results = world.query(query)
query_results

# %%
john2 = Character("John Smith", "male", "elf", location, world)
character.add_familial_edge(world, john2, RelationshipType.IS_PARENT_OF, father)
# %%
results = world.query(query)
results
# %%
query = "House"
query_results = world.query(query)
query_results
# need to handle query to convert to SPO triple
# send results to retriever
# %%
from retriever import Retriever

retriever = Retriever()
# %%
query = "Who is John Smith's father?"
results = retriever.retrieve(query, world)
for subject in results.keys():
    print(results[subject]["retrieved_edges"])

# Generate story with values to populate classes
# # Proompt it to fill in details besides positional coords
# # How do we place locations? Randomly? From a position move to nearest?
# Build world from story
# %%
print(results.values())
# %%
