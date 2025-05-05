# %%
from storyteller import Storyteller
from graph import KnowledgeGraph

# %%
graph = KnowledgeGraph()
storyteller = Storyteller(graph)
# %%
world_story, characters_story = storyteller.init_story()
# %%
print(world_story)
# %%
print(characters_story)
# %%

create_building_declaration = {
    "name": "create_building",
    "description": "Creates a building with the given name, the coordinates it occupies within a province, and the province it belongs to",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the building",
            },
            "province": {
                "type": "string",
                "description": "The province the building belongs to",
            },
        },
        "required": ["name", "province"],
    },
}
# %%
from importlib import reload
import world

reload(world)
from world import World
from google import genai
from google.genai import types
from typing import List

# %%
WORLD = World(
    world_dims=(3, 3),
    region_dims=(4, 4),
    province_dims=(10, 10),
)
tools = types.Tool(function_declarations=[create_character_declaration])
config = types.GenerateContentConfig(tools=[tools])
client = genai.Client()


# %%
def build_world(characters_story: str, world_story: str):

    world_contents = [types.Content(role="user", parts=[types.Part(text=world_story)])]
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17", config=config, contents=contents
    )

    locations = []
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=f"Valid locations in the world are: {locations}"),
                types.Part(text=characters_story),
            ],
        )
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17", config=config, contents=contents
    )
    function_calls = [
        [p.function_call for p in c.content.parts if p.function_call]
        for c in response.candidates
    ]
    for call in function_calls:
        call.args["world"] = WORLD
        world_location = WORLD.get_entity(call.args["location"])[0]
        call.args["location"] = world_location
        WORLD.create_character(**call.args)


# %%
world_triples = storyteller.extractor.extract(world)
# %%
characters_triples = storyteller.extractor.extract(characters)

# %%
for triple in characters:
    graph.add_edge(*triple)
# %%
graph.query("Theldrin Stonehand")
# %%
