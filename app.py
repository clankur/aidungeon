from flask import Flask, jsonify, request
from flask_cors import CORS
from uuid import UUID

# Your existing project imports
from world import World, Building, Tile, TerrainType  # TerrainType is in world.py
from character import Character
from enums import Direction  # Assuming Direction is now in enums/__init__.py

# graph.py for KnowledgeGraph if direct access is needed, but World is a KG
# entity.py for Entity if needed for type checking beyond Character/Building


# --- World Initialization ---
def initialize_world():
    # World(world_dims, region_dims, province_dims, graph=None)
    world_instance = World(
        world_dims=(1, 1),
        region_dims=(1, 1),  # Region internal grid for provinces
        province_dims=(10, 10),  # Province internal grid for buildings/terrain features
    )

    # Access the defaultly created region and province
    # Assuming world_dims=(1,1) means region (0,0) exists
    # Assuming region_dims=(1,1) means province (0,0) within that region exists
    try:
        default_region = world_instance.regions[(0, 0)]
        # Province __init__ takes (region, coords, province_dims, terrain, world)
        # Region __init__ creates its provinces.
        # province_dims in Region.__init__ refers to the grid size of the province itself (e.g., 10x10 tiles for buildings)
        # The province_dims passed to World constructor is used by Region for creating its Provinces.
        default_province = default_region.subregions[0][0]
    except KeyError:
        print(
            "Error: Default region or province not found during initialization. Check World/Region setup."
        )
        # Fallback or raise error
        # For now, let's try to create them if not found, though World init should handle it.
        # This indicates a potential mismatch with World/Region constructor logic if it fails.
        default_region = world_instance.create_region(
            region_coord=(0, 0)
        )  # This might need more args
        default_province = default_region.subregions[0][0]

    # Create a Building in the default province
    # Building.__init__(name, coords, province, world, internal_dims)
    # Coords are within the province for the building.
    building1 = Building(
        name="Test House",
        coords=(2, 2),  # Position of the building in the province grid
        province=default_province,
        world=world_instance,
        internal_dims=(5, 5),  # 5x5 grid inside the building
    )
    # Building's __init__ (via Entity) should add it to the world graph.

    # Add Characters
    # Character.__init__(name, sex, race, location: Tile, world, birth_time)
    tile_char1 = building1.find_unoccupied_tile()
    if tile_char1:
        Character(  # Assign to variable if you need to reference it later, e.g. char1 = ...
            name="Alice",
            sex="female",
            race="human",  # Ensure 'human' is valid for get_character_emoji
            location=tile_char1,
            world=world_instance,
        )
    else:
        print("Warning: Could not find an unoccupied tile for Alice in Test House")

    tile_char2 = building1.find_unoccupied_tile()
    if tile_char2:
        Character(
            name="Bob",
            sex="male",
            race="orc",  # Ensure 'orc' is valid
            location=tile_char2,
            world=world_instance,
        )
    else:
        print("Warning: Could not find an unoccupied tile for Bob in Test House")

    return world_instance


# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

game_world = initialize_world()


# --- Helper Functions for Serialization ---
def serialize_tile(tile: Tile):  # Removed world param, not used
    occupants_data = []
    for occupant_entity in tile.get_occupants():
        render_char = "."
        if hasattr(occupant_entity, "render") and callable(occupant_entity.render):
            render_char = occupant_entity.render()

        entity_type = occupant_entity.__class__.__name__

        occupant_info = {
            "id": str(occupant_entity.uuid),
            "name": occupant_entity.name,
            "render": render_char,
            "type": entity_type,
        }
        occupants_data.append(occupant_info)

    return {
        "id": str(tile.uuid),
        "local_coords": tile.local_coords,
        "occupants": occupants_data,
        # "type": "floor", # Could add explicit tile type if Tile has such an attribute
    }


def serialize_building(building: Building):  # Removed world param
    tiles_grid = []
    for y in range(building.internal_dims[1]):
        row_data = []
        for x in range(building.internal_dims[0]):
            tile = building.get_tile((x, y))
            if tile:
                row_data.append(serialize_tile(tile))
            else:
                row_data.append(None)
        tiles_grid.append(row_data)

    return {
        "id": str(building.uuid),
        "name": building.name,
        "internal_dims": building.internal_dims,
        "tiles_grid": tiles_grid,
        "province_coords": building.coords,
    }


def get_world_characters_info(world: World) -> list[dict]:
    characters_info = []
    if hasattr(world, "graph") and world.graph:
        for node_obj in world.graph.nodes():  # Iterate over nodes directly
            if isinstance(node_obj, Character):
                loc_edges = world.get_edges(
                    predicate="is_occupied_by", object_node=node_obj
                )
                tile_coords = None
                building_id = None
                tile_id = None
                if loc_edges:
                    tile_obj = loc_edges[0][0]
                    if hasattr(tile_obj, "local_coords") and hasattr(
                        tile_obj, "building"
                    ):
                        tile_coords = tile_obj.local_coords
                        tile_id = str(tile_obj.uuid)
                        building_id = str(tile_obj.building.uuid)

                characters_info.append(
                    {
                        "id": str(node_obj.uuid),
                        "name": node_obj.name,
                        "render": node_obj.render(),
                        "race": node_obj.physical_attributes.race,
                        "sex": node_obj.physical_attributes.sex,
                        "current_tile_coords": tile_coords,
                        "current_tile_id": tile_id,
                        "current_building_id": building_id,
                    }
                )
    return characters_info


# --- API Endpoints ---
@app.route("/world_state", methods=["GET"])
def get_world_state_endpoint():  # Renamed to avoid conflict with any variable
    buildings_data = []
    if hasattr(game_world, "graph") and game_world.graph:
        for node_obj in game_world.graph.nodes():
            if isinstance(node_obj, Building):
                buildings_data.append(serialize_building(node_obj))

    characters_data = get_world_characters_info(game_world)

    return jsonify(
        {
            "world_time": game_world.get_current_world_time(),
            "buildings": buildings_data,
            "characters": characters_data,
        }
    )


def _find_entity_by_uuid(world: World, entity_uuid: UUID, entity_type: type):
    if hasattr(world, "graph") and world.graph:
        for node_obj in world.graph.nodes():
            if isinstance(node_obj, entity_type) and node_obj.uuid == entity_uuid:
                return node_obj
    return None


@app.route("/action/move", methods=["POST"])
def action_move_endpoint():  # Renamed
    data = request.get_json()
    character_id_str = data.get("character_id")
    direction_str = data.get("direction")

    if not character_id_str or not direction_str:
        return (
            jsonify({"success": False, "message": "Missing character_id or direction"}),
            400,
        )

    try:
        character_uuid = UUID(character_id_str)
        # Ensure direction_str is a valid key in Direction enum
        if direction_str.upper() not in Direction.__members__:
            raise KeyError(f"Invalid direction string: {direction_str}")
        move_direction = Direction[direction_str.upper()]
    except (ValueError, KeyError) as e:  # ValueError for UUID, KeyError for Direction
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Invalid character_id or direction format/value: {str(e)}",
                }
            ),
            400,
        )
    except Exception as e:  # Catch any other unexpected errors during conversion
        return (
            jsonify({"success": False, "message": f"Error processing input: {str(e)}"}),
            400,
        )

    character_to_move = _find_entity_by_uuid(game_world, character_uuid, Character)

    if not character_to_move:
        return jsonify({"success": False, "message": "Character not found"}), 404

    success = character_to_move.move(move_direction)

    if success:
        game_world.curr_time += 1
        return jsonify(
            {
                "success": True,
                "message": f"{character_to_move.name} moved {move_direction.name}.",
            }
        )  # Use .name for Enum
    else:
        return jsonify(
            {
                "success": False,
                "message": f"{character_to_move.name} could not move {move_direction.name}.",
            }
        )


@app.route("/action/chat", methods=["POST"])
def action_chat_endpoint():  # Renamed
    data = request.get_json()
    source_char_id_str = data.get("source_character_id")
    target_char_id_str = data.get("target_character_id")
    message = data.get("message")

    if not source_char_id_str or not target_char_id_str or message is None:
        return (
            jsonify({"success": False, "message": "Missing required chat parameters"}),
            400,
        )

    try:
        source_uuid = UUID(source_char_id_str)
        target_uuid = UUID(target_char_id_str)
    except ValueError:
        return (
            jsonify({"success": False, "message": "Invalid character ID format"}),
            400,
        )

    source_char = _find_entity_by_uuid(game_world, source_uuid, Character)
    target_char = _find_entity_by_uuid(game_world, target_uuid, Character)

    if not source_char:
        return jsonify({"success": False, "message": "Source character not found"}), 404
    if not target_char:
        return jsonify({"success": False, "message": "Target character not found"}), 404

    if not isinstance(message, str):
        return jsonify({"success": False, "message": "Message must be a string"}), 400

    success = source_char.chat(target_char, message)
    if success:
        game_world.curr_time += 1
        return jsonify({"success": True, "message": "Chat successful."})
    else:
        return jsonify(
            {
                "success": False,
                "message": "Chat failed (e.g., not adjacent or other issue).",
            }
        )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
