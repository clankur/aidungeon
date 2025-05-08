import urwid
from typing import Dict, Tuple, Optional, Any
from world import Building, Tile, Entity  # Assuming these are accessible
from character import Character  # Assuming Character is an entity type

# Define a placeholder color mapping for entities
# We can expand this later based on actual entity types in the project
ENTITY_COLOR_MAP: Dict[str, str] = {
    "Character": "dark red",
    "Wall": "dark gray",
    "Door": "brown",
    "default_tile": "black",
    "default_entity": "light gray",
}

# Define a character to represent a tile
TILE_CHAR = "█"


class BuildingRenderer:
    def __init__(
        self, building: Building, screen: Optional[urwid.display.raw.Screen] = None
    ):
        self.building = building
        self.screen = screen or urwid.display.raw.Screen()
        self._setup_palette()

    def _setup_palette(self) -> None:
        palette = [
            ("default", "white", "black"),  # Default text color
        ]
        for entity_name, color_name_str in ENTITY_COLOR_MAP.items():
            palette.append((entity_name, color_name_str, "black"))
            palette.append((f"{entity_name}_bg", "white", color_name_str))

        try:
            default_bg = ENTITY_COLOR_MAP.get("default_tile", "black")
            urwid.AttrSpec("white", default_bg, colors=256)
            palette.append(("empty_tile", "white", default_bg))
        except urwid.AttrSpecError as e:
            print(f"Warning: Could not register background color for default_tile: {e}")
            palette.append(("empty_tile", "white", "black"))
        self.screen.register_palette(palette)

    def _get_tile_display(self, tile: Optional[Tile]) -> Tuple[str, str]:
        """Determines the character and attribute for a tile."""
        if not tile:
            return (" ", "empty_tile")  # Should not happen if grid is full

        # Check occupants
        # This logic will need to align with how occupants are stored and retrieved.
        # Assuming world.get_edges() can be used, or tile.occupants attribute.
        # For now, let's assume a simple check.
        # We need to get entities that have an "is_located_at" or "is_occupied_by" relationship with the tile.

        occupants = tile.get_occupants()  # Use the new method from Tile class

        if occupants:
            # For simplicity, take the first occupant.
            # You might want more complex logic if multiple entities can be on one tile.
            occupant = occupants[0]
            occupant_type_name = occupant.__class__.__name__

            char_to_display = TILE_CHAR
            if hasattr(occupant, "render") and callable(occupant.render):
                try:
                    rendered_char = occupant.render()
                    if rendered_char and len(rendered_char) > 0:
                        char_to_display = rendered_char
                    else:
                        # Optional: Log that render() returned empty/None
                        print(
                            f"Warning: {occupant_type_name} '{getattr(occupant, 'name', 'Unnamed')}'.render() returned empty."
                        )
                        char_to_display = TILE_CHAR  # Fallback to default tile char
                except AttributeError as e:
                    # This specifically catches AttributeError if .name or similar is missing inside render()
                    print(
                        f"Warning: AttributeError in {occupant_type_name}.render(): {e}. Using default char."
                    )
                    char_to_display = TILE_CHAR  # Fallback
                except Exception as e:
                    # Catch any other unexpected errors during render
                    print(
                        f"Warning: Error in {occupant_type_name}.render(): {e}. Using default char."
                    )
                    char_to_display = TILE_CHAR  # Fallback

            # Use specific color if defined, otherwise fallback
            attr_name = ENTITY_COLOR_MAP.get(
                occupant_type_name, ENTITY_COLOR_MAP["default_entity"]
            )
            # We need to use the background version of the palette entry
            return (char_to_display, f"{attr_name}_bg")
        else:
            return (TILE_CHAR, "empty_tile")

    def render_grid_widget(self) -> urwid.Widget:
        """Creates an Urwid widget for the building grid."""
        grid_width = self.building.internal_dims[0]
        grid_height = self.building.internal_dims[1]

        text_markup = []
        for y in range(grid_height):
            row_markup = []
            for x in range(grid_width):
                tile = self.building.get_tile((x, y))
                char, attr = self._get_tile_display(tile)
                row_markup.append((attr, char))
            if y < grid_height - 1:
                row_markup.append("\n")  # Add newline for all but the last row
            text_markup.extend(row_markup)

        grid_text = urwid.Text(text_markup)

        # To center the grid, we can wrap it in a Filler or Overlay
        # For Overlay, we need to specify width and height for the grid_text
        # Let's assume each character is 1 cell wide.
        # For TILE_CHAR = "█", it might take more, depending on font.
        # For simplicity, let's use a fixed width for each cell for now.

        content_width = grid_width
        content_height = grid_height

        # Wrap the text in a Filler. The Filler will handle providing the correct
        # size to the Text widget. Then LineBox it.
        from urwid import BoxAdapter

        filled_text = BoxAdapter(
            urwid.Filler(grid_text, valign="top"), height=content_height
        )
        lined_box = urwid.LineBox(filled_text, title=self.building.name)

        # Center using Overlay
        # Overlay positions the widget against a backdrop (another widget, often a SolidFill)
        # and can align it.
        centered_widget = urwid.Overlay(
            lined_box,
            urwid.SolidFill(" "),  # Background character for the terminal
            align="center",
            width=(
                "relative",
                80,
            ),  # Or a fixed width, e.g., content_width + 2 for LineBox borders
            valign="middle",
            height=("relative", 80),  # Or a fixed height, e.g., content_height + 2
        )
        return centered_widget

    def run(self) -> None:
        """Runs the Urwid main loop to display the building grid."""
        main_widget = self.render_grid_widget()

        def unhandled_input(key: Any) -> None:
            if key in ("q", "Q", "esc"):
                raise urwid.ExitMainLoop()

        loop = urwid.MainLoop(
            main_widget, screen=self.screen, unhandled_input=unhandled_input
        )
        loop.run()


if __name__ == "__main__":
    # This is placeholder for testing.
    # You'll need to create a World, Province, and Building instance.
    print(
        "To test BuildingRenderer, you need to instantiate it with a Building object."
    )
    print("Example (conceptual):")
    print("# from world import World")
    print(
        "# world_instance = World(world_dims=(1,1), region_dims=(1,1), province_dims=(10,10))"
    )
    print("# region = world_instance.create_region((0,0))")
    print("# province = region.subregions[0][0]")
    print(
        "# house = province.create_building('Test House', (0,0), internal_dims=(5,5))"
    )

    # Example: Add a character to a tile
    # print("# character = Character(name='TestChar', world=world_instance)")
    # print("# tile_to_occupy = house.get_tile((1,1))")
    # print("# if tile_to_occupy:")
    # print("#     # This is conceptual. Actual mechanism for placing character needs to be used.")
    # print("#     world_instance.add_edge(character, tile_to_occupy, 'is_located_at') ")
    # print("#     world_instance.add_edge(tile_to_occupy, character, 'is_occupied_by')")

    print("# renderer = BuildingRenderer(house)")
    print("# renderer.run()")
    pass
