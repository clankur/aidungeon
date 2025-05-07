from relationships import RelationshipType, ActionType, HistoryPredicateTypes

predicates = (
    [item.value for item in RelationshipType]
    + [item.value for item in ActionType]
    + [item.value for item in HistoryPredicateTypes]
)


create_character_declaration = {
    "name": "create_character",
    "description": "Creates a character with the given name, sex, race, location, world, birth_time",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the character",
            },
            "sex": {
                "type": "string",
                "enum": ["male", "female"],
                "description": "The sex of the character",
            },
            "race": {
                "type": "string",
                "enum": ["human", "elf", "dwarf", "orc", "troll"],
                "description": "The race of the character",
            },
            "location": {
                "type": "string",
                "description": "The location of the character",
            },
            "birth_time": {
                "type": "integer",
                "description": "The birth time of the character since the start of the world time 0 BGB (Before the Game Begins)",
            },
        },
        "required": ["name", "sex", "race", "location", "birth_time"],
    },
}

add_edge_declaration = {
    "name": "add_edge",
    "description": "Creates subject-predicate-object relationship between two entities.",
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "The subject of the triple",
            },
            "predicate": {
                "type": "string",
                "enum": predicates,
                "description": "The predicate of the triple",
            },
            "object": {
                "type": "string",
                "description": "The object of the triple",
            },
            "weight": {
                "type": "number",
                "description": "The weighting of an edge. Higher means positive relationship, lower means negative relationship. (Optional: default value = 0.0)",
            },
        },
        "required": ["subject", "predicate", "object"],
    },
}
