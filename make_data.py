from datasets import load_dataset
import json
import os


def make_characters():
    if not os.path.exists("data"):
        os.makedirs("data")
    dataset = load_dataset("IlyaGusev/gpt_roleplay_realm", split="en")
    for character in dataset:
        name = character["name"]
        file_name = f"{name.lower().replace(' ', '_')}.json"
        context = character["context"]
        topics = character["topics"]

        character_data = {
            "name": name,
            "context": context,
            "topics": topics,
        }
        json.dump(character_data, open(os.path.join("data", file_name), "w"))
        break


make_characters()
