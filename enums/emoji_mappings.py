from typing import Dict, Optional, Tuple
from enum import Enum


class CharacterEmoji(Enum):
    # Human races
    HUMAN_MALE = "👨"
    HUMAN_FEMALE = "👩"
    ELF_MALE = "🧝‍♂️"
    ELF_FEMALE = "🧝‍♀️"
    DWARF_MALE = "🧙‍♂️"
    DWARF_FEMALE = "🧙‍♀️"
    ORC_MALE = "👹"
    ORC_FEMALE = "👺"
    HALFLING_MALE = "🧒"
    HALFLING_FEMALE = "👧"

    # Default fallback
    DEFAULT = "👤"


class ItemEmoji(Enum):
    # Weapons
    SWORD = "⚔️"
    BOW = "🏹"
    STAFF = "🪄"
    DAGGER = "🗡️"
    AXE = "🪓"
    SPEAR = "🔱"
    CROSSBOW = "🏹"
    MACE = "⚒️"
    HAMMER = "🔨"
    WHIP = "🪢"

    # Armor
    SHIELD = "🛡️"
    HELMET = "⛑️"
    ARMOR = "🥋"
    BOOTS = "👢"
    GLOVES = "🧤"
    CLOAK = "🧥"
    BELT = "👔"
    RING = "💍"
    AMULET = "📿"
    BRACERS = "⌚"

    # Magic items
    WAND = "🪄"
    SCROLL = "📜"
    POTION = "🧪"
    CRYSTAL = "💎"
    ORB = "🔮"
    TALISMAN = "🧿"
    RUNE = "🔰"
    GEM = "💠"
    BOOK = "📚"
    STAFF_MAGIC = "🪄"

    # Treasure
    GOLD = "💰"
    COIN = "🪙"
    CROWN = "👑"
    CHEST = "🧰"
    BAG = "👜"
    BACKPACK = "🎒"
    POUCH = "👝"
    VASE = "🏺"
    STATUE = "🗿"
    CANDLE = "🕯️"

    # Tools
    KEY = "🔑"
    MAP = "🗺️"
    COMPASS = "🧭"
    TORCH = "🔥"
    LANTERN = "🏮"
    ROPE = "🧶"
    PICKAXE = "⛏️"
    SHOVEL = "🪚"
    SAW = "🪚"
    HAMMER_TOOL = "🔨"

    # Food & Drink
    APPLE = "🍎"
    BREAD = "🍞"
    WATER = "💧"
    WINE = "🍷"
    MEAT = "🍖"
    FISH = "🐟"
    SOUP = "🥣"
    CAKE = "🍰"
    HONEY = "🍯"
    MUSHROOM = "🍄"

    # Materials
    WOOD = "🪵"
    STONE = "🪨"
    IRON = "⚙️"
    GOLD_ORE = "🪙"
    SILVER = "🥈"
    COPPER = "🥉"
    CLOTH = "🧵"
    LEATHER = "🧶"
    BONE = "🦴"
    CRYSTAL_SHARD = "💎"

    # Default fallback
    DEFAULT = "📦"


# Mapping of (race, sex) tuples to emojis
RACE_SEX_TO_EMOJI: Dict[Tuple[str, str], str] = {
    # Humans
    ("human", "male"): CharacterEmoji.HUMAN_MALE.value,
    ("human", "female"): CharacterEmoji.HUMAN_FEMALE.value,
    # Elves
    ("elf", "male"): CharacterEmoji.ELF_MALE.value,
    ("elf", "female"): CharacterEmoji.ELF_FEMALE.value,
    # Dwarves
    ("dwarf", "male"): CharacterEmoji.DWARF_MALE.value,
    ("dwarf", "female"): CharacterEmoji.DWARF_FEMALE.value,
    # Orcs
    ("orc", "male"): CharacterEmoji.ORC_MALE.value,
    ("orc", "female"): CharacterEmoji.ORC_FEMALE.value,
    # Halflings
    ("halfling", "male"): CharacterEmoji.HALFLING_MALE.value,
    ("halfling", "female"): CharacterEmoji.HALFLING_FEMALE.value,
}

# Mapping of item names to emojis
ITEM_TO_EMOJI: Dict[str, str] = {
    # Weapons
    "sword": ItemEmoji.SWORD.value,
    "bow": ItemEmoji.BOW.value,
    "staff": ItemEmoji.STAFF.value,
    "dagger": ItemEmoji.DAGGER.value,
    "axe": ItemEmoji.AXE.value,
    "spear": ItemEmoji.SPEAR.value,
    "crossbow": ItemEmoji.CROSSBOW.value,
    "mace": ItemEmoji.MACE.value,
    "hammer": ItemEmoji.HAMMER.value,
    "whip": ItemEmoji.WHIP.value,
    # Armor
    "shield": ItemEmoji.SHIELD.value,
    "helmet": ItemEmoji.HELMET.value,
    "armor": ItemEmoji.ARMOR.value,
    "boots": ItemEmoji.BOOTS.value,
    "gloves": ItemEmoji.GLOVES.value,
    "cloak": ItemEmoji.CLOAK.value,
    "belt": ItemEmoji.BELT.value,
    "ring": ItemEmoji.RING.value,
    "amulet": ItemEmoji.AMULET.value,
    "bracers": ItemEmoji.BRACERS.value,
    # Magic items
    "wand": ItemEmoji.WAND.value,
    "scroll": ItemEmoji.SCROLL.value,
    "potion": ItemEmoji.POTION.value,
    "crystal": ItemEmoji.CRYSTAL.value,
    "orb": ItemEmoji.ORB.value,
    "talisman": ItemEmoji.TALISMAN.value,
    "rune": ItemEmoji.RUNE.value,
    "gem": ItemEmoji.GEM.value,
    "book": ItemEmoji.BOOK.value,
    "staff_magic": ItemEmoji.STAFF_MAGIC.value,
    # Treasure
    "gold": ItemEmoji.GOLD.value,
    "coin": ItemEmoji.COIN.value,
    "crown": ItemEmoji.CROWN.value,
    "chest": ItemEmoji.CHEST.value,
    "bag": ItemEmoji.BAG.value,
    "backpack": ItemEmoji.BACKPACK.value,
    "pouch": ItemEmoji.POUCH.value,
    "vase": ItemEmoji.VASE.value,
    "statue": ItemEmoji.STATUE.value,
    "candle": ItemEmoji.CANDLE.value,
    # Tools
    "key": ItemEmoji.KEY.value,
    "map": ItemEmoji.MAP.value,
    "compass": ItemEmoji.COMPASS.value,
    "torch": ItemEmoji.TORCH.value,
    "lantern": ItemEmoji.LANTERN.value,
    "rope": ItemEmoji.ROPE.value,
    "pickaxe": ItemEmoji.PICKAXE.value,
    "shovel": ItemEmoji.SHOVEL.value,
    "saw": ItemEmoji.SAW.value,
    "hammer_tool": ItemEmoji.HAMMER_TOOL.value,
    # Food & Drink
    "apple": ItemEmoji.APPLE.value,
    "bread": ItemEmoji.BREAD.value,
    "water": ItemEmoji.WATER.value,
    "wine": ItemEmoji.WINE.value,
    "meat": ItemEmoji.MEAT.value,
    "fish": ItemEmoji.FISH.value,
    "soup": ItemEmoji.SOUP.value,
    "cake": ItemEmoji.CAKE.value,
    "honey": ItemEmoji.HONEY.value,
    "mushroom": ItemEmoji.MUSHROOM.value,
    # Materials
    "wood": ItemEmoji.WOOD.value,
    "stone": ItemEmoji.STONE.value,
    "iron": ItemEmoji.IRON.value,
    "gold_ore": ItemEmoji.GOLD_ORE.value,
    "silver": ItemEmoji.SILVER.value,
    "copper": ItemEmoji.COPPER.value,
    "cloth": ItemEmoji.CLOTH.value,
    "leather": ItemEmoji.LEATHER.value,
    "bone": ItemEmoji.BONE.value,
    "crystal_shard": ItemEmoji.CRYSTAL_SHARD.value,
}


def get_character_emoji(race: str, sex: str) -> str:
    """Get the emoji for a character based on their race and sex."""
    key = (race.lower(), sex.lower())
    return RACE_SEX_TO_EMOJI.get(key, CharacterEmoji.DEFAULT.value)


def get_item_emoji(item_name: str) -> str:
    """Get the emoji for an item based on its name."""
    return ITEM_TO_EMOJI.get(item_name.lower(), ItemEmoji.DEFAULT.value)
