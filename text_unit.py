# %% markdown
# - [ ] Slice up an input corpus into a series of TextUnits, which act as analyzable units for the rest of the process, and provide fine-grained references in our outputs.
# %%
from typing import List
import nltk

# %%


class TextUnit:
    def __init__(
        self, content: str, unit_type: str, start: int, end: int, index: int
    ) -> None:
        self.content = content
        self.unit_type = unit_type
        self.start = start
        self.end = end
        self.index = index

    @staticmethod
    def get_text_units(corpus: str) -> List["TextUnit"]:
        sentences = nltk.sent_tokenize(corpus)
        units = []
        start = 0
        for i, sentence in enumerate(sentences):
            end = start + len(sentence)
            units.append(TextUnit(sentence.strip(), "sentence", start, end, i))
            start = end + 1
        return units


corpus = """
**Setting:** The bustling, yet slightly seedier, port town of *Saltwind*. The air is thick with the smell of brine, fish, and exotic spices. A grand annual *Sailor's Moon Festival* is underway, filling the streets with music, food stalls, and boisterous crowds.

**The Twelve Characters:**

1.  **Captain Elara "Stormblade" Vane:** A renowned (and slightly arrogant) human pirate captain, known for her daring raids and sharp wit. She arrives in Saltwind seeking to resupply and enjoy the festival's festivities with her loyal first mate.
2.  **Finnigan "Flicker" Quickfoot:** A nimble halfling rogue and information broker, always with an ear to the ground and a few secrets up his sleeve. He's mingling in the crowds, looking for lucrative opportunities.
3.  **Sister Agnes:** A devout human cleric of a benevolent sea goddess, recently arrived in Saltwind to establish a new temple and offer aid to the needy. She's trying to navigate the chaotic festival.
4.  **Grizelda "Grizz" Ironhand:** A gruff dwarven blacksmith and former adventurer, now running a forge in Saltwind. She prefers the company of metal to people but can't entirely ignore the festival noise.
5.  **Lysandra Meadowlight:** A graceful half-elf bard, traveling the coast, collecting local legends and performing enchanting melodies. She's currently entertaining a small crowd in the town square.
6.  **Kaelen Stoneheart:** A stoic human ranger, tracking a dangerous beast that has been sighted near the town's outskirts. He's trying to gather information without getting caught up in the celebrations.
7.  **Zephyr Whisperwind:** A reclusive gnome wizard, studying ancient runes etched on some of the older buildings in Saltwind, seemingly oblivious to the surrounding revelry.
8.  **Jaxx "The Shadow" Thorne:** A mysterious tiefling sorcerer, cloaked and hooded, observing the crowds from the shadows of a dimly lit alleyway. His motives are unclear.
9.  **Bronwyn Battleaxe:** A boisterous human mercenary captain, leading a small company of sellswords who are in Saltwind seeking new contracts and enjoying the festival's ale.
10. **Silas Blackwood:** A wealthy and influential human merchant, known for his shrewd business dealings and connections to various factions within the town. He's hosting a private gathering at his opulent manor overlooking the harbor.
11. **Rhea Swiftcurrent:** A young and eager half-orc sailor, newly arrived in Saltwind seeking her first proper ship assignment. She's wide-eyed at the festival's grandeur.
12. **Elder Theron Oakheart:** A wise and enigmatic human druid, who has journeyed to Saltwind due to unusual disturbances in the local flora and fauna, which seem to coincide with the festival.

**Narrative Outline & Potential Plot Premise:**

**Act I: Convergence and Initial Interactions**

* **The Festival as a Catalyst:** The Sailor's Moon Festival brings these diverse individuals together in Saltwind. Their initial interactions are largely coincidental and driven by their personal goals:
    * Captain Vane might try to recruit some strong arms (potentially Bronwyn's mercenaries or even a capable-looking Rhea).
    * Flicker could be attempting to pickpocket Silas or gather rumors from the drunken sailors.
    * Sister Agnes might offer aid to someone injured in the festival crowds, perhaps encountering Grizz who needs a specific tool repaired.
    * Lysandra's music might attract the attention of Kaelen, who seeks information about the beast, or even the withdrawn Zephyr.
    * Jaxx could be subtly observing Silas's manor or any displays of magical energy.
    * Elder Theron might question locals about the strange behavior of the wildlife, perhaps encountering Kaelen or even the knowledgeable Grizz.
    * Rhea could be trying to impress Captain Vane or Bronwyn for a spot on their crew.
* **Seeds of Conflict/Mystery:** Small, seemingly unrelated incidents begin to occur during the festival:
    * A valuable artifact goes missing from Silas's manor.
    * Strange symbols are found scrawled on buildings (perhaps noticed by Zephyr).
    * Whispers of unusual magical occurrences or unsettling dreams spread among the townsfolk (potentially sensed by Sister Agnes or Jaxx).
    * Kaelen's quarry makes a bolder appearance within the town limits, causing minor chaos.

**Act II: Rising Action and Intertwined Paths**

* **The Catalyst Incident:** A more significant event occurs that forces some of the characters to interact more directly. This could be:
    * The beast Kaelen is tracking attacks a prominent figure (perhaps Silas or someone close to Captain Vane).
    * The missing artifact is linked to a dangerous prophecy or a powerful magical entity.
    * The strange symbols are revealed to be a warning or a key to something significant.
* **Forming Alliances (and Rivalries):** Characters with shared goals or facing a common threat begin to band together. Others might find themselves in opposition due to conflicting interests or personalities.
    * Captain Vane might reluctantly team up with Kaelen to hunt the beast if it threatens her crew or ship.
    * Flicker's information network could become crucial in uncovering the truth behind the missing artifact or the magical disturbances, potentially leading him to work with Sister Agnes or Lysandra.
    * Zephyr's knowledge of ancient runes might be vital in deciphering the symbols, requiring him to interact with the more worldly characters.
    * Jaxx's shadowy investigations might intersect with Silas's attempts to recover his stolen property, leading to suspicion or conflict.
    * Bronwyn's mercenaries could be hired by Silas or another faction, potentially putting them at odds with other characters.
    * Rhea's eagerness and skills might prove useful to an unexpected group.
    * Elder Theron's connection to nature might reveal a deeper underlying cause for the unsettling events, requiring him to seek help from those with different skills.

**Act III: Climax and Resolution (Open-Ended)**

* **Confrontation and Revelation:** The various plot threads converge, leading to a confrontation with the source of the problems. This could involve:
    * A showdown with a powerful antagonist seeking the artifact or exploiting the magical disturbances.
    * A desperate attempt to contain the rampaging beast.
    * Unraveling a conspiracy that reaches the highest levels of Saltwind society (potentially involving Silas).
* **Character Choices and Consequences:** The individual characters face difficult choices that will shape their futures and their relationships with each other.
    * Will Captain Vane prioritize her crew or the greater good?
    * Will Flicker betray his newfound allies for personal gain?
    * Will Sister Agnes's faith be tested by the darkness they face?
    * Will Grizz put aside her solitude to help?
    * Will Lysandra's music hold the key to calming the chaos?
    * Will Kaelen fulfill his duty as a ranger, even if it means sacrificing personal connections?
    * Will Zephyr's arcane knowledge be enough to save them?
    * Will Jaxx reveal his true motives and allegiance?
    * Will Bronwyn's loyalty be bought or earned?
    * Will Silas's influence be a help or a hindrance?
    * Will Rhea find her place among these seasoned individuals?
    * Will Elder Theron restore the balance of nature?
* **Open Endings:** The scenario could conclude with some plot threads resolved while others remain open, leaving room for future adventures and the continued development of the characters' relationships. Perhaps the artifact is recovered but its true purpose is still unknown. Maybe the beast is defeated, but signs of a larger threat remain. Or perhaps the characters, having faced adversity together, form a lasting (if sometimes uneasy) alliance.
"""
units = TextUnit.get_text_units(corpus)
print(len(units))
# %%
