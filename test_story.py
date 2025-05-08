# %%
import time
from importlib import reload
import random

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
from building_renderer import BuildingRenderer

# %%
use_cached_story = True
random.seed(42)
world = World(
    world_dims=(3, 3),
    region_dims=(4, 4),
    province_dims=(10, 10),
)
region = world.create_region((0, 0))
province = region.subregions[0][0]
house = province.create_building("House", (0, 0), (10, 10))

storyteller = Storyteller(world)
# %%
if not use_cached_story:
    characters_story = storyteller.generate_character_story()
    print(characters_story)
else:
    characters_story = "## The Genesis of the Simulation: A Historical Account of the Initial Inhabitants\n\nThis volume chronicles the early days of the Simulated Habitation Experiment, focusing on the four individuals selected to inaugurate this groundbreaking endeavor. Their histories, meticulously documented, provide invaluable insight into the human condition as it existed prior to the controlled environment of the Simulation.\n\n### Subject Alpha: Elara Vance\n\n**Family History:** The Vance lineage traces its roots back several generations in the bustling metropolis of Veridia. Her father, Thomas Vance, was a respected architect known for his innovative designs, while her mother, Eleanor Vance (née Sterling), hailed from a family of renowned academics. Elara has one younger sibling, a brother named Liam, who pursued a career in engineering.\n\n**Backstory (Up to 0 BGB):** Born in 25 BGB, Elara displayed an early aptitude for the arts. Her childhood was marked by a blend of structured learning and creative exploration. She excelled in her studies, particularly in history and literature, but found her true passion in painting. After completing her formal education, she dedicated herself to her artistic pursuits, exhibiting her work in local galleries and gaining a modest following. The call to participate in the Simulation presented an opportunity for a new form of inspiration and a break from her established routine.\n\n**Physical Attributes:** Elara Vance is a female of Caucasian descent, born in 25 BGB. She possesses a slender build, with striking green eyes and a cascade of auburn hair. Her demeanor is often described as graceful and contemplative.\n\n**Personality:** Elara is characterized by her introspective and sensitive nature. She is deeply empathetic and often finds solace in solitude and creative expression. While generally reserved, she possesses a quiet determination and a strong sense of personal integrity.\n\n**Goals and Motivations:** Elara's primary motivation for entering the Simulation is to seek new experiences and perspectives that will inform her artistic endeavors. She hopes to observe human interaction in a controlled environment and translate these observations into her work. Her goal is to create art that resonates with a deeper understanding of the human psyche.\n\n### Subject Beta: Marcus \"Brick\" O'Connell\n\n**Family History:** The O'Connell family has a long-standing tradition of manual labor and craftsmanship. His father, Patrick O'Connell, was a skilled carpenter, and his mother, Mary O'Connell (née Murphy), worked in the local textile mill. Marcus is the eldest of three siblings, with two younger sisters, Fiona and Siobhan, who followed in their mother's footsteps.\n\n**Backstory (Up to 0 BGB):** Born in 28 BGB, Marcus's early life was shaped by the demands of a working-class family. He learned the value of hard work and practical skills from a young age, assisting his father in various construction projects. While not academically inclined, he possessed a natural talent for building and problem-solving. He spent his formative years working in construction, gaining a reputation for his strength and reliability. The Simulation offered a chance for a different kind of challenge and a potential escape from the physically demanding nature of his profession.\n\n**Physical Attributes:** Marcus O'Connell is a male of Caucasian descent, born in 28 BGB. He is of a robust build, with broad shoulders and calloused hands. His hair is dark and often unkempt, and his eyes are a piercing blue. His nickname, \"Brick,\" is a testament to his sturdy physique.\n\n**Personality:** Marcus is known for his straightforward and pragmatic approach to life. He is loyal, dependable, and possesses a strong sense of duty. While not overly talkative, he is quick to offer assistance and is fiercely protective of those he cares about.\n\n**Goals and Motivations:** Marcus's primary motivation for participating in the Simulation is to provide a better future for himself and potentially his family. He sees it as an opportunity to learn new skills and potentially transition into a less physically demanding career. His goal is to prove his adaptability and resourcefulness in an unfamiliar environment.\n\n### Subject Gamma: Anya Sharma\n\n**Family History:** The Sharma family has a distinguished history in the field of medicine. Her father, Dr. Rajesh Sharma, is a renowned surgeon, and her mother, Dr. Priya Sharma (née Kapoor), is a leading researcher in genetics. Anya is an only child, a fact that often placed significant expectations upon her.\n\n**Backstory (Up to 0 BGB):** Born in 26 BGB, Anya was raised in an environment that prioritized academic excellence and scientific inquiry. She displayed an exceptional intellect from a young age, excelling in all subjects, particularly science and mathematics. She pursued a degree in psychology, driven by a fascination with the complexities of the human mind. Her research focused on social dynamics and group behavior. The Simulation presented a unique opportunity to observe these dynamics in a controlled and isolated setting.\n\n**Physical Attributes:** Anya Sharma is a female of South Asian descent, born in 26 BGB. She is of average height and build, with sharp, intelligent eyes and dark, neatly styled hair. Her demeanor is often described as focused and analytical.\n\n**Personality:** Anya is characterized by her sharp intellect and analytical mind. She is highly observant and possesses a natural curiosity about human behavior. While sometimes perceived as reserved, she is capable of deep empathy and is driven by a desire to understand the world around her.\n\n**Goals and Motivations:** Anya's primary motivation for entering the Simulation is scientific observation and research. She aims to study the psychological and social interactions of the participants in a controlled environment. Her goal is to gather data that will contribute to a greater understanding of human behavior under unique circumstances.\n\n### Subject Delta: Kai Tanaka\n\n**Family History:** The Tanaka family has a history rooted in technology and innovation. His father, Kenji Tanaka, was a software engineer, and his mother, Hiroko Tanaka (née Sato), was a graphic designer. Kai has one older sister, Sakura, who followed in their mother's footsteps.\n\n**Backstory (Up to 0 BGB):** Born in 27 BGB, Kai grew up surrounded by the latest technological advancements. He developed an early fascination with computers and programming, spending countless hours exploring the digital world. He pursued a degree in computer science and quickly established himself as a skilled programmer and system administrator. His interest in the Simulation stemmed from a desire to understand the human element within complex systems and to potentially contribute to the development of future simulated environments.\n\n**Physical Attributes:** Kai Tanaka is a male of East Asian descent, born in 27 BGB. He is of a lean build, with dark, expressive eyes and neatly styled black hair. His demeanor is often described as calm and focused.\n\n**Personality:** Kai is known for his quiet intelligence and meticulous attention to detail. He is a problem-solver by nature and enjoys the challenge of complex systems. While not overtly social, he is capable of forming strong connections with those who share his interests.\n\n**Goals and Motivations:** Kai's primary motivation for participating in the Simulation is to observe and analyze the technical aspects of the environment and the human interaction within it. He is interested in the algorithms and programming that govern the simulation and how human behavior influences these systems. His goal is to gain a deeper understanding of the intersection of technology and human experience.\n\nThese four individuals, with their diverse backgrounds and motivations, represent the initial cohort of the Simulated Habitation Experiment. Their interactions and experiences within the controlled environment will undoubtedly provide invaluable data for future historical analysis of this groundbreaking project."
# %%
function_calls = storyteller.init_story(characters_story)
function_calls
# %%
house.get_tile((3, 1)).get_occupants()
# %%
renderer = BuildingRenderer(house)
renderer.run()
for i in range(10):
    people = world.living_entities
    for p in people:
        p.move()
    world.curr_time += 1
    time.sleep(1)
    renderer.run()
# %%
# %%
