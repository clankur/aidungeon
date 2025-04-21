from enum import Enum


class RelationshipType(Enum):
    # Familial Relationships
    IS_PARENT_OF = "is_parent_of"
    IS_CHILD_OF = "is_child_of"
    IS_SIBLING_OF = "is_sibling_of"
    IS_GRANDPARENT_OF = "is_grandparent_of"
    IS_GRANDCHILD_OF = "is_grandchild_of"
    IS_AUNT_OR_UNCLE_OF = "is_aunt_or_uncle_of"
    IS_NIECE_OR_NEPHEW_OF = "is_niece_or_nephew_of"
    IS_COUSIN_OF = "is_cousin_of"
    IS_SPOUSE_OF = "is_spouse_of"
    IS_LOVER_OF = "is_lover_of"

    # Social Relationships
    HAS_PASSING_ACQUAINTANCE_WITH = "has_passing_acquaintance_with"
    HAS_LONG_TERM_ACQUAINTANCE_WITH = "has_long_term_acquaintance_with"
    IS_FRIEND_OF = "is_friend_of"
    IS_CLOSE_FRIEND_OF = "is_close_friend_of"
    IS_KINDRED_SPIRIT_OF = "is_kindred_spirit_of"
    HAS_GRUDGE_AGAINST = "has_grudge_against"
    IS_COMPANION_OF = "is_companion_of"

    # Professional Relationships
    IS_MENTOR_OF = "is_mentor_of"
    IS_APPRENTICE_OF = "is_apprentice_of"
    IS_FORMER_MENTOR_OF = "is_former_mentor_of"
    IS_FORMER_APPRENTICE_OF = "is_former_apprentice_of"

    # Spiritual Relationships
    WORSHIPS = "worships"
    WORSHIPS_EVIL_ENTITY = "worships_evil_entity"
    IS_MEMBER_OF_RELIGIOUS_GROUP = "is_member_of_religious_group"

    # Group Membership
    IS_MEMBER_OF_GROUP = "is_member_of_group"
    IS_FORMER_MEMBER_OF_GROUP = "is_former_member_of_group"

    # Animal Bonds
    IS_BONDED_WITH_ANIMAL = "is_bonded_with_animal"
    IS_PET_OF = "is_pet_of"

    # Antagonistic / Conflict Relationships
    IS_AT_WAR_WITH = "is_at_war_with"
    IS_ENEMY_OF = "is_enemy_of"
    IS_AGENT_OF = "is_agent_of"
    IS_SPYING_ON = "is_spying_on"
    HAS_COMMITTED_CRIME_AGAINST = "has_committed_crime_against"
    SEEKS_REVENGE_AGAINST = "seeks_revenge_against"


class ActionType(Enum):
    # Warfare and Conflict
    BATTLE = "battle"
    SIEGE = "siege"
    DUEL = "duel"
    CONQUEST = "conquest"
    REVOLT = "revolt"
    ASSASSINATION = "assassination"
    RAID = "raid"
    AMBUSH = "ambush"
    MASSACRE = "massacre"
    EXECUTION = "execution"
    ATTACK = "attack"

    # Political and Diplomatic
    CROWNED = "crowned"
    ABDICATION = "abdication"
    USURPATION = "usurpation"
    TREATY_SIGNED = "treaty_signed"
    ALLIANCE_FORMED = "alliance_formed"
    PEACE_DECLARED = "peace_declared"
    WAR_DECLARED = "war_declared"
    TRIBUTE_PAID = "tribute_paid"
    DIPLOMATIC_MEETING = "diplomatic_meeting"

    # Exploration and Discovery
    DISCOVERED_LOCATION = "discovered_location"
    EXPEDITION_LAUNCHED = "expedition_launched"
    MAP_CREATED = "map_created"

    # Construction and Foundation
    ESTABLISHED_SITE_OR_STRUCTURE = "established_site_or_structure"
    BUILT_INFRASTRUCTURE = "built_infrastructure"

    # Cultural and Religious
    CREATED_ARTIFACT = "created_artifact"
    CREATED_WRITTEN_WORK = "created_written_work"
    PERFORMED_MUSIC = "performed_music"
    ESTABLISHED_RELIGION = "established_religion"
    BUILT_TEMPLE = "built_temple"
    BECAME_PRIEST = "became_priest"
    SACRIFICE_MADE = "sacrifice_made"

    # Technological and Magical
    DISCOVERED_TECHNOLOGY = "discovered_technology"
    LEARNED_SECRET = "learned_secret"
    BECAME_NECROMANCER = "became_necromancer"
    CREATED_GOLEM = "created_golem"
    OPENED_PORTAL = "opened_portal"

    # Natural and Supernatural
    EARTHQUAKE_OCCURRED = "earthquake_occurred"
    VOLCANO_ERUPTED = "volcano_erupted"
    PLAGUE_SPREAD = "plague_spread"
    METEOR_IMPACT = "meteor_impact"
    APPEARANCE_OF_BEAST = "appearance_of_beast"
    DIVINE_INTERVENTION = "divine_intervention"

    # Personal and Miscellaneous
    BIRTH = "birth"
    DEATH = "death"
    MARRIAGE = "marriage"
    DIVORCE = "divorce"
    ADOPTION = "adoption"
    NAME_CHANGED = "name_changed"
    BECAME_HERO = "became_hero"
    RETIRED = "retired"
    WENT_MISSING = "went_missing"
    RETURNED = "returned"


class HistoryPredicateTypes(Enum):
    # Temporal relations
    OCCURRED_ON = "occurred_on"
    OCCURRED_IN = "occurred_in"
    STARTED_ON = "started_on"
    ENDED_ON = "ended_on"
    DURATION = "duration"

    # Spatial relations
    LOCATED_IN = "located_in"
    TOOK_PLACE_IN = "took_place_in"
    MOVED_TO = "moved_to"
    MOVED_FROM = "moved_from"

    # Causal relations
    CAUSED = "caused"
    RESULTED_IN = "resulted_in"
    LED_TO = "led_to"
    MOTIVATED = "motivated"

    # Participation and roles
    PARTICIPATED_IN = "participated_in"
    FOUGHT_IN = "fought_in"
    RULED = "ruled"
    GOVERNED = "governed"
    BELONGED_TO = "belonged_to"
    REPRESENTED = "represented"
    ALLIED_WITH = "allied_with"
    OPPOSED = "opposed"

    # Events and actions
    DECLARED = "declared"
    SIGNED = "signed"
    FOUNDED = "founded"
    BUILT = "built"
    DISCOVERED = "discovered"
    INVENTED = "invented"
    WROTE = "wrote"
    DIED_IN = "died_in"
    BORN_IN = "born_in"
    EXECUTED = "executed"
    ASSASSINATED = "assassinated"

    # Properties and identity
    HELD_TITLE = "held_title"
    WAS = "was"
    KNOWN_FOR = "known_for"
    MEMBER_OF = "member_of"
    DESCENDANT_OF = "descendant_of"
    SUCCESSOR_OF = "successor_of"
    PREDECESSOR_OF = "predecessor_of"

    # Treaties and documents
    SIGNATORY_OF = "signatory_of"
    AUTHORED = "authored"
    CONTAINED_IN = "contained_in"
    RATIFIED = "ratified"

    # Economic and political
    ENACTED = "enacted"
    REFORMED = "reformed"
    TAXED = "taxed"
    TRADED_WITH = "traded_with"
    COLONIZED = "colonized"
    LIBERATED = "liberated"
    OCCUPIED = "occupied"

    # Social and cultural
    INFLUENCED = "influenced"
    BELIEVED_IN = "believed_in"
    PRACTICED = "practiced"
    PERSECUTED = "persecuted"
    MIGRATED_TO = "migrated_to"
    MIGRATED_FROM = "migrated_from"

    # Miscellaneous
    NAMED_AFTER = "named_after"
    RENAMED = "renamed"
    WON = "won"
    LOST = "lost"
