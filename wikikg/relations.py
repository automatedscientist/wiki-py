"""Relation types for WikiKG knowledge graph.

Contains 80+ relation types across multiple domains:
- General: TYPE_OF, IS_A, HAS_PART, PART_OF, etc.
- Biography: BORN_IN, DIED_IN, OCCUPATION, etc.
- History: PRECEDED_BY, FOLLOWED_BY, etc.
- Geography: LOCATED_IN, CAPITAL_OF, etc.
- And more...
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .entities import Entity


class Relation:
    """Base class for all relation types.

    Relations represent typed edges between entities in the knowledge graph.
    """
    relation_type: str = "RELATION"

    def __init__(self, subject: Entity, obj: Entity):
        self.subject = subject
        self.obj = obj

    def __repr__(self) -> str:
        return f"{self.relation_type}({self.subject.name!r}, {self.obj.name!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Relation):
            return (
                self.relation_type == other.relation_type
                and self.subject == other.subject
                and self.obj == other.obj
            )
        return False

    def __hash__(self) -> int:
        return hash((self.relation_type, self.subject.name, self.obj.name))


# Complete list of relation types organized by domain
RELATION_TYPES = [
    # General/Foundational
    "TYPE_OF",
    "IS_A",
    "HAS_PART",
    "PART_OF",
    "INSTANCE_OF",
    "SUBCLASS_OF",
    "MEMBER_OF",
    "BELONGS_TO",
    "CONTAINS",
    "COMPRISES",
    "RELATED_TO",
    "SIMILAR_TO",
    "DIFFERENT_FROM",
    "OPPOSITE_OF",
    "EQUIVALENT_TO",
    "SAME_AS",

    # Causation/Influence
    "CAUSES",
    "CAUSED_BY",
    "INFLUENCES",
    "INFLUENCED_BY",
    "AFFECTS",
    "AFFECTED_BY",
    "ENABLES",
    "PREVENTS",
    "REQUIRES",
    "DEPENDS_ON",
    "RESULTS_IN",

    # Temporal/Sequence
    "PRECEDED_BY",
    "FOLLOWED_BY",
    "BEFORE",
    "AFTER",
    "DURING",
    "CONTEMPORARY_OF",
    "SUCCEEDS",
    "PRECEDES",

    # Spatial/Geography
    "LOCATED_IN",
    "LOCATION_OF",
    "CAPITAL_OF",
    "HAS_CAPITAL",
    "BORDERS",
    "NEAR",
    "CONTAINS_LOCATION",
    "REGION_OF",
    "COUNTRY_OF",
    "CONTINENT_OF",

    # Biography/People
    "BORN_IN",
    "DIED_IN",
    "BIRTHPLACE_OF",
    "DEATHPLACE_OF",
    "NATIONALITY",
    "CITIZEN_OF",
    "OCCUPATION",
    "PROFESSION_OF",
    "EMPLOYER",
    "EMPLOYED_BY",
    "EDUCATED_AT",
    "ALMA_MATER_OF",
    "SPOUSE_OF",
    "CHILD_OF",
    "PARENT_OF",
    "SIBLING_OF",
    "RELATIVE_OF",
    "FRIEND_OF",
    "COLLEAGUE_OF",
    "MENTOR_OF",
    "STUDENT_OF",

    # Creation/Production
    "CREATED_BY",
    "CREATOR_OF",
    "INVENTED_BY",
    "INVENTOR_OF",
    "DISCOVERED_BY",
    "DISCOVERER_OF",
    "FOUNDED_BY",
    "FOUNDER_OF",
    "AUTHORED_BY",
    "AUTHOR_OF",
    "COMPOSED_BY",
    "COMPOSER_OF",
    "DIRECTED_BY",
    "DIRECTOR_OF",
    "PRODUCED_BY",
    "PRODUCER_OF",
    "DESIGNED_BY",
    "DESIGNER_OF",
    "BUILT_BY",
    "BUILDER_OF",

    # Organization/Institution
    "AFFILIATED_WITH",
    "HEADQUARTERS_IN",
    "SUBSIDIARY_OF",
    "PARENT_COMPANY_OF",
    "DEPARTMENT_OF",
    "DIVISION_OF",
    "BRANCH_OF",
    "LED_BY",
    "LEADER_OF",
    "CEO_OF",
    "PRESIDENT_OF",
    "CHAIRMAN_OF",

    # Science/Technology
    "DERIVES_FROM",
    "DERIVED_FROM",
    "APPLIED_TO",
    "APPLICATION_OF",
    "FIELD_OF",
    "STUDIES",
    "STUDIED_BY",
    "PROPERTY_OF",
    "HAS_PROPERTY",
    "CHARACTERISTIC_OF",
    "COMPOSED_OF",
    "MADE_OF",
    "MATERIAL_OF",

    # Biology
    "SPECIES_OF",
    "GENUS_OF",
    "FAMILY_OF",
    "ORDER_OF",
    "CLASS_OF",
    "PHYLUM_OF",
    "KINGDOM_OF",
    "HABITAT_OF",
    "LIVES_IN",
    "EATS",
    "EATEN_BY",
    "PREDATOR_OF",
    "PREY_OF",
    "SYMBIONT_OF",
    "HOST_OF",
    "PARASITE_OF",

    # Arts/Culture
    "GENRE_OF",
    "STYLE_OF",
    "MOVEMENT_OF",
    "INSPIRED_BY",
    "INSPIRATION_FOR",
    "PERFORMED_BY",
    "PERFORMER_OF",
    "PORTRAYED_BY",
    "PORTRAYAL_OF",
    "ADAPTATION_OF",
    "BASED_ON",

    # Economics/Business
    "CURRENCY_OF",
    "TRADED_IN",
    "MARKET_OF",
    "INDUSTRY_OF",
    "SECTOR_OF",
    "PRODUCT_OF",
    "MANUFACTURER_OF",
    "SUPPLIER_OF",
    "CUSTOMER_OF",
    "COMPETITOR_OF",

    # Politics/Government
    "GOVERNED_BY",
    "GOVERNS",
    "CAPITAL_OF",
    "OFFICIAL_LANGUAGE_OF",
    "MEMBER_STATE_OF",
    "SIGNATORY_OF",
    "ALLY_OF",
    "ENEMY_OF",
    "COLONY_OF",
    "COLONIZED_BY",

    # Events
    "TIMELINE_EVENT",
    "PARTICIPANT_IN",
    "OCCURRED_IN",
    "VENUE_OF",
    "ORGANIZER_OF",
    "WINNER_OF",
    "AWARD_FOR",

    # Language/Linguistics
    "LANGUAGE_OF",
    "SPOKEN_IN",
    "WRITTEN_IN",
    "TRANSLATED_FROM",
    "TRANSLATION_OF",
    "ETYMOLOGY_OF",
    "DERIVED_WORD_OF",

    # Mathematics
    "THEOREM_OF",
    "PROOF_OF",
    "COROLLARY_OF",
    "GENERALIZATION_OF",
    "SPECIAL_CASE_OF",
    "DUAL_OF",
    "INVERSE_OF",

    # Additional relations found in wiki-kg-dataset (high frequency)
    "VARIANT_OF",
    "ASSOCIATED_WITH",
    "POSITION_HELD",
    "FIELD_OF_WORK",
    "AWARDED_TO",
    "COMPONENT_OF",
    "BORN_ON",
    "WORKS_FOR",
    "GENRE",
    "COLLABORATED_WITH",
    "DIED_ON",
    "CATEGORY_OF",
    "COMPETED_IN",
    "USED_FOR",
    "RESIDED_IN",
    "ALMA_MATER",
    "RECIPIENT_OF",
    "MANAGED_BY",
    "USED_IN",
    "ETHNICITY",
    "SUCCEEDED_BY",
    "SUBSET_OF",
    "PUBLISHED_BY",
    "USES",
    "PLAYS_FOR_TEAM",
    "CONTEMPORARY_WITH",
    "RELIGION",
    "WON_MEDAL_AT",
    "ALBUM_OF",
    "USED_BY",
    "BURIED_AT",
    "ADAPTED_FROM",
    "SUBJECT_OF",
    "RELEASED_ON_LABEL",
    "TRACK_OF",
    "ALIAS_OF",
    "CONTRIBUTED_TO",
    "HEADQUARTERED_IN",
    "MARRIED_TO",
    "NAMED_AFTER",
    "SERVED_IN",
    "NOMINATED_FOR",
    "MENTIONED_IN",
    "RELATION_OF",
    "AWARDED_RANK",
    "KNOWN_FOR",
    "OBSERVED_BY",
    "ABOUT",
    "INDUSTRY",
    "CITES",
    "CASES",
    "SUPPORTED_BY",
    "FOCUSES_ON",
    "ACQUIRED_BY",
    "COACH_OF",
    "SUPPORTS",
    "SPECIALIZATION_OF",
    "FLOWS_INTO",
    "COMMANDER_OF",
    "WEST_OF",
    "OPPOSED_BY",
    "INVOLVED_IN",
    "MENTIONED_BY",
    "TRANSLATED_BY",
    "NORTH_OF",
    "OPPOSED_TO",
    "USED_WITH",
    "CITED_BY",
    "APPLIES_TO",
    "TREATS",
    "SOUTH_OF",
    "INSPIRES",
    "DEFINED_BY",
    "TRIBUTARY_OF",
    "EAST_OF",
    "CONSUMES",
    "DESCRIBED_BY",
    "WITHIN",
    "REACTS_WITH",
    "OPPOSES",
    "BETWEEN",
    "EPISODE_OF",
    "CONTEXT_OF",
    "APPLIED_IN",
    "CONFLICT_WITH",
    "REPRESENTS",
    "TREATED_BY",
    "RECORDED_AT",
    "FEEDS_ON",
    "FUNCTION_OF",
    "ALSO_KNOWN_AS",
    "POPULAR_IN",
    "OWNED_BY",
    "PRODUCES",
    "TOPIC_OF",
    "POPULATION",
    "TRAVELED_TO",
    "ALLIED_WITH",
    "CONNECTED_TO",
    "COMPETES_WITH",
    "OVERLAPS_WITH",
    "RESULT_OF",
    "RECOGNIZED_BY",
    "MERGED_WITH",
    "INFLUENCED",
    "EVENT_OF",
    "INTERACTS_WITH",
    "APPEARS_IN",
    "INTRODUCED_BY",
    "PARTNER_OF",
    "OCCUPIED_BY",
    "PROTECTS_AGAINST",
    "INVOLVES",
    "COMPARED_WITH",
    "PROTECTS",
    "ILLUSTRATED_BY",
    "COORDINATES",
    "SET_IN",
    "RESPONSIBLE_FOR",
    "REPLACED_BY",
    "SPINOFF_OF",
    "CREATES",
    "PROVIDES",
    "ANALOGOUS_TO",
    "CONNECTS",
    "OWNER_OF",
    "APPOINTED_BY",
    "TEACHES",
    "TREATMENT_FOR",
    "POWERED_BY",
    "MECHANISM_OF",
    "MAINTAINS",
    "FEATURES",
    "PROPOSED_BY",
]


def _create_relation_class(name: str) -> type[Relation]:
    """Create a relation class with the given name."""
    return type(name, (Relation,), {"relation_type": name})


# Dynamically create all relation classes and add to module globals
for _rel_name in RELATION_TYPES:
    globals()[_rel_name] = _create_relation_class(_rel_name)


def get_relation_class(name: str) -> type[Relation] | None:
    """Get a relation class by name."""
    return globals().get(name)


def all_relation_types() -> list[str]:
    """Return list of all defined relation type names."""
    return RELATION_TYPES.copy()


# Export all relation classes
__all__ = ["Relation", "RELATION_TYPES", "get_relation_class", "all_relation_types"] + RELATION_TYPES
