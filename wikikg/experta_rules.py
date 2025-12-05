"""Experta inference rules for WikiKG knowledge graph.

Contains auto-generated transitive and implication rules for reasoning
over the knowledge graph.
"""

from __future__ import annotations

from experta import KnowledgeEngine, Rule, AS, MATCH, NOT

from .experta_facts import RelationFact


class WikiKGEngine(KnowledgeEngine):
    """Knowledge engine with transitive inference rules.

    Implements common inference patterns:
    - Transitive closure for IS_A, TYPE_OF, PART_OF, SUBCLASS_OF, etc.
    - Inverse relations (HAS_PART <-> PART_OF)
    - Implication rules (CAUSES -> INFLUENCES)
    """

    # Transitive IS_A: A is-a B, B is-a C => A is-a C
    @Rule(
        AS.r1 << RelationFact(rel_type="IS_A", subject=MATCH.a, obj=MATCH.b),
        AS.r2 << RelationFact(rel_type="IS_A", subject=MATCH.b, obj=MATCH.c),
        NOT(RelationFact(rel_type="IS_A", subject=MATCH.a, obj=MATCH.c))
    )
    def transitive_is_a(self, r1, r2, a, c):
        """Infer transitive IS_A relationships."""
        self.declare(RelationFact(rel_type="IS_A", subject=a, obj=c))

    # Transitive TYPE_OF
    @Rule(
        AS.r1 << RelationFact(rel_type="TYPE_OF", subject=MATCH.a, obj=MATCH.b),
        AS.r2 << RelationFact(rel_type="TYPE_OF", subject=MATCH.b, obj=MATCH.c),
        NOT(RelationFact(rel_type="TYPE_OF", subject=MATCH.a, obj=MATCH.c))
    )
    def transitive_type_of(self, r1, r2, a, c):
        """Infer transitive TYPE_OF relationships."""
        self.declare(RelationFact(rel_type="TYPE_OF", subject=a, obj=c))

    # Transitive PART_OF
    @Rule(
        AS.r1 << RelationFact(rel_type="PART_OF", subject=MATCH.a, obj=MATCH.b),
        AS.r2 << RelationFact(rel_type="PART_OF", subject=MATCH.b, obj=MATCH.c),
        NOT(RelationFact(rel_type="PART_OF", subject=MATCH.a, obj=MATCH.c))
    )
    def transitive_part_of(self, r1, r2, a, c):
        """Infer transitive PART_OF relationships."""
        self.declare(RelationFact(rel_type="PART_OF", subject=a, obj=c))

    # Transitive SUBCLASS_OF
    @Rule(
        AS.r1 << RelationFact(rel_type="SUBCLASS_OF", subject=MATCH.a, obj=MATCH.b),
        AS.r2 << RelationFact(rel_type="SUBCLASS_OF", subject=MATCH.b, obj=MATCH.c),
        NOT(RelationFact(rel_type="SUBCLASS_OF", subject=MATCH.a, obj=MATCH.c))
    )
    def transitive_subclass_of(self, r1, r2, a, c):
        """Infer transitive SUBCLASS_OF relationships."""
        self.declare(RelationFact(rel_type="SUBCLASS_OF", subject=a, obj=c))

    # Transitive LOCATED_IN
    @Rule(
        AS.r1 << RelationFact(rel_type="LOCATED_IN", subject=MATCH.a, obj=MATCH.b),
        AS.r2 << RelationFact(rel_type="LOCATED_IN", subject=MATCH.b, obj=MATCH.c),
        NOT(RelationFact(rel_type="LOCATED_IN", subject=MATCH.a, obj=MATCH.c))
    )
    def transitive_located_in(self, r1, r2, a, c):
        """Infer transitive LOCATED_IN relationships."""
        self.declare(RelationFact(rel_type="LOCATED_IN", subject=a, obj=c))

    # Transitive MEMBER_OF
    @Rule(
        AS.r1 << RelationFact(rel_type="MEMBER_OF", subject=MATCH.a, obj=MATCH.b),
        AS.r2 << RelationFact(rel_type="MEMBER_OF", subject=MATCH.b, obj=MATCH.c),
        NOT(RelationFact(rel_type="MEMBER_OF", subject=MATCH.a, obj=MATCH.c))
    )
    def transitive_member_of(self, r1, r2, a, c):
        """Infer transitive MEMBER_OF relationships."""
        self.declare(RelationFact(rel_type="MEMBER_OF", subject=a, obj=c))

    # HAS_PART inverse of PART_OF
    @Rule(
        AS.r << RelationFact(rel_type="HAS_PART", subject=MATCH.a, obj=MATCH.b),
        NOT(RelationFact(rel_type="PART_OF", subject=MATCH.b, obj=MATCH.a))
    )
    def has_part_implies_part_of(self, r, a, b):
        """HAS_PART(A, B) implies PART_OF(B, A)."""
        self.declare(RelationFact(rel_type="PART_OF", subject=b, obj=a))

    # PART_OF inverse to HAS_PART
    @Rule(
        AS.r << RelationFact(rel_type="PART_OF", subject=MATCH.a, obj=MATCH.b),
        NOT(RelationFact(rel_type="HAS_PART", subject=MATCH.b, obj=MATCH.a))
    )
    def part_of_implies_has_part(self, r, a, b):
        """PART_OF(A, B) implies HAS_PART(B, A)."""
        self.declare(RelationFact(rel_type="HAS_PART", subject=b, obj=a))

    # CAUSES implies INFLUENCES
    @Rule(
        AS.r << RelationFact(rel_type="CAUSES", subject=MATCH.a, obj=MATCH.b),
        NOT(RelationFact(rel_type="INFLUENCES", subject=MATCH.a, obj=MATCH.b))
    )
    def causes_implies_influences(self, r, a, b):
        """CAUSES(A, B) implies INFLUENCES(A, B)."""
        self.declare(RelationFact(rel_type="INFLUENCES", subject=a, obj=b))

    # CAUSED_BY inverse of CAUSES
    @Rule(
        AS.r << RelationFact(rel_type="CAUSED_BY", subject=MATCH.a, obj=MATCH.b),
        NOT(RelationFact(rel_type="CAUSES", subject=MATCH.b, obj=MATCH.a))
    )
    def caused_by_implies_causes(self, r, a, b):
        """CAUSED_BY(A, B) implies CAUSES(B, A)."""
        self.declare(RelationFact(rel_type="CAUSES", subject=b, obj=a))

    # PARENT_OF inverse of CHILD_OF
    @Rule(
        AS.r << RelationFact(rel_type="PARENT_OF", subject=MATCH.a, obj=MATCH.b),
        NOT(RelationFact(rel_type="CHILD_OF", subject=MATCH.b, obj=MATCH.a))
    )
    def parent_of_implies_child_of(self, r, a, b):
        """PARENT_OF(A, B) implies CHILD_OF(B, A)."""
        self.declare(RelationFact(rel_type="CHILD_OF", subject=b, obj=a))

    # CHILD_OF inverse of PARENT_OF
    @Rule(
        AS.r << RelationFact(rel_type="CHILD_OF", subject=MATCH.a, obj=MATCH.b),
        NOT(RelationFact(rel_type="PARENT_OF", subject=MATCH.b, obj=MATCH.a))
    )
    def child_of_implies_parent_of(self, r, a, b):
        """CHILD_OF(A, B) implies PARENT_OF(B, A)."""
        self.declare(RelationFact(rel_type="PARENT_OF", subject=b, obj=a))

    # CREATED_BY inverse of CREATOR_OF
    @Rule(
        AS.r << RelationFact(rel_type="CREATED_BY", subject=MATCH.a, obj=MATCH.b),
        NOT(RelationFact(rel_type="CREATOR_OF", subject=MATCH.b, obj=MATCH.a))
    )
    def created_by_implies_creator_of(self, r, a, b):
        """CREATED_BY(A, B) implies CREATOR_OF(B, A)."""
        self.declare(RelationFact(rel_type="CREATOR_OF", subject=b, obj=a))

    # CREATOR_OF inverse of CREATED_BY
    @Rule(
        AS.r << RelationFact(rel_type="CREATOR_OF", subject=MATCH.a, obj=MATCH.b),
        NOT(RelationFact(rel_type="CREATED_BY", subject=MATCH.b, obj=MATCH.a))
    )
    def creator_of_implies_created_by(self, r, a, b):
        """CREATOR_OF(A, B) implies CREATED_BY(B, A)."""
        self.declare(RelationFact(rel_type="CREATED_BY", subject=b, obj=a))

    # Symmetric SIBLING_OF
    @Rule(
        AS.r << RelationFact(rel_type="SIBLING_OF", subject=MATCH.a, obj=MATCH.b),
        NOT(RelationFact(rel_type="SIBLING_OF", subject=MATCH.b, obj=MATCH.a))
    )
    def symmetric_sibling_of(self, r, a, b):
        """SIBLING_OF is symmetric."""
        self.declare(RelationFact(rel_type="SIBLING_OF", subject=b, obj=a))

    # Symmetric SPOUSE_OF
    @Rule(
        AS.r << RelationFact(rel_type="SPOUSE_OF", subject=MATCH.a, obj=MATCH.b),
        NOT(RelationFact(rel_type="SPOUSE_OF", subject=MATCH.b, obj=MATCH.a))
    )
    def symmetric_spouse_of(self, r, a, b):
        """SPOUSE_OF is symmetric."""
        self.declare(RelationFact(rel_type="SPOUSE_OF", subject=b, obj=a))

    # Symmetric BORDERS
    @Rule(
        AS.r << RelationFact(rel_type="BORDERS", subject=MATCH.a, obj=MATCH.b),
        NOT(RelationFact(rel_type="BORDERS", subject=MATCH.b, obj=MATCH.a))
    )
    def symmetric_borders(self, r, a, b):
        """BORDERS is symmetric."""
        self.declare(RelationFact(rel_type="BORDERS", subject=b, obj=a))

    # Symmetric EQUIVALENT_TO
    @Rule(
        AS.r << RelationFact(rel_type="EQUIVALENT_TO", subject=MATCH.a, obj=MATCH.b),
        NOT(RelationFact(rel_type="EQUIVALENT_TO", subject=MATCH.b, obj=MATCH.a))
    )
    def symmetric_equivalent_to(self, r, a, b):
        """EQUIVALENT_TO is symmetric."""
        self.declare(RelationFact(rel_type="EQUIVALENT_TO", subject=b, obj=a))

    # Symmetric SAME_AS
    @Rule(
        AS.r << RelationFact(rel_type="SAME_AS", subject=MATCH.a, obj=MATCH.b),
        NOT(RelationFact(rel_type="SAME_AS", subject=MATCH.b, obj=MATCH.a))
    )
    def symmetric_same_as(self, r, a, b):
        """SAME_AS is symmetric."""
        self.declare(RelationFact(rel_type="SAME_AS", subject=b, obj=a))


def create_engine() -> WikiKGEngine:
    """Create and return a new WikiKGEngine instance."""
    return WikiKGEngine()
