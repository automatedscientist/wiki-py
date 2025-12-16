# WikiKG

Python package for Wikipedia knowledge graphs. Converts Mathematica-based knowledge graph definitions to Python, provides SQLite persistence, and integrates with Experta for rule-based inference.

**Python >= 3.10** | **45,416 Wikipedia articles** | **180+ relation types**

## Overview

WikiKG provides:

- **Knowledge Graph Storage**: SQLite-backed persistence for entities, relations, properties, and events
- **Mathematica Converter**: Transform `.m` files from wiki-kg-dataset to Python
- **Citation Tracking**: Source attribution for all facts
- **Dataset Generation**: Export to JSONL for LLM training
- **Experta Integration**: Rule-based reasoning with transitive inference

## Installation

```bash
uv sync
```

**Dependencies**: datasets, experta, pydantic>=2.0, tqdm

## Quick Start

```python
from wikikg import *

# Initialize database
init_db("knowledge.db")

# Create entities
Wing = AddEntity("Wing", {})
Fin = AddEntity("Fin", {})

# Set properties with citations
SetPropertyCited(
    Wing,
    "Medium",
    "air",
    Citation("air or some other fluid", "Wing", {"Snippet": "air"})
)

# Assert relations
AssertCited(
    TYPE_OF(Wing, Fin),
    Citation("A wing is a type of fin", "Wing", {})
)

# Query relations
relations = query_relations(subject="Wing")
for rel in relations:
    print(f"{rel['subject']} -{rel['type']}-> {rel['object']}")

close_db()
```

## Core Concepts

### Entity

Knowledge graph nodes with properties:

```python
entity = Entity("Wing", {"color": "white"})
Wing = AddEntity("Wing", {"color": "white"})  # Creates and persists
```

### Citation

Source attribution for facts:

```python
Citation(
    regex="air or some other fluid",  # Text pattern from source
    source="Wing",                     # Article name
    meta={"Snippet": "air"}            # Additional metadata
)
```

### Relation

Typed connections between entities. Use relation functions:

```python
TYPE_OF(Wing, Fin)           # Wing is a type of Fin
PART_OF(Engine, Car)         # Engine is part of Car
BORN_IN(Einstein, Germany)   # Einstein was born in Germany
```

### Event

Timeline events with dates and labels:

```python
event = AddEvent(
    "WorldWarII",
    {"Date": "1939-1945", "Label": "World War II"},
    Citation("...", "WWII", {})
)
```

## API Reference

### Database Management

```python
init_db(path: str = ":memory:") -> Storage
    # Initialize SQLite database (file or in-memory)

get_db() -> Storage
    # Get current database instance

close_db() -> None
    # Close database connection

clear_all() -> None
    # Clear all entities and close database
```

### Entity Operations

```python
AddEntity(name: str, props: dict | None = None) -> Entity
    # Create entity and persist to database

SetPropertyCited(entity: Entity, key: str, value: Any,
                 citation: Citation | None = None) -> None
    # Set entity property with optional citation

get_entity(name: str) -> Entity | None
    # Retrieve entity by name

get_or_create_entity(name: str) -> Entity
    # Get existing or create new entity
```

### Relation Operations

```python
AssertCited(relation: Relation, citation: Citation | None = None) -> None
    # Assert relation with optional citation

query_relations(
    subject: str | None = None,
    obj: str | None = None,
    rel_type: str | None = None
) -> list[dict]
    # Query relations with filters
    # Returns: [{"type", "subject", "object", "citation"}]
```

### Query Functions

```python
query_properties(entity_name: str) -> list[dict]
    # Returns: [{"key", "value", "citation"}]

get_neighbors(entity_name: str, direction: str = "both") -> list[dict]
    # direction: "forward", "backward", "both"
    # Returns: [{"neighbor", "relation_type", "direction"}]

random_walk(start: str | None = None, steps: int = 5,
            allow_cycles: bool = False) -> list[dict]
    # Generate random path through graph

format_path(path: list[dict]) -> str
    # Convert path to readable string
    # Example: "Einstein -[FIELD_OF_WORK]-> Physics"

get_random_entity() -> str | None
    # Get random entity name
```

## Relation Types

180+ relation types organized by domain:

**General**: `TYPE_OF`, `IS_A`, `HAS_PART`, `PART_OF`, `INSTANCE_OF`, `SUBCLASS_OF`, `MEMBER_OF`, `RELATED_TO`, `SAME_AS`, `EQUIVALENT_TO`

**Temporal**: `PRECEDED_BY`, `FOLLOWED_BY`, `BEFORE`, `AFTER`, `DURING`, `CONTEMPORARY_OF`

**Causation**: `CAUSES`, `CAUSED_BY`, `INFLUENCES`, `ENABLES`, `PREVENTS`, `REQUIRES`, `DEPENDS_ON`

**Geography**: `LOCATED_IN`, `CAPITAL_OF`, `BORDERS`, `NEAR`, `CONTAINS_LOCATION`, `COUNTRY_OF`

**Biography**: `BORN_IN`, `DIED_IN`, `NATIONALITY`, `OCCUPATION`, `SPOUSE_OF`, `CHILD_OF`, `PARENT_OF`, `SIBLING_OF`, `MENTOR_OF`, `STUDENT_OF`

**Creation**: `CREATED_BY`, `CREATOR_OF`, `INVENTED_BY`, `AUTHORED_BY`, `COMPOSED_BY`, `DIRECTED_BY`

**Organization**: `AFFILIATED_WITH`, `HEADQUARTERS_IN`, `SUBSIDIARY_OF`, `LED_BY`, `CEO_OF`

**Science**: `DERIVES_FROM`, `FIELD_OF`, `STUDIES`, `PROPERTY_OF`, `COMPOSED_OF`, `MADE_OF`

**Biology**: `SPECIES_OF`, `GENUS_OF`, `HABITAT_OF`, `EATS`, `PREDATOR_OF`, `PREY_OF`

**Arts**: `GENRE_OF`, `STYLE_OF`, `INSPIRED_BY`, `PERFORMED_BY`, `ADAPTATION_OF`

**Politics**: `GOVERNED_BY`, `OFFICIAL_LANGUAGE_OF`, `MEMBER_STATE_OF`, `ALLY_OF`, `COLONY_OF`

**Mathematics**: `THEOREM_OF`, `PROOF_OF`, `COROLLARY_OF`, `GENERALIZATION_OF`, `SPECIAL_CASE_OF`

See `wikikg/relations.py` for complete list.

## Database Schema

```sql
entities(id, name UNIQUE, props JSON)
citations(id, regex, source, snippet)
properties(id, entity_id FK, key, value, citation_id FK)
relations(id, type, subject_id FK, object_id FK, citation_id FK)
events(id, name, date, label, props JSON, citation_id FK)
timeline_links(id, entity_id FK, event_id FK)
```

Indexes on: `entities.name`, `properties.entity_id`, `properties.key`, `relations.type`, `relations.subject_id`, `relations.object_id`, `events.date`

## Mathematica Converter

Convert `.m` files from wiki-kg-dataset to Python:

### CLI Usage

```bash
# Single file
python -m wikikg.converter input.m -o output.py

# Directory (recursive)
python -m wikikg.converter input_dir/ -o output_dir/ -r
```

### Programmatic API

```python
from wikikg.converter import convert_mathematica_to_python, convert_file

# Convert string
py_code = convert_mathematica_to_python(mathematica_code)

# Convert file
output_path = convert_file("input.m", "output.py")
```

### Conversion Steps

1. Unicode normalization (smart quotes, dashes, symbols)
2. Comments: `(* ... *)` -> `# ...`
3. Brackets: `Func[args]` -> `Func(args)`
4. Associations: `<| ... |>` -> `{ ... }`
5. Rules: `key -> value` -> `key: value`
6. AddEntity pattern: `AddEntity[Wing, ...]` -> `Wing = AddEntity("Wing", ...)`
7. Adds `from wikikg import *` header

## Dataset Generation

Export knowledge graph for LLM training.

### Quick Start: Generate Triplets

```python
from wikikg.datasets import DatasetGenerator
from wikikg.storage import Storage
from pathlib import Path

storage = Storage("wikikg.db")
gen = DatasetGenerator(storage)
gen.to_jsonl(Path("triplets.jsonl"), "triplets")
```

**Sample output** (`triplets.jsonl`):

```json
{"subject":"GirardDesargues","relation":"BORN_ON","object":"1591-02-21"}
{"subject":"GirardDesargues","relation":"CREATOR_OF","object":"DesarguesianPlane"}
{"subject":"MatsWilander","relation":"BORN_IN","object":"VaxjoSweden"}
{"subject":"MatsWilander","relation":"COACHED_BY","object":"JohnAndersSjogren"}
{"subject":"ClydeTolson","relation":"SIBLING_OF","object":"HilloryAlfredTolson"}
```

### Quick Start: Generate Paths

```python
from wikikg.datasets import DatasetGenerator
from wikikg.storage import Storage
from pathlib import Path

storage = Storage("wikikg.db")
gen = DatasetGenerator(storage)
gen.to_jsonl(Path("paths.jsonl"), "paths", num_paths=1000, min_length=3, max_length=6)
```

**Sample output** (`paths.jsonl`):

```json
{"entities":["Einstein","Physics","Feynman"],"relations":["FIELD_OF_WORK","STUDIED_BY"],"directions":["forward","backward"]}
{"entities":["Wing","Fin","Fish","Ocean"],"relations":["TYPE_OF","PART_OF","HABITAT_OF"],"directions":["forward","forward","forward"]}
```

**Human-readable format** using `GraphPath.to_text()`:

```
Einstein -[FIELD_OF_WORK]-> Physics -[STUDIED_BY]<- Feynman
Wing -[TYPE_OF]-> Fin -[PART_OF]-> Fish -[HABITAT_OF]-> Ocean
```

### Quick Start: Generate Entity Profiles

```python
from wikikg.datasets import DatasetGenerator
from wikikg.storage import Storage
from pathlib import Path

storage = Storage("wikikg.db")
gen = DatasetGenerator(storage)
gen.to_jsonl(Path("profiles.jsonl"), "profiles", min_relations=3)
```

**Sample output** (`profiles.jsonl`):

```json
{"name":"MatsWilander","outgoing":[{"subject":"MatsWilander","relation":"BORN_IN","object":"VaxjoSweden"},{"subject":"MatsWilander","relation":"COACHED_BY","object":"JohnAndersSjogren"}],"incoming":[{"subject":"JerringAward","relation":"AWARDED_TO","object":"MatsWilander"}],"properties":{"birthdate":"1964-08-22"}}
```

### Quick Start: Generate Neighborhoods

```python
from wikikg.datasets import DatasetGenerator
from wikikg.storage import Storage
from pathlib import Path

storage = Storage("wikikg.db")
gen = DatasetGenerator(storage)
gen.to_jsonl(Path("neighborhoods.jsonl"), "neighborhoods", limit=1000)
```

**Sample output** (`neighborhoods.jsonl`):

```json
{"center":"PhysicsEducation","neighbors":[{"subject":"PhysicsEducation","relation":"HAS_PART","object":"LectureMethod"},{"subject":"PhysicsEducation","relation":"HAS_PART","object":"LaboratoryExercises"},{"subject":"PhysicsEducationResearch","relation":"SUBCLASS_OF","object":"PhysicsEducation"}]}
```

### CLI Scripts

```bash
# Triplets
uv run python scripts/export_dataset.py triplets -o data/triplets.jsonl

# Paths (random walks)
uv run python scripts/export_dataset.py paths --num-paths 10000 -o data/paths.jsonl

# Entity profiles
uv run python scripts/export_dataset.py profiles --min-relations 5 -o data/profiles.jsonl

# Neighborhoods
uv run python scripts/export_dataset.py neighborhoods --limit 1000 -o data/neighborhoods.jsonl
```

### Tool-Calling Dataset (Multihop)

Generate replayable multihop tool-call trajectories from `paths.jsonl`:

```bash
uv run python scripts/generate_tool_dataset.py --db data/wikikg.db --paths data/paths.jsonl -o data/tool_calls.jsonl --num-examples 1000
uv run python scripts/verify_tool_dataset.py --db data/wikikg.db --dataset data/tool_calls.jsonl
```

### Full Example with Filters

```python
from wikikg.datasets import DatasetGenerator
from wikikg.storage import Storage
from pathlib import Path

storage = Storage("wikikg.db")
gen = DatasetGenerator(storage)

# Filter to specific relation types
gen.filter_by_relation_types(["BORN_IN", "DIED_IN", "OCCUPATION", "CREATED_BY"])

# Only entities with at least 5 connections
gen.filter_connected_only(min_degree=5)

# Export filtered triplets
gen.to_jsonl(Path("filtered_triplets.jsonl"), "triplets")

# Clear filters for next export
gen.clear_filters()
```

### Pydantic Schemas

```python
from wikikg.schemas import Triplet, EntityProfile, GraphPath, Neighborhood, Subgraph

Triplet(subject="Wing", relation="TYPE_OF", object="Fin")

EntityProfile(
    name="Einstein",
    outgoing=[Triplet(subject="Einstein", relation="FIELD_OF_WORK", object="Physics")],
    incoming=[Triplet(subject="Nobel Prize", relation="AWARDED_TO", object="Einstein")],
    properties={"birthdate": "1879-03-14"}
)

GraphPath(
    entities=["Einstein", "Physics", "Feynman"],
    relations=["FIELD_OF_WORK", "STUDIED_BY"],
    directions=["forward", "backward"]
)

Neighborhood(
    center="PhysicsEducation",
    neighbors=[Triplet(subject="PhysicsEducation", relation="HAS_PART", object="LectureMethod")]
)
```

## Experta Integration

Rule-based reasoning with transitive inference:

```python
from wikikg.experta_rules import create_engine
from wikikg.experta_facts import load_from_db_path

engine = create_engine()
engine.reset()
num_facts = load_from_db_path(engine, "knowledge.db")
engine.run()  # Triggers inference rules
```

### Fact Classes

```python
from wikikg.experta_facts import EntityFact, PropertyFact, RelationFact, EventFact

EntityFact(name="Wing")
PropertyFact(entity="Wing", key="Medium", value="air")
RelationFact(rel_type="TYPE_OF", subject="Wing", obj="Fin")
EventFact(name="WWII", date="1939-1945", label="World War II")
```

### Inference Rules

**Transitive**: IS_A, TYPE_OF, PART_OF, SUBCLASS_OF, LOCATED_IN, MEMBER_OF

**Inverse**: HAS_PART <-> PART_OF, CHILD_OF <-> PARENT_OF, CREATED_BY <-> CREATOR_OF

**Symmetric**: SIBLING_OF, SPOUSE_OF, BORDERS, EQUIVALENT_TO, SAME_AS

**Implication**: CAUSES -> INFLUENCES

## Data Assets

### Converted Dataset

- **Location**: `data/converted_valid/`
- **Format**: `{index}_{article_name}.py`
- **Count**: 45,416 Wikipedia articles

### Database Files

- **Main**: `wikikg.db` (~690MB) - Full knowledge graph
- **Test**: `test_wikikg.db` (~13MB) - Sample for testing

## Testing

```bash
pytest tests/test_wikikg.py -v
```

Test coverage includes:
- Entity creation and registry
- Citation handling
- Relation types and assertions
- SQLite storage operations
- Mathematica conversion
- Experta integration
- File I/O operations
