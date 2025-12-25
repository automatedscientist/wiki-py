#!/usr/bin/env python3
"""Generate million-scale trajectories and triplets from WikiKG.

Usage:
    # Generate 1M trajectories with query_relations style (~5 min)
    uv run python scripts/generate_million.py --style query_relations

    # Generate 1M trajectories with get_neighbors style (~5 min)
    uv run python scripts/generate_million.py --style get_neighbors

    # Generate both styles (run twice)
    uv run python scripts/generate_million.py --style query_relations --skip-triplets
    uv run python scripts/generate_million.py --style get_neighbors --skip-paths --skip-triplets

    # Triplets only (~24 sec)
    uv run python scripts/generate_million.py --skip-paths --skip-trajectories

    # Resume after interruption (automatic via .progress_<style>.json)
    uv run python scripts/generate_million.py --style query_relations

Output files:
    data/million/
    ├── paths_1m.jsonl                          # 1.5M random walk paths
    ├── trajectories_query_relations_1m.jsonl   # 1M tool-call trajectories
    ├── trajectories_get_neighbors_1m.jsonl     # 1M tool-call trajectories
    └── triplets_all.jsonl                      # All valid triplets (~366k)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

# Import ALL wikikg modules
from wikikg import (
    Entity,
    Citation,
    Event,
    Relation,
    Storage,
    init_db,
    get_db,
    close_db,
    clear_all,
    AddEntity,
    SetPropertyCited,
    AssertCited,
    AddEvent,
    query_relations,
    query_properties,
    get_neighbors,
    get_random_entity,
    random_walk,
    format_path,
    RELATION_TYPES,
    get_relation_class,
    all_relation_types,
    # All relation types
    TYPE_OF, IS_A, HAS_PART, PART_OF, INSTANCE_OF, SUBCLASS_OF,
    MEMBER_OF, BELONGS_TO, CONTAINS, COMPRISES, RELATED_TO, SIMILAR_TO,
    DIFFERENT_FROM, OPPOSITE_OF, EQUIVALENT_TO, SAME_AS,
    CAUSES, CAUSED_BY, INFLUENCES, INFLUENCED_BY, AFFECTS, AFFECTED_BY,
    ENABLES, PREVENTS, REQUIRES, DEPENDS_ON, RESULTS_IN,
    PRECEDED_BY, FOLLOWED_BY, BEFORE, AFTER, DURING, CONTEMPORARY_OF,
    SUCCEEDS, PRECEDES,
    LOCATED_IN, LOCATION_OF, CAPITAL_OF, HAS_CAPITAL, BORDERS, NEAR,
    CONTAINS_LOCATION, REGION_OF, COUNTRY_OF, CONTINENT_OF,
    BORN_IN, DIED_IN, BIRTHPLACE_OF, DEATHPLACE_OF, NATIONALITY, CITIZEN_OF,
    OCCUPATION, PROFESSION_OF, EMPLOYER, EMPLOYED_BY, EDUCATED_AT,
    ALMA_MATER_OF, SPOUSE_OF, CHILD_OF, PARENT_OF, SIBLING_OF, RELATIVE_OF,
    FRIEND_OF, COLLEAGUE_OF, MENTOR_OF, STUDENT_OF,
    CREATED_BY, CREATOR_OF, INVENTED_BY, INVENTOR_OF, DISCOVERED_BY,
    DISCOVERER_OF, FOUNDED_BY, FOUNDER_OF, AUTHORED_BY, AUTHOR_OF,
    COMPOSED_BY, COMPOSER_OF, DIRECTED_BY, DIRECTOR_OF, PRODUCED_BY,
    PRODUCER_OF, DESIGNED_BY, DESIGNER_OF, BUILT_BY, BUILDER_OF,
    AFFILIATED_WITH, HEADQUARTERS_IN, SUBSIDIARY_OF, PARENT_COMPANY_OF,
    DEPARTMENT_OF, DIVISION_OF, BRANCH_OF, LED_BY, LEADER_OF, CEO_OF,
    PRESIDENT_OF, CHAIRMAN_OF,
    DERIVES_FROM, DERIVED_FROM, APPLIED_TO, APPLICATION_OF, FIELD_OF,
    STUDIES, STUDIED_BY, PROPERTY_OF, HAS_PROPERTY, CHARACTERISTIC_OF,
    COMPOSED_OF, MADE_OF, MATERIAL_OF,
    SPECIES_OF, GENUS_OF, FAMILY_OF, ORDER_OF, CLASS_OF, PHYLUM_OF,
    KINGDOM_OF, HABITAT_OF, LIVES_IN, EATS, EATEN_BY, PREDATOR_OF,
    PREY_OF, SYMBIONT_OF, HOST_OF, PARASITE_OF,
    GENRE_OF, STYLE_OF, MOVEMENT_OF, INSPIRED_BY, INSPIRATION_FOR,
    PERFORMED_BY, PERFORMER_OF, PORTRAYED_BY, PORTRAYAL_OF, ADAPTATION_OF,
    BASED_ON,
    CURRENCY_OF, TRADED_IN, MARKET_OF, INDUSTRY_OF, SECTOR_OF, PRODUCT_OF,
    MANUFACTURER_OF, SUPPLIER_OF, CUSTOMER_OF, COMPETITOR_OF,
    GOVERNED_BY, GOVERNS, OFFICIAL_LANGUAGE_OF, MEMBER_STATE_OF,
    SIGNATORY_OF, ALLY_OF, ENEMY_OF, COLONY_OF, COLONIZED_BY,
    TIMELINE_EVENT, PARTICIPANT_IN, OCCURRED_IN, VENUE_OF, ORGANIZER_OF,
    WINNER_OF, AWARD_FOR,
    LANGUAGE_OF, SPOKEN_IN, WRITTEN_IN, TRANSLATED_FROM, TRANSLATION_OF,
    ETYMOLOGY_OF, DERIVED_WORD_OF,
    THEOREM_OF, PROOF_OF, COROLLARY_OF, GENERALIZATION_OF, SPECIAL_CASE_OF,
    DUAL_OF, INVERSE_OF,
)
from wikikg.schemas import GraphPath, Triplet, EntityProfile, Neighborhood, Subgraph
from wikikg.datasets import DatasetGenerator
from wikikg.tool_dataset import (
    ToolCallStyle,
    ToolCallingDatasetGenerator,
    generate_from_paths,
    load_paths_jsonl,
    write_jsonl,
    default_provenance,
    tool_schema,
)
from wikikg.tool_dataset_verify import verify_example
from wikikg.torch_dataloader import (
    ToolCallsJsonlDataset,
    ToolCallsJsonlIterableDataset,
    ToolCallsRenderConfig,
    build_tool_calls_dataloader,
)


def load_progress(progress_file: Path) -> dict[str, int]:
    """Load progress from checkpoint file."""
    if progress_file.exists():
        return json.loads(progress_file.read_text())
    return {}


def save_progress(progress_file: Path, progress: dict[str, int]) -> None:
    """Save progress to checkpoint file."""
    progress_file.write_text(json.dumps(progress))


def fast_random_walk(
    storage: Storage,
    entity_names: list[str],
    rng: random.Random,
    steps: int = 5,
    allow_cycles: bool = False,
) -> list[dict[str, Any]]:
    """Optimized random walk using pre-loaded entity list.

    Avoids slow ORDER BY RANDOM() by using Python's random.choice().
    """
    start = rng.choice(entity_names)

    path = [{"entity": start}]
    visited = {start}
    current = start

    for _ in range(steps):
        neighbors = storage.get_neighbors(current, direction="both")

        if not allow_cycles:
            neighbors = [n for n in neighbors if n["neighbor"] not in visited]

        if not neighbors:
            break

        chosen = rng.choice(neighbors)
        current = chosen["neighbor"]
        visited.add(current)

        path.append({
            "entity": current,
            "relation": chosen["relation_type"],
            "direction": chosen["direction"]
        })

    return path


def generate_paths_resumable(
    storage: Storage,
    output_path: Path,
    num_paths: int,
    progress_file: Path,
    min_length: int = 3,
    max_length: int = 7,
    seed: int = 42,
    flush_every: int = 10_000,
) -> int:
    """Generate paths in chunks, with resumability.

    Returns number of paths generated.
    """
    progress = load_progress(progress_file)
    start_count = progress.get("paths", 0)

    # Pre-load all entity names for fast random selection
    print("Pre-loading entity names...")
    conn = storage._get_conn()
    entity_names = [row["name"] for row in conn.execute("SELECT name FROM entities")]
    print(f"Loaded {len(entity_names):,} entity names")

    rng = random.Random(seed)
    # Advance RNG state to match checkpoint (approximate)
    for _ in range(start_count * 3):
        rng.random()

    mode = "a" if start_count > 0 else "w"
    generated = start_count
    attempts = 0
    max_attempts = num_paths * 20

    with open(output_path, mode) as f:
        pbar = tqdm(
            total=num_paths,
            initial=start_count,
            desc="Paths",
            unit="path",
            ncols=80,
        )

        while generated < num_paths and attempts < max_attempts:
            attempts += 1

            steps = rng.randint(min_length - 1, max_length - 1)
            path_data = fast_random_walk(
                storage, entity_names, rng, steps=steps, allow_cycles=False
            )

            if len(path_data) < min_length:
                continue

            entities = [step["entity"] for step in path_data]
            relations = [step["relation"] for step in path_data[1:]]
            directions = [step["direction"] for step in path_data[1:]]

            gp = GraphPath(
                entities=entities,
                relations=relations,
                directions=directions,
            )

            f.write(gp.model_dump_json() + "\n")
            generated += 1
            pbar.update(1)

            if generated % flush_every == 0:
                f.flush()
                progress["paths"] = generated
                save_progress(progress_file, progress)

        pbar.close()

    progress["paths"] = generated
    save_progress(progress_file, progress)
    return generated


def generate_trajectories_resumable(
    storage: Storage,
    paths_file: Path,
    output_path: Path,
    num_trajectories: int,
    progress_file: Path,
    style: str = "query_relations",
    min_hops: int = 2,
    max_hops: int = 6,
    seed: int = 42,
    flush_every: int = 1_000,
) -> int:
    """Generate trajectories from paths, with resumability.

    Returns number of trajectories written.
    """
    progress = load_progress(progress_file)
    start_count = progress.get("trajectories", 0)
    path_index_start = progress.get("trajectory_path_index", 0)

    # Load paths and shuffle deterministically
    print(f"Loading paths from {paths_file}...")
    paths = load_paths_jsonl(paths_file)
    print(f"Loaded {len(paths):,} paths")

    # Filter by hop count
    candidates = [p for p in paths if min_hops <= p.length <= max_hops]
    print(f"Filtered to {len(candidates):,} paths with {min_hops}-{max_hops} hops")

    rng = random.Random(seed)
    rng.shuffle(candidates)

    generator = ToolCallingDatasetGenerator(storage, max_tool_results=50, seed=seed)
    provenance = default_provenance(
        Path(storage.db_path), repo_root=Path(__file__).parent.parent
    )
    tool_style = ToolCallStyle(style)

    seen: set[tuple] = set()
    mode = "a" if start_count > 0 else "w"

    with open(output_path, mode) as f:
        pbar = tqdm(
            total=num_trajectories,
            initial=start_count,
            desc="Trajectories",
            unit="traj",
            ncols=80,
        )

        written = start_count
        for i, path in enumerate(candidates):
            if i < path_index_start:
                continue
            if written >= num_trajectories:
                break

            # Dedup by signature
            signature = (
                style,
                tuple(path.entities),
                tuple(path.relations),
                tuple(path.directions),
            )
            if signature in seen:
                continue
            seen.add(signature)

            example = generator.build_example(
                path,
                style=tool_style,
                drop_if_answer_is_direct_neighbor=True,
                example_id=f"{style}-{i}",
                provenance=provenance,
            )

            if example is not None:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1
                pbar.update(1)

                if written % flush_every == 0:
                    f.flush()
                    progress["trajectories"] = written
                    progress["trajectory_path_index"] = i + 1
                    save_progress(progress_file, progress)

        pbar.close()

    progress["trajectories"] = written
    save_progress(progress_file, progress)
    return written


def export_triplets_all(
    storage: Storage,
    output_path: Path,
    limit: int = 0,
) -> int:
    """Export all triplets from database.

    Args:
        storage: Database storage
        output_path: Output JSONL file
        limit: Max triplets (0 = all)

    Returns:
        Number of triplets exported
    """
    gen = DatasetGenerator(storage)
    total = storage.count_relations() if limit == 0 else limit

    written = 0
    with open(output_path, "w") as f:
        for triplet in tqdm(
            gen.extract_triplets(limit=limit if limit > 0 else None, show_progress=False),
            total=total,
            desc="Triplets",
            unit="triplet",
            ncols=80,
        ):
            f.write(triplet.model_dump_json() + "\n")
            written += 1

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Generate 1M trajectories and triplets from WikiKG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db", type=Path, default=Path("data/wikikg.db"),
        help="Path to WikiKG database"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/million"),
        help="Output directory"
    )
    parser.add_argument(
        "--num-trajectories", type=int, default=1_000_000,
        help="Number of trajectories to generate"
    )
    parser.add_argument(
        "--num-triplets", type=int, default=0,
        help="Number of triplets (0 = export ALL from DB)"
    )
    parser.add_argument(
        "--num-paths", type=int, default=1_500_000,
        help="Number of paths to generate (need extra for filtering)"
    )
    parser.add_argument(
        "--min-hops", type=int, default=2,
        help="Minimum hops per trajectory"
    )
    parser.add_argument(
        "--max-hops", type=int, default=6,
        help="Maximum hops per trajectory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--style", choices=["query_relations", "get_neighbors"],
        default="query_relations",
        help="Tool calling style"
    )
    parser.add_argument(
        "--skip-paths", action="store_true",
        help="Skip path generation"
    )
    parser.add_argument(
        "--skip-trajectories", action="store_true",
        help="Skip trajectory generation"
    )
    parser.add_argument(
        "--skip-triplets", action="store_true",
        help="Skip triplet export"
    )

    args = parser.parse_args()

    # Validate
    if not args.db.exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths_file = args.output_dir / "paths_1m.jsonl"
    trajectories_file = args.output_dir / f"trajectories_{args.style}_1m.jsonl"
    triplets_file = args.output_dir / "triplets_all.jsonl"
    progress_file = args.output_dir / f".progress_{args.style}.json"

    storage = Storage(args.db)

    entity_count = storage.count_entities()
    relation_count = storage.count_relations()
    print(f"Database: {entity_count:,} entities, {relation_count:,} relations")

    try:
        # 1. Generate paths
        if not args.skip_paths:
            print(f"\n=== Generating {args.num_paths:,} paths ===")
            path_count = generate_paths_resumable(
                storage,
                paths_file,
                args.num_paths,
                progress_file,
                min_length=args.min_hops + 1,
                max_length=args.max_hops + 1,
                seed=args.seed,
            )
            print(f"Generated {path_count:,} paths to {paths_file}")

        # 2. Generate trajectories
        if not args.skip_trajectories:
            print(f"\n=== Generating {args.num_trajectories:,} trajectories ===")
            if not paths_file.exists():
                print(f"Error: Paths file not found: {paths_file}", file=sys.stderr)
                print("Run without --skip-paths first", file=sys.stderr)
                sys.exit(1)

            traj_count = generate_trajectories_resumable(
                storage,
                paths_file,
                trajectories_file,
                args.num_trajectories,
                progress_file,
                style=args.style,
                min_hops=args.min_hops,
                max_hops=args.max_hops,
                seed=args.seed,
            )
            print(f"Generated {traj_count:,} trajectories to {trajectories_file}")

        # 3. Export triplets
        if not args.skip_triplets:
            limit_str = "ALL" if args.num_triplets == 0 else f"{args.num_triplets:,}"
            print(f"\n=== Exporting {limit_str} triplets ===")
            triplet_count = export_triplets_all(
                storage,
                triplets_file,
                limit=args.num_triplets,
            )
            print(f"Exported {triplet_count:,} triplets to {triplets_file}")

        # Summary
        print("\n=== Summary ===")
        for f in [paths_file, trajectories_file, triplets_file]:
            if f.exists():
                size_mb = f.stat().st_size / (1024 * 1024)
                lines = sum(1 for _ in open(f))
                print(f"{f.name}: {lines:,} records, {size_mb:.1f} MB")

    finally:
        storage.close()


if __name__ == "__main__":
    main()
