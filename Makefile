# WikiKG Dataset Generation
# Usage: make assets SIZE=100

SIZE ?= 100
DB_PATH = data/wikikg.db
ASSETS_DIR = assets

.PHONY: all assets triplets paths db clean clean-db help

all: assets

help:
	@echo "WikiKG Dataset Generation"
	@echo ""
	@echo "Usage:"
	@echo "  make assets SIZE=100    Generate triplets and paths (default: 100)"
	@echo "  make db SIZE=100        Create database from HuggingFace dataset"
	@echo "  make triplets SIZE=100  Generate only triplets"
	@echo "  make paths SIZE=100     Generate only paths"
	@echo "  make clean              Remove generated assets"
	@echo "  make clean-db           Remove database"
	@echo ""
	@echo "Variables:"
	@echo "  SIZE      Number of samples/records (default: 100)"
	@echo "  DB_PATH   Database path (default: data/wikikg.db)"

# Create database from HuggingFace dataset
$(DB_PATH):
	@echo "Creating database with $(SIZE) samples..."
	@mkdir -p data
	uv run python scripts/convert_dataset.py --db-path $(DB_PATH) --max-samples $(SIZE) --load-db

db: $(DB_PATH)

# Generate triplets
$(ASSETS_DIR)/triplets.jsonl: $(DB_PATH)
	@echo "Generating $(SIZE) triplets..."
	@mkdir -p $(ASSETS_DIR)
	uv run python scripts/export_dataset.py triplets --db $(DB_PATH) --limit $(SIZE) -o $(ASSETS_DIR)/triplets.jsonl

triplets: $(ASSETS_DIR)/triplets.jsonl

# Generate paths
$(ASSETS_DIR)/paths.jsonl: $(DB_PATH)
	@echo "Generating $(SIZE) paths..."
	@mkdir -p $(ASSETS_DIR)
	uv run python scripts/export_dataset.py paths --db $(DB_PATH) --num-paths $(SIZE) -o $(ASSETS_DIR)/paths.jsonl

paths: $(ASSETS_DIR)/paths.jsonl

# Generate all assets
assets: $(ASSETS_DIR)/triplets.jsonl $(ASSETS_DIR)/paths.jsonl
	@echo "Generated assets in $(ASSETS_DIR)/"

# Clean targets
clean:
	rm -rf $(ASSETS_DIR)

clean-db:
	rm -f $(DB_PATH) data/errors.log data/success.log

clean-all: clean clean-db
