# Nomad Job Spec Agent - Makefile
# ================================

.PHONY: help install install-dev test test-cov lint format clean build publish publish-test

# Default target
help:
	@echo "Nomad Job Spec Agent - Available targets:"
	@echo ""
	@echo "  install       Install the package"
	@echo "  install-dev   Install with development dependencies"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  lint          Run linting (ruff)"
	@echo "  format        Format code (ruff)"
	@echo "  typecheck     Run type checking (mypy)"
	@echo "  clean         Remove build artifacts"
	@echo "  build         Build distribution packages"
	@echo "  publish       Publish to PyPI"
	@echo "  publish-test  Publish to TestPyPI"
	@echo "  run           Run the CLI (use CMD= to pass args)"
	@echo ""
	@echo "Examples:"
	@echo "  make install-dev"
	@echo "  make test"
	@echo "  make run CMD='analyze --path ./my-app'"

# Python and pip
PYTHON ?= python3
PIP ?= pip

# Package info
PACKAGE_NAME = nomad-job-spec
VERSION = $(shell $(PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")

# Directories
SRC_DIR = src
TEST_DIR = tests
DIST_DIR = dist
BUILD_DIR = build

# ================================
# Installation
# ================================

install:
	$(PIP) install .

install-dev:
	$(PIP) install -e ".[dev]"

# ================================
# Testing
# ================================

test:
	$(PYTHON) -m pytest $(TEST_DIR) -v --tb=short

test-cov:
	$(PYTHON) -m pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html

test-fast:
	$(PYTHON) -m pytest $(TEST_DIR) -v --tb=short -x

# ================================
# Code Quality
# ================================

lint:
	$(PYTHON) -m ruff check $(SRC_DIR) $(TEST_DIR)

lint-fix:
	$(PYTHON) -m ruff check $(SRC_DIR) $(TEST_DIR) --fix

format:
	$(PYTHON) -m ruff format $(SRC_DIR) $(TEST_DIR)

format-check:
	$(PYTHON) -m ruff format $(SRC_DIR) $(TEST_DIR) --check

typecheck:
	$(PYTHON) -m mypy $(SRC_DIR)

# Run all checks
check: lint format-check typecheck test
	@echo "All checks passed!"

# ================================
# Building
# ================================

clean:
	rm -rf $(DIST_DIR) $(BUILD_DIR)
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: clean
	$(PYTHON) -m build

# ================================
# Publishing
# ================================

# Publish to PyPI (production)
publish: build
	@echo "Publishing $(PACKAGE_NAME) v$(VERSION) to PyPI..."
	$(PYTHON) -m twine upload $(DIST_DIR)/*
	@echo "Published successfully!"

# Publish to TestPyPI (for testing)
publish-test: build
	@echo "Publishing $(PACKAGE_NAME) v$(VERSION) to TestPyPI..."
	$(PYTHON) -m twine upload --repository testpypi $(DIST_DIR)/*
	@echo "Published to TestPyPI successfully!"
	@echo ""
	@echo "Install from TestPyPI with:"
	@echo "  pip install --index-url https://test.pypi.org/simple/ $(PACKAGE_NAME)"

# Install build/publish dependencies
publish-deps:
	$(PIP) install build twine

# ================================
# Local Installation
# ================================

# Install to user site-packages (no virtualenv needed)
install-user:
	$(PIP) install --user .

# Install system-wide (requires sudo)
install-system:
	sudo $(PIP) install .

# Uninstall the package
uninstall:
	$(PIP) uninstall -y $(PACKAGE_NAME)

# ================================
# Development
# ================================

# Run the CLI
run:
	$(PYTHON) -m src.main $(CMD)

# Run with example
run-example:
	$(PYTHON) -m src.main analyze --path tests/fixtures/sample_repos/express-app

# Start development environment (install deps + run tests)
dev: install-dev test
	@echo "Development environment ready!"

# ================================
# Docker (optional)
# ================================

docker-build:
	docker build -t $(PACKAGE_NAME):$(VERSION) .
	docker tag $(PACKAGE_NAME):$(VERSION) $(PACKAGE_NAME):latest

docker-run:
	docker run --rm -it \
		-v $(PWD):/app \
		-e VLLM_BASE_URL \
		-e NOMAD_ADDRESS \
		$(PACKAGE_NAME):latest $(CMD)

# ================================
# Version Management
# ================================

version:
	@echo "$(VERSION)"

bump-patch:
	@echo "Bumping patch version..."
	@$(PYTHON) -c "import tomllib, re; \
		data = open('pyproject.toml').read(); \
		v = tomllib.loads(data)['project']['version']; \
		parts = v.split('.'); \
		parts[2] = str(int(parts[2]) + 1); \
		new_v = '.'.join(parts); \
		new_data = re.sub(r'version = \"[^\"]+\"', f'version = \"{new_v}\"', data); \
		open('pyproject.toml', 'w').write(new_data); \
		print(f'Bumped version: {v} -> {new_v}')"

bump-minor:
	@echo "Bumping minor version..."
	@$(PYTHON) -c "import tomllib, re; \
		data = open('pyproject.toml').read(); \
		v = tomllib.loads(data)['project']['version']; \
		parts = v.split('.'); \
		parts[1] = str(int(parts[1]) + 1); \
		parts[2] = '0'; \
		new_v = '.'.join(parts); \
		new_data = re.sub(r'version = \"[^\"]+\"', f'version = \"{new_v}\"', data); \
		open('pyproject.toml', 'w').write(new_data); \
		print(f'Bumped version: {v} -> {new_v}')"

bump-major:
	@echo "Bumping major version..."
	@$(PYTHON) -c "import tomllib, re; \
		data = open('pyproject.toml').read(); \
		v = tomllib.loads(data)['project']['version']; \
		parts = v.split('.'); \
		parts[0] = str(int(parts[0]) + 1); \
		parts[1] = '0'; \
		parts[2] = '0'; \
		new_v = '.'.join(parts); \
		new_data = re.sub(r'version = \"[^\"]+\"', f'version = \"{new_v}\"', data); \
		open('pyproject.toml', 'w').write(new_data); \
		print(f'Bumped version: {v} -> {new_v}')"
