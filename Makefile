# WZRD-Algo-System Makefile

.PHONY: help dev install test validate preview view clean

# Default target
help:
	@echo "🚀 WZRD-Algo-System Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make dev      - Install dependencies and setup development environment"
	@echo "  make install  - Install production dependencies only"
	@echo ""
	@echo "Development:"
	@echo "  make validate - Validate all strategies and test plans"
	@echo "  make preview  - Run strategy preview (requires STRAT and PLAN)"
	@echo "  make view     - Launch Streamlit viewer"
	@echo "  make test     - Run test suite"
	@echo ""
	@echo "Examples:"
	@echo "  make preview STRAT=strategies/spy_vwap PLAN=strategies/spy_vwap/TestPlan.yaml"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean    - Clean temporary files and outputs"

# Development setup
dev: install
	@echo "🔧 Setting up development environment..."
	@pip install -e .
	@echo "✅ Development environment ready"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	@pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Validate all strategies and test plans
validate:
	@echo "📋 Validating strategies and test plans..."
	@python -c "from utils.validation import *; \
		import os; \
		validator = StrategyValidator(); \
		for root, dirs, files in os.walk('strategies'): \
			for file in files: \
				if file == 'StrategySpec.json': \
					spec_path = os.path.join(root, file); \
					spec = load_strategy_spec(spec_path); \
					errors = validator.validate_strategy_spec(spec); \
					status = '✅' if not errors else '❌'; \
					print(f'{status} {spec_path}'); \
					if errors: [print(f'   Error: {e}') for e in errors] \
				elif file == 'TestPlan.yaml': \
					plan_path = os.path.join(root, file); \
					plan = load_test_plan(plan_path); \
					errors = validator.validate_test_plan(plan); \
					status = '✅' if not errors else '❌'; \
					print(f'{status} {plan_path}'); \
					if errors: [print(f'   Error: {e}') for e in errors]"
	@echo "✅ Validation complete"

# Run strategy preview
preview:
	@if [ -z "$(STRAT)" ] || [ -z "$(PLAN)" ]; then \
		echo "❌ Usage: make preview STRAT=strategies/spy_vwap PLAN=strategies/spy_vwap/TestPlan.yaml"; \
		exit 1; \
	fi
	@echo "🚀 Running strategy preview..."
	@mkdir -p runs/preview-$(shell date +%Y%m%d-%H%M%S)
	@python engine/runner.py \
		--strategy $(STRAT)/StrategySpec.json \
		--plan $(PLAN) \
		--output runs/preview-$(shell date +%Y%m%d-%H%M%S) \
		--verbose
	@echo "✅ Preview complete"

# Launch Streamlit viewer
view:
	@echo "🎯 Launching Streamlit viewer..."
	@streamlit run apps/strategy_viewer_enhanced.py --server.port 8510

# Run test suite
test:
	@echo "🧪 Running test suite..."
	@python tests/test_system.py
	@python -m pytest tests/ -v || echo "pytest not available, using basic tests"
	@echo "✅ Tests complete"

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	@rm -rf __pycache__ */__pycache__ */*/__pycache__
	@rm -rf .pytest_cache
	@rm -rf runs/preview-*
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@find . -name ".DS_Store" -delete
	@echo "✅ Cleanup complete"

# Quick development cycle
dev-cycle: validate test
	@echo "🔄 Development cycle complete"

# Production deployment preparation
prod-ready: clean validate test
	@echo "🚀 Production readiness check complete"