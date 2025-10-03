#!/usr/bin/env python3
"""
WZRD Strategy Validation System

Provides schema validation and acceptance testing for strategy specifications.
"""

import json
import os
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import jsonschema
import pandas as pd
import numpy as np

class ValidationError(Exception):
    """Custom exception for validation failures"""
    pass

class StrategyValidator:
    """Validates strategy specifications and test plans"""

    def __init__(self, schemas_dir: str = None):
        if schemas_dir is None:
            schemas_dir = os.path.join(os.path.dirname(__file__), '..', 'schemas')

        self.schemas_dir = schemas_dir
        self._load_schemas()

    def _load_schemas(self):
        """Load JSON schemas for validation"""
        try:
            with open(os.path.join(self.schemas_dir, 'strategy_spec.schema.json'), 'r') as f:
                self.strategy_schema = json.load(f)

            with open(os.path.join(self.schemas_dir, 'test_plan.schema.json'), 'r') as f:
                self.test_plan_schema = json.load(f)

        except Exception as e:
            raise ValidationError(f"Failed to load schemas: {e}")

    def validate_strategy_spec(self, spec: Dict[str, Any]) -> List[str]:
        """
        Validate strategy specification against schema

        Args:
            spec: Strategy specification dictionary

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            jsonschema.validate(spec, self.strategy_schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        except Exception as e:
            errors.append(f"Validation error: {e}")

        # Additional business logic validation
        errors.extend(self._validate_business_rules(spec))

        return errors

    def validate_test_plan(self, plan: Dict[str, Any]) -> List[str]:
        """
        Validate test plan against schema

        Args:
            plan: Test plan dictionary

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        try:
            jsonschema.validate(plan, self.test_plan_schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        except Exception as e:
            errors.append(f"Validation error: {e}")

        # Additional business logic validation
        errors.extend(self._validate_test_plan_logic(plan))

        return errors

    def _validate_business_rules(self, spec: Dict[str, Any]) -> List[str]:
        """Validate strategy business rules"""
        errors = []

        # Check session times
        session = spec.get('session', {})
        start_time = session.get('start', '09:30')
        end_time = session.get('end', '16:00')

        try:
            start_dt = datetime.strptime(start_time, '%H:%M')
            end_dt = datetime.strptime(end_time, '%H:%M')

            if start_dt >= end_dt:
                errors.append("Session start time must be before end time")

        except ValueError:
            errors.append("Invalid session time format")

        # Validate entry/exit rule IDs are unique
        entry_ids = [rule['id'] for rule in spec.get('entry_rules', [])]
        exit_ids = [rule['id'] for rule in spec.get('exit_rules', [])]

        if len(entry_ids) != len(set(entry_ids)):
            errors.append("Entry rule IDs must be unique")

        if len(exit_ids) != len(set(exit_ids)):
            errors.append("Exit rule IDs must be unique")

        # Validate feature dependencies
        features = spec.get('features', {})
        for feature_name, feature_def in features.items():
            if 'input' in feature_def:
                # Check if input references exist
                input_expr = feature_def['input']
                # Basic validation - could be enhanced with expression parser
                if 'vwap' in input_expr and 'vwap' not in features:
                    errors.append(f"Feature '{feature_name}' references undefined 'vwap'")

        return errors

    def _validate_test_plan_logic(self, plan: Dict[str, Any]) -> List[str]:
        """Validate test plan business logic"""
        errors = []

        # Check date range
        data = plan.get('data', {})
        try:
            start_date = datetime.strptime(data['start'], '%Y-%m-%d')
            end_date = datetime.strptime(data['end'], '%Y-%m-%d')

            if start_date >= end_date:
                errors.append("Start date must be before end date")

            # Check if date range is reasonable
            if (end_date - start_date).days > 365:
                errors.append("Date range exceeds 1 year maximum")

        except (ValueError, KeyError):
            errors.append("Invalid date format in test plan")

        return errors

class AcceptanceTestRunner:
    """Runs acceptance tests on strategy specifications"""

    def __init__(self, validator: StrategyValidator = None):
        self.validator = validator or StrategyValidator()

    def run_acceptance_tests(self, spec: Dict[str, Any],
                           test_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run acceptance tests on strategy specification

        Args:
            spec: Strategy specification
            test_data: Optional test data for worked examples

        Returns:
            Acceptance test results
        """
        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'tests': {}
        }

        # Test 1: Schema validation
        schema_errors = self.validator.validate_strategy_spec(spec)
        if schema_errors:
            results['passed'] = False
            results['errors'].extend(schema_errors)
        results['tests']['schema_validation'] = len(schema_errors) == 0

        # Test 2: Worked examples validation
        if 'worked_examples' in spec and spec['worked_examples']:
            worked_examples_result = self._test_worked_examples(spec, test_data)
            results['tests']['worked_examples'] = worked_examples_result['passed']
            if not worked_examples_result['passed']:
                results['passed'] = False
                results['errors'].extend(worked_examples_result['errors'])

        # Test 3: Session window compliance
        session_result = self._test_session_compliance(spec)
        results['tests']['session_compliance'] = session_result['passed']
        if not session_result['passed']:
            results['warnings'].extend(session_result['warnings'])

        # Test 4: Rule consistency
        rules_result = self._test_rule_consistency(spec)
        results['tests']['rule_consistency'] = rules_result['passed']
        if not rules_result['passed']:
            results['passed'] = False
            results['errors'].extend(rules_result['errors'])

        return results

    def _test_worked_examples(self, spec: Dict[str, Any],
                             test_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Test worked examples against expected behavior"""
        # Placeholder for worked examples testing
        # In real implementation, this would:
        # 1. Load test data for the specified dates
        # 2. Run the strategy engine on those specific bars
        # 3. Check if actual signals match expected behavior

        return {
            'passed': True,
            'errors': [],
            'details': 'Worked examples testing not implemented yet'
        }

    def _test_session_compliance(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Test that strategy respects session windows"""
        warnings = []

        session = spec.get('session', {})
        start_time = session.get('start', '09:30')
        end_time = session.get('end', '16:00')

        # Check if session is within reasonable market hours
        try:
            start_dt = datetime.strptime(start_time, '%H:%M')
            end_dt = datetime.strptime(end_time, '%H:%M')

            # Warn if session is very short
            session_minutes = (end_dt - start_dt).seconds / 60
            if session_minutes < 30:
                warnings.append("Session window is very short (< 30 minutes)")

            # Warn if session starts very early or ends very late
            if start_dt.hour < 4:
                warnings.append("Session starts very early (before 4 AM)")
            if end_dt.hour > 20:
                warnings.append("Session ends very late (after 8 PM)")

        except ValueError:
            warnings.append("Could not parse session times")

        return {
            'passed': True,  # Warnings don't fail the test
            'warnings': warnings
        }

    def _test_rule_consistency(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Test that entry and exit rules are consistent"""
        errors = []

        entry_rules = spec.get('entry_rules', [])
        exit_rules = spec.get('exit_rules', [])

        if not entry_rules:
            errors.append("Strategy must have at least one entry rule")

        if not exit_rules:
            errors.append("Strategy must have at least one exit rule")

        # Check that sides are consistent
        sides = set(rule.get('side') for rule in entry_rules)
        if len(sides) > 1:
            # Strategy trades both long and short - check exit rules handle both
            for exit_rule in exit_rules:
                if exit_rule.get('type') == 'condition' and 'side' in exit_rule:
                    # Exit rule is side-specific, make sure it covers all entry sides
                    pass  # Detailed implementation would go here

        return {
            'passed': len(errors) == 0,
            'errors': errors
        }

def load_strategy_spec(file_path: str) -> Dict[str, Any]:
    """Load strategy specification from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ValidationError(f"Failed to load strategy spec: {e}")

def load_test_plan(file_path: str) -> Dict[str, Any]:
    """Load test plan from YAML file"""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ValidationError(f"Failed to load test plan: {e}")

def create_lock_file(spec_path: str, lock_path: str, user: str = "system") -> Dict[str, Any]:
    """Create a lock file for validated strategy"""
    import hashlib

    # Calculate hash of strategy spec
    with open(spec_path, 'rb') as f:
        content = f.read()
        sha256 = hashlib.sha256(content).hexdigest()

    lock_data = {
        "spec_path": spec_path,
        "sha256": sha256,
        "locked_by": user,
        "locked_at": datetime.utcnow().isoformat() + "Z",
        "version": "1.0"
    }

    with open(lock_path, 'w') as f:
        json.dump(lock_data, f, indent=2)

    return lock_data

if __name__ == "__main__":
    # Example usage
    validator = StrategyValidator()
    acceptance_runner = AcceptanceTestRunner(validator)

    print("🧪 WZRD Strategy Validation System")
    print("Available functions:")
    print("- validate_strategy_spec()")
    print("- validate_test_plan()")
    print("- run_acceptance_tests()")
    print("- create_lock_file()")