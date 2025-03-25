# Coding Standards for Hydrological Modeling Framework

This document outlines the coding standards and conventions used in the hydrological modeling framework to ensure consistency and maintainability.

## Naming Conventions

### Configuration Attributes

Configuration attributes should follow Python's standard naming conventions:

- Use **lowercase snake_case** for attribute names (e.g., `group_identifier`, `forcing_features`)
- Avoid uppercase names for configuration attributes, unless they are truly constants

```python
# Good
config.group_identifier
config.forcing_features

# Avoid
config.GROUP_IDENTIFIER
config.FORCING_FEATURES
```

> **Note**: For backward compatibility, some configuration classes provide property getters with uppercase names, but new code should use the lowercase versions.

### Class Names

- Use **PascalCase** for class names (e.g., `HydroDataModule`, `BaseExperimentConfig`)
- Configuration classes should end with `Config` (e.g., `TiDEConfig`, `ExperimentConfig`)

### Function and Method Names

- Use **lowercase snake_case** for function and method names (e.g., `load_data`, `train_model`)
- Verb phrases are preferred for functions that perform actions (e.g., `create_model`, not `model_creator`)

### Constants

- Use **UPPERCASE** with underscores for true constants (e.g., `MAX_SEQUENCE_LENGTH`)
- Constants should be defined at module level, not within classes

## Code Organization

### Configuration Classes

Configuration classes should:

1. Use `dataclass` for clean definition of attributes
2. Include type hints for all attributes
3. Provide default values where appropriate
4. Include validation logic in a `validate` method
5. Document attributes in class docstring
6. Group related attributes together with blank lines between groups

### Type Hints

All functions should include appropriate type hints:

```python
def load_data(
    config: ExperimentConfig,
    country: Optional[str] = None
) -> Dict[str, Any]:
    """Load data for a specific country."""
    # Function implementation
```

## Documentation

### Docstrings

Use Google-style docstrings for all functions and classes:

```python
def train_model(
    model_type: str,
    config: ExperimentConfig,
    **kwargs
) -> Dict[str, Any]:
    """Train a model with the given configuration.
    
    Args:
        model_type: Type of model to train
        config: Experiment configuration
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary with training results
        
    Raises:
        ValueError: If model_type is not supported
    """
    # Function implementation
```

## Exception Handling

- Be specific about the exceptions you catch
- Include meaningful error messages that help diagnose the issue
- Re-raise exceptions with context where appropriate
- Avoid catching broad exceptions without logging or re-raising

## Logging

- Use the `logging` module instead of print statements
- Include appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Structure log messages to be informative but concise
