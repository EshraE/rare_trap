"""Guard production numerical code against accidental release refactors."""
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def syntax_fingerprint(node):
    # Canonical AST encoding independent of ast.dump formatting/Python 3.12's
    # new empty type_params field. Nonempty type parameters are never ignored.
    def encode(value):
        if isinstance(value, ast.AST):
            return {"node": type(value).__name__, "fields": {
                name: encode(item) for name, item in ast.iter_fields(value)
                if not (name == "type_params" and not item)}}
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        return value
    return hashlib.sha256(json.dumps(encode(node), sort_keys=True).encode()).hexdigest()


def test_production_modules_are_byte_identical():
    reference = json.loads((ROOT / "tests/production_fingerprints.json").read_text())
    for path, expected in reference["files"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path


def test_wrapped_production_functions_have_identical_syntax_trees():
    reference = json.loads((ROOT / "tests/production_fingerprints.json").read_text())
    for path, entries in reference["functions"].items():
        functions = {node.name: node for node in ast.parse((ROOT / path).read_text()).body
                     if isinstance(node, ast.FunctionDef)}
        for name, expected in entries.items():
            actual = syntax_fingerprint(functions[name])
            assert actual == expected, (path, name)


def test_projection_module_has_identical_syntax_tree():
    reference = json.loads((ROOT / "tests/production_fingerprints.json").read_text())
    for path, expected in reference["modules"].items():
        assert syntax_fingerprint(ast.parse((ROOT / path).read_text())) == expected
