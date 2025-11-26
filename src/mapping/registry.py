"""Registry for declarative Triton→Gluon lowering specs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass(slots=True)
class MappingSpec:
    """Represents a single lowering rule loaded from YAML/JSON."""

    name: str
    gluon_template: str
    requirements: Dict[str, str]


class MappingRegistry:
    """Loads mapping specs from ``mapping/specs`` and exposes lookup APIs."""

    def __init__(self, specs_dir: Optional[Path] = None) -> None:
        base_dir = Path(__file__).resolve().parent
        self.specs_dir = specs_dir or base_dir / "specs"
        self._specs: Dict[str, MappingSpec] = {}
        self._load_specs()

    def lookup(self, op_name: str) -> Optional[MappingSpec]:
        """Return the :class:`MappingSpec` for ``op_name`` if present."""

        return self._specs.get(op_name)

    # ------------------------------------------------------------------ helpers
    def _load_specs(self) -> None:
        for path in self.specs_dir.glob("*.yml"):
            data = self._parse_stub_yaml(path)
            spec = MappingSpec(
                name=data.get("triton_op", path.stem),
                gluon_template=data.get("gluon_template", ""),
                requirements=data.get("requirements", {}),
            )
            self._specs[spec.name] = spec

    def _parse_stub_yaml(self, path: Path) -> Dict[str, object]:
        """Parse ``path`` using a permissive fallback parser.

        We intentionally avoid introducing a YAML dependency during scaffolding.
        The placeholder parser expects each file to contain a single JSON object
        or key-value pairs separated by ``:``.
        """

        text = path.read_text().strip()
        if text.startswith("{"):
            return json.loads(text)

        if yaml is not None:
            return yaml.safe_load(text)

        result: Dict[str, object] = {}
        current_key: Optional[str] = None
        for raw_line in text.splitlines():
            if not raw_line.strip() or raw_line.strip().startswith("#"):
                continue
            if raw_line.startswith(" ") and current_key:
                sub_key, sub_value = raw_line.split(":", 1)
                section = result.setdefault(current_key, {})
                if isinstance(section, dict):
                    section[sub_key.strip()] = sub_value.strip()
                continue
            key, value = raw_line.split(":", 1)
            value = value.strip()
            if value:
                result[key.strip()] = value
                current_key = None
            else:
                result[key.strip()] = {}
                current_key = key.strip()
        return result
