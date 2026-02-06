#!/usr/bin/env python3
"""Generate component_path_info.json from part_config.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


CONFIG_NAME = "part_config.json"
OUTPUT_NAME = "component_path_info.json"
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan part_config.json files and generate component name -> path mapping "
            "in component_path_info.json."
        )
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory used to resolve subsystem/component paths.",
    )
    parser.add_argument(
        "--parts-dir",
        help="Directory containing part_config.json files (recursive). Defaults to --root-dir.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_NAME,
        help=(
            "Output JSON file path. Relative paths are written under cfi_check_tools/ "
            "(default: component_path_info.json)."
        ),
    )
    return parser.parse_args()


def collect_parts(parts_dir: Path) -> Set[str]:
    parts: Set[str] = set()
    for config_path in sorted(parts_dir.rglob(CONFIG_NAME)):
        try:
            content = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            raise ValueError(f"Invalid JSON in {config_path}: {err}") from err

        if not isinstance(content, dict):
            continue
        config_parts = content.get("parts")
        if not isinstance(config_parts, dict):
            continue

        for part_key in config_parts.keys():
            if isinstance(part_key, str) and ":" in part_key:
                parts.add(part_key)
    return parts


def find_component_path(root: Path, subsystem: str, component: str) -> Path | None:
    rel_target = Path(subsystem) / component

    direct = root / rel_target
    if direct.is_dir():
        return rel_target

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        nested = child / rel_target
        if nested.is_dir():
            return Path(child.name) / rel_target

    return None


def resolve_component_path(root: Path, subsystem: str, component: str) -> Path | None:
    rel_path = find_component_path(root, subsystem, component)
    if rel_path is not None:
        return rel_path

    new_root = root / "vendor" / "huawei"
    if not new_root.is_dir():
        return None

    new_root_rel_path = find_component_path(new_root, subsystem, component)
    if new_root_rel_path is None:
        return None

    return Path("vendor") / "huawei" / new_root_rel_path


def normalize_subsystem(subsystem: str) -> str:
    if subsystem == "thirdparty":
        return "third_party"
    return subsystem


def build_mapping(root: Path, parts: Set[str]) -> Tuple[Dict[str, str], List[str]]:
    mapping: Dict[str, str] = {}
    errors: List[str] = []

    for full_name in sorted(parts):
        subsystem, component = full_name.split(":", 1)
        normalized_subsystem = normalize_subsystem(subsystem)
        rel_path = resolve_component_path(root, normalized_subsystem, component)
        if rel_path is None:
            errors.append(
                "Path not found for "
                f"{full_name} (expected {normalized_subsystem}/{component}, "
                "or under vendor/huawei)"
            )
            continue

        if component in mapping and mapping[component] != rel_path.as_posix():
            errors.append(
                "Duplicate component name with different paths: "
                f"{component} -> {mapping[component]} and {rel_path.as_posix()}"
            )
            continue

        mapping[component] = rel_path.as_posix()

    return mapping, errors


def main() -> int:
    args = parse_args()
    root = Path(args.root_dir).resolve()
    parts_dir = Path(args.parts_dir).resolve() if args.parts_dir else root

    if not root.is_dir():
        raise ValueError(f"Root directory does not exist: {root}")
    if not parts_dir.is_dir():
        raise ValueError(f"Parts directory does not exist: {parts_dir}")

    parts = collect_parts(parts_dir)
    mapping, errors = build_mapping(root, parts)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if errors:
        print("Found errors while resolving component paths:")
        for err in errors:
            print(f"- {err}")
        print(f"Generated partial mapping at: {output_path}")
        return 1

    print(f"Generated mapping for {len(mapping)} components: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
