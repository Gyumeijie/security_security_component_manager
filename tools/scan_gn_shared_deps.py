#!/usr/bin/env python3
"""Scan BUILD.gn files and list dependencies for ohos_shared_library targets."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Set, Tuple

TARGET_RE = re.compile(r'(ohos_shared_library|ohos_static_library|ohos_source_set|ohos_executable)\("([^"]+)"\)\s*\{', re.M)
DEPS_RE = re.compile(r'(deps|public_deps|external_deps)\s*\+?=\s*\[(.*?)\]', re.S)
LABEL_RE = re.compile(r'"([^"]+)"')
VAR_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"')
BRACE_INTERPOLATION_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')
PLAIN_INTERPOLATION_RE = re.compile(r'(?<!\$)\$([A-Za-z_][A-Za-z0-9_]*)')
IMPORT_RE = re.compile(r'import\(\s*"([^"]+)"\s*\)')
EVENT_RE = re.compile(r'if\s*\(([^)]*)\)\s*\{|else\s*\{|\{|\}|(deps|public_deps|external_deps)\s*\+?=\s*\[(.*?)\]', re.S)
SANITIZE_RE = re.compile(r'sanitize\s*=\s*\{', re.S)
CFI_TRUE_RE = re.compile(r'\bcfi\s*=\s*true\b')


@dataclass(frozen=True)
class DepEdge:
    label: str
    condition: str


@dataclass(frozen=True)
class Target:
    name: str
    kind: str
    dir_path: str
    deps: List[DepEdge]
    cfi_enabled: bool
    defined_line: int

    @property
    def key(self) -> str:
        return f"//{self.dir_path}:{self.name}" if self.dir_path else f"//:{self.name}"


def strip_comments(text: str) -> str:
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)

    lines = []
    for line in text.splitlines():
        out = []
        in_string = False
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == '"' and (i == 0 or line[i - 1] != '\\'):
                in_string = not in_string
                out.append(ch)
                i += 1
                continue
            if not in_string and ch == '/' and i + 1 < len(line) and line[i + 1] == '/':
                break
            out.append(ch)
            i += 1
        lines.append(''.join(out))
    return '\n'.join(lines)


def block_end(text: str, start_index: int) -> int:
    depth = 0
    for i in range(start_index, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def normalize_dir(root: str, file_path: str) -> str:
    directory = os.path.dirname(os.path.relpath(file_path, root))
    return '' if directory == '.' else directory.replace('\\', '/')


def get_git_root(start_path: str) -> str | None:
    cur = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(cur, '.git')):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def parse_string_variables(content: str) -> Dict[str, str]:
    variables: Dict[str, str] = {}
    for match in VAR_RE.finditer(content):
        variables[match.group(1)] = match.group(2)
    return variables


def resolve_import_path(import_label: str, current_file: str, git_root: str | None) -> str | None:
    label = import_label.strip()
    if label.startswith('//'):
        if git_root is None:
            return None
        path = os.path.join(git_root, label[2:])
    else:
        path = os.path.join(os.path.dirname(current_file), label)

    path = os.path.normpath(path)
    if not os.path.isfile(path):
        return None
    return path


def collect_variables_for_file(
    gn_path: str,
    git_root: str | None,
    cache: Dict[str, Dict[str, str]],
    visiting: Set[str],
) -> Dict[str, str]:
    if gn_path in cache:
        return cache[gn_path]
    if gn_path in visiting:
        return {}
    visiting.add(gn_path)

    try:
        content = strip_comments(open(gn_path, 'r', encoding='utf-8').read())
    except OSError:
        visiting.remove(gn_path)
        cache[gn_path] = {}
        return {}

    variables: Dict[str, str] = {}
    for m in IMPORT_RE.finditer(content):
        import_file = resolve_import_path(m.group(1), gn_path, git_root)
        if import_file is None:
            continue
        variables.update(collect_variables_for_file(import_file, git_root, cache, visiting))

    variables.update(parse_string_variables(content))
    visiting.remove(gn_path)
    cache[gn_path] = variables
    return variables


def collect_global_variables(root: str) -> Dict[str, str]:
    variables: Dict[str, str] = {}
    for dir_path, _, files in os.walk(root):
        for file_name in files:
            full_path = os.path.join(dir_path, file_name)
            if file_name.endswith(('.gn', '.gni')):
                try:
                    content = strip_comments(open(full_path, 'r', encoding='utf-8').read())
                except OSError:
                    continue
                variables.update(parse_string_variables(content))
            elif file_name == 'bundle.json':
                try:
                    data = json.load(open(full_path, 'r', encoding='utf-8'))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, str) and v.startswith('//') and (k.startswith('path_') or k.endswith('_path')):
                            variables[k] = v
    return variables


def normalize_leading_slashes(label: str) -> str:
    if label.startswith('//'):
        return '//' + label.lstrip('/')
    return label


def interpolate_label(raw_label: str, variables: Dict[str, str], component_prefix: str = '') -> str:
    fallback = f"//{component_prefix}" if component_prefix else ''

    def repl_brace(m: re.Match[str]) -> str:
        return variables.get(m.group(1), fallback if fallback else m.group(0))

    def repl_plain(m: re.Match[str]) -> str:
        return variables.get(m.group(1), fallback if fallback else m.group(0))

    label = raw_label
    for _ in range(8):
        expanded = BRACE_INTERPOLATION_RE.sub(repl_brace, label)
        expanded = PLAIN_INTERPOLATION_RE.sub(repl_plain, expanded)
        expanded = normalize_leading_slashes(expanded)
        if expanded == label:
            break
        label = expanded
    return label


def resolve_label(raw_label: str, current_dir: str, variables: Dict[str, str], component_prefix: str = '') -> str | None:
    label = interpolate_label(raw_label.strip(), variables, component_prefix)
    if not label:
        return None

    if label.startswith('@'):
        return None

    if label.startswith(':'):
        return f"//{current_dir}{label}" if current_dir else f"//{label}"

    if label.startswith('//'):
        if ':' in label:
            return normalize_leading_slashes(label)
        last = label.rstrip('/').split('/')[-1]
        return f"{label}:{last}"

    if ':' in label:
        path, name = label.split(':', 1)
        path = path.strip('./')
        return f"//{path}:{name}" if path else f"//:{name}"

    return None


def current_condition(stack: List[Tuple[str, str]]) -> str:
    conditions: List[str] = []
    for t, c in stack:
        if t == 'if_true':
            conditions.append(f'{c} = true')
        elif t == 'if_false':
            conditions.append(f'{c} = false')
    return ' && '.join(conditions)


def resolve_external_dep_label(raw_label: str, component_paths: Dict[str, str]) -> str | None:
    raw = raw_label.strip()
    if ':' not in raw:
        return None

    component_name, target_name = raw.split(':', 1)
    component_name = component_name.strip()
    target_name = target_name.strip()
    if not component_name or not target_name:
        return None

    component_dir = component_paths.get(component_name)
    if not component_dir:
        return None

    return f"{component_dir.strip('/')}:{target_name}"


def parse_target_deps(
    body: str,
    current_dir: str,
    variables: Dict[str, str],
    component_prefix: str,
    include_external_deps: bool,
    component_paths: Dict[str, str],
) -> List[DepEdge]:
    dep_edges: List[DepEdge] = []
    stack: List[Tuple[str, str]] = []  # (block_type, cond_expr)
    pending_else_cond = ''

    for m in EVENT_RE.finditer(body):
        token = m.group(0)
        if_expr = m.group(1)
        dep_kind = m.group(2)
        dep_block = m.group(3)

        if if_expr is not None and token.startswith('if'):
            stack.append(('if_true', if_expr.strip()))
            pending_else_cond = ''
            continue

        if token.startswith('else'):
            if pending_else_cond:
                stack.append(('if_false', pending_else_cond))
                pending_else_cond = ''
            else:
                stack.append(('block', ''))
            continue

        if token == '{':
            stack.append(('block', ''))
            pending_else_cond = ''
            continue

        if token == '}':
            if stack:
                block_type, cond_expr = stack.pop()
                if block_type == 'if_true':
                    pending_else_cond = cond_expr
                else:
                    pending_else_cond = ''
            else:
                pending_else_cond = ''
            continue

        # dependency assignment event
        if dep_kind == 'external_deps' and not include_external_deps:
            continue

        cond = current_condition(stack)
        pending_else_cond = ''
        for label_match in LABEL_RE.finditer(dep_block or ''):
            raw_label = label_match.group(1)
            if dep_kind == 'external_deps':
                resolved = resolve_external_dep_label(raw_label, component_paths)
            else:
                resolved = resolve_label(raw_label, current_dir, variables, component_prefix)
            if resolved is not None:
                dep_edges.append(DepEdge(resolved, cond))

    return dep_edges




def parse_target_cfi(body: str) -> bool:
    m = SANITIZE_RE.search(body)
    if not m:
        return False

    brace_index = m.end() - 1
    end = block_end(body, brace_index)
    sanitize_block = body[brace_index + 1:end]
    return CFI_TRUE_RE.search(sanitize_block) is not None

def parse_targets(
    root: str,
    component_prefix: str = '',
    include_external_deps: bool = False,
    component_paths: Dict[str, str] | None = None,
) -> Dict[str, Target]:
    targets: Dict[str, Target] = {}
    git_root = get_git_root(root)
    var_cache: Dict[str, Dict[str, str]] = {}
    global_vars = collect_global_variables(root)

    if component_paths is None:
        component_paths = {}

    for dir_path, _, files in os.walk(root):
        if 'BUILD.gn' not in files:
            continue
        gn_path = os.path.join(dir_path, 'BUILD.gn')
        content = strip_comments(open(gn_path, 'r', encoding='utf-8').read())
        rel_dir = normalize_dir(root, gn_path)

        vars_in_file = dict(global_vars)
        vars_in_file.update(collect_variables_for_file(gn_path, git_root, var_cache, set()))

        rel_dir_git = ''
        if git_root is not None:
            rel_dir_git = normalize_dir(git_root, gn_path)

        for match in TARGET_RE.finditer(content):
            kind, name = match.group(1), match.group(2)
            body_start = match.end() - 1
            body_end = block_end(content, body_start)
            body = content[body_start + 1:body_end]
            dep_edges = parse_target_deps(
                body,
                rel_dir,
                vars_in_file,
                component_prefix,
                include_external_deps,
                component_paths,
            )

            target_line = content.count("\n", 0, match.start()) + 1
            target = Target(
                name=name,
                kind=kind,
                dir_path=rel_dir,
                deps=dep_edges,
                cfi_enabled=parse_target_cfi(body),
                defined_line=target_line,
            )
            targets[target.key] = target
            if rel_dir_git and rel_dir_git != rel_dir:
                targets[f"//{rel_dir_git}:{name}"] = target

    return targets


def to_component_relative(label: str, component_prefix: str) -> str:
    if not component_prefix:
        return label
    prefix = f"//{component_prefix}/"
    if label.startswith(prefix):
        return f"//{label[len(prefix):]}"
    return label


def candidate_suffix_labels(label: str) -> List[str]:
    if not label.startswith('//') or ':' not in label:
        return []
    path_part, name = label[2:].split(':', 1)
    parts = [p for p in path_part.split('/') if p]
    return [f"//{'/'.join(parts[i:])}:{name}" for i in range(1, len(parts))]


def find_target_by_label(targets: Dict[str, Target], label: str, component_prefix: str = '') -> Target | None:
    normalized_label = label
    if not normalized_label.startswith('//') and ':' in normalized_label:
        normalized_label = f"//{normalized_label.lstrip('/')}"

    target = targets.get(normalized_label)
    if target is not None:
        return target

    if component_prefix:
        target = targets.get(to_component_relative(normalized_label, component_prefix))
        if target is not None:
            return target

    for candidate in candidate_suffix_labels(normalized_label):
        target = targets.get(candidate)
        if target is not None:
            return target

    return None




def target_kind_tag(kind: str) -> str | None:
    mapping = {
        'ohos_static_library': 'static_library',
        'ohos_source_set': 'source_set',
        'ohos_shared_library': 'shared_library',
        'ohos_executable': 'executable',
    }
    return mapping.get(kind)


def format_target_compact(kind: str, name: str) -> str:
    return f'{kind}("{name}")'


def format_target_path(root_arg: str, dir_path: str, defined_line: int) -> str:
    base = root_arg.rstrip("/")
    rel = f"{dir_path}/BUILD.gn" if dir_path else "BUILD.gn"
    rel_with_line = f"{rel}:L{defined_line}"
    if base in ("", "."):
        return rel_with_line
    return f"{base}/{rel_with_line}"

def merge_condition(parent: str, edge: str) -> str:
    if parent and edge:
        return f'{parent} && {edge}'
    return edge or parent


def collect_dependency_entries(
    targets: Dict[str, Target], shared_key: str, component_prefix: str = '', deps_all: bool = False
) -> List[Tuple[int, str, str, str, bool, str, str, str, int]]:
    # (depth, label, kind_tag, condition, cfi_enabled, kind, name, dir_path, defined_line)
    entries: List[Tuple[int, str, str, str, bool, str, str, str, int]] = []
    best_depth: Dict[str, int] = {}

    stack: Deque[Tuple[str, int, str, Set[Tuple[str, str]]]] = deque([(shared_key, 0, '', set())])

    while stack:
        cur, depth, cond, path_seen = stack.pop()

        target = find_target_by_label(targets, cur, component_prefix)
        canonical_label = target.key if target is not None else cur
        state_key = (canonical_label, cond)
        if state_key in path_seen:
            continue

        if target is None:
            continue

        if depth > 0:
            kind_tag = target_kind_tag(target.kind)
            if kind_tag is not None:
                allowed = deps_all or target.kind in ('ohos_static_library', 'ohos_source_set')
                if allowed:
                    prev_depth = best_depth.get(canonical_label)
                    if prev_depth is None or depth < prev_depth:
                        best_depth[canonical_label] = depth
                        entries.append((depth, canonical_label, kind_tag, cond, target.cfi_enabled, target.kind, target.name, target.dir_path, target.defined_line))

        next_seen = set(path_seen)
        next_seen.add(state_key)

        for dep in reversed(target.deps):
            stack.append((dep.label, depth + 1, dep.condition, next_seen))

    return entries



def format_condition(cond: str) -> str:
    if not cond:
        return ''
    normalized = cond.replace(' = ', '=')
    return f' [{normalized}]'


def format_kind(kind_tag: str) -> str:
    return f' [{kind_tag}]'


def format_cfi(cfi_enabled: bool, mismatch: bool = False) -> str:
    label = 'cfi=true' if cfi_enabled else 'cfi=false'
    if mismatch:
        return f' [\033[33m{label}\033[0m]'
    if cfi_enabled:
        return f' [\033[32m{label}\033[0m]'
    return f' [\033[31m{label}\033[0m]'






def collect_auto_dep_root_map(
    targets: Dict[str, Target],
    root_targets: List[Target],
    component_prefix: str,
) -> Dict[str, Set[str]]:
    dep_roots: Dict[str, Set[str]] = {}

    for root in root_targets:
        root_name = root.name
        seen_for_root: Set[str] = set()
        stack: Deque[Tuple[str, Set[str]]] = deque([(root.key, set())])

        while stack:
            cur, path_seen = stack.pop()
            target = find_target_by_label(targets, cur, component_prefix)
            canonical_label = target.key if target is not None else cur
            if canonical_label in path_seen:
                continue
            if target is None:
                continue

            if target.kind in ('ohos_static_library', 'ohos_source_set'):
                seen_for_root.add(canonical_label)

            next_seen = set(path_seen)
            next_seen.add(canonical_label)
            for dep in reversed(target.deps):
                stack.append((dep.label, next_seen))

        for dep_label in seen_for_root:
            dep_roots.setdefault(dep_label, set()).add(root_name)

    return dep_roots


def format_common_dep_suffix(target_names: List[str], show_common_targets: bool) -> str:
    if show_common_targets:
        text = f"common_targets:{','.join(target_names)}"
    else:
        text = f"common_targets:{len(target_names)}"
    return f" [\033[1;35m{text}\033[0m]"

def is_test_scope(target: Target) -> bool:
    if 'test' in target.name.lower():
        return True
    if target.dir_path:
        for seg in target.dir_path.split('/'):
            if 'test' in seg.lower():
                return True
    return False


def collect_auto_roots(targets: Dict[str, Target]) -> List[Target]:
    roots: List[Target] = []
    seen: Set[str] = set()
    for t in targets.values():
        if t.key in seen:
            continue
        seen.add(t.key)
        if t.kind not in ('ohos_shared_library', 'ohos_executable'):
            continue
        if is_test_scope(t):
            continue
        roots.append(t)
    return sorted(roots, key=lambda x: x.key)


def load_component_paths(component_path_file: str) -> Dict[str, str]:
    data = json.load(open(component_path_file, 'r', encoding='utf-8'))
    if not isinstance(data, dict):
        return {}
    paths: Dict[str, str] = {}
    for name, path in data.items():
        if isinstance(name, str) and isinstance(path, str):
            paths[name] = path.strip('/').replace('\\', '/')
    return paths


def merge_external_component_targets(
    targets: Dict[str, Target],
    git_root: str | None,
    root: str,
    component_paths: Dict[str, str],
    include_external_deps: bool,
) -> Dict[str, Target]:
    merged = dict(targets)
    if not include_external_deps:
        return merged

    parse_root = git_root if git_root is not None else os.path.dirname(root)
    seen_component_roots: Set[str] = {os.path.normpath(root)}
    for component_dir in component_paths.values():
        component_root = component_dir
        if not os.path.isabs(component_root):
            component_root = os.path.join(parse_root, component_root)
        component_root = os.path.normpath(component_root)
        if component_root in seen_component_roots or not os.path.isdir(component_root):
            continue
        seen_component_roots.add(component_root)

        component_prefix = ''
        if git_root is not None:
            component_prefix = os.path.relpath(component_root, git_root).replace('\\', '/')
            if component_prefix == '.':
                component_prefix = ''

        merged.update(parse_targets(component_root, component_prefix, include_external_deps, component_paths))

    return merged

def main() -> int:
    parser = argparse.ArgumentParser(description='Scan ohos_shared_library dependencies in BUILD.gn files.')
    parser.add_argument('--root', default='.', help='Repository/component root path (default: current directory).')
    parser.add_argument('--target', help='Only print dependencies for one target name (shared library or executable).')
    parser.add_argument('--all-targets', action='store_true', help='Auto scan all ohos_shared_library/ohos_executable (excluding test dirs/targets).')
    parser.add_argument('--deps-all', action='store_true', help='Print all dependency kinds (include shared_library/executable).')
    parser.add_argument('--details', action='store_true', help='Show detailed label format (full //path:target and [kind]).')
    parser.add_argument('--show-common-targets', action='store_true', help='Show common root target names instead of needs count in all-targets mode.')
    parser.add_argument('--show-path', action='store_true', help='Append BUILD.gn path for each printed target.')
    parser.add_argument('--external_deps', '--external-deps', dest='external_deps', action='store_true', help='Include and traverse external_deps dependencies.')
    parser.add_argument('--component-path', help='Path to component_path.json (component name -> component directory).')
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    git_root = get_git_root(root)
    component_prefix = ''
    if git_root is not None:
        component_prefix = os.path.relpath(root, git_root).replace('\\', '/')
        if component_prefix == '.':
            component_prefix = ''

    if args.external_deps and not args.component_path:
        print('Missing required argument: --component-path must be specified when using --external_deps.')
        return 1

    component_paths: Dict[str, str] = {}
    if args.component_path:
        try:
            component_paths = load_component_paths(args.component_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f'Failed to load --component-path file: {exc}')
            return 1

    targets = parse_targets(root, component_prefix, args.external_deps, component_paths)
    targets = merge_external_component_targets(targets, git_root, root, component_paths, args.external_deps)

    if args.all_targets:
        root_targets = collect_auto_roots(targets)
    else:
        root_map: Dict[str, Target] = {}
        unsupported_found = False
        for t in targets.values():
            if args.target is not None:
                if t.name != args.target:
                    continue
                if t.kind in ('ohos_shared_library', 'ohos_executable'):
                    root_map[t.key] = t
                else:
                    unsupported_found = True
            else:
                if t.kind == 'ohos_shared_library':
                    root_map[t.key] = t
        root_targets = sorted(list(root_map.values()), key=lambda x: x.key)

        if args.target is not None and not root_targets and unsupported_found:
            print('Unsupported target kind: only ohos_executable and ohos_shared_library are supported for --target.')
            return 1

    if not root_targets:
        print('No matching target found.')
        return 1

    auto_dep_root_map: Dict[str, Set[str]] = {}
    if args.all_targets:
        auto_dep_root_map = collect_auto_dep_root_map(targets, root_targets, component_prefix)

    total_roots = len(root_targets)
    for root_index, root_target in enumerate(root_targets, start=1):
        entries = collect_dependency_entries(targets, root_target.key, component_prefix, args.deps_all)
        root_cfi = root_target.cfi_enabled

        root_label = root_target.key if args.details else format_target_compact(root_target.kind, root_target.name)
        root_prefix = f"{root_index}. " if args.all_targets else ''
        root_path_part = f" [\033[1;34mpath:{format_target_path(args.root, root_target.dir_path, root_target.defined_line)}\033[0m]" if args.show_path else ''
        print(f"{root_prefix}{root_label}{format_cfi(root_cfi)}{root_path_part}")
        for depth, label, kind_tag, cond, cfi_enabled, kind, name, dir_path, defined_line in entries:
            indent = '    ' + '  ' * max(depth - 1, 0)
            mismatch = (cfi_enabled != root_cfi)
            dep_label = label if args.details else format_target_compact(kind, name)
            kind_part = format_kind(kind_tag) if args.details else ''
            path_part = f" [\033[1;34mpath:{format_target_path(args.root, dir_path, defined_line)}\033[0m]" if args.show_path else ''
            targets_suffix = ''
            if args.all_targets and kind in ('ohos_static_library', 'ohos_source_set'):
                root_names = sorted(auto_dep_root_map.get(label, set()))
                if len(root_names) >= 2:
                    targets_suffix = format_common_dep_suffix(root_names, args.show_common_targets)
            print(f"{indent}- {dep_label}{kind_part}{format_condition(cond)}{format_cfi(cfi_enabled, mismatch)}{path_part}{targets_suffix}")
        if args.all_targets and root_index < total_roots:
            print()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
