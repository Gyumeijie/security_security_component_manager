#!/usr/bin/env python3
"""Scan BUILD.gn files and list dependencies for ohos_shared_library targets."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
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


VARIABLE_SCAN_CACHE: Dict[Tuple[str, str], Tuple[List[Tuple[str, str]], List[str]]] = {}


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
    external_deps: List[DepEdge]
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
    def repl_brace(m: re.Match[str]) -> str:
        return variables.get(m.group(1), m.group(0))

    def repl_plain(m: re.Match[str]) -> str:
        return variables.get(m.group(1), m.group(0))

    label = raw_label
    for _ in range(8):
        expanded = BRACE_INTERPOLATION_RE.sub(repl_brace, label)
        expanded = PLAIN_INTERPOLATION_RE.sub(repl_plain, expanded)
        expanded = normalize_leading_slashes(expanded)
        if expanded == label:
            break
        label = expanded
    return label


def unresolved_interpolation_step(raw_label: str, variables: Dict[str, str]) -> Tuple[int, str, str] | None:
    label = raw_label
    for step in range(1, 9):
        vars_in_step: List[str] = []
        for m in BRACE_INTERPOLATION_RE.finditer(label):
            vars_in_step.append(m.group(1))
        for m in PLAIN_INTERPOLATION_RE.finditer(label):
            vars_in_step.append(m.group(1))

        if not vars_in_step:
            return None

        for var_name in vars_in_step:
            if var_name not in variables:
                return (step, var_name, label)

        expanded = BRACE_INTERPOLATION_RE.sub(lambda m: variables.get(m.group(1), m.group(0)), label)
        expanded = PLAIN_INTERPOLATION_RE.sub(lambda m: variables.get(m.group(1), m.group(0)), expanded)
        expanded = normalize_leading_slashes(expanded)
        if expanded == label:
            return (step, vars_in_step[0], label)
        label = expanded

    if '$' in label:
        return (8, 'unknown', label)
    return None


def scan_variable_definitions(scan_root: str, variable_name: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    cache_key = (os.path.abspath(scan_root), variable_name)
    cached = VARIABLE_SCAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    exact_defs: List[Tuple[str, str]] = []
    non_string_defs: List[str] = []
    exact_re = re.compile(rf'\b{re.escape(variable_name)}\s*=\s*"([^"]*)"')
    assign_re = re.compile(rf'\b{re.escape(variable_name)}\s*=')

    for dir_path, _, files in os.walk(scan_root):
        for file_name in files:
            if not file_name.endswith(('.gn', '.gni')):
                continue
            full_path = os.path.join(dir_path, file_name)
            rel_path = os.path.relpath(full_path, scan_root).replace('\\', '/')
            try:
                content = strip_comments(open(full_path, 'r', encoding='utf-8').read())
            except OSError:
                continue

            found_exact = False
            for m in exact_re.finditer(content):
                exact_defs.append((rel_path, m.group(1)))
                found_exact = True

            if not found_exact and assign_re.search(content):
                non_string_defs.append(rel_path)

    result = (exact_defs, sorted(non_string_defs))
    VARIABLE_SCAN_CACHE[cache_key] = result
    return result


def resolve_label_candidates(
    raw_label: str,
    current_dir: str,
    variables: Dict[str, str],
    component_prefix: str = '',
    scan_root: str = '',
    current_file: str = '',
) -> List[str]:
    base = resolve_label(raw_label, current_dir, variables, component_prefix, '', current_file)
    candidates: List[str] = []
    if base and '$' not in base:
        candidates.append(base)

    if '$' not in raw_label or not scan_root:
        return candidates

    vars_in_label: List[str] = []
    for m in BRACE_INTERPOLATION_RE.finditer(raw_label):
        vars_in_label.append(m.group(1))
    for m in PLAIN_INTERPOLATION_RE.finditer(raw_label):
        vars_in_label.append(m.group(1))

    if not vars_in_label:
        return candidates

    var_name = vars_in_label[0]
    exact_defs, _ = scan_variable_definitions(scan_root, var_name)
    if exact_defs:
        for _, var_value in exact_defs:
            substituted = BRACE_INTERPOLATION_RE.sub(lambda m: var_value if m.group(1) == var_name else m.group(0), raw_label)
            substituted = PLAIN_INTERPOLATION_RE.sub(lambda m: var_value if m.group(1) == var_name else m.group(0), substituted)
            resolved = resolve_label(substituted, current_dir, variables, component_prefix, '', current_file)
            if resolved and resolved not in candidates:
                candidates.append(resolved)

    if not candidates:
        # Keep warning behavior for truly unresolved labels.
        resolve_label(raw_label, current_dir, variables, component_prefix, scan_root, current_file)

    return candidates


def explain_missing_variable(variable_name: str, current_file: str, scan_root: str) -> str:
    if not scan_root:
        return (
            'no --root context available; variables are resolved only from current BUILD.gn, '
            'its import() chain, and parsed global string assignments'
        )

    if not os.path.isdir(scan_root):
        return f'--root path does not exist or is not a directory: {scan_root}'

    exact_defs, non_string_defs = scan_variable_definitions(scan_root, variable_name)
    current_display = current_file if current_file else 'unknown'
    if exact_defs:
        examples = ', '.join([f'{f}={v}' for f, v in exact_defs[:3]])
        return (
            f'found definitions under --root and fallback expansion is enabled; '
            f'current_file={current_display}, tried_values=[{examples}]'
        )

    if non_string_defs:
        examples = ', '.join(non_string_defs[:3])
        return (
            f'found assignment in [{examples}] but value is not a plain quoted string; '
            'current parser only captures VAR = "..." string assignments'
        )

    return (
        f'no assignment found under --root={scan_root}; this usually means the variable is defined outside '
        '--root or behind unsupported GN constructs'
    )


def resolve_label(
    raw_label: str,
    current_dir: str,
    variables: Dict[str, str],
    component_prefix: str = '',
    scan_root: str = '',
    current_file: str = '',
) -> str | None:
    if '$' in raw_label and scan_root:
        unresolved = unresolved_interpolation_step(raw_label.strip(), variables)
        if unresolved is not None:
            step, var_name, expr = unresolved
            reason = explain_missing_variable(var_name, current_file, scan_root)
            print(
                f"[WARN] unresolved deps variable: step={step}, variable={var_name}, expression={expr}",
                file=sys.stderr,
            )
            print(
                f"[WARN] unresolved deps variable reason: variable={var_name}, current_file={current_file or 'unknown'}, reason={reason}",
                file=sys.stderr,
            )

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
        path = path.strip()
        if path.startswith('//'):
            normalized_path = normalize_leading_slashes(path)
            return f"{normalized_path}:{name}"

        if path in ('.', '..') or path.startswith('./') or path.startswith('../'):
            resolved_path = os.path.normpath(os.path.join(current_dir, path)).replace('\\', '/')
            if resolved_path == '.':
                return f"//:{name}"
            return f"//{resolved_path}:{name}"

        normalized_path = path.strip('/')
        return f"//{normalized_path}:{name}" if normalized_path else f"//:{name}"

    return None


def current_condition(stack: List[Tuple[str, str]]) -> str:
    conditions: List[str] = []
    for t, c in stack:
        if t == 'if_true':
            conditions.append(f'{c} = true')
        elif t == 'if_false':
            conditions.append(f'{c} = false')
    return ' && '.join(conditions)


def parse_target_deps(
    body: str,
    current_dir: str,
    variables: Dict[str, str],
    component_prefix: str,
    scan_root: str,
    current_file: str,
) -> Tuple[List[DepEdge], List[DepEdge]]:
    dep_edges: List[DepEdge] = []
    external_dep_edges: List[DepEdge] = []
    stack: List[Tuple[str, str]] = []  # (block_type, cond_expr)
    pending_else_cond = ''

    for m in EVENT_RE.finditer(body):
        token = m.group(0)
        if_expr = m.group(1)
        dep_type = m.group(2)
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
        cond = current_condition(stack)
        pending_else_cond = ''
        for label_match in LABEL_RE.finditer(dep_block or ''):
            if dep_type == 'external_deps':
                external_dep_edges.append(DepEdge(label_match.group(1), cond))
            else:
                resolved_candidates = resolve_label_candidates(
                    label_match.group(1),
                    current_dir,
                    variables,
                    component_prefix,
                    scan_root,
                    current_file,
                )
                for resolved in resolved_candidates:
                    dep_edges.append(DepEdge(resolved, cond))

    return dep_edges, external_dep_edges




def parse_target_cfi(body: str) -> bool:
    m = SANITIZE_RE.search(body)
    if not m:
        return False

    brace_index = m.end() - 1
    end = block_end(body, brace_index)
    sanitize_block = body[brace_index + 1:end]
    return CFI_TRUE_RE.search(sanitize_block) is not None

def parse_targets(root: str, component_prefix: str = '', diagnostic_root: str | None = None) -> Dict[str, Target]:
    targets: Dict[str, Target] = {}
    git_root = get_git_root(root)
    var_cache: Dict[str, Dict[str, str]] = {}
    global_vars = collect_global_variables(root)

    if diagnostic_root is None:
        diagnostic_root = root
    diagnostic_root = os.path.abspath(diagnostic_root)

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
            current_file = os.path.relpath(gn_path, diagnostic_root).replace('\\', '/')
            dep_edges, external_dep_edges = parse_target_deps(body, rel_dir, vars_in_file, component_prefix, diagnostic_root, current_file)

            target_line = content.count("\n", 0, match.start()) + 1
            target = Target(
                name=name,
                kind=kind,
                dir_path=rel_dir,
                deps=dep_edges,
                external_deps=external_dep_edges,
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
    target = targets.get(label)
    if target is not None:
        return target

    if component_prefix:
        target = targets.get(to_component_relative(label, component_prefix))
        if target is not None:
            return target

    for candidate in candidate_suffix_labels(label):
        target = targets.get(candidate)
        if target is not None:
            return target

    return None


def find_target_by_name(targets: Dict[str, Target], name: str, allowed_kinds: Set[str]) -> Target | None:
    seen: Set[str] = set()
    for target in targets.values():
        if target.key in seen:
            continue
        seen.add(target.key)
        if target.name == name and target.kind in allowed_kinds:
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
    targets: Dict[str, Target],
    shared_key: str,
    component_prefix: str = '',
    deps_all: bool = False,
    include_external_deps: bool = False,
    component_path_map: Dict[str, str] | None = None,
    cur_dir: str = '',
    scan_root: str = '',
) -> List[Tuple[int, str, str, str, bool, str, str, str, int, str, str]]:
    # (depth, label, kind_tag, condition, cfi_enabled, kind, name, dir_path, defined_line, root_arg, ext_component)
    entries: List[Tuple[int, str, str, str, bool, str, str, str, int, str, str]] = []
    best_depth: Dict[Tuple[str, str], int] = {}
    target_cache: Dict[str, Tuple[Dict[str, Target], str, str]] = {}

    def ensure_parsed_targets(component_root: str) -> Tuple[Dict[str, Target], str, str] | None:
        component_root = os.path.abspath(component_root)
        if not os.path.isdir(component_root):
            return None
        if component_root in target_cache:
            return target_cache[component_root]

        ext_git_root = get_git_root(component_root)
        ext_prefix = ''
        if ext_git_root is not None:
            ext_prefix = os.path.relpath(component_root, ext_git_root).replace('\\', '/')
            if ext_prefix == '.':
                ext_prefix = ''

        parsed = (parse_targets(component_root, ext_prefix, diagnostic_root=scan_root), ext_prefix, component_root)
        target_cache[component_root] = parsed
        return parsed

    def resolve_dep_target(
        cur_targets: Dict[str, Target],
        cur_component_prefix: str,
        dep_label: str,
        cur_root: str,
    ) -> Tuple[Dict[str, Target], str, str, str] | None:
        target = find_target_by_label(cur_targets, dep_label, cur_component_prefix)
        if target is not None:
            return (cur_targets, cur_component_prefix, cur_root, target.key)

        # deps = ["//path:target"] or resolved variable-path labels:
        # always try component-path lookup first so multiple variable definitions are attempted in order.
        if not dep_label.startswith('//') or ':' not in dep_label:
            print(
                f"[WARN] unresolved deps label: {dep_label}",
                file=sys.stderr,
            )
            return None

        dep_path, dep_name = dep_label[2:].split(':', 1)
        if not dep_path:
            print(
                f"[WARN] unresolved deps label: {dep_label}",
                file=sys.stderr,
            )
            return None

        dep_component_root = os.path.abspath(os.path.join(cur_dir, dep_path))
        parsed_targets = ensure_parsed_targets(dep_component_root)
        if parsed_targets is not None:
            ext_targets, ext_component_prefix, ext_root = parsed_targets
            ext_target = find_target_by_name(
                ext_targets,
                dep_name,
                {'ohos_executable', 'ohos_shared_library', 'ohos_static_library', 'ohos_source_set'},
            )
            if ext_target is not None:
                return (ext_targets, ext_component_prefix, ext_root, ext_target.key)

        print(
            f"[WARN] deps not found, component_path={dep_component_root}, dep_name={dep_name}",
            file=sys.stderr,
        )

        # Fallback for historical behavior (e.g. deps = [":name"]).
        by_name = find_target_by_name(
            cur_targets,
            dep_name,
            {'ohos_executable', 'ohos_shared_library', 'ohos_static_library', 'ohos_source_set'},
        )
        if by_name is not None:
            print(
                f"[WARN] deps path lookup failed, fallback to name lookup in current scope: dep_name={dep_name}, matched={by_name.key}",
                file=sys.stderr,
            )
            return (cur_targets, cur_component_prefix, cur_root, by_name.key)
        return None

    # path_seen stores nodes in current DFS path, avoiding cycles even if condition text changes.
    stack: Deque[Tuple[Dict[str, Target], str, str, int, str, Set[Tuple[str, str]], str]] = deque(
        [(targets, component_prefix, shared_key, 0, '', set(), scan_root or cur_dir)]
    )

    while stack:
        cur_targets, cur_component_prefix, cur, depth, cond, path_seen, cur_root = stack.pop()

        target = find_target_by_label(cur_targets, cur, cur_component_prefix)
        canonical_label = target.key if target is not None else cur
        state_key = (cur_root, canonical_label)
        if state_key in path_seen:
            continue

        if target is None:
            continue

        if depth > 0:
            kind_tag = target_kind_tag(target.kind)
            if kind_tag is not None:
                allowed = deps_all or target.kind in ('ohos_static_library', 'ohos_source_set')
                if allowed:
                    depth_key = (cur_root, canonical_label)
                    prev_depth = best_depth.get(depth_key)
                    if prev_depth is None or depth < prev_depth:
                        best_depth[depth_key] = depth
                        entries.append((depth, canonical_label, kind_tag, cond, target.cfi_enabled, target.kind, target.name, target.dir_path, target.defined_line, cur_root, ''))

        next_seen = set(path_seen)
        next_seen.add(state_key)

        for dep in reversed(target.deps):
            resolved_dep = resolve_dep_target(cur_targets, cur_component_prefix, dep.label, cur_root)
            if resolved_dep is None:
                continue
            dep_targets, dep_component_prefix, dep_root, dep_key = resolved_dep
            merged_cond = merge_condition(cond, dep.condition)
            stack.append((dep_targets, dep_component_prefix, dep_key, depth + 1, merged_cond, next_seen, dep_root))

        if include_external_deps and component_path_map is not None:
            for ext_dep in reversed(target.external_deps):
                if ':' not in ext_dep.label:
                    continue
                component_name, dep_name = ext_dep.label.split(':', 1)
                component_path = component_path_map.get(component_name)
                if component_path is None:
                    continue

                component_root = os.path.abspath(os.path.join(cur_dir, component_path))
                parsed_targets = ensure_parsed_targets(component_root)
                if parsed_targets is None:
                    continue

                ext_targets, ext_component_prefix, ext_root = parsed_targets
                ext_target = find_target_by_name(
                    ext_targets,
                    dep_name,
                    {'ohos_executable', 'ohos_shared_library', 'ohos_static_library', 'ohos_source_set'},
                )
                if ext_target is None:
                    continue

                merged_cond = merge_condition(cond, ext_dep.condition)
                kind_tag = target_kind_tag(ext_target.kind)
                if kind_tag is None:
                    continue
                allowed = deps_all or ext_target.kind in ('ohos_static_library', 'ohos_source_set')
                if not allowed:
                    continue

                depth_key = (ext_root, ext_target.key)
                prev_depth = best_depth.get(depth_key)
                ext_depth = depth + 1
                if prev_depth is None or ext_depth < prev_depth:
                    best_depth[depth_key] = ext_depth
                    entries.append((
                        ext_depth,
                        ext_target.key,
                        kind_tag,
                        merged_cond,
                        ext_target.cfi_enabled,
                        ext_target.kind,
                        ext_target.name,
                        ext_target.dir_path,
                        ext_target.defined_line,
                        ext_root,
                        component_name,
                    ))

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

def main() -> int:
    parser = argparse.ArgumentParser(description='Scan ohos_shared_library dependencies in BUILD.gn files.')
    parser.add_argument('--root', default='.', help='Repository/component root path (default: current directory).')
    parser.add_argument('--target', help='Only print dependencies for one target name (shared library or executable).')
    parser.add_argument('--all-targets', action='store_true', help='Auto scan all ohos_shared_library/ohos_executable (excluding test dirs/targets).')
    parser.add_argument('--deps-all', action='store_true', help='Print all dependency kinds (include shared_library/executable).')
    parser.add_argument('--details', action='store_true', help='Show detailed label format (full //path:target and [kind]).')
    parser.add_argument('--show-common-targets', action='store_true', help='Show common root target names instead of needs count in all-targets mode.')
    parser.add_argument('--show-path', action='store_true', help='Append BUILD.gn path for each printed target.')
    parser.add_argument('--external_deps', '--external-deps', dest='external_deps', action='store_true', help='Traverse supported targets referenced by external_deps.')
    parser.add_argument('--component-path', help='Path to component_path.json that maps component name to relative path.')
    parser.add_argument('--cur-dir', default='.', help='Root directory used with --component-path to resolve external_deps component paths.')
    args = parser.parse_args()

    component_path_map: Dict[str, str] | None = None
    if args.external_deps:
        if not args.component_path:
            print('--component-path is required when --external_deps is set.')
            return 1
        try:
            with open(args.component_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as ex:
            print(f'Failed to load --component-path file: {ex}')
            return 1

        if not isinstance(data, dict):
            print('--component-path file must be a JSON object mapping component name to path.')
            return 1

        component_path_map = {}
        for k, v in data.items():
            if isinstance(k, str) and isinstance(v, str):
                component_path_map[k] = v

    root = os.path.abspath(args.root)
    git_root = get_git_root(root)
    component_prefix = ''
    if git_root is not None:
        component_prefix = os.path.relpath(root, git_root).replace('\\', '/')
        if component_prefix == '.':
            component_prefix = ''

    targets = parse_targets(root, component_prefix)

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
        entries = collect_dependency_entries(
            targets,
            root_target.key,
            component_prefix,
            args.deps_all,
            include_external_deps=args.external_deps,
            component_path_map=component_path_map,
            cur_dir=os.path.abspath(args.cur_dir),
            scan_root=root,
        )
        root_cfi = root_target.cfi_enabled

        root_label = root_target.key if args.details else format_target_compact(root_target.kind, root_target.name)
        root_prefix = f"{root_index}. " if args.all_targets else ''
        root_path_part = f" [\033[1;34mpath:{format_target_path(args.root, root_target.dir_path, root_target.defined_line)}\033[0m]" if args.show_path else ''
        print(f"{root_prefix}{root_label}{format_cfi(root_cfi)}{root_path_part}")
        for depth, label, kind_tag, cond, cfi_enabled, kind, name, dir_path, defined_line, entry_root, ext_component in entries:
            indent = '    ' + '  ' * max(depth - 1, 0)
            mismatch = (cfi_enabled != root_cfi)
            dep_label = label if args.details else format_target_compact(kind, name)
            kind_part = format_kind(kind_tag) if args.details else ''
            path_part = f" [\033[1;34mpath:{format_target_path(entry_root, dir_path, defined_line)}\033[0m]" if args.show_path else ''
            targets_suffix = ''
            if args.all_targets and kind in ('ohos_static_library', 'ohos_source_set'):
                root_names = sorted(auto_dep_root_map.get(label, set()))
                if len(root_names) >= 2:
                    targets_suffix = format_common_dep_suffix(root_names, args.show_common_targets)
            ext_suffix = f' [ext:{ext_component}]' if ext_component else ''
            print(f"{indent}- {dep_label}{kind_part}{format_condition(cond)}{format_cfi(cfi_enabled, mismatch)}{path_part}{targets_suffix}{ext_suffix}")
        if args.all_targets and root_index < total_roots:
            print()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
