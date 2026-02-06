# GN 依赖扫描工具

`scan_gn_shared_deps.py` 用于扫描一个部件目录下所有 `BUILD.gn`，并输出根目标（`ohos_shared_library` 或 `ohos_executable`）的依赖。

- 依赖目标（`ohos_static_library` / `ohos_source_set`），并在每条依赖后标记类型

依赖来源包含 `deps`、`public_deps`、`external_deps`、`public_external_deps`，并按传递依赖遍历。

## external_deps 解析

默认会解析 `external_deps`，需要提供 `--component-path-info` 指向 `component_path_info.json`。

- `--no-external`：不解析 `external_deps`

## 变量路径补全

脚本会在 `--root-dir` 目录下搜索并解析：

- `.gn`
- `.gni`
- `bundle.json`

并结合 `BUILD.gn` 里的 `import("*.gni")` 递归收集字符串变量（如 `path_backup`），用于展开：

```gn
"${path_backup}/services/backup_sa:backup_sa_ipc"
"$path_backup/services/backup_sa:backup_sa_ipc"
```

## if/else 分支依赖分析

如果依赖定义在 `if (...) { deps += [...] } else { ... }` 分支中，脚本会把两个分支都输出，并附带条件：

- `libark_llvmcodegen [if (is_ohos) => true]`
- `libark_llvmcodegen_set [if (is_ohos) => false]`

并继续递归分析这两个分支目标各自的下游依赖。

说明：每一条依赖只显示“当前这一层边”上的条件，不叠加父层条件。

## 递归依赖输出

输出会按递归层级缩进，并为每条依赖标注：

- 类型标签：`[static_library]` 或 `[source_set]`

## CFI 标记输出

当依赖目标（`ohos_static_library` / `ohos_source_set`）包含：

```gn
sanitize = {
  cfi=true
}
```

启用 `--show-cfi-status` 后，脚本会为根目标（`--target` 或 `--all-targets` 选出的目标）及其依赖输出 CFI 状态：

- `cfi=true`：绿色
- `cfi=false`：默认红色
- 若依赖与根目标（`--target` 或 `--all-targets` 选中目标）CFI 状态不一致：依赖的 `cfi` 状态改为黄色

## 详细输出模式（--verbose）

默认使用简洁格式输出目标：

- `ohos_executable("target")`
- `ohos_shared_library("target")`
- `ohos_static_library("target")`
- `ohos_source_set("target")`

启用 `--verbose` 后，改为完整标签格式（`//path:target`）并附加类型标签（如 `[source_set]`）。

## 调试日志（--debug）

默认不打印解析日志。

启用 `--debug` 后，会输出 unresolved deps / deps not found 的诊断日志。

## 条件显示模式（--show-dep-condition）

默认不显示条件信息。

启用 `--show-dep-condition` 后，会在每个已打印依赖后追加：

- `[if (is_ohos) => true]`

## CFI 状态显示（--show-cfi-status）

默认不显示 CFI 状态。

启用 `--show-cfi-status` 后，会在每个已打印目标后追加：

- `[cfi=true]` / `[cfi=false]`

## 路径显示模式（--show-dep-path）

默认不显示路径信息。

启用 `--show-dep-path` 后，会在每个已打印目标后追加：

- `[path:<root>/<target_dir>/BUILD.gn:L123]`

其中 `<root>` 为 `--root-dir` 传入目录。

## 全量依赖模式（--all-deps-type）

默认仅输出 `ohos_static_library` 与 `ohos_source_set`。

启用 `--all-deps-type` 后，会输出所有可识别的依赖目标类型，包括：

- `[static_library]`
- `[source_set]`
- `[shared_library]`
- `[executable]`

## 单目标扫描（--target）

- `--target <name>`：扫描指定根目标（支持 `ohos_shared_library`、`ohos_executable`、`ohos_static_library`、`ohos_source_set`）
- 如果目标名称对应类型不在上述四种类型中，会提示不支持该类型查询

默认（不加 `--all-deps-type`）仅输出 `ohos_static_library` / `ohos_source_set`；
开启 `--all-deps-type` 后可输出所有依赖类型。

## 自动扫描模式（--all-targets）

启用 `--all-targets` 后，脚本会自动扫描组件目录下所有：

- `ohos_shared_library`
- `ohos_executable`

并从这些根目标递归分析其 `ohos_static_library` / `ohos_source_set` 依赖。

输出格式：

- 每个根目标前使用数字编号（从 1 开始）
- 根目标之间空一行

过滤规则：

- 目标名称包含 `test`（不区分大小写）会被忽略
- 目标所在目录路径任一段包含 `test`（不区分大小写）会被忽略

共享依赖标记规则：

- 在 `--all-targets` 模式下，如果某个 `ohos_static_library` / `ohos_source_set` 被 2 个及以上根目标依赖：
  - 指定 `--show-common-targets-num` 后，追加紫色加粗标记：`common_targets:{count}`（count 为根目标数量）

## 用法

```bash
python3 cfi_check_tools/scan_gn_shared_deps.py --root-dir .
python3 cfi_check_tools/scan_gn_shared_deps.py --root-dir . --target libso
python3 cfi_check_tools/scan_gn_shared_deps.py --root-dir . --target app_main
python3 scan_gn_shared_deps.py --root-dir foundation/filemanagement/app_file_service --target backup_sa
python3 scan_gn_shared_deps.py --root-dir foundation/filemanagement/app_file_service --target app_main
python3 scan_gn_shared_deps.py --root-dir foundation/filemanagement/app_file_service --all-targets
python3 scan_gn_shared_deps.py --root-dir foundation/filemanagement/app_file_service --target libA --all-deps-type
python3 scan_gn_shared_deps.py --root-dir foundation/filemanagement/app_file_service --all-targets --verbose
python3 scan_gn_shared_deps.py --root-dir foundation/filemanagement/app_file_service --all-targets --show-common-targets-num
python3 scan_gn_shared_deps.py --root-dir foundation/filemanagement/app_file_service --target libcompiler_service --show-dep-path
```

## 输出示例

```text
1. ohos_shared_library("libA") [cfi=true]
    - ohos_static_library("libB") [cfi=false] common_targets:2
      - ohos_static_library("libD") [cfi=false]
      - ohos_source_set("libE") [cfi=false]
    - ohos_source_set("libC") [cfi=false]

2. ohos_executable("app_main") [cfi=false]
    - ohos_source_set("libC") [cfi=false] common_targets:2
```

说明：以上示例在开启 `--show-cfi-status` 与 `--show-common-targets-num` 时输出。
