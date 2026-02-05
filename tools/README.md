# GN 依赖扫描工具

`scan_gn_shared_deps.py` 用于扫描一个部件目录下所有 `BUILD.gn`，并输出根目标（`ohos_shared_library` 或 `ohos_executable`）的依赖。

- 依赖目标（`ohos_static_library` / `ohos_source_set`），并在每条依赖后标记类型

默认依赖来源包含 `deps`、`public_deps`，并按传递依赖遍历。

启用 `--external_deps`（兼容 `--external-deps`）后，会额外遍历 `external_deps`。


## external_deps 跨部件遍历（--external_deps）

启用 `--external_deps` 后，脚本会按与 `deps` 相同的递归逻辑继续遍历 `external_deps` 中的依赖，支持类型：

- `ohos_executable`
- `ohos_shared_library`
- `ohos_static_library`
- `ohos_source_set`

当 `external_deps` 依赖格式为 `"部件名:目标"` 时：

1. 通过 `--component-path <component_path.json>` 读取部件名到目录路径映射。
2. 将标签转换为 `//<component_dir>:<target>`。
3. 在目标部件目录的 `BUILD.gn` 中继续解析并递归遍历。
4. 查询该 external 目标时会优先且仅在映射目录（如 `third_party/icu`）下查找，不回退到 `--root` 下的同名目标。

`--external_deps`（或 `--external-deps`）需要配合 `--component-path` 使用。

## 变量路径补全

脚本会在 `--root` 目录下搜索并解析：

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

- `libark_llvmcodegen [is_ohos=true]`
- `libark_llvmcodegen_set [is_ohos=false]`

并继续递归分析这两个分支目标各自的下游依赖。

说明：每一条依赖只显示“当前这一层边”上的条件，不叠加父层条件。

## 递归依赖输出

输出会按递归层级缩进，并为每条依赖标注：

- 类型标签：`[static_library]` 或 `[source_set]`
- 分支条件（如有）：`[is_ohos=true]`
- CFI 状态

## CFI 标记输出

当依赖目标（`ohos_static_library` / `ohos_source_set`）包含：

```gn
sanitize = {
  cfi=true
}
```

脚本会为根目标（`--target` 或 `--all-targets` 选出的目标）及其依赖输出 CFI 状态：

- `cfi=true`：绿色
- `cfi=false`：默认红色
- 若依赖与根目标（`--target` 或 `--all-targets` 选中目标）CFI 状态不一致：依赖的 `cfi` 状态改为黄色

## 详细输出模式（--details）

默认使用简洁格式输出目标：

- `ohos_executable("target")`
- `ohos_shared_library("target")`
- `ohos_static_library("target")`
- `ohos_source_set("target")`

启用 `--details` 后，改为完整标签格式（`//path:target`）并附加类型标签（如 `[source_set]`）。

## 路径显示模式（--show-path）

默认不显示路径信息。

启用 `--show-path` 后，会在每个已打印目标后追加：

- `[path:<root>/<target_dir>/BUILD.gn:L123]`

其中 `<root>` 为 `--root` 传入目录。

## 全量依赖模式（--deps-all）

默认仅输出 `ohos_static_library` 与 `ohos_source_set`。

启用 `--deps-all` 后，会输出所有可识别的依赖目标类型，包括：

- `[static_library]`
- `[source_set]`
- `[shared_library]`
- `[executable]`

## 单目标扫描（--target）

- `--target <name>`：扫描指定根目标（支持 `ohos_shared_library` 或 `ohos_executable`）
- 如果目标名称对应类型不是 `ohos_shared_library`/`ohos_executable`，会提示不支持该类型查询

默认（不加 `--deps-all`）仅输出 `ohos_static_library` / `ohos_source_set`；
开启 `--deps-all` 后可输出所有依赖类型。

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
  - 默认追加紫色加粗标记：`common_targets:{count}`（count 为根目标数量）
  - 指定 `--show-common-targets` 后，改为紫色加粗：`common_targets:A,B,C`

## 用法

```bash
python3 tools/scan_gn_shared_deps.py --root .
python3 tools/scan_gn_shared_deps.py --root . --target libso
python3 tools/scan_gn_shared_deps.py --root . --target app_main
python3 scan_gn_shared_deps.py --root foundation/filemanagement/app_file_service --target backup_sa
python3 scan_gn_shared_deps.py --root foundation/filemanagement/app_file_service --target app_main
python3 scan_gn_shared_deps.py --root foundation/filemanagement/app_file_service --all-targets
python3 scan_gn_shared_deps.py --root foundation/filemanagement/app_file_service --target libA --deps-all
python3 scan_gn_shared_deps.py --root foundation/filemanagement/app_file_service --all-targets --details
python3 scan_gn_shared_deps.py --root foundation/filemanagement/app_file_service --all-targets --show-common-targets
python3 scan_gn_shared_deps.py --root foundation/filemanagement/app_file_service --target libcompiler_service --show-path
python3 scan_gn_shared_deps.py --root foundation/filemanagement/app_file_service --target libcompiler_service --external_deps --component-path out/component_path.json
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
