# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the OpenHarmony Security Component Manager Service - a system ability (SA ID: 3506) that allows applications to access temporary permissions through UI components without permanent authorization. When a user clicks a security component (e.g., PasteButton, SaveButton), the app receives temporary permission to access sensitive data.

## Build System

This project uses OpenHarmony's GN (Generate Ninja) build system.

**Main build files:**
- `BUILD.gn` - Main build configuration
- `security_component.gni` - Feature flags for enhancement capabilities
- `bundle.json` - Component metadata and dependencies

**Build targets:**
- `security_component_build_module` - Main service module
- `security_component_build_module_test` - Unit tests
- `security_component_build_fuzz_test` - Fuzz tests

**Build commands:**
```bash
# Build main service
./build.sh --product-name <product> --build-variant root --build-target security_component_manager

# Build tests
./build.sh --product-name <product> --build-variant root --build-target security_component_build_module_test
./build.sh --product-name <product> --build-variant root --build-target security_component_build_fuzz_test
```

## Architecture

The codebase follows OpenHarmony's System Ability pattern with three main layers:

### 1. Service Layer (`services/security_component_service/sa/`)
- `SecCompService` - Main SystemAbility class, inherits from `SystemAbility`
- `SecCompManager` - Core business logic manager (singleton pattern)
- `SecCompEntity` - Represents individual security components
- `AppStateObserver` - Monitors app lifecycle (foreground/background transitions)

### 2. Framework Layer (`frameworks/`)
- `security_component/` - Core framework implementation
- `enhance_adapter/` - Adaptation layer for vendor enhancements
- `common/` - Shared utilities and base classes

### 3. Interface Layer (`interfaces/inner_api/`)
- `security_component/` - Main service APIs (register, update, unregister, report click)
- `enhance_kits/` - Enhancement configuration APIs
- `security_component_common/` - Common utilities and definitions

## Component Types

- `PasteButton` - Clipboard access
- `SaveButton` - Save operations

## Testing

**Unit tests:** `frameworks/*/test/` and `services/*/test/unittest/`
**Fuzz tests:** `test/fuzztest/security_component/` - Tests all major APIs
**Test mocks:** `test/fuzztest/security_component/mock/`

## Important Design Patterns

- **Singleton**: `SecCompManager` manages global state
- **Observer**: `AppStateObserver` for app lifecycle events
- **Factory**: Component creation and type handling
- **Delayed tasks**: `DelayExitTask` for cleanup operations

## Dependencies

Key OpenHarmony components:
- `samgr` - System Ability Manager
- `ipc_core` - IPC communication
- `access_token` - Permission tokens
- `ability_runtime` - Application lifecycle
- `window_manager` - Window management
- `hilog` - Logging
- `hisysevent` - System events
