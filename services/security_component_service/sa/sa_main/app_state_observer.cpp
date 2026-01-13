/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "app_state_observer.h"

#include "sec_comp_log.h"
#include "sec_comp_manager.h"

namespace OHOS {
namespace Security {
namespace SecurityComponent {
namespace {
constexpr OHOS::HiviewDFX::HiLogLabel LABEL = {LOG_CORE, SECURITY_DOMAIN_SECURITY_COMPONENT, "AppStateObserver"};
constexpr int32_t APP_STATE_CACHED = 100;
}

AppStateObserver::AppStateObserver()
{
}

AppStateObserver::~AppStateObserver()
{
}

bool AppStateObserver::IsProcessForeground(int32_t pid, int32_t uid)
{
    std::unique_lock<ffrt::shared_mutex> infoGuard(this->fgProcLock_);
    for (auto iter = foregrandProcList_.begin(); iter != foregrandProcList_.end(); ++iter) {
        if (pid == iter->pid) {
            return true;
        }

        if ((iter->pid == -1) && (uid == iter->uid)) {
            iter->pid = pid;
            return true;
        }
    }
    return false;
}

void AppStateObserver::AddProcessToForegroundSet(int32_t pid, const SecCompProcessData& data)
{
    std::unique_lock<ffrt::shared_mutex> infoGuard(this->fgProcLock_);
    for (auto iter = foregrandProcList_.begin(); iter != foregrandProcList_.end(); ++iter) {
        if (pid == -1) {
            if (iter->uid == data.uid) {
                SC_LOG_INFO(LABEL, "uid %{public}d is already in foreground", data.uid);
                return;
            }
        } else if (pid == iter->pid) {
            SC_LOG_INFO(LABEL, "pid %{public}d is already in foreground", pid);
            return;
        }
    }
    foregrandProcList_.emplace_back(data);
}

void AppStateObserver::AddProcessToForegroundSet(const AppExecFwk::AppStateData& stateData)
{
    SecCompProcessData proc = {
        .bundleName = stateData.bundleName,
        .pid = stateData.pid,
        .uid = stateData.uid
    };
    AddProcessToForegroundSet(stateData.pid, proc);
}

void AppStateObserver::AddProcessToForegroundSet(const AppExecFwk::ProcessData &processData)
{
    SecCompProcessData proc = {
        .bundleName = processData.bundleName,
        .pid = processData.pid,
        .uid = processData.uid
    };
    AddProcessToForegroundSet(processData.pid, proc);
}

void AppStateObserver::RemoveProcessFromForegroundSet(int32_t pid)
{
    std::unique_lock<ffrt::shared_mutex> infoGuard(this->fgProcLock_);
    for (auto iter = foregrandProcList_.begin(); iter != foregrandProcList_.end(); ++iter) {
        if (pid == iter->pid) {
            foregrandProcList_.erase(iter);
            return;
        }
    }
}

void AppStateObserver::OnProcessStateChanged(const AppExecFwk::ProcessData &processData)
{
    if (processData.state == AppExecFwk::AppProcessState::APP_STATE_FOREGROUND) {
        AddProcessToForegroundSet(processData);
        SecCompManager::GetInstance().NotifyProcessForeground(processData.pid);
    } else if (processData.state == AppExecFwk::AppProcessState::APP_STATE_BACKGROUND) {
        RemoveProcessFromForegroundSet(processData.pid);
        SecCompManager::GetInstance().NotifyProcessBackground(processData.pid);
    }
}

void AppStateObserver::OnProcessDied(const AppExecFwk::ProcessData& processData)
{
    SC_LOG_INFO(LABEL, "OnProcessDied die %{public}s pid %{public}d",
        processData.bundleName.c_str(), processData.pid);
    RemoveProcessFromForegroundSet(processData.pid);
    SecCompManager::GetInstance().NotifyProcessDied(processData.pid, false);
}

void AppStateObserver::OnAppCacheStateChanged(const AppExecFwk::AppStateData &appStateData)
{
    SC_LOG_INFO(LABEL, "OnAppCacheStateChanged pid %{public}d",
        appStateData.pid);
    if (appStateData.state != APP_STATE_CACHED) {
        return;
    }
    RemoveProcessFromForegroundSet(appStateData.pid);
    SecCompManager::GetInstance().NotifyProcessDied(appStateData.pid, true);
}

void AppStateObserver::DumpProcess(std::string& dumpStr)
{
    std::unique_lock<ffrt::shared_mutex> infoGuard(this->fgProcLock_);
    for (auto iter = foregrandProcList_.begin(); iter != foregrandProcList_.end(); ++iter) {
        dumpStr.append("uid:" + std::to_string(iter->uid) + ", pid:" + std::to_string(iter->pid));
        dumpStr.append(", procName:" + iter->bundleName + "\n");
    }
}
}  // namespace SecurityComponent
}  // namespace Security
}  // namespace OHOS

