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
#ifndef SECURITY_COMPONENT_SA_APP_STATE_OBSERVER_APP_STATE_OBSERVER_H
#define SECURITY_COMPONENT_SA_APP_STATE_OBSERVER_APP_STATE_OBSERVER_H

#include <set>
#include <vector>
#include "app_mgr_interface.h"
#include "application_state_observer_stub.h"
#include "ffrt.h"
#include "iremote_object.h"

namespace OHOS {
namespace Security {
namespace SecurityComponent {
struct SecCompProcessData {
    std::string bundleName;
    int32_t pid = -1;
    int32_t uid = -1;
};

class AppStateObserver : public AppExecFwk::ApplicationStateObserverStub {
public:
    explicit AppStateObserver();
    virtual ~AppStateObserver();

    void OnProcessStateChanged(const AppExecFwk::ProcessData& processData) override;
    void OnProcessDied(const AppExecFwk::ProcessData& processData) override;
    bool IsProcessForeground(int32_t pid, int32_t uid);
    void AddProcessToForegroundSet(int32_t pid, const SecCompProcessData& data);
    void AddProcessToForegroundSet(const AppExecFwk::ProcessData& processData);
    void AddProcessToForegroundSet(const AppExecFwk::AppStateData& stateData);
    void DumpProcess(std::string& dumpStr);
    void OnAppCacheStateChanged(const AppExecFwk::AppStateData &appStateData) override;

private:
    void RemoveProcessFromForegroundSet(int32_t pid);
    std::vector<SecCompProcessData> foregrandProcList_;
    ffrt::shared_mutex fgProcLock_;
};
}  // namespace SecurityComponent
}  // namespace Security
}  // namespace OHOS

#endif  // SECURITY_COMPONENT_SA_APP_STATE_OBSERVER_APP_STATE_OBSERVER_H

