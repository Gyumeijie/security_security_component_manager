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
#ifndef SECURITY_COMPONENT_MANAGER_H
#define SECURITY_COMPONENT_MANAGER_H

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "accesstoken_kit.h"
#include "app_state_observer.h"
#include "ffrt.h"
#include "first_use_dialog.h"
#include "nocopyable.h"
#include "sec_comp_base.h"
#include "sec_comp_entity.h"
#include "sec_comp_info.h"
#include "sec_comp_malicious_apps.h"
#include "sec_event_handler.h"

namespace OHOS {
namespace Security {
namespace SecurityComponent {
struct SecCompCallerInfo {
    AccessToken::AccessTokenID tokenId;
    int32_t uid;
    int32_t pid;
};

struct ProcessCompInfos {
    std::vector<std::shared_ptr<SecCompEntity>> compList;
    bool isForeground = false;
    AccessToken::AccessTokenID tokenId;
};

class SecCompManager {
public:
    static SecCompManager& GetInstance();
    virtual ~SecCompManager() = default;

    int32_t RegisterSecurityComponent(SecCompType type, const nlohmann::json& jsonComponent,
        const SecCompCallerInfo& caller, int32_t& scId);
    int32_t UpdateSecurityComponent(int32_t scId, const nlohmann::json& jsonComponent,
        const SecCompCallerInfo& caller);
    int32_t UnregisterSecurityComponent(int32_t scId, const SecCompCallerInfo& caller);
    int32_t StartDialog(SecCompInfo& info, std::shared_ptr<SecCompEntity>& sc,
        const std::vector<sptr<IRemoteObject>>& remote, std::string& message);
    int32_t ReportSecurityComponentClickEvent(SecCompInfo& secCompInfo, const nlohmann::json& jsonComponent,
        const SecCompCallerInfo& caller, const std::vector<sptr<IRemoteObject>>& remote, std::string& message);
    int32_t CheckClickEventParams(const SecCompCallerInfo& caller, const std::vector<sptr<IRemoteObject>>& remote);
    void NotifyProcessForeground(int32_t pid);
    void NotifyProcessBackground(int32_t pid);
    void NotifyProcessDied(int32_t pid, bool isProcessCached);
    void DumpSecComp(std::string& dumpStr);
    bool Initialize();
    void ExitSaProcess();
    void ExitWhenAppMgrDied();
    int32_t AddSecurityComponentProcess(const SecCompCallerInfo& caller);
    bool HasCustomPermissionForSecComp();

private:
    SecCompManager();
    bool IsForegroundCompExist();
    bool IsCompExist();
    bool IsScIdExist(int32_t scId);
    int32_t AddSecurityComponentToList(int32_t pid,
        AccessToken::AccessTokenID tokenId, std::shared_ptr<SecCompEntity> newEntity);
    int32_t DeleteSecurityComponentFromList(int32_t pid, int32_t scId);
    std::shared_ptr<SecCompEntity> GetSecurityComponentFromList(int32_t pid, int32_t scId);
    int32_t CheckClickSecurityComponentInfo(std::shared_ptr<SecCompEntity> sc, int32_t scId,
        const nlohmann::json& jsonComponent,  const SecCompCallerInfo& caller, std::string& message);
    void SendCheckInfoEnhanceSysEvent(int32_t scId,
        SecCompType type, const std::string& scene, int32_t res);
    int32_t CreateScId();
    void GetFoldOffsetY(const CrossAxisState crossAxisState);

    ffrt::shared_mutex componentInfoLock_;
    std::mutex scIdMtx_;
    std::mutex superFoldOffsetMtx_;
    std::unordered_map<int32_t, ProcessCompInfos> componentMap_;
    int32_t scIdStart_;
    bool isSaExit_ = false;
    int32_t superFoldOffsetY_ = 0;

    std::shared_ptr<AppExecFwk::EventRunner> secRunner_;
    std::shared_ptr<SecEventHandler> secHandler_;
    SecCompMaliciousApps malicious_;

    std::function<void ()> exitSaProcessFunc_ = []() { return; };
    DISALLOW_COPY_AND_MOVE(SecCompManager);
};
}  // namespace SecurityComponent
}  // namespace Security
}  // namespace OHOS
#endif  // SECURITY_COMPONENT_MANAGER_H
