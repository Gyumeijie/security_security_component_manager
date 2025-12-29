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
#ifndef SECURITY_COMPONENT_PERMISSION_MANAGER_H
#define SECURITY_COMPONENT_PERMISSION_MANAGER_H

#include <deque>
#include <map>
#include <set>
#include "accesstoken_kit.h"
#include "ffrt.h"
#include "sec_comp_base.h"
#include "sec_event_handler.h"

namespace OHOS {
namespace Security {
namespace SecurityComponent {
class SecCompPermManager {
public:
    SecCompPermManager() = default;
    virtual ~SecCompPermManager() = default;
    static SecCompPermManager& GetInstance();

    int32_t GrantTempPermission(AccessToken::AccessTokenID tokenId,
        const std::shared_ptr<SecCompBase>& componentInfo);
    int32_t GrantTempSavePermission(AccessToken::AccessTokenID tokenId);
    void RevokeTempSavePermission(AccessToken::AccessTokenID tokenId);
    bool VerifySavePermission(AccessToken::AccessTokenID tokenId);
    bool VerifyPermission(AccessToken::AccessTokenID tokenId, SecCompType type);

    int32_t GrantAppPermission(AccessToken::AccessTokenID tokenId, const std::string& permissionName);
    int32_t RevokeAppPermission(AccessToken::AccessTokenID tokenId, const std::string& permissionName);
    void RevokeAppPermissions(AccessToken::AccessTokenID tokenId);

    void InitEventHandler(const std::shared_ptr<SecEventHandler>& secHandler);
    std::shared_ptr<SecEventHandler> GetSecEventHandler() const;

    void RevokeAppPermisionsDelayed(AccessToken::AccessTokenID tokenId);
    void CancelAppRevokingPermisions(AccessToken::AccessTokenID tokenId);

private:
    bool DelaySaveRevokePermission(AccessToken::AccessTokenID tokenId, const std::string& taskName);
    bool RevokeSavePermissionTask(const std::string& taskName);
    void RevokeTempSavePermissionCount(AccessToken::AccessTokenID tokenId);
    void RevokeAppPermisionsImmediately(AccessToken::AccessTokenID tokenId);

    void AddAppGrantPermissionRecord(AccessToken::AccessTokenID tokenId,
        const std::string& permissionName);
    void RemoveAppGrantPermissionRecord(AccessToken::AccessTokenID tokenId,
        const std::string& permissionName);

    std::unordered_map<AccessToken::AccessTokenID, int32_t> applySaveCountMap_;
    std::unordered_map<AccessToken::AccessTokenID, std::deque<std::string>> saveTaskDequeMap_;
    std::mutex mutex_;
    std::shared_ptr<SecEventHandler> secHandler_;

    std::mutex grantMtx_;
    std::unordered_map<int32_t, std::set<std::string>> grantMap_;
};
}  // namespace SecurityComponent
}  // namespace Security
}  // namespace OHOS
#endif  // SECURITY_COMPONENT_PERMISSION_MANAGER_H
