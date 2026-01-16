/*
 * Copyright (c) 2023-2024 Huawei Device Co., Ltd.
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
#ifndef INTERFACES_INNER_API_SECURITY_COMPONENT_KIT_H
#define INTERFACES_INNER_API_SECURITY_COMPONENT_KIT_H
#define SECURITY_COMPONENT_API_CALLER __attribute__((optnone))

#include <string>
#include "accesstoken_kit.h"
#include "iremote_object.h"
#include "sec_comp_info.h"
#include "sec_comp_ui_register.h"

namespace OHOS {
namespace Security {
namespace SecurityComponent {
class __attribute__((visibility("default"))) SecCompKit {
public:
    static int32_t RegisterSecurityComponent(SecCompType type, std::string& componentInfo, int32_t& scId);
    static int32_t UpdateSecurityComponent(int32_t scId, std::string& componentInfo);
    static int32_t UnregisterSecurityComponent(int32_t scId);
    static int32_t ReportSecurityComponentClickEvent(SecCompInfo& SecCompInfo, sptr<IRemoteObject> callerToken,
        OnFirstUseDialogCloseFunc&& callback, std::string& message);
    static bool VerifySavePermission(AccessToken::AccessTokenID tokenId);
    static int32_t PreRegisterSecCompProcess();
    static bool IsServiceExist();
    static bool LoadService();
    static bool IsSystemAppCalling();
    static bool HasCustomPermissionForSecComp();
};
}  // namespace SecurityComponent
}  // namespace Security
}  // namespace OHOS
#endif  // INTERFACES_INNER_API_SECURITY_COMPONENT_KIT_H
