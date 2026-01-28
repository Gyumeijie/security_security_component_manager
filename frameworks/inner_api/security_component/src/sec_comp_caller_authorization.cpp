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
#include "sec_comp_caller_authorization.h"

#include <mutex>
#include "sec_comp_log.h"

namespace OHOS {
namespace Security {
namespace SecurityComponent {
namespace {
#ifdef TDD_COVERAGE
static constexpr int32_t MAX_FUNC_ASM_SIZE = 0x1388;
#elif defined(TDD_MACRO)
static constexpr int32_t MAX_FUNC_ASM_SIZE = 0x380;
#else
static constexpr int32_t MAX_FUNC_ASM_SIZE = 0x46c;
#endif
static constexpr OHOS::HiviewDFX::HiLogLabel LABEL = {
    LOG_CORE, SECURITY_DOMAIN_SECURITY_COMPONENT, "SecCompCallerAuthorization"};
static constexpr size_t MAX_CALLER_SIZE = 10;
static std::mutex g_instanceMutex;
}

void SecCompCallerAuthorization::RegisterSecCompKitCaller(std::vector<uintptr_t>& callerList)
{
    if (isInit_) {
        SC_LOG_ERROR(LABEL, "can not init repeatly");
        return;
    }

    isInit_ = true;
    if ((callerList.size() == 0) || (callerList.size() > MAX_CALLER_SIZE)) {
        SC_LOG_ERROR(LABEL, "caller size is invalid");
        return;
    }

    kitCallerList_ = callerList;
}

bool SecCompCallerAuthorization::IsKitCaller(uintptr_t callerAddr)
{
    if (!isInit_) {
        SC_LOG_INFO(LABEL, "caller authorization has not init");
        return true;
    }
    for (size_t i = 0; i < kitCallerList_.size(); i++) {
        if ((callerAddr > kitCallerList_[i]) && (callerAddr < kitCallerList_[i] + MAX_FUNC_ASM_SIZE)) {
            return true;
        }
    }
    return false;
}

SecCompCallerAuthorization& SecCompCallerAuthorization::GetInstance()
{
    static SecCompCallerAuthorization* instance = nullptr;
    if (instance == nullptr) {
        std::lock_guard<std::mutex> lock(g_instanceMutex);
        if (instance == nullptr) {
            instance = new SecCompCallerAuthorization();
        }
    }
    return *instance;
}
}  // namespace SecurityComponent
}  // namespace Security
}  // namespace OHOS

