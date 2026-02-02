/*
 * Copyright (c) 2026 Huawei Device Co., Ltd.
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

#include "securitycomponentstub_fuzzer.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include "accesstoken_kit.h"
#include "fuzz_common.h"
#include "isec_comp_service.h"
#include "sec_comp_click_event_parcel.h"
#include "sec_comp_dialog_callback.h"
#include "sec_comp_enhance_adapter.h"
#include "sec_comp_info.h"
#include "sec_comp_service.h"
#include "securec.h"
#include "token_setproc.h"


using namespace OHOS::Security::SecurityComponent;
using namespace OHOS::Security::AccessToken;

static constexpr uint64_t MAX_TOUCH_INTERVAL = 5000000L; // 5000ms
static constexpr uint64_t TIME_CONVERSION_UNIT = 1000;
static int32_t g_scId = 0;
static std::shared_ptr<SecCompService> g_service;
static uint32_t g_type;
static std::string g_compoInfo = "";

namespace OHOS {
void EmptyCallback(int32_t input)
{
    return;
}

bool ParseDimension(const nlohmann::json& json, const std::string& tag, DimensionT& res)
{
    if ((json.find(tag) == json.end()) || !json.at(tag).is_number_float()) {
        return false;
    }

    res = json.at(tag).get<double>();
    return true;
}

void GetCompoRect(nlohmann::json &jsonComponent, SecCompRect& rect)
{
    if ((jsonComponent.find(JsonTagConstants::JSON_RECT) == jsonComponent.end()) ||
        !jsonComponent.at(JsonTagConstants::JSON_RECT).is_object()) {
        return;
    }

    auto jsonSize = jsonComponent.at(JsonTagConstants::JSON_RECT);
    if (!ParseDimension(jsonSize, JsonTagConstants::JSON_RECT_X, rect.x_)) {
        return;
    }

    if (!ParseDimension(jsonSize, JsonTagConstants::JSON_RECT_Y, rect.y_)) {
        return;
    }

    if (!ParseDimension(jsonSize, JsonTagConstants::JSON_RECT_WIDTH, rect.width_)) {
        return;
    }

    if (!ParseDimension(jsonSize, JsonTagConstants::JSON_RECT_HEIGHT, rect.height_)) {
        return;
    }

    return;
}

void ConstructClickEvent(CompoRandomGenerator &generator, SecCompClickEvent& clickEvent)
{
    int32_t randomType = generator.GetData<int32_t>() % 2 + 1; // generate type in range
    clickEvent.type = static_cast<ClickEventType>(randomType);

    SecCompKeyEvent keyEvent;
    SecCompPointEvent pointEvent;
    SecCompRect rect;
    uint64_t current = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()) / TIME_CONVERSION_UNIT;
    GetCompoRect(generator.compoJson_, rect);
    if (clickEvent.type == ClickEventType::POINT_EVENT_TYPE) {
        pointEvent.touchX = std::fmod(generator.GetData<double>(), rect.width_) + rect.x_;
        pointEvent.touchY = std::fmod(generator.GetData<double>(), rect.height_) + rect.y_;
        pointEvent.timestamp = generator.GetData<uint64_t>() % MAX_TOUCH_INTERVAL + current;
        clickEvent.point = pointEvent;
    } else if (clickEvent.type == ClickEventType::KEY_EVENT_TYPE) {
        keyEvent.timestamp = generator.GetData<uint64_t>() % MAX_TOUCH_INTERVAL + current;
        keyEvent.keyCode = generator.GetData<uint32_t>();
        clickEvent.key = keyEvent;
    }
}

static void PreRegisterSecCompProcessStubFuzzTest(const uint8_t *data, size_t size)
{
    uint32_t code =
        static_cast<uint32_t>(ISecCompServiceIpcCode::COMMAND_PRE_REGISTER_SEC_COMP_PROCESS);
    MessageParcel rawData;
    MessageParcel input;
    SecCompRawdata inputData;
    MessageParcel reply;
    CompoRandomGenerator generator(data, size);
    uint32_t type = generator.GetScType();

    if (!input.WriteInterfaceToken(ISecCompService::GetDescriptor())) {
        return;
    }
    rawData.WriteUint32(type);
    SecCompEnhanceAdapter::EnhanceClientSerialize(rawData, inputData);
    input.WriteUint32(inputData.size);
    input.WriteRawData(inputData.data, inputData.size);
    MessageOption option(MessageOption::TF_SYNC);
    g_service->OnRemoteRequest(code, input, reply, option);
    g_service->OnStart();
    g_service->OnStop();
}

static void RegisterSecurityComponentStubFuzzTest(const uint8_t *data, size_t size)
{
    uint32_t code =
        static_cast<uint32_t>(ISecCompServiceIpcCode::COMMAND_REGISTER_SECURITY_COMPONENT);
    MessageParcel rawData;
    MessageParcel input;
    SecCompRawdata inputData;
    MessageParcel reply;
    SecCompRawdata replyData;
    CompoRandomGenerator generator(data, size);

    g_type = generator.GetScType();
    g_compoInfo = generator.GenerateRandomCompoStr(g_type);

    if (!input.WriteInterfaceToken(ISecCompService::GetDescriptor())) {
        return;
    }
    if (!rawData.WriteUint32(g_type)) {
        return;
    }

    if (!rawData.WriteString(g_compoInfo)) {
        return;
    }
    SecCompEnhanceAdapter::EnhanceClientSerialize(rawData, inputData);
    input.WriteUint32(inputData.size);
    input.WriteRawData(inputData.data, inputData.size);
    MessageOption option(MessageOption::TF_SYNC);
    g_service->OnRemoteRequest(code, input, reply, option);
    int32_t errCode = reply.ReadInt32();
    if (errCode != 0) {
        return;
    }
    if (!reply.ReadUint32(replyData.size)) {
        return;
    }
    auto readRawReply = reply.ReadRawData(replyData.size);
    if (readRawReply == nullptr) {
        return;
    }
    int32_t res = replyData.RawDataCpy(readRawReply);
    if (res != SC_OK) {
        return;
    }
    MessageParcel deserializedReply;
    SecCompEnhanceAdapter::EnhanceClientDeserialize(replyData, deserializedReply);
    if (!deserializedReply.ReadInt32(res)) {
        return;
    }
    if (!deserializedReply.ReadInt32(g_scId)) {
        return;
    }
}

static void UpdateSecurityComponentStubFuzzTest(const uint8_t *data, size_t size)
{
    uint32_t code = static_cast<uint32_t>(ISecCompServiceIpcCode::COMMAND_UPDATE_SECURITY_COMPONENT);
    MessageParcel rawData;
    MessageParcel input;
    SecCompRawdata inputData;
    MessageParcel reply;
    CompoRandomGenerator generator(data, size);
    if (!input.WriteInterfaceToken(ISecCompService::GetDescriptor())) {
        return;
    }
    if (!rawData.WriteInt32(g_scId)) {
        return;
    }
    g_compoInfo = generator.GenerateRandomCompoStr(g_type);
    if (!rawData.WriteString(g_compoInfo)) {
        return;
    }
    SecCompEnhanceAdapter::EnhanceClientSerialize(rawData, inputData);
    input.WriteUint32(inputData.size);
    input.WriteRawData(inputData.data, inputData.size);

    MessageOption option(MessageOption::TF_SYNC);
    g_service->OnRemoteRequest(code, input, reply, option);
}

static void UnRegisterSecurityComponentStubFuzzTest(const uint8_t *data, size_t size)
{
    uint32_t code =
        static_cast<uint32_t>(ISecCompServiceIpcCode::COMMAND_UNREGISTER_SECURITY_COMPONENT);
    MessageParcel rawData;
    MessageParcel input;
    SecCompRawdata inputData;
    MessageParcel reply;
    SecCompRawdata replyData;

    if (!input.WriteInterfaceToken(ISecCompService::GetDescriptor())) {
        return;
    }
    if (!rawData.WriteInt32(g_scId)) {
        return;
    }
    SecCompEnhanceAdapter::EnhanceClientSerialize(rawData, inputData);
    input.WriteUint32(inputData.size);
    input.WriteRawData(inputData.data, inputData.size);
    MessageOption option(MessageOption::TF_SYNC);
    g_service->OnRemoteRequest(code, input, reply, option);
    return;
}

static void ReportSecurityComponentClickEventStubFuzzTest(const uint8_t *data, size_t size)
{
    uint32_t code = static_cast<uint32_t>(ISecCompServiceIpcCode::COMMAND_REPORT_SECURITY_COMPONENT_CLICK_EVENT);
    MessageParcel rawData;
    MessageParcel input;
    SecCompRawdata inputData;
    MessageParcel reply;
    CompoRandomGenerator generator(data, size);

    if (!input.WriteInterfaceToken(ISecCompService::GetDescriptor())) {
        return;
    }
    if (!rawData.WriteInt32(g_scId)) {
        return;
    }
    if (!rawData.WriteString(g_compoInfo)) {
        return;
    }

    if (!rawData.WriteString(generator.GetMessage())) {
        return;
    }
    sptr<SecCompClickEventParcel> parcel =  new (std::nothrow) SecCompClickEventParcel();
    ConstructClickEvent(generator, parcel->clickInfoParams_);
    if (!rawData.WriteParcelable(parcel)) {
        return;
    }

    sptr<SecCompDialogCallback> callbackObj = sptr<SecCompDialogCallback>::MakeSptr(EmptyCallback);
    if (!input.WriteRemoteObject(callbackObj->AsObject())) {
        return;
    }
    if (!input.WriteRemoteObject(callbackObj->AsObject())) {
        return;
    }

    SecCompEnhanceAdapter::EnhanceClientSerialize(rawData, inputData);
    input.WriteUint32(inputData.size);
    input.WriteRawData(inputData.data, inputData.size);

    MessageOption option(MessageOption::TF_SYNC);
    g_service->OnRemoteRequest(code, input, reply, option);
}

static void VerifySavePermissionStubFuzzTest(const uint8_t *data, size_t size)
{
    uint32_t code = static_cast<uint32_t>(ISecCompServiceIpcCode::COMMAND_VERIFY_SAVE_PERMISSION);
    MessageParcel rawData;
    MessageParcel input;
    MessageParcel reply;
    CompoRandomGenerator generator(data, size);

    if (!input.WriteInterfaceToken(ISecCompService::GetDescriptor())) {
        return;
    }

    uint32_t tokenid = generator.GetData<uint32_t>();
    if (!input.WriteUint32(tokenid)) {
        return;
    }

    MessageOption option(MessageOption::TF_SYNC);
    g_service->OnRemoteRequest(code, input, reply, option);
}

static void SecurityComponentFuzzTest(const uint8_t *data, size_t size)
{
    PreRegisterSecCompProcessStubFuzzTest(data, size);
    RegisterSecurityComponentStubFuzzTest(data, size);
    ReportSecurityComponentClickEventStubFuzzTest(data, size);
    UpdateSecurityComponentStubFuzzTest(data, size);
    UnRegisterSecurityComponentStubFuzzTest(data, size);
    VerifySavePermissionStubFuzzTest(data, size);
}
} // namespace OHOS

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv)
{
    g_service =
        std::make_shared<SecCompService>(SA_ID_SECURITY_COMPONENT_SERVICE, true);
    return 0;
}

/* Fuzzer entry point */
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    /* Run your code on data */
    OHOS::SecurityComponentFuzzTest(data, size);
    return 0;
}
