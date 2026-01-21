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

#include "sec_comp_manager_test.h"

#include "sec_comp_log.h"
#define private public
#include "sec_comp_manager.h"
#undef private
#include "ipc_skeleton.h"
#include "iservice_registry.h"
#include "location_button.h"
#include "mock_system_ability_proxy.h"
#include "save_button.h"
#include "sec_comp_err.h"
#include "service_test_common.h"
#include "system_ability.h"

using namespace testing::ext;
using namespace OHOS;
using namespace OHOS::Security::SecurityComponent;
using namespace OHOS::Security::AccessToken;

namespace {
static constexpr OHOS::HiviewDFX::HiLogLabel LABEL = {
    LOG_CORE, SECURITY_DOMAIN_SECURITY_COMPONENT, "SecCompManagerTest"};

static LocationButton BuildInvalidLocationComponent()
{
    LocationButton button;
    button.fontSize_ = ServiceTestCommon::TEST_INVALID_SIZE;
    button.iconSize_ = ServiceTestCommon::TEST_INVALID_SIZE;
    button.padding_.top = ServiceTestCommon::TEST_INVALID_SIZE;
    button.padding_.right = ServiceTestCommon::TEST_INVALID_SIZE;
    button.padding_.bottom = ServiceTestCommon::TEST_INVALID_SIZE;
    button.padding_.left = ServiceTestCommon::TEST_INVALID_SIZE;
    button.textIconSpace_ = ServiceTestCommon::TEST_INVALID_SIZE;
    button.fontColor_.value = ServiceTestCommon::TEST_COLOR_WHITE;
    button.iconColor_.value = ServiceTestCommon::TEST_COLOR_WHITE;
    button.bgColor_.value = ServiceTestCommon::TEST_COLOR_WHITE;
    button.borderWidth_ = ServiceTestCommon::TEST_INVALID_SIZE;
    button.type_ = LOCATION_COMPONENT;
    button.rect_.x_ = ServiceTestCommon::TEST_COORDINATE;
    button.rect_.y_ = ServiceTestCommon::TEST_COORDINATE;
    button.rect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    button.rect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    button.windowRect_.x_ = ServiceTestCommon::TEST_COORDINATE;
    button.windowRect_.y_ = ServiceTestCommon::TEST_COORDINATE;
    button.windowRect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    button.windowRect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    button.text_ = UNKNOWN_TEXT;
    button.icon_ = UNKNOWN_ICON;
    button.bg_ = SecCompBackground::UNKNOWN_BG;
    return button;
}

static LocationButton BuildValidLocationComponent()
{
    LocationButton button;
    button.fontSize_ = ServiceTestCommon::TEST_SIZE;
    button.iconSize_ = ServiceTestCommon::TEST_SIZE;
    button.padding_.top = ServiceTestCommon::TEST_SIZE;
    button.padding_.right = ServiceTestCommon::TEST_SIZE;
    button.padding_.bottom = ServiceTestCommon::TEST_SIZE;
    button.padding_.left = ServiceTestCommon::TEST_SIZE;
    button.textIconSpace_ = ServiceTestCommon::TEST_SIZE;
    button.fontColor_.value = ServiceTestCommon::TEST_COLOR_YELLOW;
    button.iconColor_.value = ServiceTestCommon::TEST_COLOR_RED;
    button.bgColor_.value = ServiceTestCommon::TEST_COLOR_BLUE;
    button.borderWidth_ = ServiceTestCommon::TEST_SIZE;
    button.type_ = LOCATION_COMPONENT;
    button.rect_.x_ = ServiceTestCommon::TEST_COORDINATE;
    button.rect_.y_ = ServiceTestCommon::TEST_COORDINATE;
    button.rect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    button.rect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    button.windowRect_.x_ = ServiceTestCommon::TEST_COORDINATE;
    button.windowRect_.y_ = ServiceTestCommon::TEST_COORDINATE;
    button.windowRect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    button.windowRect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    button.text_ = static_cast<int32_t>(LocationDesc::SELECT_LOCATION);
    button.icon_ = static_cast<int32_t>(LocationIcon::LINE_ICON);
    button.bg_ = SecCompBackground::CIRCLE;
    return button;
}
static const int32_t SLEEP_TIME = 5;
}

void SecCompManagerTest::SetUpTestCase()
{}

void SecCompManagerTest::TearDownTestCase()
{
    if (SecCompManager::GetInstance().secRunner_ != nullptr) {
        SecCompManager::GetInstance().secRunner_->queue_.reset();
        SecCompManager::GetInstance().secRunner_ = nullptr;
    }
    if (SecCompManager::GetInstance().secHandler_ != nullptr) {
        SecCompManager::GetInstance().secHandler_->eventRunner_.reset();
        SecCompManager::GetInstance().secHandler_ = nullptr;
    }
    sleep(SLEEP_TIME);
}

void SecCompManagerTest::SetUp()
{
    SC_LOG_INFO(LABEL, "setup");
}

void SecCompManagerTest::TearDown()
{
    SecCompManager::GetInstance().componentMap_.clear();
}


/**
 * @tc.name: CreateScId001
 * @tc.desc: Test create sc id
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, CreateScId001, TestSize.Level0)
{
    SecCompManager::GetInstance().scIdStart_ = ServiceTestCommon::MAX_INT_NUM;
    ASSERT_EQ(ServiceTestCommon::SC_ID_START, SecCompManager::GetInstance().CreateScId());

    SecCompManager::GetInstance().scIdStart_ = ServiceTestCommon::SC_ID_START;
    ASSERT_EQ(ServiceTestCommon::SC_ID_START + 1, SecCompManager::GetInstance().CreateScId());
}

/**
 * @tc.name: CreateScId002
 * @tc.desc: Test create sc id
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, CreateScId002, TestSize.Level0)
{
    std::shared_ptr<LocationButton> compPtr = std::make_shared<LocationButton>();
    ASSERT_NE(nullptr, compPtr);
    compPtr->rect_.x_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.y_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        compPtr, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::SC_ID_START, 1, 1);
    ASSERT_EQ(SC_OK, SecCompManager::GetInstance().AddSecurityComponentToList(1, 0, entity));

    SecCompManager::GetInstance().scIdStart_ = ServiceTestCommon::MAX_INT_NUM;
    ASSERT_EQ(ServiceTestCommon::SC_ID_START + 1, SecCompManager::GetInstance().CreateScId());
}

/**
 * @tc.name: AddSecurityComponentToList001
 * @tc.desc: Test add security component to list
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, AddSecurityComponentToList001, TestSize.Level0)
{
    std::shared_ptr<LocationButton> compPtr = std::make_shared<LocationButton>();
    ASSERT_NE(nullptr, compPtr);
    compPtr->rect_.x_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.y_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        compPtr, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_1, 1, 1);

    ASSERT_EQ(SC_OK, SecCompManager::GetInstance().AddSecurityComponentToList(1, 0, entity));

    std::shared_ptr<LocationButton> compPtrNew = std::make_shared<LocationButton>();
    ASSERT_NE(nullptr, compPtrNew);
    compPtrNew->rect_.x_ = ServiceTestCommon::TEST_COORDINATE * 2; // not overlap
    compPtrNew->rect_.y_ = ServiceTestCommon::TEST_COORDINATE * 2; // not overlap
    compPtrNew->rect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    compPtrNew->rect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    std::shared_ptr<SecCompEntity> entityNew =
        std::make_shared<SecCompEntity>(
        compPtrNew, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_2, 1, 1);
    ASSERT_EQ(SC_OK, SecCompManager::GetInstance().AddSecurityComponentToList(1, 0, entityNew));
}

/**
 * @tc.name: AddSecurityComponentToList002
 * @tc.desc: Test add security component to list sa not exit
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, AddSecurityComponentToList002, TestSize.Level0)
{
    bool isSaExit = SecCompManager::GetInstance().isSaExit_;
    std::shared_ptr<LocationButton> compPtr = std::make_shared<LocationButton>();
    ASSERT_NE(nullptr, compPtr);
    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        compPtr, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_1, 1, 1);

    ASSERT_NE(SC_SERVICE_ERROR_SERVICE_NOT_EXIST,
        SecCompManager::GetInstance().AddSecurityComponentToList(ServiceTestCommon::TEST_PID_1, 0, entity));
    SecCompManager::GetInstance().isSaExit_ = isSaExit;
}

/**
 * @tc.name: DeleteSecurityComponentFromList001
 * @tc.desc: Test delete security component
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, DeleteSecurityComponentFromList001, TestSize.Level0)
{
    ASSERT_EQ(SC_SERVICE_ERROR_COMPONENT_NOT_EXIST, SecCompManager::GetInstance().DeleteSecurityComponentFromList(
        ServiceTestCommon::TEST_PID_1, ServiceTestCommon::TEST_SC_ID_1));

    std::shared_ptr<LocationButton> compPtr = std::make_shared<LocationButton>();
    ASSERT_NE(nullptr, compPtr);
    compPtr->rect_.x_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.y_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        compPtr, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_1, 1, 1);
    ASSERT_EQ(SC_OK,
        SecCompManager::GetInstance().AddSecurityComponentToList(ServiceTestCommon::TEST_PID_1, 0, entity));

    ASSERT_EQ(SC_SERVICE_ERROR_COMPONENT_NOT_EXIST, SecCompManager::GetInstance().DeleteSecurityComponentFromList(
        ServiceTestCommon::TEST_PID_1, ServiceTestCommon::TEST_SC_ID_2));

    ASSERT_EQ(SC_OK, SecCompManager::GetInstance().DeleteSecurityComponentFromList(
        ServiceTestCommon::TEST_PID_1, ServiceTestCommon::TEST_SC_ID_1));
}

/**
 * @tc.name: GetSecurityComponentFromList001
 * @tc.desc: Test get security component
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, GetSecurityComponentFromList001, TestSize.Level0)
{
    ASSERT_EQ(nullptr, SecCompManager::GetInstance().GetSecurityComponentFromList(
        ServiceTestCommon::TEST_PID_1, ServiceTestCommon::TEST_SC_ID_1));

    std::shared_ptr<LocationButton> compPtr = std::make_shared<LocationButton>();
    ASSERT_NE(nullptr, compPtr);
    compPtr->rect_.x_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.y_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        compPtr, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_1, 1, 1);
    ASSERT_EQ(SC_OK,
        SecCompManager::GetInstance().AddSecurityComponentToList(ServiceTestCommon::TEST_PID_1, 0, entity));
    ASSERT_EQ(nullptr, SecCompManager::GetInstance().GetSecurityComponentFromList(
        ServiceTestCommon::TEST_PID_1, ServiceTestCommon::TEST_SC_ID_2));
    ASSERT_NE(nullptr, SecCompManager::GetInstance().GetSecurityComponentFromList(
        ServiceTestCommon::TEST_PID_1, ServiceTestCommon::TEST_SC_ID_1));
}

/**
 * @tc.name: NotifyProcessBackground001
 * @tc.desc: Test notify process background
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, NotifyProcessBackground001, TestSize.Level0)
{
    SecCompManager::GetInstance().NotifyProcessBackground(ServiceTestCommon::TEST_PID_1);

    std::shared_ptr<LocationButton> compPtr = std::make_shared<LocationButton>();
    ASSERT_NE(nullptr, compPtr);
    compPtr->rect_.x_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.y_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        compPtr, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_1, 1, 1);
    ASSERT_EQ(SC_OK,
        SecCompManager::GetInstance().AddSecurityComponentToList(ServiceTestCommon::TEST_PID_1, 0, entity));
    auto component = SecCompManager::GetInstance().GetSecurityComponentFromList(
        ServiceTestCommon::TEST_PID_1, ServiceTestCommon::TEST_SC_ID_1);
    ASSERT_NE(nullptr, component);
}

/**
 * @tc.name: NotifyProcessDied001
 * @tc.desc: Test notify process died
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, NotifyProcessDied001, TestSize.Level0)
{
    std::shared_ptr<LocationButton> compPtr = std::make_shared<LocationButton>();
    ASSERT_NE(nullptr, compPtr);
    compPtr->rect_.x_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.y_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        compPtr, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_1, 1, 1);
    ASSERT_EQ(SC_OK,
        SecCompManager::GetInstance().AddSecurityComponentToList(ServiceTestCommon::TEST_PID_1, 0, entity));

    std::shared_ptr<LocationButton> compPtr2 = std::make_shared<LocationButton>();
    ASSERT_NE(nullptr, compPtr2);
    compPtr2->rect_.x_ = ServiceTestCommon::TEST_COORDINATE * 2; // not overlap
    compPtr2->rect_.y_ = ServiceTestCommon::TEST_COORDINATE * 2; // not overlap
    compPtr2->rect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr2->rect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    std::shared_ptr<SecCompEntity> entity2 =
        std::make_shared<SecCompEntity>(
        compPtr2, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_2, 1, 1);
    ASSERT_EQ(SC_OK,
        SecCompManager::GetInstance().AddSecurityComponentToList(ServiceTestCommon::TEST_PID_2, 0, entity2));

    SecCompManager::GetInstance().NotifyProcessDied(ServiceTestCommon::TEST_PID_3, false);
    ASSERT_NE(nullptr, SecCompManager::GetInstance().GetSecurityComponentFromList(
        ServiceTestCommon::TEST_PID_1, ServiceTestCommon::TEST_SC_ID_1));

    SecCompManager::GetInstance().NotifyProcessDied(ServiceTestCommon::TEST_PID_1, false);
    ASSERT_EQ(nullptr, SecCompManager::GetInstance().GetSecurityComponentFromList(
        ServiceTestCommon::TEST_PID_1, ServiceTestCommon::TEST_SC_ID_1));
}

/**
 * @tc.name: RegisterSecurityComponent001
 * @tc.desc: Test register security component
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, RegisterSecurityComponent001, TestSize.Level0)
{
    SecCompCallerInfo caller = {
        .tokenId = ServiceTestCommon::TEST_TOKEN_ID,
        .uid = 1,
        .pid = ServiceTestCommon::TEST_PID_1
    };
    int32_t scId;
    SecCompManager::GetInstance().malicious_.maliciousAppList_.insert(ServiceTestCommon::TEST_PID_1);
    nlohmann::json jsonInvalid;
    EXPECT_EQ(SC_ENHANCE_ERROR_IN_MALICIOUS_LIST,
        SecCompManager::GetInstance().RegisterSecurityComponent(LOCATION_COMPONENT, jsonInvalid, caller, scId));
    SecCompManager::GetInstance().malicious_.maliciousAppList_.clear();

    LocationButton buttonInvalid = BuildInvalidLocationComponent();
    buttonInvalid.ToJson(jsonInvalid);
    EXPECT_EQ(SC_SERVICE_ERROR_COMPONENT_INFO_INVALID,
        SecCompManager::GetInstance().RegisterSecurityComponent(LOCATION_COMPONENT, jsonInvalid, caller, scId));

    nlohmann::json jsonValid;
    LocationButton buttonValid = BuildValidLocationComponent();
    buttonValid.ToJson(jsonValid);

    EXPECT_EQ(SC_OK,
        SecCompManager::GetInstance().RegisterSecurityComponent(LOCATION_COMPONENT, jsonValid, caller, scId));
    SecCompManager::GetInstance().malicious_.maliciousAppList_.clear();
}

/**
 * @tc.name: UpdateSecurityComponent001
 * @tc.desc: Test update security component
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, UpdateSecurityComponent001, TestSize.Level0)
{
    SecCompManager::GetInstance().malicious_.maliciousAppList_.insert(ServiceTestCommon::TEST_PID_1);
    nlohmann::json jsonValid;
    LocationButton buttonValid = BuildValidLocationComponent();
    buttonValid.ToJson(jsonValid);
    SecCompCallerInfo caller = {
        .tokenId = ServiceTestCommon::TEST_TOKEN_ID,
        .uid = 1,
        .pid = ServiceTestCommon::TEST_PID_1
    };
    EXPECT_EQ(SC_ENHANCE_ERROR_IN_MALICIOUS_LIST,
        SecCompManager::GetInstance().UpdateSecurityComponent(ServiceTestCommon::TEST_SC_ID_1, jsonValid, caller));
    SecCompManager::GetInstance().malicious_.maliciousAppList_.clear();

    std::shared_ptr<LocationButton> compPtr = std::make_shared<LocationButton>();
    compPtr->type_ = LOCATION_COMPONENT;
    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        compPtr, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_1, 1, 1);
    EXPECT_EQ(SC_OK,
        SecCompManager::GetInstance().AddSecurityComponentToList(ServiceTestCommon::TEST_PID_1, 0, entity));

    // wrong json
    nlohmann::json jsonInvalid;
    LocationButton buttonInvalid = BuildInvalidLocationComponent();
    buttonInvalid.ToJson(jsonInvalid);
    EXPECT_EQ(SC_SERVICE_ERROR_COMPONENT_INFO_INVALID,
        SecCompManager::GetInstance().UpdateSecurityComponent(ServiceTestCommon::TEST_SC_ID_1, jsonInvalid, caller));

    // no enhance data
    EXPECT_EQ(SC_OK,
        SecCompManager::GetInstance().UpdateSecurityComponent(ServiceTestCommon::TEST_SC_ID_1, jsonValid, caller));
    SecCompManager::GetInstance().malicious_.maliciousAppList_.clear();
}

/**
 * @tc.name: UnregisterSecurityComponent001
 * @tc.desc: Test unregister security component
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, UnregisterSecurityComponent001, TestSize.Level0)
{
    SecCompCallerInfo caller = {
        .tokenId = ServiceTestCommon::TEST_TOKEN_ID,
        .pid = ServiceTestCommon::TEST_PID_1
    };

    ASSERT_EQ(SC_SERVICE_ERROR_VALUE_INVALID,
        SecCompManager::GetInstance().UnregisterSecurityComponent(ServiceTestCommon::TEST_INVALID_SC_ID, caller));
}

/**
 * @tc.name: ReportSecurityComponentClickEvent001
 * @tc.desc: Test report security component click
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, ReportSecurityComponentClickEvent001, TestSize.Level0)
{
    SecCompCallerInfo caller = {
        .tokenId = ServiceTestCommon::TEST_TOKEN_ID,
        .pid = ServiceTestCommon::TEST_PID_1
    };
    SecCompClickEvent clickInfo = {};
    nlohmann::json jsonValid;
    LocationButton buttonValid = BuildValidLocationComponent();
    buttonValid.ToJson(jsonValid);
    std::vector<sptr<IRemoteObject>> remote = { nullptr, nullptr };
    int32_t scId = 1;
    SecCompInfo secCompInfo{ scId, "", clickInfo};
    std::string message;
    ASSERT_NE(SC_OK, SecCompManager::GetInstance().ReportSecurityComponentClickEvent(secCompInfo,
        jsonValid, caller, remote, message));
}

/**
 * @tc.name: AddSecurityComponentToList004
 * @tc.desc: Test add security component too many components
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, AddSecurityComponentToList004, TestSize.Level0)
{
    std::shared_ptr<SecCompManager> managerInstance = std::make_shared<SecCompManager>();
    managerInstance->isSaExit_ = false;
    int pid = 1;
    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        nullptr, 0, 0, 1, 1);
    const int MAX_COMPONENT_SIZE = 500;
    for (int i = 0; i < MAX_COMPONENT_SIZE; i++) {
        managerInstance->componentMap_[pid].compList.emplace_back(entity);
    }

    ASSERT_NE(managerInstance->AddSecurityComponentToList(pid, 0, entity), SC_SERVICE_ERROR_VALUE_INVALID);
}

/**
 * @tc.name: CheckClickSecurityComponentInfo001
 * @tc.desc: Test check click security component info failed
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, CheckClickSecurityComponentInfo001, TestSize.Level0)
{
    SecCompCallerInfo caller = {
        .tokenId = ServiceTestCommon::TEST_TOKEN_ID,
        .pid = ServiceTestCommon::TEST_PID_1
    };
    std::shared_ptr<LocationButton> compPtr = std::make_shared<LocationButton>();
    ASSERT_NE(nullptr, compPtr);

    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        compPtr, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_1, 1, 1);
    ASSERT_EQ(SC_OK,
        SecCompManager::GetInstance().AddSecurityComponentToList(ServiceTestCommon::TEST_PID_1, 0, entity));

    SecCompClickEvent clickInfo = {};
    nlohmann::json jsonValid;
    LocationButton buttonValid = BuildValidLocationComponent();
    buttonValid.ToJson(jsonValid);
    jsonValid[JsonTagConstants::JSON_SC_TYPE] = UNKNOWN_SC_TYPE;
    std::vector<sptr<IRemoteObject>> remote = { nullptr, nullptr };
    SecCompInfo secCompInfo{ ServiceTestCommon::TEST_SC_ID_1, "", clickInfo };
    std::string message;
    ASSERT_EQ(SC_SERVICE_ERROR_COMPONENT_INFO_INVALID, SecCompManager::GetInstance().ReportSecurityComponentClickEvent(
        secCompInfo, jsonValid, caller, remote, message));

    jsonValid[JsonTagConstants::JSON_SC_TYPE] = LOCATION_COMPONENT;
    jsonValid[JsonTagConstants::JSON_RECT][JsonTagConstants::JSON_RECT_X] = ServiceTestCommon::TEST_INVALID_DIMENSION;
    ASSERT_EQ(SC_SERVICE_ERROR_COMPONENT_INFO_INVALID, SecCompManager::GetInstance().ReportSecurityComponentClickEvent(
        secCompInfo, jsonValid, caller, remote, message));

    jsonValid[JsonTagConstants::JSON_RECT][JsonTagConstants::JSON_RECT_X] = ServiceTestCommon::TEST_COORDINATE - 1;
    ASSERT_EQ(SC_SERVICE_ERROR_COMPONENT_INFO_INVALID, SecCompManager::GetInstance().ReportSecurityComponentClickEvent(
        secCompInfo, jsonValid, caller, remote, message));
}

/**
 * @tc.name: AddSecurityComponentToList003
 * @tc.desc: Test add security component to list sa not exit
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, AddSecurityComponentToList003, TestSize.Level0)
{
    SecCompManager::GetInstance().isSaExit_ = true;
    std::shared_ptr<LocationButton> compPtr = std::make_shared<LocationButton>();
    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        compPtr, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_1, 1, 1);
    ASSERT_EQ(SC_SERVICE_ERROR_SERVICE_NOT_EXIST,
        SecCompManager::GetInstance().AddSecurityComponentToList(ServiceTestCommon::TEST_PID_1, 0, entity));
    SecCompManager::GetInstance().isSaExit_ = false;
}

/**
 * @tc.name: DeleteSecurityComponentFromList002
 * @tc.desc: Test delete security component
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, DeleteSecurityComponentFromList002, TestSize.Level0)
{
    std::shared_ptr<LocationButton> compPtr = std::make_shared<LocationButton>();
    ASSERT_NE(nullptr, compPtr);
    compPtr->rect_.x_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.y_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        compPtr, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_1, 1, 1);
    ASSERT_EQ(SC_OK,
        SecCompManager::GetInstance().AddSecurityComponentToList(ServiceTestCommon::TEST_PID_1, 0, entity));

    auto component = SecCompManager::GetInstance().GetSecurityComponentFromList(1, 1);
    ASSERT_NE(nullptr, component);
    ASSERT_EQ(SC_OK, SecCompManager::GetInstance().DeleteSecurityComponentFromList(1, 1));
}

/**
 * @tc.name: UpdateSecurityComponent002
 * @tc.desc: Test update security component
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, UpdateSecurityComponent002, TestSize.Level0)
{
    nlohmann::json jsonValid;
    SecCompCallerInfo caller = {
        .tokenId = ServiceTestCommon::TEST_TOKEN_ID,
        .pid = ServiceTestCommon::TEST_PID_1
    };
    ASSERT_NE(SC_SERVICE_ERROR_COMPONENT_INFO_INVALID, SecCompManager::GetInstance().UpdateSecurityComponent(
        ServiceTestCommon::TEST_SC_ID_1, jsonValid, caller));

    SecCompManager::GetInstance().malicious_.AddAppToMaliciousAppList(ServiceTestCommon::TEST_PID_1);
    LocationButton buttonValid = BuildValidLocationComponent();
    buttonValid.ToJson(jsonValid);
    ASSERT_NE(SC_ENHANCE_ERROR_IN_MALICIOUS_LIST, SecCompManager::GetInstance().UpdateSecurityComponent(
        ServiceTestCommon::TEST_SC_ID_1, jsonValid, caller));
}

/**
 * @tc.name: ExitSaProcess001
 * @tc.desc: Test check ExitSaProcess
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, ExitSaProcess001, TestSize.Level0)
{
    std::shared_ptr<LocationButton> compPtr = std::make_shared<LocationButton>();
    ASSERT_NE(nullptr, compPtr);
    compPtr->rect_.x_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.y_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        compPtr, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_1, 1, 1);
    std::shared_ptr<SecCompManager> instance = std::make_shared<SecCompManager>();
    ASSERT_EQ(SC_OK,
        instance->AddSecurityComponentToList(ServiceTestCommon::TEST_PID_1, 0, entity));
    instance->ExitSaProcess();
    EXPECT_FALSE(instance->isSaExit_);

    instance->componentMap_.clear();
    instance->ExitSaProcess();
    EXPECT_TRUE(instance->isSaExit_);
    instance->isSaExit_ = false;

    std::shared_ptr<SystemAbilityManagerClient> saClient = std::make_shared<SystemAbilityManagerClient>();
    ASSERT_NE(nullptr, saClient);
    SystemAbilityManagerClient::clientInstance = saClient.get();
    sptr<SystemAbilityManagerProxy> proxy = new SystemAbilityManagerProxy(nullptr);
    EXPECT_CALL(*saClient, GetSystemAbilityManager()).WillOnce(testing::Return(proxy));
    instance->ExitSaProcess();
    EXPECT_TRUE(instance->isSaExit_);
    instance->isSaExit_ = false;

    EXPECT_CALL(*saClient, GetSystemAbilityManager()).WillOnce(testing::Return(proxy));
    EXPECT_CALL(*proxy, UnloadSystemAbility(testing::_)).WillOnce(testing::Return(-1));
    instance->ExitSaProcess();
    EXPECT_TRUE(instance->isSaExit_);
}

/**
 * @tc.name: ExitWhenAppMgrDied001
 * @tc.desc: Test check ExitWhenAppMgrDied
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, ExitWhenAppMgrDied001, TestSize.Level0)
{
    std::shared_ptr<LocationButton> compPtr = std::make_shared<LocationButton>();
    ASSERT_NE(nullptr, compPtr);
    compPtr->rect_.x_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.y_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        compPtr, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_1, 1, 1);

    std::shared_ptr<SystemAbilityManagerClient> saClient = std::make_shared<SystemAbilityManagerClient>();
    ASSERT_NE(nullptr, saClient);
    SystemAbilityManagerClient::clientInstance = saClient.get();

    EXPECT_CALL(*saClient, GetSystemAbilityManager()).WillOnce(testing::Return(nullptr));
    std::shared_ptr<SecCompManager> instance = std::make_shared<SecCompManager>();
    EXPECT_EQ(SC_OK, instance->AddSecurityComponentToList(1, 0, entity));
    instance->ExitWhenAppMgrDied();
    EXPECT_TRUE(instance->isSaExit_);

    sptr<SystemAbilityManagerProxy> proxy = new SystemAbilityManagerProxy(nullptr);
    EXPECT_CALL(*saClient, GetSystemAbilityManager()).WillOnce(testing::Return(proxy));
    instance->ExitWhenAppMgrDied();
    EXPECT_TRUE(instance->isSaExit_);

    EXPECT_CALL(*saClient, GetSystemAbilityManager()).WillOnce(testing::Return(proxy));
    instance->ExitWhenAppMgrDied();
    EXPECT_TRUE(instance->isSaExit_);

    EXPECT_CALL(*saClient, GetSystemAbilityManager()).WillOnce(testing::Return(proxy));
    EXPECT_CALL(*proxy, UnloadSystemAbility(testing::_)).WillOnce(testing::Return(-1));
    instance->ExitWhenAppMgrDied();
    EXPECT_TRUE(instance->isSaExit_);
}

/**
 * @tc.name: SendCheckInfoEnhanceSysEvent001
 * @tc.desc: Test check SendCheckInfoEnhanceSysEvent
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, SendCheckInfoEnhanceSysEvent001, TestSize.Level0)
{
    SecCompManager::GetInstance().malicious_.maliciousAppList_.clear();
    ASSERT_TRUE(SecCompManager::GetInstance().malicious_.IsMaliciousAppListEmpty());
    int32_t scId = INVALID_SC_ID;
    const std::string scene = "";
    int32_t res = SC_ENHANCE_ERROR_CHALLENGE_CHECK_FAIL;
    SecCompManager::GetInstance().SendCheckInfoEnhanceSysEvent(scId, LOCATION_COMPONENT, scene, res);
}

/**
 * @tc.name: DumpSecComp001
 * @tc.desc: Test check DumpSecComp
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, DumpSecComp001, TestSize.Level0)
{
    std::shared_ptr<LocationButton> compPtr = std::make_shared<LocationButton>();
    ASSERT_NE(nullptr, compPtr);
    compPtr->rect_.x_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.y_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.width_ = ServiceTestCommon::TEST_COORDINATE;
    compPtr->rect_.height_ = ServiceTestCommon::TEST_COORDINATE;
    std::shared_ptr<SecCompEntity> entity =
        std::make_shared<SecCompEntity>(
        compPtr, ServiceTestCommon::TEST_TOKEN_ID, ServiceTestCommon::TEST_SC_ID_1, 1, 1);
    SecCompManager::GetInstance().isSaExit_ = false;
    ASSERT_EQ(SC_OK,
        SecCompManager::GetInstance().AddSecurityComponentToList(ServiceTestCommon::TEST_PID_1, 0, entity));
    std::string dumpStr;
    SecCompManager::GetInstance().DumpSecComp(dumpStr);
}

/**
 * @tc.name: TransformCallBackResult001
 * @tc.desc: Test check TransformCallBackResult
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, TransformCallBackResult001, TestSize.Level0)
{
    int32_t scId = INVALID_SC_ID;
    const std::string scene = "REGISTER";
    int32_t res = SC_ENHANCE_ERROR_NOT_EXIST_ENHANCE;
    SecCompManager::GetInstance().SendCheckInfoEnhanceSysEvent(scId, LOCATION_COMPONENT, scene, res);
    res = SC_ENHANCE_ERROR_VALUE_INVALID;
    SecCompManager::GetInstance().SendCheckInfoEnhanceSysEvent(scId, LOCATION_COMPONENT, scene, res);
    res = SC_ENHANCE_ERROR_CALLBACK_OPER_FAIL;
    SecCompManager::GetInstance().SendCheckInfoEnhanceSysEvent(scId, LOCATION_COMPONENT, scene, res);
    res = SC_SERVICE_ERROR_COMPONENT_INFO_INVALID;
    SecCompManager::GetInstance().SendCheckInfoEnhanceSysEvent(scId, LOCATION_COMPONENT, scene, res);
    res = SC_ENHANCE_ERROR_CALLBACK_CHECK_FAIL;
    SecCompManager::GetInstance().SendCheckInfoEnhanceSysEvent(scId, LOCATION_COMPONENT, scene, res);
    res = SC_SERVICE_ERROR_VALUE_INVALID;
    SecCompManager::GetInstance().SendCheckInfoEnhanceSysEvent(scId, LOCATION_COMPONENT, scene, res);
}

/**
 * @tc.name: AddSecurityComponentProcess001
 * @tc.desc: Test AddSecurityComponentProcess
 * @tc.type: FUNC
 * @tc.require:
 */
HWTEST_F(SecCompManagerTest, AddSecurityComponentProcess001, TestSize.Level0)
{
    bool isSaExit = SecCompManager::GetInstance().isSaExit_;
    SecCompManager::GetInstance().isSaExit_ = true;
    SecCompCallerInfo info = {
        .pid = 0,
        .tokenId = 0,
        .uid = 0,
    };
    EXPECT_EQ(SC_SERVICE_ERROR_SERVICE_NOT_EXIST,
        SecCompManager::GetInstance().AddSecurityComponentProcess(info));
    auto oldmap = SecCompManager::GetInstance().componentMap_;
    SecCompManager::GetInstance().componentMap_.clear();
    SecCompManager::GetInstance().isSaExit_ = false;
    EXPECT_EQ(SC_OK, SecCompManager::GetInstance().AddSecurityComponentProcess(info));
    SecCompManager::GetInstance().componentMap_ = oldmap;
    SecCompManager::GetInstance().isSaExit_ = isSaExit;
}