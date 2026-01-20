/*
 * Copyright (c) 2023-2026 Huawei Device Co., Ltd.
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
#include "sec_comp_info_helper.h"

#include <cstdint>
#include <iomanip>
#include <sstream>

#include "accesstoken_kit.h"
#include "display.h"
#include "display_info.h"
#include "display_manager.h"
#include "ipc_skeleton.h"
#include "location_button.h"
#include "paste_button.h"
#include "save_button.h"
#include "sec_comp_err.h"
#include "sec_comp_info.h"
#include "sec_comp_log.h"
#include "sec_comp_tool.h"
#include "tokenid_kit.h"
#include "window_info_helper.h"

namespace OHOS {
namespace Security {
namespace SecurityComponent {
namespace {
constexpr OHOS::HiviewDFX::HiLogLabel LABEL = {LOG_CORE, SECURITY_DOMAIN_SECURITY_COMPONENT, "SecCompInfoHelper"};
static constexpr double MAX_RECT_PERCENT = 0.3F; // 30%
static constexpr double ZERO_OFFSET = 0.0F;
static constexpr double MAX_THRESHOLD_PERCENT = 0.1F; // 10%
const std::string OUT_OF_SCREEN = ", security component is out of screen, security component(x = ";
const std::string OUT_OF_WINDOW = ", security component is out of window, security component(x = ";
const std::string SECURITY_COMPONENT_IS_TOO_LARGER =
    ", security component is too larger, security component(width = ";
const int HEX_FIELD_WIDTH = 2;
const int NUMBER_TWO = 2;
const char HEX_FILL_CHAR = '0';
const static uint8_t DEFAULT_TRANSPARENCY_THRESHOLD = 0x1A;
const static std::set<uint32_t> RELEASE_ATTRIBUTE_LIST = {
    0x0C000000,
};
}

void SecCompInfoHelper::AdjustSecCompRect(SecCompBase* comp, const Scales scales, bool isCompatScaleMode,
    SecCompRect& windowRect)
{
    comp->rect_.x_ = windowRect.x_ + (comp->rect_.x_ - comp->windowRect_.x_) * scales.scaleX;
    comp->rect_.y_ = windowRect.y_ + (comp->rect_.y_ - comp->windowRect_.y_) * scales.scaleY;
    comp->rect_.width_ *= scales.scaleX;
    comp->rect_.height_ *= scales.scaleY;
    comp->windowRect_.x_ = windowRect.x_;
    comp->windowRect_.y_ = windowRect.y_;
    comp->windowRect_.width_ = windowRect.width_;
    comp->windowRect_.height_ = windowRect.height_;
    if (scales.scaleX > scales.scaleY) {
        comp->scale_ = scales.scaleX;
    } else {
        comp->scale_ = scales.scaleY;
    }
    SC_LOG_DEBUG(LABEL, "After adjust x %{public}f, y %{public}f, width %{public}f, height %{public}f",
        comp->rect_.x_, comp->rect_.y_, comp->rect_.width_, comp->rect_.height_);
}

SecCompBase* SecCompInfoHelper::ParseComponent(SecCompType type, const nlohmann::json& jsonComponent,
    std::string& message, bool isClicked)
{
    SecCompBase* comp = nullptr;
    message.clear();
    switch (type) {
        case LOCATION_COMPONENT:
            comp = ConstructComponent<LocationButton>(jsonComponent, message, isClicked);
            break;
        case PASTE_COMPONENT:
            comp = ConstructComponent<PasteButton>(jsonComponent, message, isClicked);
            break;
        case SAVE_COMPONENT:
            comp = ConstructComponent<SaveButton>(jsonComponent, message, isClicked);
            break;
        default:
            SC_LOG_ERROR(LABEL, "Parse component type unknown");
            break;
    }
    if (comp == nullptr) {
        SC_LOG_ERROR(LABEL, "Parse component failed");
        return comp;
    }

    comp->SetValid(CheckComponentValid(comp, message));
    comp->isClickEvent_ = isClicked;
    return comp;
}

static bool GetScreenSize(double& width, double& height, SecCompInfoHelper::ScreenInfo& screenInfo)
{
    sptr<OHOS::Rosen::Display> display =
        OHOS::Rosen::DisplayManager::GetInstance().GetDisplayById(screenInfo.displayId);
    if (display == nullptr) {
        SC_LOG_ERROR(LABEL, "Get display manager failed");
        return false;
    }

    auto info = display->GetDisplayInfo();
    if (info == nullptr) {
        SC_LOG_ERROR(LABEL, "Get display info failed");
        return false;
    }

    screenInfo.screenShape = info->GetScreenShape();
    width = static_cast<double>(info->GetWidth());
    if (screenInfo.crossAxisState == CrossAxisState::STATE_CROSS) {
        height = static_cast<double>(info->GetPhysicalHeight());
    } else {
        height = static_cast<double>(info->GetHeight());
    }
    SC_LOG_DEBUG(LABEL, "display manager Screen width %{public}f height %{public}f",
        width, height);
    return true;
}

double SecCompInfoHelper::GetDistance(DimensionT x1, DimensionT y1, DimensionT x2, DimensionT y2)
{
    return sqrt(pow(x1 - x2, NUMBER_TWO) + pow(y1 -y2, NUMBER_TWO));
}

bool SecCompInfoHelper::IsOutOfWatchScreen(const SecCompRect& rect, double radius, std::string& message)
{
    double diagonal = sqrt(pow(rect.width_, NUMBER_TWO) + pow(rect.height_, NUMBER_TWO));
    double threshold = std::max(diagonal * MAX_THRESHOLD_PERCENT, 1.0);
    double leftTop = GetDistance(rect.x_ + rect.borderRadius_.leftTop,
        rect.y_ + rect.borderRadius_.leftTop, radius, radius);
    double leftBottom = GetDistance(rect.x_ + rect.borderRadius_.leftBottom,
        rect.y_ + rect.height_ - rect.borderRadius_.leftBottom, radius, radius);
    double rightTop = GetDistance(rect.x_ + rect.width_ - rect.borderRadius_.rightTop,
        rect.y_ + rect.borderRadius_.rightTop, radius, radius);
    double rightBottom = GetDistance(rect.x_ + rect.width_ - rect.borderRadius_.rightBottom,
        rect.y_ + rect.height_ - rect.borderRadius_.rightBottom, radius, radius);
    if (GreatNotEqual(leftTop, radius - rect.borderRadius_.leftTop + threshold) ||
        GreatNotEqual(leftBottom, radius - rect.borderRadius_.leftBottom + threshold) ||
        GreatNotEqual(rightTop, radius - rect.borderRadius_.rightTop + threshold) ||
        GreatNotEqual(rightBottom, radius - rect.borderRadius_.rightBottom + threshold)) {
        SC_LOG_ERROR(LABEL, "SecurityComponentCheckFail: security component is out of screen");
        message = OUT_OF_SCREEN + std::to_string(rect.x_) + ", y = " + std::to_string(rect.y_) +
            ", width = " + std::to_string(rect.width_) + ", height = " + std::to_string(rect.height_) +
            "), current screen(width = " + std::to_string(radius * NUMBER_TWO) +
            ", height = " + std::to_string(radius * NUMBER_TWO) + ")";
        return true;
    }
    return false;
}

bool SecCompInfoHelper::IsOutOfScreen(const SecCompRect& rect, double curScreenWidth, double curScreenHeight,
    std::string& message, const ScreenInfo& screenInfo)
{
    if ((screenInfo.isWearable) && (screenInfo.screenShape == Rosen::ScreenShape::ROUND)) {
        if (IsOutOfWatchScreen(rect, curScreenHeight / NUMBER_TWO, message)) {
            return true;
        }
    } else {
        double thresholdX = std::max(rect.width_ * MAX_THRESHOLD_PERCENT, 1.0);
        double thresholdY = std::max(rect.height_ * MAX_THRESHOLD_PERCENT, 1.0);
        if (GreatNotEqual(ZERO_OFFSET - thresholdX, rect.x_) ||
            GreatNotEqual(ZERO_OFFSET - thresholdY, rect.y_) ||
            GreatNotEqual(rect.x_, curScreenWidth + thresholdX) ||
            GreatNotEqual(rect.y_, curScreenHeight + thresholdY)) {
            SC_LOG_ERROR(LABEL, "SecurityComponentCheckFail: security component is out of screen");
            message = OUT_OF_SCREEN + std::to_string(rect.x_) + ", y = " + std::to_string(rect.y_) +
                ", width = " + std::to_string(rect.width_) + ", height = " + std::to_string(rect.height_) +
                "), current screen(width = " + std::to_string(curScreenWidth) +
                ", height = " + std::to_string(curScreenHeight) + ")";
            return true;
        }

        if (GreatNotEqual((rect.x_ + rect.width_), curScreenWidth + thresholdX) ||
            GreatNotEqual((rect.y_ + rect.height_), curScreenHeight + thresholdY)) {
            SC_LOG_ERROR(LABEL, "SecurityComponentCheckFail: security component is out of screen");
            message = OUT_OF_SCREEN + std::to_string(rect.x_) + ", y = " + std::to_string(rect.y_) +
                ", width = " + std::to_string(rect.width_) + ", height = " + std::to_string(rect.height_) +
                "), current screen(width = " + std::to_string(curScreenWidth) +
                ", height = " + std::to_string(curScreenHeight) + ")";
            return true;
        }
    }

    return false;
}

bool SecCompInfoHelper::CheckRectValid(const SecCompRect& rect, const SecCompRect& windowRect,
    ScreenInfo& screenInfo, std::string& message, const float scale)
{
    double curScreenWidth = 0.0F;
    double curScreenHeight = 0.0F;
    if (!GetScreenSize(curScreenWidth, curScreenHeight, screenInfo)) {
        SC_LOG_ERROR(LABEL, "Get screen size is invalid");
        return false;
    }

    if (GreatOrEqual(0.0, rect.width_) || GreatOrEqual(0.0, rect.height_)) {
        SC_LOG_ERROR(LABEL, "width or height is <= 0");
        return false;
    }

    message.clear();
    if (IsOutOfScreen(rect, curScreenWidth, curScreenHeight, message, screenInfo)) {
        return false;
    }

    if (GreatNotEqual(windowRect.x_, rect.x_ + 1.0 + scale) || GreatNotEqual(windowRect.y_, rect.y_ + 1.0 + scale) ||
        GreatNotEqual(rect.x_ + rect.width_, windowRect.x_ + windowRect.width_ + 1.0 + scale) ||
        GreatNotEqual(rect.y_ + rect.height_, windowRect.y_ + windowRect.height_ + 1.0 + scale)) {
        SC_LOG_ERROR(LABEL, "SecurityComponentCheckFail: security component is out of window");
        message = OUT_OF_WINDOW + std::to_string(rect.x_) + ", y = " + std::to_string(rect.y_) +
            ", width = " + std::to_string(rect.width_) + ", height = " + std::to_string(rect.height_) +
            "), current window(x = " + std::to_string(windowRect.x_) + ", y = " + std::to_string(windowRect.y_) +
            ", width = " + std::to_string(windowRect.width_) + ", height = " +
            std::to_string(windowRect.height_) + ")";
        return false;
    }

    // check rect > 30%
    if (GreatOrEqual((rect.width_ * rect.height_), (curScreenWidth * curScreenHeight * MAX_RECT_PERCENT))) {
        SC_LOG_INFO(LABEL, "security component is larger than 30 percent of screen");
    }
    SC_LOG_DEBUG(LABEL, "check component rect success.");
    return true;
}

std::string ColorToHexString(const SecCompColor& color)
{
    std::stringstream ss;
    ss << std::hex;
    ss << std::setw(HEX_FIELD_WIDTH) << std::setfill(HEX_FILL_CHAR) << static_cast<int>(color.argb.alpha);
    ss << std::setw(HEX_FIELD_WIDTH) << std::setfill(HEX_FILL_CHAR) << static_cast<int>(color.argb.red);
    ss << std::setw(HEX_FIELD_WIDTH) << std::setfill(HEX_FILL_CHAR) << static_cast<int>(color.argb.green);
    ss << std::setw(HEX_FIELD_WIDTH) << std::setfill(HEX_FILL_CHAR) << static_cast<int>(color.argb.blue);

    return "#" + ss.str();
}

static bool CheckSecCompBaseButtonColorsimilar(const SecCompBase* comp, std::string& message)
{
    if ((comp->bg_ != SecCompBackground::NO_BG_TYPE) && !IsColorFullTransparent(comp->bgColor_) &&
        (comp->icon_ != NO_ICON) && (comp->iconColor_.value == comp->bgColor_.value)) {
        SC_LOG_INFO(LABEL, "SecurityComponentCheckFail: iconColor is the same with backgroundColor.");
        message = ", icon color is similar with background color, icon color = " +
            ColorToHexString(comp->iconColor_) + ", background color = " + ColorToHexString(comp->bgColor_);
        return false;
    }

    if ((comp->bg_ != SecCompBackground::NO_BG_TYPE) && !IsColorFullTransparent(comp->bgColor_) &&
        (comp->text_ != NO_TEXT) && (comp->fontColor_.value == comp->bgColor_.value)) {
        SC_LOG_INFO(LABEL, "SecurityComponentCheckFail: fontColor is the same with backgroundColor.");
        message = ", font color is similar with background color, font color = " +
            ColorToHexString(comp->fontColor_) + ", background color = " + ColorToHexString(comp->bgColor_);
        return false;
    }

    return true;
}

static bool IsInReleaseList(uint32_t value)
{
    return RELEASE_ATTRIBUTE_LIST.find(value) != RELEASE_ATTRIBUTE_LIST.end();
}

static bool IsBelowThreshold(const SecCompColor& value)
{
    return value.argb.alpha < DEFAULT_TRANSPARENCY_THRESHOLD;
}

static bool IsSystemAppCalling()
{
    auto callerToken = IPCSkeleton::GetCallingFullTokenID();
    return Security::AccessToken::TokenIdKit::IsSystemAppByFullTokenID(callerToken);
}

static bool CheckSecCompBaseButton(const SecCompBase* comp, std::string& message)
{
    if ((comp->text_ < 0) && (comp->icon_ < 0)) {
        SC_LOG_INFO(LABEL, "both text and icon do not exist.");
        return false;
    }
    if (comp->text_ >= 0) {
        if (LessOrEqual(comp->fontSize_, 0.0)) {
            SC_LOG_INFO(LABEL, "SecurityComponentCheckFail: fontSize is too small.");
            message = ", font size is too small, font size = " +
                std::to_string(comp->fontSize_);
            return false;
        }
    }
    if ((comp->icon_ >= 0) && comp->iconSize_ < MIN_ICON_SIZE) {
        SC_LOG_INFO(LABEL, "SecurityComponentCheckFail: iconSize is too small.");
        message = ", icon size is too small, icon size = " +
            std::to_string(comp->iconSize_);
        return false;
    }

    if (comp->bg_ == SecCompBackground::NO_BG_TYPE) {
        SC_LOG_ERROR(LABEL, "SecurityComponentCheckFail: background must be set.");
        return false;
    }

    bool res = IsSystemAppCalling();
    res |= comp->isArkuiComponent_;
    if (!res && !IsInReleaseList(comp->bgColor_.value) && IsBelowThreshold(comp->bgColor_)) {
        SC_LOG_ERROR(LABEL, "SecurityComponentCheckFail: the transparency of the background color cannot be too low.");
        message = ", the transparency of the background color is too low.";
        return false;
    }

    return CheckSecCompBaseButtonColorsimilar(comp, message);
}

static bool CheckSecCompBase(const SecCompBase* comp, std::string& message)
{
    if (comp->parentEffect_) {
        SC_LOG_ERROR(LABEL,
            "SecurityComponentCheckFail: the parents of security component have invalid effect.");
        message = "PARENT_HAVE_INVALID_EFFECT";
        return false;
    }

    if ((comp->padding_.top < MIN_PADDING_SIZE) || (comp->padding_.right < MIN_PADDING_SIZE) ||
        (comp->padding_.bottom < MIN_PADDING_SIZE) || (comp->padding_.left < MIN_PADDING_SIZE)) {
        SC_LOG_ERROR(LABEL, "SecurityComponentCheckFail: padding is too small.");
        message = ", padding is too small, padding(top = " + std::to_string(comp->padding_.top) +
            ", bottom = " + std::to_string(comp->padding_.bottom) +
            ", left = " + std::to_string(comp->padding_.left) +
            ", right = " + std::to_string(comp->padding_.right) + ")";
        return false;
    }

    if ((comp->textIconSpace_ < MIN_PADDING_SIZE)) {
        SC_LOG_ERROR(LABEL, "SecurityComponentCheckFail: textIconSpace is too small.");
        message = ", textIconSpace is too small, textIconSpace = " + std::to_string(comp->textIconSpace_);
        return false;
    }

    if ((comp->text_ != NO_TEXT) && (IsColorTransparent(comp->fontColor_))) {
        SC_LOG_ERROR(LABEL, "SecurityComponentCheckFail: fontColor is too transparent.");
        message = ", font color is too transparent, font color = " + ColorToHexString(comp->fontColor_);
        return false;
    }

    if ((comp->icon_ != NO_ICON) && (IsColorTransparent(comp->iconColor_))) {
        SC_LOG_ERROR(LABEL, "SecurityComponentCheckFail: iconColor is too transparent.");
        message = ", icon color is too transparent, icon color = " + ColorToHexString(comp->iconColor_);
        return false;
    }

    if (!CheckSecCompBaseButton(comp, message)) {
        return false;
    }
    return true;
}

bool SecCompInfoHelper::CheckComponentValid(SecCompBase* comp, std::string& message)
{
    if ((comp == nullptr) || !IsComponentTypeValid(comp->type_)) {
        SC_LOG_INFO(LABEL, "comp is null or type is invalid.");
        return false;
    }

    bool isCompatScaleMode = false;
    SecCompRect scaleRect;
    Scales scales = WindowInfoHelper::GetWindowScale(comp->windowId_, isCompatScaleMode, scaleRect);
    if ((!IsEqual(scales.floatingScale, WindowInfoHelper::FULL_SCREEN_SCALE) && !IsEqual(scales.floatingScale, 0.0)) ||
        isCompatScaleMode) {
        AdjustSecCompRect(comp, scales, isCompatScaleMode, scaleRect);
    }

    if (!CheckSecCompBase(comp, message)) {
        SC_LOG_INFO(LABEL, "SecComp base is invalid.");
        return false;
    }

    return true;
}
}  // namespace SecurityComponent
}  // namespace Security
}  // namespace OHOS
