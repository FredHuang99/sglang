// SPDX-License-Identifier: Apache-2.0

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "sfwan_vae_kernels_v2.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;

namespace
{

constexpr char const* kNamespace = "sglang.sfwan";
constexpr char const* kVersion = "2";
constexpr char const* kPluginName = "SfWanResidualBoundaryV2Plugin";

template <size_t N>
bool readIntArray(PluginFieldCollection const* fields, char const* name,
    std::array<int32_t, N>& output)
{
    if (fields == nullptr || fields->fields == nullptr)
    {
        return false;
    }
    for (int32_t index = 0; index < fields->nbFields; ++index)
    {
        PluginField const& field = fields->fields[index];
        if (std::strcmp(field.name, name) == 0)
        {
            if (field.type != PluginFieldType::kINT32
                || field.length != static_cast<int32_t>(N)
                || field.data == nullptr)
            {
                return false;
            }
            std::memcpy(output.data(), field.data, sizeof(int32_t) * N);
            return true;
        }
    }
    return false;
}

bool readInt(PluginFieldCollection const* fields, char const* name,
    int32_t& output)
{
    std::array<int32_t, 1> values{};
    if (!readIntArray(fields, name, values))
    {
        return false;
    }
    output = values[0];
    return true;
}

bool readFloat(PluginFieldCollection const* fields, char const* name,
    float& output)
{
    if (fields == nullptr || fields->fields == nullptr)
    {
        return false;
    }
    for (int32_t index = 0; index < fields->nbFields; ++index)
    {
        PluginField const& field = fields->fields[index];
        if (std::strcmp(field.name, name) == 0)
        {
            if (field.type != PluginFieldType::kFLOAT32 || field.length != 1
                || field.data == nullptr)
            {
                return false;
            }
            output = *static_cast<float const*>(field.data);
            return true;
        }
    }
    return false;
}

bool positiveShape(std::array<int32_t, 5> const& shape)
{
    for (int32_t value : shape)
    {
        if (value <= 0)
        {
            return false;
        }
    }
    return true;
}

bool zeroShape(std::array<int32_t, 5> const& shape)
{
    for (int32_t value : shape)
    {
        if (value != 0)
        {
            return false;
        }
    }
    return true;
}

bool descMatches(PluginTensorDesc const& desc, DataType type,
    TensorFormat format, std::array<int32_t, 5> const& shape)
{
    if (desc.type != type || desc.format != format || desc.dims.nbDims != 5)
    {
        return false;
    }
    for (int32_t index = 0; index < 5; ++index)
    {
        if (desc.dims.d[index] != shape[index])
        {
            return false;
        }
    }
    return true;
}

int64_t descVolume(PluginTensorDesc const& desc)
{
    if (desc.dims.nbDims <= 0)
    {
        return -1;
    }
    int64_t result = 1;
    for (int32_t index = 0; index < desc.dims.nbDims; ++index)
    {
        if (desc.dims.d[index] <= 0)
        {
            return -1;
        }
        result *= desc.dims.d[index];
    }
    return result;
}

class ResidualBoundaryV2Plugin final : public IPluginV3,
                                       public IPluginV3OneCore,
                                       public IPluginV3OneBuild,
                                       public IPluginV3OneRuntime
{
public:
    ResidualBoundaryV2Plugin(int32_t mode, float consumeScale,
        float produceScale, std::array<int32_t, 5> currentShape,
        std::array<int32_t, 5> cacheShape,
        std::array<int32_t, 5> packedShape,
        std::array<int32_t, 10> pads)
        : mMode(mode)
        , mConsumeScale(consumeScale)
        , mProduceScale(produceScale)
        , mCurrentShape(currentShape)
        , mCacheShape(cacheShape)
        , mPackedShape(packedShape)
        , mPads(pads)
    {
        initFields();
    }

    ResidualBoundaryV2Plugin(ResidualBoundaryV2Plugin const& other)
        : ResidualBoundaryV2Plugin(other.mMode, other.mConsumeScale,
            other.mProduceScale, other.mCurrentShape, other.mCacheShape,
            other.mPackedShape, other.mPads)
    {
    }

    IPluginCapability* getCapabilityInterface(
        PluginCapabilityType type) noexcept override
    {
        if (type == PluginCapabilityType::kCORE)
        {
            return static_cast<IPluginV3OneCore*>(this);
        }
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    IPluginV3* clone() noexcept override
    {
        try
        {
            return new ResidualBoundaryV2Plugin(*this);
        }
        catch (...)
        {
            return nullptr;
        }
    }

    char const* getPluginName() const noexcept override
    {
        return kPluginName;
    }

    char const* getPluginVersion() const noexcept override
    {
        return kVersion;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return kNamespace;
    }

    int32_t numInputs() const noexcept
    {
        if (mMode == 2)
        {
            return 4;
        }
        return mMode == 3 ? 2 : 3;
    }

    int32_t getNbOutputs() const noexcept override
    {
        if (mMode == 2)
        {
            return 3;
        }
        return mMode == 3 ? 1 : 2;
    }

    std::array<int32_t, 5> const& outputShape(int32_t index) const noexcept
    {
        if (mMode == 2)
        {
            return index == 0 ? mCurrentShape
                              : (index == 1 ? mPackedShape : mCacheShape);
        }
        if (mMode == 3)
        {
            return mCurrentShape;
        }
        return index == 0 ? mPackedShape : mCacheShape;
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
        DataType const*, int32_t) const noexcept override
    {
        if (outputTypes == nullptr || nbOutputs != getNbOutputs())
        {
            return -1;
        }
        if (mMode == 2)
        {
            outputTypes[0] = DataType::kHALF;
            outputTypes[1] = DataType::kINT8;
            outputTypes[2] = DataType::kINT8;
        }
        else if (mMode == 3)
        {
            outputTypes[0] = DataType::kHALF;
        }
        else
        {
            outputTypes[0] = DataType::kINT8;
            outputTypes[1] = DataType::kINT8;
        }
        return 0;
    }

    bool supportsFormatCombination(int32_t pos,
        DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != numInputs()
            || nbOutputs != getNbOutputs() || pos < 0
            || pos >= nbInputs + nbOutputs)
        {
            return false;
        }
        auto const isHalf = [&](int32_t value) {
            return inOut[value].desc.type == DataType::kHALF
                && inOut[value].desc.format == TensorFormat::kLINEAR;
        };
        auto const isInt8 = [&](int32_t value) {
            return inOut[value].desc.type == DataType::kINT8
                && inOut[value].desc.format == TensorFormat::kCDHW32;
        };
        if (mMode == 0)
        {
            return pos == 0 || pos == 1 ? isHalf(pos) : isInt8(pos);
        }
        if (mMode == 1)
        {
            return pos == 1 ? isHalf(pos) : isInt8(pos);
        }
        if (mMode == 2)
        {
            if (pos == 1 || pos == 2 || pos == nbInputs)
            {
                return isHalf(pos);
            }
            return isInt8(pos);
        }
        return pos == 0 ? isInt8(pos) : isHalf(pos);
    }

    int32_t getOutputShapes(DimsExprs const*, int32_t, DimsExprs const*,
        int32_t, DimsExprs* outputs, int32_t nbOutputs,
        IExprBuilder& exprBuilder) noexcept override
    {
        if (outputs == nullptr || nbOutputs != getNbOutputs())
        {
            return -1;
        }
        for (int32_t outputIndex = 0; outputIndex < nbOutputs; ++outputIndex)
        {
            outputs[outputIndex].nbDims = 5;
            auto const& shape = outputShape(outputIndex);
            for (int32_t dimension = 0; dimension < 5; ++dimension)
            {
                outputs[outputIndex].d[dimension]
                    = exprBuilder.constant(shape[dimension]);
            }
        }
        return 0;
    }

    bool validateConcrete(PluginTensorDesc const* inputs, int32_t nbInputs,
        PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != numInputs()
            || nbOutputs != getNbOutputs())
        {
            return false;
        }
        bool valid = false;
        if (mMode == 0)
        {
            valid = descMatches(inputs[0], DataType::kHALF,
                        TensorFormat::kLINEAR, mCurrentShape)
                && inputs[1].type == DataType::kHALF
                && inputs[1].format == TensorFormat::kLINEAR
                && descVolume(inputs[1]) == mCurrentShape[1]
                && descMatches(inputs[2], DataType::kINT8,
                    TensorFormat::kCDHW32, mCacheShape)
                && descMatches(outputs[0], DataType::kINT8,
                    TensorFormat::kCDHW32, mPackedShape)
                && descMatches(outputs[1], DataType::kINT8,
                    TensorFormat::kCDHW32, mCacheShape);
        }
        else if (mMode == 1)
        {
            valid = descMatches(inputs[0], DataType::kINT8,
                        TensorFormat::kCDHW32, mCurrentShape)
                && inputs[1].type == DataType::kHALF
                && inputs[1].format == TensorFormat::kLINEAR
                && descVolume(inputs[1]) == mCurrentShape[1]
                && descMatches(inputs[2], DataType::kINT8,
                    TensorFormat::kCDHW32, mCacheShape)
                && descMatches(outputs[0], DataType::kINT8,
                    TensorFormat::kCDHW32, mPackedShape)
                && descMatches(outputs[1], DataType::kINT8,
                    TensorFormat::kCDHW32, mCacheShape);
        }
        else if (mMode == 2)
        {
            valid = descMatches(inputs[0], DataType::kINT8,
                        TensorFormat::kCDHW32, mCurrentShape)
                && descMatches(inputs[1], DataType::kHALF,
                    TensorFormat::kLINEAR, mCurrentShape)
                && inputs[2].type == DataType::kHALF
                && inputs[2].format == TensorFormat::kLINEAR
                && descVolume(inputs[2]) == mCurrentShape[1]
                && descMatches(inputs[3], DataType::kINT8,
                    TensorFormat::kCDHW32, mCacheShape)
                && descMatches(outputs[0], DataType::kHALF,
                    TensorFormat::kLINEAR, mCurrentShape)
                && descMatches(outputs[1], DataType::kINT8,
                    TensorFormat::kCDHW32, mPackedShape)
                && descMatches(outputs[2], DataType::kINT8,
                    TensorFormat::kCDHW32, mCacheShape);
        }
        else
        {
            valid = descMatches(inputs[0], DataType::kINT8,
                        TensorFormat::kCDHW32, mCurrentShape)
                && descMatches(inputs[1], DataType::kHALF,
                    TensorFormat::kLINEAR, mCurrentShape)
                && descMatches(outputs[0], DataType::kHALF,
                    TensorFormat::kLINEAR, mCurrentShape);
        }
        return valid;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs, DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr)
        {
            return -1;
        }
        std::vector<PluginTensorDesc> concreteInputs;
        std::vector<PluginTensorDesc> concreteOutputs;
        for (int32_t index = 0; index < nbInputs; ++index)
        {
            concreteInputs.push_back(inputs[index].desc);
        }
        for (int32_t index = 0; index < nbOutputs; ++index)
        {
            concreteOutputs.push_back(outputs[index].desc);
        }
        return validateConcrete(concreteInputs.data(), nbInputs,
                   concreteOutputs.data(), nbOutputs)
            ? 0
            : -1;
    }

    int32_t onShapeChange(PluginTensorDesc const* inputs, int32_t nbInputs,
        PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override
    {
        return validateConcrete(inputs, nbInputs, outputs, nbOutputs) ? 0 : -1;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t,
        DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const*, PluginTensorDesc const*,
        void const* const* inputs, void* const* outputs, void*,
        cudaStream_t stream) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr)
        {
            return -1;
        }
        return sfwanV2LaunchBoundary(mMode, inputs[0], inputs[1],
            mMode == 3 ? nullptr : inputs[2],
            mMode == 2 ? inputs[3] : nullptr, outputs[0],
            getNbOutputs() > 1 ? outputs[1] : nullptr,
            getNbOutputs() > 2 ? outputs[2] : nullptr, mConsumeScale,
            mProduceScale, mCurrentShape.data(), mCacheShape.data(),
            mPackedShape.data(), mPads.data(), stream);
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override
    {
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFields;
    }

private:
    void initFields()
    {
        mSerialized = {
            {"mode", &mMode, PluginFieldType::kINT32, 1},
            {"consume_scale", &mConsumeScale, PluginFieldType::kFLOAT32, 1},
            {"produce_scale", &mProduceScale, PluginFieldType::kFLOAT32, 1},
            {"current_shape", mCurrentShape.data(), PluginFieldType::kINT32, 5},
            {"cache_shape", mCacheShape.data(), PluginFieldType::kINT32, 5},
            {"packed_shape", mPackedShape.data(), PluginFieldType::kINT32, 5},
            {"pads", mPads.data(), PluginFieldType::kINT32, 10},
        };
        mFields = {static_cast<int32_t>(mSerialized.size()), mSerialized.data()};
    }

    int32_t mMode{};
    float mConsumeScale{};
    float mProduceScale{};
    std::array<int32_t, 5> mCurrentShape{};
    std::array<int32_t, 5> mCacheShape{};
    std::array<int32_t, 5> mPackedShape{};
    std::array<int32_t, 10> mPads{};
    std::vector<PluginField> mSerialized;
    PluginFieldCollection mFields{};
};

class ResidualBoundaryV2Creator final : public IPluginCreatorV3One
{
public:
    ResidualBoundaryV2Creator()
    {
        mAttributes = {
            {"mode", nullptr, PluginFieldType::kINT32, 1},
            {"consume_scale", nullptr, PluginFieldType::kFLOAT32, 1},
            {"produce_scale", nullptr, PluginFieldType::kFLOAT32, 1},
            {"current_shape", nullptr, PluginFieldType::kINT32, 5},
            {"cache_shape", nullptr, PluginFieldType::kINT32, 5},
            {"packed_shape", nullptr, PluginFieldType::kINT32, 5},
            {"pads", nullptr, PluginFieldType::kINT32, 10},
        };
        mFields = {static_cast<int32_t>(mAttributes.size()), mAttributes.data()};
    }

    char const* getPluginName() const noexcept override
    {
        return kPluginName;
    }

    char const* getPluginVersion() const noexcept override
    {
        return kVersion;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return kNamespace;
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFields;
    }

    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fields,
        TensorRTPhase) noexcept override
    {
        try
        {
            int32_t mode{};
            float consumeScale{}, produceScale{};
            std::array<int32_t, 5> current{}, cache{}, packed{};
            std::array<int32_t, 10> pads{};
            if (!readInt(fields, "mode", mode)
                || !readFloat(fields, "consume_scale", consumeScale)
                || !readFloat(fields, "produce_scale", produceScale)
                || !readIntArray(fields, "current_shape", current)
                || !readIntArray(fields, "cache_shape", cache)
                || !readIntArray(fields, "packed_shape", packed)
                || !readIntArray(fields, "pads", pads) || mode < 0 || mode > 3
                || !std::isfinite(consumeScale) || consumeScale <= 0.0F
                || !std::isfinite(produceScale) || produceScale <= 0.0F
                || !positiveShape(current) || current[1] % 32 != 0)
            {
                return nullptr;
            }
            if (mode != 3)
            {
                if (!positiveShape(cache) || !positiveShape(packed)
                    || cache[0] != current[0] || cache[1] != current[1]
                    || cache[3] != current[3] || cache[4] != current[4]
                    || packed[0] != current[0] || packed[1] != current[1]
                    || packed[2] != pads[2] + cache[2] + current[2]
                            + pads[7]
                    || packed[3] != pads[3] + current[3] + pads[8]
                    || packed[4] != pads[4] + current[4] + pads[9])
                {
                    return nullptr;
                }
            }
            else if (!zeroShape(cache) || !zeroShape(packed))
            {
                return nullptr;
            }
            return new ResidualBoundaryV2Plugin(mode, consumeScale,
                produceScale, current, cache, packed, pads);
        }
        catch (...)
        {
            return nullptr;
        }
    }

private:
    std::vector<PluginField> mAttributes;
    PluginFieldCollection mFields{};
};

ResidualBoundaryV2Creator gCreator;
std::once_flag gRegisterOnce;
bool gRegistered{false};

} // namespace

extern "C" bool initSfWanVaeTrtFusionV2Plugins()
{
    std::call_once(gRegisterOnce, [] {
        IPluginRegistry* registry = getPluginRegistry();
        if (registry != nullptr)
        {
            gRegistered = registry->registerCreator(gCreator, kNamespace);
        }
    });
    return gRegistered;
}
