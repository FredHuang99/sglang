// SPDX-License-Identifier: Apache-2.0

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;

extern "C" int32_t sfwanLaunchPackQuant(half const*, half const*, int8_t*, half*, float, int32_t const*,
    int32_t const*, int32_t const*, int32_t const*, int32_t const*, cudaStream_t);
extern "C" int32_t sfwanLaunchCacheUpdate(
    half const*, half const*, half*, int32_t const*, int32_t const*, int32_t const*, cudaStream_t);
extern "C" int32_t sfwanLaunchEpilogue(
    int32_t, int8_t const*, half const*, half*, float, int32_t const*, cudaStream_t);

namespace
{

constexpr char const* kNamespace = "sglang.sfwan";
constexpr char const* kVersion = "1";

template <size_t N>
bool readIntArray(PluginFieldCollection const* fields, char const* name, std::array<int32_t, N>& output)
{
    if (fields == nullptr || fields->fields == nullptr || name == nullptr)
    {
        return false;
    }
    for (int32_t index = 0; index < fields->nbFields; ++index)
    {
        PluginField const& field = fields->fields[index];
        if (std::strcmp(field.name, name) == 0)
        {
            if (field.type != PluginFieldType::kINT32 || field.length != static_cast<int32_t>(N) || field.data == nullptr)
            {
                return false;
            }
            std::memcpy(output.data(), field.data, sizeof(int32_t) * N);
            return true;
        }
    }
    return false;
}

bool readInt(PluginFieldCollection const* fields, char const* name, int32_t& output)
{
    std::array<int32_t, 1> value{};
    if (!readIntArray(fields, name, value))
    {
        return false;
    }
    output = value[0];
    return true;
}

bool readFloat(PluginFieldCollection const* fields, char const* name, float& output)
{
    if (fields == nullptr || fields->fields == nullptr || name == nullptr)
    {
        return false;
    }
    for (int32_t index = 0; index < fields->nbFields; ++index)
    {
        PluginField const& field = fields->fields[index];
        if (std::strcmp(field.name, name) == 0)
        {
            if (field.type != PluginFieldType::kFLOAT32 || field.length != 1 || field.data == nullptr)
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

bool compatibleFeatureShapes(std::array<int32_t, 5> const& current, std::array<int32_t, 5> const& cache)
{
    return current[0] == cache[0] && current[1] == cache[1] && current[3] == cache[3]
        && current[4] == cache[4];
}

bool descMatches(PluginTensorDesc const& desc, DataType type, TensorFormat format, std::array<int32_t, 5> const& shape)
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
    int64_t volume = 1;
    for (int32_t index = 0; index < desc.dims.nbDims; ++index)
    {
        if (desc.dims.d[index] <= 0)
        {
            return -1;
        }
        volume *= desc.dims.d[index];
    }
    return volume;
}

template <typename Derived>
class PluginCommon : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
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
            return new Derived(static_cast<Derived const&>(*this));
        }
        catch (...)
        {
            return nullptr;
        }
    }

    char const* getPluginVersion() const noexcept override
    {
        return kVersion;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return kNamespace;
    }

    int32_t getNbOutputs() const noexcept override
    {
        return static_cast<Derived const*>(this)->numOutputs();
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr
            || nbOutputs != static_cast<Derived const*>(this)->numOutputs())
        {
            return -1;
        }
        return static_cast<Derived*>(this)->validateDynamic(inputs, nbInputs, outputs, nbOutputs) ? 0 : -1;
    }

    int32_t onShapeChange(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr
            || nbOutputs != static_cast<Derived const*>(this)->numOutputs())
        {
            return -1;
        }
        return static_cast<Derived*>(this)->validateConcrete(inputs, nbInputs, outputs, nbOutputs) ? 0 : -1;
    }

    int32_t getOutputShapes(DimsExprs const*, int32_t, DimsExprs const*, int32_t, DimsExprs* outputs,
        int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
    {
        int32_t const expectedOutputs = static_cast<Derived const*>(this)->numOutputs();
        if (outputs == nullptr || nbOutputs != expectedOutputs)
        {
            return -1;
        }
        for (int32_t outputIndex = 0; outputIndex < expectedOutputs; ++outputIndex)
        {
            outputs[outputIndex].nbDims = 5;
            auto const& shape = static_cast<Derived const*>(this)->outputShape(outputIndex);
            for (int32_t dimension = 0; dimension < 5; ++dimension)
            {
                outputs[outputIndex].d[dimension] = exprBuilder.constant(shape[dimension]);
            }
        }
        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override
    {
        return clone();
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
        int32_t) const noexcept override
    {
        return 0;
    }
};

class PackQuantPlugin final : public PluginCommon<PackQuantPlugin>
{
public:
    PackQuantPlugin(float scale, int32_t hasCache, int32_t emitCache, std::array<int32_t, 10> pads,
        std::array<int32_t, 5> outputShape, std::array<int32_t, 5> cacheOutputShape,
        std::array<int32_t, 5> currentShape, std::array<int32_t, 5> cacheShape)
        : mScale(scale)
        , mHasCache(hasCache)
        , mEmitCache(emitCache)
        , mPads(pads)
        , mOutputShape(outputShape)
        , mCacheOutputShape(cacheOutputShape)
        , mCurrentShape(currentShape)
        , mCacheShape(cacheShape)
    {
        initFields();
    }

    PackQuantPlugin(PackQuantPlugin const& other)
        : PackQuantPlugin(other.mScale, other.mHasCache, other.mEmitCache, other.mPads, other.mOutputShape,
            other.mCacheOutputShape, other.mCurrentShape, other.mCacheShape)
    {
    }

    char const* getPluginName() const noexcept override
    {
        return "SfWanCausalPackQuantPlugin";
    }

    int32_t numOutputs() const noexcept
    {
        return mEmitCache != 0 ? 2 : 1;
    }

    std::array<int32_t, 5> const& outputShape(int32_t index) const noexcept
    {
        return index == 0 ? mOutputShape : mCacheOutputShape;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const*, int32_t) const noexcept override
    {
        if (outputTypes == nullptr || nbOutputs != numOutputs())
        {
            return -1;
        }
        outputTypes[0] = DataType::kINT8;
        if (mEmitCache != 0)
        {
            outputTypes[1] = DataType::kHALF;
        }
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        int32_t const expectedInputs = mHasCache != 0 ? 2 : 1;
        int32_t const expectedOutputs = numOutputs();
        if (inOut == nullptr || nbInputs != expectedInputs || nbOutputs != expectedOutputs || pos < 0
            || pos >= nbInputs + nbOutputs)
        {
            return false;
        }
        if (pos < nbInputs)
        {
            return inOut[pos].desc.type == DataType::kHALF && inOut[pos].desc.format == TensorFormat::kLINEAR;
        }
        if (pos == nbInputs)
        {
            return inOut[pos].desc.type == DataType::kINT8 && inOut[pos].desc.format == TensorFormat::kCDHW32;
        }
        return mEmitCache != 0 && inOut[pos].desc.type == DataType::kHALF
            && inOut[pos].desc.format == TensorFormat::kLINEAR;
    }

    bool validateDynamic(
        DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t) const
    {
        std::vector<PluginTensorDesc> inputs;
        inputs.reserve(nbInputs);
        for (int32_t index = 0; index < nbInputs; ++index)
        {
            inputs.push_back(in[index].desc);
        }
        std::vector<PluginTensorDesc> outputs;
        outputs.reserve(numOutputs());
        for (int32_t index = 0; index < numOutputs(); ++index)
        {
            outputs.push_back(out[index].desc);
        }
        return validateConcrete(inputs.data(), nbInputs, outputs.data(), numOutputs());
    }

    bool validateConcrete(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) const
    {
        int32_t const expectedInputs = mHasCache != 0 ? 2 : 1;
        return nbInputs == expectedInputs && nbOutputs == numOutputs()
            && descMatches(in[0], DataType::kHALF, TensorFormat::kLINEAR, mCurrentShape)
            && (mHasCache == 0 || descMatches(in[1], DataType::kHALF, TensorFormat::kLINEAR, mCacheShape))
            && descMatches(out[0], DataType::kINT8, TensorFormat::kCDHW32, mOutputShape)
            && (mEmitCache == 0
                || descMatches(out[1], DataType::kHALF, TensorFormat::kLINEAR, mCacheOutputShape));
    }

    int32_t enqueue(PluginTensorDesc const*, PluginTensorDesc const*, void const* const* inputs, void* const* outputs,
        void*, cudaStream_t stream) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr)
        {
            return -1;
        }
        auto const* cache = mHasCache != 0 ? static_cast<half const*>(inputs[1]) : nullptr;
        if (inputs[0] == nullptr || outputs[0] == nullptr || (mHasCache != 0 && cache == nullptr)
            || (mEmitCache != 0 && outputs[1] == nullptr))
        {
            return -1;
        }
        auto* cacheOutput = mEmitCache != 0 ? static_cast<half*>(outputs[1]) : nullptr;
        return sfwanLaunchPackQuant(static_cast<half const*>(inputs[0]), cache,
            static_cast<int8_t*>(outputs[0]), cacheOutput, mScale, mCurrentShape.data(), mCacheShape.data(),
            mOutputShape.data(), mCacheOutputShape.data(), mPads.data(), stream);
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFields;
    }

private:
    void initFields()
    {
        mSerialized = {
            {"scale", &mScale, PluginFieldType::kFLOAT32, 1},
            {"has_cache", &mHasCache, PluginFieldType::kINT32, 1},
            {"emit_cache", &mEmitCache, PluginFieldType::kINT32, 1},
            {"pads", mPads.data(), PluginFieldType::kINT32, 10},
            {"output_shape", mOutputShape.data(), PluginFieldType::kINT32, 5},
            {"cache_output_shape", mCacheOutputShape.data(), PluginFieldType::kINT32, 5},
            {"current_shape", mCurrentShape.data(), PluginFieldType::kINT32, 5},
            {"cache_shape", mCacheShape.data(), PluginFieldType::kINT32, 5},
        };
        mFields = {static_cast<int32_t>(mSerialized.size()), mSerialized.data()};
    }

    float mScale{};
    int32_t mHasCache{};
    int32_t mEmitCache{};
    std::array<int32_t, 10> mPads{};
    std::array<int32_t, 5> mOutputShape{};
    std::array<int32_t, 5> mCacheOutputShape{};
    std::array<int32_t, 5> mCurrentShape{};
    std::array<int32_t, 5> mCacheShape{};
    std::vector<PluginField> mSerialized;
    PluginFieldCollection mFields{};
};

class CacheUpdatePlugin final : public PluginCommon<CacheUpdatePlugin>
{
public:
    CacheUpdatePlugin(int32_t hasCache, std::array<int32_t, 5> outputShape, std::array<int32_t, 5> currentShape,
        std::array<int32_t, 5> cacheShape)
        : mHasCache(hasCache)
        , mOutputShape(outputShape)
        , mCurrentShape(currentShape)
        , mCacheShape(cacheShape)
    {
        initFields();
    }

    CacheUpdatePlugin(CacheUpdatePlugin const& other)
        : CacheUpdatePlugin(other.mHasCache, other.mOutputShape, other.mCurrentShape, other.mCacheShape)
    {
    }

    char const* getPluginName() const noexcept override
    {
        return "SfWanCacheUpdatePlugin";
    }

    int32_t numOutputs() const noexcept
    {
        return 1;
    }

    std::array<int32_t, 5> const& outputShape(int32_t) const noexcept
    {
        return mOutputShape;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const*, int32_t) const noexcept override
    {
        if (outputTypes == nullptr || nbOutputs != 1)
        {
            return -1;
        }
        outputTypes[0] = DataType::kHALF;
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        int32_t const expectedInputs = mHasCache != 0 ? 2 : 1;
        return inOut != nullptr && nbInputs == expectedInputs && nbOutputs == 1 && pos >= 0 && pos <= nbInputs
            && inOut[pos].desc.type == DataType::kHALF && inOut[pos].desc.format == TensorFormat::kLINEAR;
    }

    bool validateDynamic(
        DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t) const
    {
        std::vector<PluginTensorDesc> inputs;
        for (int32_t index = 0; index < nbInputs; ++index)
        {
            inputs.push_back(in[index].desc);
        }
        return validateConcrete(inputs.data(), nbInputs, &out[0].desc, 1);
    }

    bool validateConcrete(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t) const
    {
        int32_t const expectedInputs = mHasCache != 0 ? 2 : 1;
        return nbInputs == expectedInputs && descMatches(in[0], DataType::kHALF, TensorFormat::kLINEAR, mCurrentShape)
            && (mHasCache == 0 || descMatches(in[1], DataType::kHALF, TensorFormat::kLINEAR, mCacheShape))
            && descMatches(out[0], DataType::kHALF, TensorFormat::kLINEAR, mOutputShape);
    }

    int32_t enqueue(PluginTensorDesc const*, PluginTensorDesc const*, void const* const* inputs, void* const* outputs,
        void*, cudaStream_t stream) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr)
        {
            return -1;
        }
        auto const* cache = mHasCache != 0 ? static_cast<half const*>(inputs[1]) : nullptr;
        if (inputs[0] == nullptr || outputs[0] == nullptr || (mHasCache != 0 && cache == nullptr))
        {
            return -1;
        }
        return sfwanLaunchCacheUpdate(static_cast<half const*>(inputs[0]), cache, static_cast<half*>(outputs[0]),
            mCurrentShape.data(), mCacheShape.data(), mOutputShape.data(), stream);
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFields;
    }

private:
    void initFields()
    {
        mSerialized = {
            {"has_cache", &mHasCache, PluginFieldType::kINT32, 1},
            {"output_shape", mOutputShape.data(), PluginFieldType::kINT32, 5},
            {"current_shape", mCurrentShape.data(), PluginFieldType::kINT32, 5},
            {"cache_shape", mCacheShape.data(), PluginFieldType::kINT32, 5},
        };
        mFields = {static_cast<int32_t>(mSerialized.size()), mSerialized.data()};
    }

    int32_t mHasCache{};
    std::array<int32_t, 5> mOutputShape{};
    std::array<int32_t, 5> mCurrentShape{};
    std::array<int32_t, 5> mCacheShape{};
    std::vector<PluginField> mSerialized;
    PluginFieldCollection mFields{};
};

class EpiloguePlugin final : public PluginCommon<EpiloguePlugin>
{
public:
    EpiloguePlugin(int32_t mode, float scale, std::array<int32_t, 5> outputShape)
        : mMode(mode)
        , mScale(scale)
        , mOutputShape(outputShape)
    {
        initFields();
    }

    EpiloguePlugin(EpiloguePlugin const& other)
        : EpiloguePlugin(other.mMode, other.mScale, other.mOutputShape)
    {
    }

    char const* getPluginName() const noexcept override
    {
        return "SfWanInt8EpiloguePlugin";
    }

    int32_t numOutputs() const noexcept
    {
        return 1;
    }

    std::array<int32_t, 5> const& outputShape(int32_t) const noexcept
    {
        return mOutputShape;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const*, int32_t) const noexcept override
    {
        if (outputTypes == nullptr || nbOutputs != 1)
        {
            return -1;
        }
        outputTypes[0] = DataType::kHALF;
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != 2 || nbOutputs != 1 || pos < 0 || pos > 2)
        {
            return false;
        }
        if (pos == 0)
        {
            return inOut[pos].desc.type == DataType::kINT8 && inOut[pos].desc.format == TensorFormat::kCDHW32;
        }
        return inOut[pos].desc.type == DataType::kHALF && inOut[pos].desc.format == TensorFormat::kLINEAR;
    }

    bool validateDynamic(
        DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t) const
    {
        std::vector<PluginTensorDesc> inputs;
        for (int32_t index = 0; index < nbInputs; ++index)
        {
            inputs.push_back(in[index].desc);
        }
        return validateConcrete(inputs.data(), nbInputs, &out[0].desc, 1);
    }

    bool validateConcrete(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t) const
    {
        if (nbInputs != 2 || (mMode != 0 && mMode != 1))
        {
            return false;
        }
        bool const auxiliaryShapeValid = mMode == 0
            ? descVolume(in[1]) == static_cast<int64_t>(mOutputShape[1])
            : descMatches(in[1], DataType::kHALF, TensorFormat::kLINEAR, mOutputShape);
        return descMatches(in[0], DataType::kINT8, TensorFormat::kCDHW32, mOutputShape)
            && descMatches(out[0], DataType::kHALF, TensorFormat::kLINEAR, mOutputShape)
            && in[1].type == DataType::kHALF && in[1].format == TensorFormat::kLINEAR && auxiliaryShapeValid;
    }

    int32_t enqueue(PluginTensorDesc const*, PluginTensorDesc const*, void const* const* inputs, void* const* outputs,
        void*, cudaStream_t stream) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr
            || outputs[0] == nullptr)
        {
            return -1;
        }
        return sfwanLaunchEpilogue(mMode, static_cast<int8_t const*>(inputs[0]), static_cast<half const*>(inputs[1]),
            static_cast<half*>(outputs[0]), mScale, mOutputShape.data(), stream);
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
            {"scale", &mScale, PluginFieldType::kFLOAT32, 1},
            {"output_shape", mOutputShape.data(), PluginFieldType::kINT32, 5},
        };
        mFields = {static_cast<int32_t>(mSerialized.size()), mSerialized.data()};
    }

    int32_t mMode{};
    float mScale{};
    std::array<int32_t, 5> mOutputShape{};
    std::vector<PluginField> mSerialized;
    PluginFieldCollection mFields{};
};

template <typename Plugin>
class CreatorCommon : public IPluginCreatorV3One
{
public:
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

protected:
    void setFields(std::vector<PluginField> fields)
    {
        mAttributes = std::move(fields);
        mFields = {static_cast<int32_t>(mAttributes.size()), mAttributes.data()};
    }

private:
    std::vector<PluginField> mAttributes;
    PluginFieldCollection mFields{};
};

class PackQuantCreator final : public CreatorCommon<PackQuantPlugin>
{
public:
    PackQuantCreator()
    {
        setFields({{"scale", nullptr, PluginFieldType::kFLOAT32, 1},
            {"has_cache", nullptr, PluginFieldType::kINT32, 1}, {"pads", nullptr, PluginFieldType::kINT32, 10},
            {"emit_cache", nullptr, PluginFieldType::kINT32, 1},
            {"output_shape", nullptr, PluginFieldType::kINT32, 5},
            {"cache_output_shape", nullptr, PluginFieldType::kINT32, 5},
            {"current_shape", nullptr, PluginFieldType::kINT32, 5},
            {"cache_shape", nullptr, PluginFieldType::kINT32, 5}});
    }

    char const* getPluginName() const noexcept override
    {
        return "SfWanCausalPackQuantPlugin";
    }

    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fields, TensorRTPhase) noexcept override
    {
        try
        {
            float scale{};
            int32_t hasCache{}, emitCache{};
            std::array<int32_t, 10> pads{};
            std::array<int32_t, 5> output{}, cacheOutput{}, current{}, cache{};
            if (!readFloat(fields, "scale", scale) || !readInt(fields, "has_cache", hasCache)
                || !readInt(fields, "emit_cache", emitCache) || !readIntArray(fields, "pads", pads)
                || !readIntArray(fields, "output_shape", output)
                || !readIntArray(fields, "cache_output_shape", cacheOutput)
                || !readIntArray(fields, "current_shape", current) || !readIntArray(fields, "cache_shape", cache)
                || !std::isfinite(scale) || scale <= 0.0F || (hasCache != 0 && hasCache != 1)
                || (emitCache != 0 && emitCache != 1)
                || !positiveShape(output)
                || !positiveShape(current) || (hasCache != 0 && !positiveShape(cache))
                || (emitCache != 0 && !positiveShape(cacheOutput)))
            {
                return nullptr;
            }
            if ((hasCache != 0 && !compatibleFeatureShapes(current, cache))
                || std::any_of(pads.begin(), pads.end(), [](int32_t value) { return value < 0; })
                || pads[0] != 0 || pads[1] != 0 || pads[5] != 0 || pads[6] != 0
                || output[0] != current[0] || output[1] != current[1]
                || output[2] != current[2] + (hasCache != 0 ? cache[2] : 0) + pads[2] + pads[7]
                || output[3] != current[3] + pads[3] + pads[8]
                || output[4] != current[4] + pads[4] + pads[9]
                || (emitCache != 0
                    && (cacheOutput[0] != current[0] || cacheOutput[1] != current[1]
                        || cacheOutput[3] != current[3] || cacheOutput[4] != current[4]
                        || cacheOutput[2] > current[2] + (hasCache != 0 ? cache[2] : 0))))
            {
                return nullptr;
            }
            return new PackQuantPlugin(scale, hasCache, emitCache, pads, output, cacheOutput, current, cache);
        }
        catch (...)
        {
            return nullptr;
        }
    }
};

class CacheUpdateCreator final : public CreatorCommon<CacheUpdatePlugin>
{
public:
    CacheUpdateCreator()
    {
        setFields({{"has_cache", nullptr, PluginFieldType::kINT32, 1},
            {"output_shape", nullptr, PluginFieldType::kINT32, 5},
            {"current_shape", nullptr, PluginFieldType::kINT32, 5},
            {"cache_shape", nullptr, PluginFieldType::kINT32, 5}});
    }

    char const* getPluginName() const noexcept override
    {
        return "SfWanCacheUpdatePlugin";
    }

    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fields, TensorRTPhase) noexcept override
    {
        try
        {
            int32_t hasCache{};
            std::array<int32_t, 5> output{}, current{}, cache{};
            if (!readInt(fields, "has_cache", hasCache) || !readIntArray(fields, "output_shape", output)
                || !readIntArray(fields, "current_shape", current) || !readIntArray(fields, "cache_shape", cache)
                || (hasCache != 0 && hasCache != 1) || !positiveShape(output) || !positiveShape(current)
                || (hasCache != 0 && !positiveShape(cache)))
            {
                return nullptr;
            }
            if ((hasCache != 0 && !compatibleFeatureShapes(current, cache))
                || output[0] != current[0] || output[1] != current[1]
                || output[3] != current[3] || output[4] != current[4]
                || output[2] > current[2] + (hasCache != 0 ? cache[2] : 0))
            {
                return nullptr;
            }
            return new CacheUpdatePlugin(hasCache, output, current, cache);
        }
        catch (...)
        {
            return nullptr;
        }
    }
};

class EpilogueCreator final : public CreatorCommon<EpiloguePlugin>
{
public:
    EpilogueCreator()
    {
        setFields({{"mode", nullptr, PluginFieldType::kINT32, 1}, {"scale", nullptr, PluginFieldType::kFLOAT32, 1},
            {"output_shape", nullptr, PluginFieldType::kINT32, 5}});
    }

    char const* getPluginName() const noexcept override
    {
        return "SfWanInt8EpiloguePlugin";
    }

    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fields, TensorRTPhase) noexcept override
    {
        try
        {
            int32_t mode{};
            float scale{};
            std::array<int32_t, 5> output{};
            if (!readInt(fields, "mode", mode) || !readFloat(fields, "scale", scale)
                || !readIntArray(fields, "output_shape", output) || (mode != 0 && mode != 1)
                || !std::isfinite(scale) || scale <= 0.0F
                || !positiveShape(output))
            {
                return nullptr;
            }
            return new EpiloguePlugin(mode, scale, output);
        }
        catch (...)
        {
            return nullptr;
        }
    }
};

PackQuantCreator gPackQuantCreator;
CacheUpdateCreator gCacheUpdateCreator;
EpilogueCreator gEpilogueCreator;
std::once_flag gRegisterOnce;
bool gRegistered{false};

} // namespace

extern "C" bool initSfWanVaeTrtFusionPlugins()
{
    std::call_once(gRegisterOnce, [] {
        IPluginRegistry* registry = getPluginRegistry();
        if (registry == nullptr)
        {
            return;
        }
        bool const pack = registry->registerCreator(gPackQuantCreator, kNamespace);
        bool const cache = registry->registerCreator(gCacheUpdateCreator, kNamespace);
        bool const epilogue = registry->registerCreator(gEpilogueCreator, kNamespace);
        gRegistered = pack && cache && epilogue;
    });
    return gRegistered;
}
