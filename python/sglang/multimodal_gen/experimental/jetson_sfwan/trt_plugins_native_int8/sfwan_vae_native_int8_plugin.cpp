// SPDX-License-Identifier: Apache-2.0

#include "NvInfer.h"
#include "NvInferPlugin.h"
#ifndef SFWAN_NATIVE_INT8_KERNEL_HEADER
#define SFWAN_NATIVE_INT8_KERNEL_HEADER "sfwan_vae_native_int8_kernels.h"
#endif
#include SFWAN_NATIVE_INT8_KERNEL_HEADER

#ifndef SFWAN_NATIVE_INT8_CONFIG_TYPE
#define SFWAN_NATIVE_INT8_CONFIG_TYPE SfWanNativeInt8BlockConfig
#endif
#ifndef SFWAN_NATIVE_INT8_WORKSPACE_SIZE
#define SFWAN_NATIVE_INT8_WORKSPACE_SIZE sfwanNativeInt8WorkspaceSize
#endif
#ifndef SFWAN_NATIVE_INT8_LAUNCH
#define SFWAN_NATIVE_INT8_LAUNCH sfwanNativeInt8LaunchResidualBlock
#endif
#ifndef SFWAN_NATIVE_INT8_TILE_COUNT
#define SFWAN_NATIVE_INT8_TILE_COUNT sfwanNativeInt8TileCount
#endif
#ifndef SFWAN_NATIVE_INT8_PLUGIN_NAME
#define SFWAN_NATIVE_INT8_PLUGIN_NAME "SfWanNativeInt8ResidualBlockPlugin"
#endif
#ifndef SFWAN_NATIVE_INT8_PLUGIN_INIT
#define SFWAN_NATIVE_INT8_PLUGIN_INIT initSfWanVaeNativeInt8V1Plugins
#endif

// The V2 translation unit reuses the stable TensorRT IPluginV3 plumbing while
// binding it to a different config type and kernel entry points.  The default
// macros above preserve the byte-for-byte V1 ABI and creator identity.
#define SfWanNativeInt8BlockConfig SFWAN_NATIVE_INT8_CONFIG_TYPE
#define sfwanNativeInt8WorkspaceSize SFWAN_NATIVE_INT8_WORKSPACE_SIZE
#define sfwanNativeInt8LaunchResidualBlock SFWAN_NATIVE_INT8_LAUNCH
#define sfwanNativeInt8TileCount SFWAN_NATIVE_INT8_TILE_COUNT

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
constexpr char const* kVersion = "1";
constexpr char const* kPluginName = SFWAN_NATIVE_INT8_PLUGIN_NAME;
constexpr int32_t kInputs = 12;
constexpr int32_t kOutputs = 3;

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

template <size_t N>
bool readIntArray(PluginFieldCollection const* fields, char const* name,
    int32_t (&output)[N])
{
    std::array<int32_t, N> values{};
    if (!readIntArray(fields, name, values))
    {
        return false;
    }
    std::memcpy(output, values.data(), sizeof(int32_t) * N);
    return true;
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

bool positiveShape(int32_t const* shape, int32_t dimensions)
{
    for (int32_t index = 0; index < dimensions; ++index)
    {
        if (shape[index] <= 0)
        {
            return false;
        }
    }
    return true;
}

bool zeroShape(int32_t const* shape, int32_t dimensions)
{
    for (int32_t index = 0; index < dimensions; ++index)
    {
        if (shape[index] != 0)
        {
            return false;
        }
    }
    return true;
}

bool boolField(int32_t value)
{
    return value == 0 || value == 1;
}

bool descShape(PluginTensorDesc const& desc, int32_t const* shape,
    int32_t dimensions)
{
    if (desc.dims.nbDims != dimensions)
    {
        return false;
    }
    for (int32_t index = 0; index < dimensions; ++index)
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

class NativeResidualBlockPlugin final : public IPluginV3,
                                        public IPluginV3OneCore,
                                        public IPluginV3OneBuild,
                                        public IPluginV3OneRuntime
{
public:
    explicit NativeResidualBlockPlugin(SfWanNativeInt8BlockConfig config)
        : mConfig(config)
    {
        initFields();
    }

    NativeResidualBlockPlugin(NativeResidualBlockPlugin const& other)
        : NativeResidualBlockPlugin(other.mConfig)
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
            return new NativeResidualBlockPlugin(*this);
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

    int32_t getNbOutputs() const noexcept override
    {
        return kOutputs;
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
        DataType const*, int32_t) const noexcept override
    {
        if (outputTypes == nullptr || nbOutputs != kOutputs)
        {
            return -1;
        }
        outputTypes[0]
            = mConfig.outputIsInt8 ? DataType::kINT8 : DataType::kHALF;
        outputTypes[1] = DataType::kINT8;
        outputTypes[2] = DataType::kINT8;
        return 0;
    }

    bool matchesTypeFormat(DynamicPluginTensorDesc const& value,
        DataType type, TensorFormat format) const noexcept
    {
        return value.desc.type == type && value.desc.format == format;
    }

    bool supportsFormatCombination(int32_t pos,
        DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != kInputs || nbOutputs != kOutputs
            || pos < 0 || pos >= kInputs + kOutputs)
        {
            return false;
        }
        if (pos == 0)
        {
            return matchesTypeFormat(inOut[pos],
                mConfig.inputIsInt8 ? DataType::kINT8 : DataType::kHALF,
                mConfig.inputIsInt8 ? TensorFormat::kCDHW32
                                    : TensorFormat::kLINEAR);
        }
        if (pos == 1)
        {
            return matchesTypeFormat(inOut[pos],
                mConfig.shortcutIsInt8 ? DataType::kINT8 : DataType::kHALF,
                mConfig.shortcutIsInt8 ? TensorFormat::kCDHW32
                                       : TensorFormat::kLINEAR);
        }
        if (pos == 2 || pos == 3)
        {
            bool const hasCache = pos == 2 ? mConfig.hasCache1 != 0
                                           : mConfig.hasCache2 != 0;
            return matchesTypeFormat(inOut[pos], DataType::kINT8,
                hasCache ? TensorFormat::kCDHW32 : TensorFormat::kLINEAR);
        }
        if (pos == 4 || pos == 5)
        {
            return matchesTypeFormat(
                inOut[pos], DataType::kHALF, TensorFormat::kLINEAR);
        }
        if (pos == 6 || pos == 9)
        {
            return matchesTypeFormat(
                inOut[pos], DataType::kINT8, TensorFormat::kLINEAR);
        }
        if (pos == 7 || pos == 8 || pos == 10 || pos == 11)
        {
            return matchesTypeFormat(
                inOut[pos], DataType::kFLOAT, TensorFormat::kLINEAR);
        }
        if (pos == kInputs)
        {
            return matchesTypeFormat(inOut[pos],
                mConfig.outputIsInt8 ? DataType::kINT8 : DataType::kHALF,
                mConfig.outputIsInt8 ? TensorFormat::kCDHW32
                                     : TensorFormat::kLINEAR);
        }
        return matchesTypeFormat(
            inOut[pos], DataType::kINT8, TensorFormat::kCDHW32);
    }

    int32_t getOutputShapes(DimsExprs const*, int32_t, DimsExprs const*,
        int32_t, DimsExprs* outputs, int32_t nbOutputs,
        IExprBuilder& exprBuilder) noexcept override
    {
        if (outputs == nullptr || nbOutputs != kOutputs)
        {
            return -1;
        }
        int32_t const* shapes[kOutputs] = {mConfig.outputShape,
            mConfig.cache1OutputShape, mConfig.cache2OutputShape};
        for (int32_t output = 0; output < kOutputs; ++output)
        {
            outputs[output].nbDims = 5;
            for (int32_t dimension = 0; dimension < 5; ++dimension)
            {
                outputs[output].d[dimension]
                    = exprBuilder.constant(shapes[output][dimension]);
            }
        }
        return 0;
    }

    bool validateConcrete(PluginTensorDesc const* inputs, int32_t nbInputs,
        PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputs
            || nbOutputs != kOutputs)
        {
            return false;
        }
        if (!descShape(inputs[0], mConfig.inputShape, 5)
            || !descShape(inputs[1], mConfig.shortcutShape, 5)
            || !descShape(outputs[0], mConfig.outputShape, 5)
            || !descShape(outputs[1], mConfig.cache1OutputShape, 5)
            || !descShape(outputs[2], mConfig.cache2OutputShape, 5)
            || !descShape(inputs[6], mConfig.weight1Shape, 4)
            || !descShape(inputs[9], mConfig.weight2Shape, 4)
            || descVolume(inputs[4]) != mConfig.inputShape[1]
            || descVolume(inputs[5]) != mConfig.weight1Shape[0]
            || descVolume(inputs[7]) != mConfig.weight1Shape[0]
            || descVolume(inputs[8]) != mConfig.weight1Shape[0]
            || descVolume(inputs[10]) != mConfig.weight2Shape[0]
            || descVolume(inputs[11]) != mConfig.weight2Shape[0])
        {
            return false;
        }
        if (mConfig.hasCache1)
        {
            if (!descShape(inputs[2], mConfig.cache1InputShape, 5))
            {
                return false;
            }
        }
        else if (descVolume(inputs[2]) != 1)
        {
            return false;
        }
        if (mConfig.hasCache2)
        {
            if (!descShape(inputs[3], mConfig.cache2InputShape, 5))
            {
                return false;
            }
        }
        else if (descVolume(inputs[3]) != 1)
        {
            return false;
        }
        return true;
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
        return sfwanNativeInt8WorkspaceSize(&mConfig);
    }

    int32_t enqueue(PluginTensorDesc const*, PluginTensorDesc const*,
        void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr)
        {
            return -1;
        }
        return sfwanNativeInt8LaunchResidualBlock(&mConfig, inputs[0],
            inputs[1], mConfig.hasCache1 ? inputs[2] : nullptr,
            mConfig.hasCache2 ? inputs[3] : nullptr, inputs[4], inputs[5],
            inputs[6], inputs[7], inputs[8], inputs[9], inputs[10], inputs[11],
            outputs[0], outputs[1], outputs[2], workspace, stream);
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
            {"input_shape", mConfig.inputShape, PluginFieldType::kINT32, 5},
            {"shortcut_shape", mConfig.shortcutShape, PluginFieldType::kINT32, 5},
            {"output_shape", mConfig.outputShape, PluginFieldType::kINT32, 5},
            {"cache1_input_shape", mConfig.cache1InputShape, PluginFieldType::kINT32, 5},
            {"cache2_input_shape", mConfig.cache2InputShape, PluginFieldType::kINT32, 5},
            {"cache1_output_shape", mConfig.cache1OutputShape, PluginFieldType::kINT32, 5},
            {"cache2_output_shape", mConfig.cache2OutputShape, PluginFieldType::kINT32, 5},
            {"weight1_shape", mConfig.weight1Shape, PluginFieldType::kINT32, 4},
            {"weight2_shape", mConfig.weight2Shape, PluginFieldType::kINT32, 4},
            {"conv1_params", mConfig.conv1Params, PluginFieldType::kINT32, 6},
            {"conv2_params", mConfig.conv2Params, PluginFieldType::kINT32, 6},
            {"input_is_int8", &mConfig.inputIsInt8, PluginFieldType::kINT32, 1},
            {"shortcut_is_int8", &mConfig.shortcutIsInt8, PluginFieldType::kINT32, 1},
            {"output_is_int8", &mConfig.outputIsInt8, PluginFieldType::kINT32, 1},
            {"has_cache1", &mConfig.hasCache1, PluginFieldType::kINT32, 1},
            {"has_cache2", &mConfig.hasCache2, PluginFieldType::kINT32, 1},
            {"tile1", &mConfig.tile1, PluginFieldType::kINT32, 1},
            {"tile2", &mConfig.tile2, PluginFieldType::kINT32, 1},
            {"profile_id", &mConfig.profileId, PluginFieldType::kINT32, 1},
            {"input_scale", &mConfig.inputScale, PluginFieldType::kFLOAT32, 1},
            {"conv1_input_scale", &mConfig.conv1InputScale, PluginFieldType::kFLOAT32, 1},
            {"conv2_input_scale", &mConfig.conv2InputScale, PluginFieldType::kFLOAT32, 1},
            {"output_scale", &mConfig.outputScale, PluginFieldType::kFLOAT32, 1},
        };
        mFields = {static_cast<int32_t>(mSerialized.size()), mSerialized.data()};
    }

    SfWanNativeInt8BlockConfig mConfig{};
    std::vector<PluginField> mSerialized;
    PluginFieldCollection mFields{};
};

class NativeResidualBlockCreator final : public IPluginCreatorV3One
{
public:
    NativeResidualBlockCreator()
    {
        SfWanNativeInt8BlockConfig dummy{};
        NativeResidualBlockPlugin plugin(dummy);
        PluginFieldCollection const* serialized = plugin.getFieldsToSerialize();
        for (int32_t index = 0; index < serialized->nbFields; ++index)
        {
            PluginField field = serialized->fields[index];
            field.data = nullptr;
            mAttributes.push_back(field);
        }
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
            SfWanNativeInt8BlockConfig config{};
            bool valid = readIntArray(fields, "input_shape", config.inputShape)
                && readIntArray(fields, "shortcut_shape", config.shortcutShape)
                && readIntArray(fields, "output_shape", config.outputShape)
                && readIntArray(fields, "cache1_input_shape", config.cache1InputShape)
                && readIntArray(fields, "cache2_input_shape", config.cache2InputShape)
                && readIntArray(fields, "cache1_output_shape", config.cache1OutputShape)
                && readIntArray(fields, "cache2_output_shape", config.cache2OutputShape)
                && readIntArray(fields, "weight1_shape", config.weight1Shape)
                && readIntArray(fields, "weight2_shape", config.weight2Shape)
                && readIntArray(fields, "conv1_params", config.conv1Params)
                && readIntArray(fields, "conv2_params", config.conv2Params)
                && readInt(fields, "input_is_int8", config.inputIsInt8)
                && readInt(fields, "shortcut_is_int8", config.shortcutIsInt8)
                && readInt(fields, "output_is_int8", config.outputIsInt8)
                && readInt(fields, "has_cache1", config.hasCache1)
                && readInt(fields, "has_cache2", config.hasCache2)
                && readInt(fields, "tile1", config.tile1)
                && readInt(fields, "tile2", config.tile2)
                && readInt(fields, "profile_id", config.profileId)
                && readFloat(fields, "input_scale", config.inputScale)
                && readFloat(fields, "conv1_input_scale", config.conv1InputScale)
                && readFloat(fields, "conv2_input_scale", config.conv2InputScale)
                && readFloat(fields, "output_scale", config.outputScale);
            valid = valid && positiveShape(config.inputShape, 5)
                && positiveShape(config.shortcutShape, 5)
                && positiveShape(config.outputShape, 5)
                && positiveShape(config.cache1OutputShape, 5)
                && positiveShape(config.cache2OutputShape, 5)
                && positiveShape(config.weight1Shape, 4)
                && positiveShape(config.weight2Shape, 4)
                && boolField(config.inputIsInt8)
                && boolField(config.shortcutIsInt8)
                && boolField(config.outputIsInt8)
                && boolField(config.hasCache1) && boolField(config.hasCache2)
                && (config.hasCache1
                        ? positiveShape(config.cache1InputShape, 5)
                        : zeroShape(config.cache1InputShape, 5))
                && (config.hasCache2
                        ? positiveShape(config.cache2InputShape, 5)
                        : zeroShape(config.cache2InputShape, 5))
                && config.tile1 >= 0 && config.tile1 < sfwanNativeInt8TileCount()
                && config.tile2 >= 0 && config.tile2 < sfwanNativeInt8TileCount()
                && config.profileId >= 0 && config.profileId < 2048
                && std::isfinite(config.inputScale) && config.inputScale > 0.0F
                && std::isfinite(config.conv1InputScale)
                && config.conv1InputScale > 0.0F
                && std::isfinite(config.conv2InputScale)
                && config.conv2InputScale > 0.0F
                && std::isfinite(config.outputScale) && config.outputScale > 0.0F;
            if (!valid || sfwanNativeInt8WorkspaceSize(&config) == 0)
            {
                return nullptr;
            }
            return new NativeResidualBlockPlugin(config);
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

NativeResidualBlockCreator gCreator;
std::once_flag gRegisterOnce;
bool gRegistered{false};

} // namespace

extern "C" bool SFWAN_NATIVE_INT8_PLUGIN_INIT()
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
