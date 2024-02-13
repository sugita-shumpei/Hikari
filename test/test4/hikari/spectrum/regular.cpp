#include <hikari/spectrum/regular.h>
auto hikari::spectrum::SpectrumRegularObject::getIntensities() const noexcept -> Array<F32>
{
    return m_intensities;
}

void hikari::spectrum::SpectrumRegularObject::setIntensities(const Array<F32> &intensities)
{
    m_intensities = intensities;
}

void hikari::spectrum::SpectrumRegularObject::setIntensityCount(U32 size)
{
    m_intensities.resize(size, 0.0f);
}

auto hikari::spectrum::SpectrumRegularObject::getIntensityCount() const -> U32
{
    return m_intensities.size();
}

auto hikari::spectrum::SpectrumRegularObject::getIntensity(U32 index) const -> F32
{
    if (m_intensities.size() <= index)
    {
        return 0.0f;
    }
    return m_intensities.at(index);
}

void hikari::spectrum::SpectrumRegularObject::setIntensity(U32 index, F32 value)
{
    if (m_intensities.size() <= index)
    {
        return;
    }
    m_intensities.at(index) = value;
}

void hikari::spectrum::SpectrumRegularObject::addIntensity(F32 value)
{
    m_intensities.push_back(value);
}

void hikari::spectrum::SpectrumRegularObject::popIntensity(U32 index)
{
    if (m_intensities.size() <= index)
    {
        return;
    }
    m_intensities.erase(m_intensities.begin() + index);
}

auto hikari::spectrum::SpectrumRegularObject::getPropertyNames() const -> std::vector<Str>
{
    return {
        "min_wavelength",
        "max_wavelength",
        "intensities",
        "intensities.size",
        "color_setting",
        "color.xyz",
        "color.rgb"};
}

void hikari::spectrum::SpectrumRegularObject::getPropertyBlock(PropertyBlockBase<Object> &pb) const
{
    pb.clear();
    pb.setValue("max_wavelength", getMaxWaveLength());
    pb.setValue("min_wavelength", getMinWaveLength());
    pb.setValue("intensities", getIntensities());
    pb.setValue("color_setting", ColorSetting(getColorSetting()));
}

void hikari::spectrum::SpectrumRegularObject::setPropertyBlock(const PropertyBlockBase<Object> &pb)
{
    auto min_wavelength = pb.getValue("min_wavelength").getValueTo<F32>();
    auto max_wavelength = pb.getValue("max_wavelength").getValueTo<F32>();
    auto intensities = pb.getValue("intensities").getValue<Array<F32>>();
    auto color_setting = pb.getValue("color_setting").getValue<ColorSetting>();
    if (min_wavelength)
    {
        setMinWaveLength(*min_wavelength);
    }
    if (max_wavelength)
    {
        setMaxWaveLength(*max_wavelength);
    }
    if (!intensities.empty())
    {
        setIntensities(intensities);
    }
    if (color_setting)
    {
        setColorSetting(color_setting.getObject());
    }
}

bool hikari::spectrum::SpectrumRegularObject::hasProperty(const Str &name) const
{
    if (name == "min_wavelength")
    {
        return true;
    }
    if (name == "max_wavelength")
    {
        return true;
    }
    if (name == "intensities")
    {
        return true;
    }
    if (name == "intensities.size")
    {
        return true;
    }
    if (name == "color_setting")
    {
        return true;
    }
    return false;
}

bool hikari::spectrum::SpectrumRegularObject::getProperty(const Str &name, PropertyBase<Object> &prop) const
{
    if (name == "min_wavelength")
    {
        prop.setValue(getMinWaveLength());
        return true;
    }
    if (name == "max_wavelength")
    {
        prop.setValue(getMaxWaveLength());
        return true;
    }
    if (name == "intensities")
    {
        prop.setValue(getIntensities());
        return true;
    }
    if (name == "intensities.size")
    {
        prop.setValue(getIntensityCount());
        return true;
    }
    if (name == "color_setting")
    {
        prop.setValue(ColorSetting(getColorSetting()));
        return true;
    }
    return false;
}

bool hikari::spectrum::SpectrumRegularObject::setProperty(const Str &name, const PropertyBase<Object> &prop)
{
    if (name == "min_wavelength")
    {
        auto val = prop.getValueTo<F32>();
        if (val)
        {
            setMinWaveLength(*val);
            return true;
        }
        return false;
    }
    if (name == "max_wavelength")
    {
        auto val = prop.getValueTo<F32>();
        if (val)
        {
            setMaxWaveLength(*val);
            return true;
        }
        return false;
    }
    if (name == "intensities")
    {
        auto val = prop.getValueTo<Array<F32>>();
        if (val)
        {
            setIntensities(*val);
            return true;
        }
        return false;
    }
    if (name == "intensities.size")
    {
        auto val = prop.getValueTo<U32>();
        if (val)
        {
            setIntensityCount(*val);
            return true;
        }
        return false;
    }
    if (name == "color_setting")
    {
        auto color_setting = prop.getValue<ColorSetting>();
        setColorSetting(color_setting.getObject());
        return true;
    }
    return false;
}

auto hikari::spectrum::SpectrumRegularObject::sample(F32 wavelength) const -> F32
{
    if (m_min_wavelength > wavelength)
    {
        return 0.0f;
    }
    if (m_max_wavelength < wavelength)
    {
        return 0.0f;
    }
    if (m_intensities.size() == 0)
    {
        return 0.0f;
    }
    if (m_intensities.size() == 1)
    {
        return m_intensities[0];
    }
    auto flt_val = ((wavelength - m_min_wavelength) / (m_max_wavelength - m_min_wavelength)) * static_cast<F32>(m_intensities.size() - 1);
    auto flt_idx = floorf(flt_val);
    auto flt_off = flt_val - flt_idx;
    auto val0 = m_intensities[flt_idx];
    auto val1 = m_intensities[flt_idx + 1];
    return val0 * (1.0f - flt_off) + val1 * flt_off;
}

auto hikari::spectrum::SpectrumRegularObject::getXYZColor() const -> ColorXYZ
{
    auto min_wavelength = std::max(m_min_wavelength, 360.0f);
    auto max_wavelength = std::min(m_max_wavelength, 830.0f);
    if (m_intensities.size() == 0)
    {
        return {0.0f, 0.0f, 0.0f};
    }
    if (m_intensities.size() == 1)
    {
        auto xyz = Vec3();
        auto den = 0.0f;
        for (size_t i = 360; i <= 830 - 1; ++i)
        {
            auto xyz_i_0 = SpectrumXYZLUT::sample(i + 0);
            auto xyz_i_1 = SpectrumXYZLUT::sample(i + 1);
            den += (xyz_i_0.y + xyz_i_1.y) * 0.5f;
            if (min_wavelength > i)
            {
                continue;
            }
            if (max_wavelength < i)
            {
                continue;
            }
            xyz += Vec3{0.5f * (xyz_i_0.x + xyz_i_1.x), 0.5f * (xyz_i_0.y + xyz_i_1.y), 0.5f * (xyz_i_0.z + xyz_i_1.z)};
        }
        xyz /= den;
        return ColorXYZ{xyz.x, xyz.y, xyz.z};
    }
    auto integrate = [](F32 v0, F32 v1) -> F32
    {
        auto idx_beg = floorf(v0);
        auto idx_end = floorf(v1);
    };
    {
        for (size_t i = 0; i < m_intensities.size() - 1; ++i)
        {
            auto beg_range = (min_wavelength - min_wavelength) * (i + 0) / (m_intensities.size() - 1) + m_min_wavelength;
            auto end_range = (max_wavelength - max_wavelength) * (i + 1) / (m_intensities.size() - 1) + m_max_wavelength;
        }
    }
    return ColorXYZ{0.0f, 0.0f, 0.0f};
}

auto hikari::spectrum::SpectrumRegularSerializer::getTypeString() const noexcept -> Str
{
    return spectrum::SpectrumRegularObject::TypeString();
}

auto hikari::spectrum::SpectrumRegularSerializer::eval(const std::shared_ptr<Object> &object) const -> Json
{
    auto regular = ObjectUtils::convert<SpectrumRegularObject>(object);
    if (!regular)
    {
        return Json();
    }
    Json json = {};
    json["type"] = "SpectrumRegular";
    json["properties"] = Json();
    json["properties"]["max_wavelength"] = regular->getMaxWaveLength();
    json["properties"]["min_wavelength"] = regular->getMinWaveLength();
    json["properties"]["intensities"] = regular->getIntensities();
    return json;
}

auto hikari::spectrum::SpectrumRegularDeserializer::getTypeString() const noexcept -> Str
{
    return spectrum::SpectrumRegularObject::TypeString();
}

auto hikari::spectrum::SpectrumRegularDeserializer::eval(const Json &json) const -> std::shared_ptr<Object>
{
    auto properties = json.find("properties");
    if (properties == json.end())
    {
        return nullptr;
    }
    auto regular = SpectrumRegularObject::create();
    if (properties.value().is_null())
    {
        return regular;
    }
    if (!properties.value().is_object())
    {
        return nullptr;
    }
    auto min_wavelength = properties.value().find("min_wavelength");
    auto max_wavelength = properties.value().find("max_wavelength");
    auto intensities = properties.value().find("intensities");
    if (max_wavelength != properties.value().end())
    {
        try
        {
            auto val = max_wavelength.value().get<F32>();
            regular->setMaxWaveLength(val);
        }
        catch (...)
        {
            return nullptr;
        }
    }
    if (min_wavelength != properties.value().end())
    {
        try
        {
            auto val = min_wavelength.value().get<F32>();
            regular->setMinWaveLength(val);
        }
        catch (...)
        {
            return nullptr;
        }
    }
    if (intensities != properties.value().end())
    {
        try
        {
            auto val = intensities.value().get<Array<F32>>();
            regular->setIntensities(val);
        }
        catch (...)
        {
            return nullptr;
        }
    }
    return regular;
}
