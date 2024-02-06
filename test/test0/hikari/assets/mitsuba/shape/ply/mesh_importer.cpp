#include <hikari/assets/mitsuba/shape/ply/mesh_importer.h>
#include <tinyply.h>
#include <scn/scn.h>
#include <iostream>
#include <fstream>

auto hikari::assets::mitsuba::shape::PlyMeshImporterImpl::load() -> MeshImportOutput
{
    std::string type_str;
    std::ifstream file(m_filename, std::ios::binary);
    if (!file || file.fail())
    {
        return {};
    }
    size_t file_size = 0;

    file.seekg(0l, std::ios::end);
    file_size = file.tellg();
    file.seekg(0l, std::ios::beg);

    tinyply::PlyFile ply_file;
    ply_file.parse_header(file);

    std::shared_ptr<tinyply::PlyData> vertex_positions;
    std::shared_ptr<tinyply::PlyData> vertex_normals;
    std::shared_ptr<tinyply::PlyData> vertex_colors;
    std::shared_ptr<tinyply::PlyData> vertex_texcoords;
    std::shared_ptr<tinyply::PlyData> faces;

    try
    {
        vertex_positions = ply_file.request_properties_from_element("vertex", {"x", "y", "z"});
    }
    catch (const std::exception &e)
    {
#ifndef NDEBUG
      std::cerr << "tinyply exception: " << e.what() << std::endl;
#endif
    }
    try
    {
        vertex_normals = ply_file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
    }
    catch (const std::exception &e)
    {
#ifndef NDEBUG
        std::cerr << "tinyply exception: " << e.what() << std::endl;
#endif
    }
    try
    {
        vertex_colors = ply_file.request_properties_from_element("vertex", {"red", "green", "blue", "alpha"});
    }
    catch (const std::exception &e)
    {
#ifndef NDEBUG
        std::cerr << "tinyply exception: " << e.what() << std::endl;
#endif
    }
    try
    {
        vertex_colors = ply_file.request_properties_from_element("vertex", {"r", "g", "b", "a"});
    }
    catch (const std::exception &e)
    {
#ifndef NDEBUG
      std::cerr << "tinyply exception: " << e.what() << std::endl;
#endif
    }
    try
    {
        vertex_texcoords = ply_file.request_properties_from_element("vertex", {"u", "v"});
    }
    catch (const std::exception &e)
    {
#ifndef NDEBUG
      std::cerr << "tinyply exception: " << e.what() << std::endl;
#endif
    }
    try
    {
        faces = ply_file.request_properties_from_element("face", {"vertex_indices"}, 3);
    }
    catch (const std::exception &e)
    {
#ifndef NDEBUG
      std::cerr << "tinyply exception: " << e.what() << std::endl;
#endif
    }

    ply_file.read(file);
    if (!vertex_positions)
    {
        throw std::runtime_error("Failed To Find Verex Position In Ply File!");
    }

    auto loadFloats = [](const std::shared_ptr<tinyply::PlyData> &ply_data, size_t n) -> std::vector<float>
    {
        if (!ply_data)
        {
            return {};
        }
        if (ply_data->t == tinyply::Type::FLOAT32)
        {
            std::vector<float> res;
            res.resize(n * ply_data->count);
            std::memcpy(res.data(), ply_data->buffer.get(), sizeof(res[0]) * res.size());
            return res;
        }
        else if (ply_data->t == tinyply::Type::FLOAT64)
        {
            auto tmp = std::vector<double>(n * ply_data->count);
            std::memcpy(tmp.data(), ply_data->buffer.get(), sizeof(tmp[0]) * tmp.size());
            return std::vector<float>(tmp.begin(), tmp.end());
        }
        else
        {
            return {};
        }
    };
    auto loadUInt32 = [](const std::shared_ptr<tinyply::PlyData> &ply_data, size_t num_vertices) -> std::vector<uint32_t>
    {
        if (!ply_data)
        {
            return {};
        }
#define HK_ASSETS_MITSUBA_SHAPE_PLY_LOAD_UINT32_IMPL_INT(SIZE)                        \
    if (ply_data->t == tinyply::Type::INT##SIZE)                                      \
    {                                                                                 \
        std::vector<int##SIZE##_t> tmp(ply_data->count * 3);                          \
        std::memcpy(tmp.data(), ply_data->buffer.get(), tmp.size() * sizeof(tmp[0])); \
        std::vector<uint32_t> ply_data(ply_data->count * 3);                          \
        for (size_t i = 0; i < tmp.size(); ++i)                                       \
        {                                                                             \
            auto idx = tmp[i];                                                        \
            if (idx < 0)                                                              \
            {                                                                         \
                ply_data[i] = idx + num_vertices;                                     \
            }                                                                         \
            else                                                                      \
            {                                                                         \
                ply_data[i] = idx;                                                    \
            }                                                                         \
        }                                                                             \
        return std::vector<uint32_t>(tmp.begin(), tmp.end());                         \
    }

#define HK_ASSETS_MITSUBA_SHAPE_PLY_LOAD_UINT32_IMPL_UINT(SIZE)                       \
    if (ply_data->t == tinyply::Type::INT##SIZE)                                      \
    {                                                                                 \
        std::vector<int##SIZE##_t> tmp(ply_data->count * 3);                          \
        std::memcpy(tmp.data(), ply_data->buffer.get(), tmp.size() * sizeof(tmp[0])); \
        std::vector<uint32_t> ply_data(ply_data->count * 3);                          \
        for (size_t i = 0; i < tmp.size(); ++i)                                       \
        {                                                                             \
            auto idx = tmp[i];                                                        \
            ply_data[i] = idx;                                                        \
        }                                                                             \
        return std::vector<uint32_t>(tmp.begin(), tmp.end());                         \
    }

        if (ply_data->t == tinyply::Type::INT32)
        {
            std::vector<int32_t> tmp(ply_data->count * 3);
            std::memcpy(tmp.data(), ply_data->buffer.get(), tmp.size() * sizeof(tmp[0]));
            std::vector<uint32_t> ply_data(ply_data->count * 3);
            for (size_t i = 0; i < tmp.size(); ++i)
            {
                auto idx = tmp[i];
                if (idx < 0)
                {
                    ply_data[i] = idx + num_vertices;
                }
                else
                {
                    ply_data[i] = idx;
                }
            }
            return std::vector<uint32_t>(tmp.begin(), tmp.end());
        }

        HK_ASSETS_MITSUBA_SHAPE_PLY_LOAD_UINT32_IMPL_INT(32)
        else HK_ASSETS_MITSUBA_SHAPE_PLY_LOAD_UINT32_IMPL_INT(16) else HK_ASSETS_MITSUBA_SHAPE_PLY_LOAD_UINT32_IMPL_INT(8) else HK_ASSETS_MITSUBA_SHAPE_PLY_LOAD_UINT32_IMPL_UINT(32) else HK_ASSETS_MITSUBA_SHAPE_PLY_LOAD_UINT32_IMPL_UINT(16) else HK_ASSETS_MITSUBA_SHAPE_PLY_LOAD_UINT32_IMPL_UINT(8) else
        {
          return {};
        }
    };

    auto data_vertex_texcoords  = loadFloats(vertex_texcoords, 2);
    auto data_vertex_positions  = loadFloats(vertex_positions, 3);
    auto data_vertex_normals    = loadFloats(vertex_normals, 3);
    auto tmp_data_vertex_colors = loadFloats(vertex_colors, 4);
    auto data_vertex_colors = std::vector<float>();
    if (!tmp_data_vertex_colors.empty())
    {
        data_vertex_colors.reserve((tmp_data_vertex_colors.size() / 4) * 3);
        for (size_t i = 0; i < tmp_data_vertex_colors.size() / 4; ++i)
        {
            data_vertex_colors.push_back(tmp_data_vertex_colors[4 * i + 0]);
            data_vertex_colors.push_back(tmp_data_vertex_colors[4 * i + 1]);
            data_vertex_colors.push_back(tmp_data_vertex_colors[4 * i + 2]);
        }
    }
    auto data_faces = loadUInt32(faces, data_vertex_positions.size() / 3);

    file.close();
    if (data_vertex_colors.empty())
      return { { Mesh::create("", data_faces, data_vertex_positions, data_vertex_normals, data_vertex_texcoords, {{"colors", data_vertex_colors}})},nullptr };
    else
      return { { Mesh::create("", data_faces, data_vertex_positions, data_vertex_normals, data_vertex_texcoords,{}) },nullptr };
}

