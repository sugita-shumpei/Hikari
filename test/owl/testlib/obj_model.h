#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
namespace hikari {
    namespace test {
        namespace owl {
            namespace testlib {
                struct ObjIdx3 {
                    uint32_t x; uint32_t y; uint32_t z;
                };
                struct ObjVec2 {
                    float x; float y;
                };
                struct ObjVec3  {
                    float x; float y; float z;
                };
                struct ObjVec4 {
                    float x; float y; float z; float w;
                };
                struct ObjBBox {
                    void    addPoint(const ObjVec3& p) {
                        if (p.x > max.x) { max.x = p.x; }
                        if (p.x < min.x) { min.x = p.x; }
                        if (p.y > max.y) { max.y = p.y; }
                        if (p.y < min.y) { min.y = p.y; }
                        if (p.z > max.z) { max.z = p.z; }
                        if (p.z < min.z) { min.z = p.z; }
                    }
                    void    addBBox (const ObjBBox& b){
                        addPoint(b.max);
                        addPoint(b.min);
                    }
                    ObjVec3 getCenter() const {
                        return
                        {
                            (max.x + min.x) * 0.5f ,
                            (max.y + min.y) * 0.5f ,
                            (max.z + min.z) * 0.5f
                        };
                    }
                    ObjVec3 getRange () const {
                        return
                        {
                            (max.x - min.x) ,
                            (max.y - min.y) ,
                            (max.z - min.z) 
                        };
                    }
                    void    setCenter(const ObjVec3& c) {
                        auto r = getRange();
                        max = { c.x + r.x * 0.5f,c.y + r.y * 0.5f ,c.z + r.z * 0.5f };
                        min = { c.x - r.x * 0.5f,c.y - r.y * 0.5f ,c.z - r.z * 0.5f };
                    }
                    void    setRange(const ObjVec3& r) {
                        auto c = getCenter();
                        max = { c.x + r.x * 0.5f,c.y + r.y * 0.5f ,c.z + r.z * 0.5f };
                        min = { c.x - r.x * 0.5f,c.y - r.y * 0.5f ,c.z - r.z * 0.5f };
                    }

                    ObjVec3 max = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
                    ObjVec3 min = { +FLT_MAX,+FLT_MAX,+FLT_MAX };
                };
                struct ObjMaterial { 
                    std::string name             = "";
                    unsigned    illum            = 0;
                    ObjVec3     diffuse          = ObjVec3{ 1.0f,1.0f,1.0f };
                    ObjVec3     specular         = ObjVec3{ 1.0f,1.0f,1.0f };
                    ObjVec3     emission         = ObjVec3{ 0.0f,0.0f,0.0f };
                    float       ior              = 1.0f;
                    float       shinness         = 0.0f;
                    float       dissolve         = 0;
                    uint16_t    tex_diffuse      = 0; // 0 -> default or none
                    uint16_t    tex_specular     = 0; // 0 -> default or none
                    uint16_t    tex_shinness     = 0; // 0 -> default or none
                    uint16_t    tex_emission     = 0; // 0 -> default or none
                    uint16_t    tex_alpha        = 0; // 0 -> default or none
                    uint16_t    tex_normal       = 0; // 0 -> default or none
                    uint16_t    tex_bump         = 0; // 0 -> default or none
                    uint16_t    tex_displacement = 0; // 0 -> default or none
                    uint16_t    tex_reflection   = 0; // 0 -> default or none
                };
                struct ObjTexture  {
                    std::string filename     = "";
                };
                struct ObjMesh  {
                    auto getVisSmoothColors() const->std::vector<ObjVec3>;
                    auto getVisMaterialColors() const->std::vector<ObjVec3>;
                    auto getSubMeshIndices(uint32_t mat_idx = 0) const -> std::vector<ObjIdx3>;
                    std::vector<ObjIdx3>  tri_indices            ;
                    std::vector<ObjVec3>  positions              ;
                    std::vector<ObjVec3>  normals                ;
                    std::vector<ObjVec4>  tangents               ;
                    std::vector<ObjVec2>  uvs                    ;
                    std::vector<uint8_t>  mat_indices            ;
                    std::vector<uint16_t> materials              ;// ����1, �ꍇ�ɂ����2
                    std::vector<uint8_t>  smoothing_group_indices;
                    std::vector<uint8_t>  smoothing_groups       ;// ����1, �ꍇ�ɂ����2
                    ObjBBox               bbox;
                };
                struct ObjModel {
                    bool load(std::string filename);

                    void        setFilename(std::string filename);
                    std::string getFilename() const { return m_filename; }

                    auto  getBBox() const { return m_bbox; }
                    auto& getMaterials() const { return m_materials; }
                    auto& getMaterials()       { return m_materials; }
                    auto& getTextures() const  { return m_textures;  }
                    auto& getTextures()        { return m_textures;  }

                    auto operator[](const std::string& name) { return m_meshes[name]; }

                    auto at(const std::string& name) { return m_meshes.at(name); }
                    auto at(const std::string& name)const { return m_meshes.at(name); }

                    auto size() const  { return m_meshes.size(); }
                    auto empty() const { return m_meshes.empty(); }
                    void clear()       { 
                        m_filename = ""; 
                        m_bbox = {}; 
                        m_meshes.clear();
                        m_materials.clear();
                        m_textures.clear();
                    }

                    auto begin()      { return m_meshes.begin();  }
                    auto end()        { return m_meshes.end();    }
                    auto begin()const { return m_meshes.begin();  }
                    auto end()const   { return m_meshes.end();    }
                    auto cbegin()     { return m_meshes.cbegin(); }
                    auto cend()       { return m_meshes.cend();   }
                private:
                    std::string                              m_filename ;
                    std::unordered_map<std::string, ObjMesh> m_meshes;
                    std::vector<ObjMaterial>                 m_materials;
                    std::vector<ObjTexture>                  m_textures;
                    ObjBBox                                  m_bbox;
                };

            }
        }
    }
}
