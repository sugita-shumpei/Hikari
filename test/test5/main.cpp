#include <hikari/core/types/object.h>
#include <hikari/core/types/properties.h>
using namespace hikari;

struct Sample : public core::Properties {
  HK_CORE_TYPES_OBJECT_IMPLEM_TYPE_RELATIONSHIP(Sample, core::Properties);
  virtual ~Sample() noexcept {}
  virtual auto getPropertyNames() const->ArrayString override { return { "name" }; }
  virtual Bool hasProperty(const String& name) const { if ("name" == name) { return true; } return false; }
  virtual Bool getProperty(const String& name, Property& prop) const override {
    if (name == "name") {
      prop = m_name;
      return true;
    }
    return false;
  }
  virtual Bool setProperty(const String& name, const Property& prop) override {
    if (name == "name") {
      auto str = prop.getString();
      if (str) {
        m_name = name; return true;
      }
      return false;
    }
    return false;
  }
  inline  auto getProperty(const String& name) const-> Property {
    Property prop;
    if (getProperty(name, prop)) { return prop; }
    return Property();
  }
private:
  String m_name = "";
};



int main() {
  //o["tekitou"] = 0;
  //o["tekitou"] = 0ull;
  //o["tekitou"] = "tekitou";
  hikari::core::SRefObject sref = std::shared_ptr<Object>(new Sample());
  hikari::core::WRefObject wref = sref;
  auto sref_prop = RefObjectUtils::convert<hikari::core::SRefProperties>(sref);
  hikari::core::WRefProperties wref_prop = sref_prop;
  sref_prop["100"]   = 100;
  sref_prop["100"]   = sref;
  sref_prop["100"]   = wref;
  sref_prop["100"]   = sref_prop;
  sref_prop["100"]   = wref_prop;
  sref_prop["name"]  = "name";
  hikari::Property p = Array<Bool>{ true,false,true,false };
  auto pp            = p.getArrayBool();
  Quat q(1.0f, 0.0f, 0.0f, 0.0f);
  std::cout << q[0] << q[1] << q[2] << q[3] << std::endl;
  Transform t1(Vec3(1, 1, 1), Quat(1.0f, 0.0f, 0.0f, 0.0f), Vec3(2));
  auto m       = t1.getMat();
  Transform t2 = m;
  auto type    = t1.getType();
  auto type2   = t2.getType();
  return 0;
}
