#include <cmath>
//#include <matplot/matplot.h>
//
//int main() {
//  using namespace matplot;
//  std::vector<double> t  = iota(0, pi / 50, 10 * pi);
//  std::vector<double> st = transform(t, [](auto x) { return sin(x); });
//  std::vector<double> ct = transform(t, [](auto x) { return cos(x); });
//  auto l = plot3(st, ct, t);
//  show();
//  return 0;
//}
#include <bitset>
#include <hikari/core/node.h>
#include <hikari/core/field.h>
#include <hikari/core/spectrum.h>
#include <hikari/spectrum/regular.h>
#include <hikari/spectrum/irregular.h>
#include <hikari/spectrum/uniform.h>
#include <hikari/spectrum/blackbody.h>
#include <hikari/spectrum/color.h>
using namespace hikari;
int main() {
  ObjectSerializeManager::getInstance().add(std::make_shared<NodeSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<FieldSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<SpectrumRegularSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<SpectrumIrregularSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<SpectrumUniformSerializer>());
  ObjectSerializeManager::getInstance().add(std::make_shared<SpectrumColorSerializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<NodeDeserializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<FieldDeserializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<SpectrumRegularDeserializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<SpectrumIrregularDeserializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<SpectrumUniformDeserializer>());
  ObjectDeserializeManager::getInstance().add(std::make_shared<SpectrumColorDeserializer>());

  Node node1 = "0";
  node1.setChildCount(3);
  node1[0] = Node("0", Transform(Vec3(1, 1, 1)));
  node1[1] = Node("1", Transform(Vec3(2, 2, 2)));
  node1[2] = Node("2", Transform(Vec3(3, 3, 3)));
  std::cout << serialize(node1).dump() << std::endl;
  auto node2 = deserialize<ObjectWrapper>(serialize(node1));
  std::cout << serialize(node2).dump() << std::endl;

  SpectrumRegular regular1 = Array<F32>{ 1.0f };
  regular1.setSize(4);
  regular1[0] = 1.0f;
  regular1[1] = 0.5f;
  regular1[2] = 0.25f;
  regular1[3] = 0.125f;

  auto regular2 = ObjectWrapperUtils::convert<Spectrum>(regular1);
  std::cout << serialize(regular2).dump() << std::endl;
  auto regular3 = deserialize<ObjectWrapper>(serialize(regular2));
  std::cout << serialize(regular3).dump() << std::endl;

  SpectrumUniform uniform1   = 1.0f;
  uniform1["min_wavelength"] = 600.0f;
  uniform1["max_wavelength"] = 800.0f;
  auto uniform2 = ObjectWrapperUtils::convert<Spectrum>(uniform1);
  std::cout << serialize(uniform2).dump() << std::endl;
  auto uniform3 = deserialize<ObjectWrapper>(serialize(uniform2));
  std::cout << serialize(uniform3).dump() << std::endl;
  // RGB色空間へ変換
  auto cie_1931  = uniform1.getRGBColor(ColorSpace::eCIE1931);
  auto adobe_rgb = uniform1.getRGBColor(ColorSpace::eAdobeRGB);
  auto rec_709   = uniform1.getRGBColor(ColorSpace::eRec709);
  auto rec_2020  = uniform1.getRGBColor(ColorSpace::eRec2020);

  SpectrumIrregular irregular1 = Array<Pair<F32, F32>>{ {360.0f,1.0f},{830.0f,1.0f} };
  irregular1[400.0f] = 0.3f;
  irregular1[500.0f] = 0.4f;
  std::cout << serialize(irregular1).dump() << std::endl;
  auto irregular2 = deserialize<ObjectWrapper>(serialize(irregular1));
  std::cout << serialize(irregular2).dump() << std::endl;

  SpectrumColor color1 = ColorRGB{ 1.0f,0.3f,4.0f };
  {
    auto rgb_color = color1.getRGBColor();
    color1.setRGBColor(*rgb_color, ColorSpace::eAdobeRGB, false);
  }
  std::cout << serialize(color1).dump() << std::endl;
  auto color2 = deserialize<ObjectWrapper>(serialize(color1));
  std::cout << serialize(color2).dump() << std::endl;
  std::cout << serialize(deserialize<Vec2>(Json::parse("{\"type\":\"Vec2\",\"value\":[1, 2]}"))).dump() << std::endl;
  std::cout << serialize(deserialize<Vec2>(Json::parse("[1,2]"))).dump() << std::endl;
  std::cout << serialize(deserialize<Vec3>(Json::parse("{\"type\":\"Vec3\",\"value\":[1,2,3]}"))).dump() << std::endl;
  std::cout << serialize(deserialize<Vec3>(Json::parse("[1,2,3]"))).dump() << std::endl;
  std::cout << serialize(deserialize<Vec4>(Json::parse("{\"type\":\"Vec4\",\"value\":[1,2,3,4]}"))).dump() << std::endl;
  std::cout << serialize(deserialize<Vec4>(Json::parse("[1,2,3,4]"))).dump() << std::endl;
  std::cout << serialize(deserialize<Mat2>(Json::parse("{\"type\":\"Mat2\",\"value\":[[1,0],[0,1]]}"))).dump() << std::endl;
  std::cout << serialize(deserialize<Mat2>(Json::parse("[[1,0],[0,1]]"))).dump() << std::endl;
  std::cout << serialize(deserialize<Mat3>(Json::parse("{\"type\":\"Mat3\",\"value\":[[1,0,0],[0,1,0],[0,0,1]]}"))).dump() << std::endl;
  std::cout << serialize(deserialize<Mat3>(Json::parse("[[1,0,0],[0,1,0],[0,0,1]]"))).dump() << std::endl;
  std::cout << serialize(deserialize<Mat4>(Json::parse("{\"type\":\"Mat4\",\"value\":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}"))).dump() << std::endl;
  std::cout << serialize(deserialize<Mat4>(Json::parse("[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"))).dump() << std::endl;
  std::cout << serialize(deserialize<Quat>(Json::parse("[1,0,0,0]"))).dump() << std::endl;
  std::cout << serialize(deserialize<Quat>(Json::parse("[30,50,0]"))).dump() << std::endl;
  std::cout << serialize(deserialize<Quat>(Json::parse("{\"type\":\"Quat\",\"euler_angles\":[30,50,0]}"))).dump() << std::endl;
  std::cout << serialize(deserialize<Quat>(Json::parse("{\"type\":\"Quat\",\"value\":[1,0,0,0]}"))).dump() << std::endl;
  std::cout << serialize(deserialize<Array<Vec4>>(Json::parse("[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"))).dump() << std::endl;
  std::cout << serialize(deserialize<Transform>(Json::parse("{\"position\":[10,20,30]}"))).dump() << std::endl;
  std::cout << serialize(deserialize<Transform>(Json::parse("{\"scale\":[1,0.5,0.25]}"))).dump() << std::endl;
  std::cout << serialize(deserialize<Transform>(Json::parse("{\"rotation\":[10,20,30]}"))).dump() << std::endl;
  std::cout << serialize(deserialize<Transform>(Json::parse("{\"type\":\"Transform\",\"rotation\":[10,20,30]}"))).dump() << std::endl;
  std::cout << serialize(deserialize<Array<Quat>>(Json::parse("[{\"type\":\"Quat\",\"euler_angles\":[30,50,0]}]"))).dump() << std::endl;
  std::cout << serialize(deserialize<Array<Quat>>(Json::parse("{\"type\":\"Array<Quat>\",\"value\":[{\"type\":\"Quat\",\"euler_angles\":[30,50,0]}]}"))).dump() << std::endl;
  std::cout << serialize(deserialize<Property>(Json::parse("{\"type\":\"Array<Quat>\",\"value\":[{\"type\":\"Quat\",\"euler_angles\":[30,50,0]}]}"))).dump() << std::endl;
}
