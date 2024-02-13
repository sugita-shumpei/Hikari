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
  std::cout << serialize(Property(1.0f).getValueTo<F32>()).dump() << std::endl;
  std::cout << serialize(Property(2.0f).getValueTo<F64>()).dump() << std::endl;
  std::cout << serialize(Property(3).getValueTo<I8 >()).dump() << std::endl;
  std::cout << serialize(Property(4).getValueTo<I16>()).dump() << std::endl;
  std::cout << serialize(Property(5).getValueTo<I32>()).dump() << std::endl;
  std::cout << serialize(Property(6).getValueTo<I64>()).dump() << std::endl;
  std::cout << serialize(Property(7).getValueTo<U8 >()).dump() << std::endl;
  std::cout << serialize(Property(8).getValueTo<U16>()).dump() << std::endl;
  std::cout << serialize(Property(9).getValueTo<U32>()).dump() << std::endl;
  std::cout << serialize(Property(9).getValueTo<Vec2>()).dump() << std::endl;
  std::cout << serialize(Property(9).getValueTo<Vec4>()).dump() << std::endl;
  std::cout << serialize(Property(0xFFFFFFFFFFFFFFFF).getValueTo<U64>()).dump() << std::endl;
  std::cout << serialize(Property(0xFFFFFFFF).getValueTo<U32>()).dump() << std::endl;
  std::cout << serialize(Property(0xFFFF ).getValueTo<U16>()).dump() << std::endl;
  std::cout << serialize(Property("True" ).getValueTo<Bool>()).dump() << std::endl;
  std::cout << serialize(Property("False").getValueTo<Bool>()).dump() << std::endl;
  std::cout << serialize(Property("true" ).getValueTo<Bool>()).dump() << std::endl;
  std::cout << serialize(Property("false").getValueTo<Bool>()).dump() << std::endl;
  std::cout << serialize(Property("false").getValueTo<Array<F32>>()).dump() << std::endl;
  std::cout << serialize(Property(Array<I32>{1,2}).getValueTo<Array<F32>>()).dump() << std::endl;
  std::cout << serialize(Property(1.0f).getValueTo<Transform>()).dump() << std::endl;
  std::cout << serialize(Property(1).getValueTo<Transform>()).dump() << std::endl;
  std::cout << serialize(Property(Mat3(3.0f)).getValueTo<Transform>()).dump() << std::endl;
  return 0;
}
