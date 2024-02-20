#include <hikari/framework/core/object.h>
using namespace hikari;
int main() {
  hikari::core::ObjectWrapper o;
  o["tekitou"] = 0;
  o["tekitou"] = 0ull;
  o["tekitou"] = "tekitou";
}
