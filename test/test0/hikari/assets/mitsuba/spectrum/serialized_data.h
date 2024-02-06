#pragma once
#include <memory>
#include <string>
#include <vector>
namespace hikari {
  namespace assets {
    namespace mitsuba {
      namespace spectrum {
        struct Serialized {
          std::vector<float> wavelengths;
          std::vector<float> weights;
        };

        struct SerializedImporter {
          static auto create(const std::string& filename) -> std::shared_ptr<SerializedImporter>;
          virtual ~SerializedImporter() noexcept {}

          auto load() const -> std::shared_ptr<Serialized>;
        private:
          SerializedImporter(const std::string& filename) :m_filename{ filename } {}
        private:
          std::string m_filename;
        };
      }
    }
  }
}
