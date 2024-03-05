#pragma once
#include <optional>
#include <vector>
#include <ranges>
#include <string_view>
#include <array>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_extension_inspection.hpp>
namespace hikari {
  namespace core {
    template<typename VulkanStructT>
    concept VulkanInStructure = requires (VulkanStructT s) {
      {s.sType}-> std::convertible_to<::vk::StructureType>;
      {s.pNext}-> std::convertible_to<const void*>;
    };
    template<typename VulkanStructT>
    concept VulkanOutStructure = requires (VulkanStructT s) {
      {s.sType}-> std::convertible_to<::vk::StructureType>;
      {s.pNext}-> std::convertible_to<void*>;
    };

    template<typename VulkanStructT>
    concept VulkanStructure = VulkanInStructure<VulkanStructT> || VulkanOutStructure<VulkanStructT>;

    struct VulkanPNextChain;
    struct VulkanPNextChainStructInfo {
      vk::StructureType sType; vk::DeviceSize sSize; vk::DeviceSize sOffset;
    };
    struct VulkanPNextChain;
    struct VulkanPNextChainBuilder {
       VulkanPNextChainBuilder() noexcept {}
       VulkanPNextChainBuilder(VulkanPNextChain&& chain) noexcept;
      ~VulkanPNextChainBuilder() noexcept {}
      auto getInfos() const noexcept -> const std::vector<VulkanPNextChainStructInfo>& { return infos; }
      auto getCount() const noexcept -> size_t { return infos.size(); }
      auto getSize() const noexcept -> size_t {
        return bytes.size();
      }
      void setChain(const VulkanPNextChain& chain) noexcept;
      template<VulkanStructure VulkanStructT>
      void addStruct(VulkanStructT  vulkan_struct) noexcept {
        vk::DeviceSize sOffset = 0u;
        if (!infos.empty()) { sOffset = getSize(); }
        infos.emplace_back(VulkanStructT::structureType, sizeof(VulkanStructT), sOffset);
        vulkan_struct.pNext = nullptr;
        auto byte_storage = std::bit_cast<std::array<std::byte, sizeof(VulkanStructT)>>(vulkan_struct);
        if (bytes.size() + sizeof(byte_storage) > bytes.capacity()) {
          bytes.reserve(bytes.size() + sizeof(byte_storage));
        }
        std::ranges::copy(byte_storage, std::back_inserter(bytes));
      }
      template<VulkanStructure VulkanStructT>
      void addStruct() noexcept {
        addStruct(VulkanStructT());
      }
      template<VulkanStructure... VulkanStructs>
      void addStructs() {
        using Swallow = int[];
        (void)Swallow {
          (addStruct<VulkanStructs>(), 0)...
        };
      }
      template<VulkanStructure VulkanStructT>
      bool getStruct(VulkanStructT& vulkan_struct) const noexcept  {
        auto iter = std::ranges::find_if(infos, [](const auto& info) { return info.sType == VulkanStructT::structureType; });
        if (iter == std::end(infos)) { return false; }
        // memcpy
        auto beg_struct   = std::begin(bytes) + iter->sOffset;
        auto byte_storage = std::array<std::byte, sizeof(VulkanStructT)>{};
        std::ranges::copy(beg_struct, beg_struct + sizeof(VulkanStructT), std::begin(byte_storage));
        return true;
      }
      template<VulkanStructure VulkanStructT>
      auto getStruct() const noexcept -> std::optional<VulkanStructT> {
        VulkanStructT vulkan_struct = {};
        if (getStruct(vulkan_struct)) { return std::optional(vulkan_struct); }
        else { return std::nullopt; }
      }
      template<VulkanStructure VulkanStructT>
      void setStruct(VulkanStructT vulkan_struct)noexcept {
        vulkan_struct.pNext = nullptr;
        auto iter = std::ranges::find_if(infos, [](const auto& info) { return info.sType == VulkanStructT::structureType; });
        if (iter == std::end(infos)) {
          addStruct<VulkanStructT>(vulkan_struct);
        }
        else {
          auto byte_storage = std::bit_cast<std::array<std::byte, sizeof(VulkanStructT)>>(vulkan_struct);
          std::ranges::copy(byte_storage, bytes.begin() + iter->sOffset);
        }
      }
      template<VulkanStructure VulkanStructT>
      void popStruct() {
        popStruct(VulkanStructT::structureType);
      }
      void popStruct(vk::StructureType sType) {
        auto iter = std::ranges::find_if(infos, [sType](const auto& info) { return info.sType == sType; });
        if (iter == std::end(infos)) {
          return;
        }
        size_t cur_offset = 0;
        for (auto& info : std::span(iter + 1, infos.end())) {
          info.sOffset = cur_offset;
          cur_offset += info.sSize;
        };
        auto beg = bytes.begin();
        bytes.erase(beg + iter->sOffset, beg + iter->sOffset + iter->sSize);
        infos.erase(iter);
      }
      template<VulkanStructure... VulkanStructs>
      void popStructs() {
        using Swallow = int[];
        (void)Swallow {
          (popStruct(VulkanStructs::structureType),0)...
        };
      }
      template<VulkanStructure VulkanStructT>
      auto getStructTo() const -> VulkanStructT {
        VulkanStructT vulkan_struct = {};
        if (getStruct(vulkan_struct)) { return vulkan_struct; }
        else { throw std::invalid_argument("Failed To Find Type!"); }
      }
      template<VulkanStructure VulkanStructT>
      bool hasStruct() const noexcept {
        return hasStruct(VulkanStructT::structureType);
      }
      bool hasStruct(vk::StructureType sType) const noexcept {
        return std::ranges::find_if(infos, [sType](const auto& info) { return info.sType == sType; }) != std::end(infos);
      }
      template<VulkanStructure VulkanStructT>
      bool containBits() const noexcept {
        return containBits(VulkanStructT::structureType);
      }
      bool containBits(vk::StructureType sType) const noexcept {
        auto iter = std::ranges::find_if(infos, [sType](const auto& info) { return info.sType == sType; });
        if (iter == std::end(infos)) { return false; }
        auto tmp_storage = std::unique_ptr<std::byte[]>(new std::byte[iter->sSize]);
        auto beg = bytes.data() + iter->sOffset;
        std::memcpy(tmp_storage.get(), beg, iter->sSize);
        std::memset(tmp_storage.get() + offsetof(vk::BaseInStructure, sType), 0, sizeof(vk::StructureType));
        std::memset(tmp_storage.get() + offsetof(vk::BaseInStructure, pNext), 0, sizeof(uintptr_t));
        for (size_t i = 0; i < iter->sSize; ++i) {
          if (tmp_storage[i] != std::byte(0)) { return true; }
        }
        return false;
      }
      inline auto build() noexcept -> hikari::core::VulkanPNextChain;
    private:
      void link() noexcept {
        if (infos.empty()) { return; }
        for (size_t i = 1; i < infos.size(); ++i) {
          auto cur_off = infos[i - 1].sOffset;
          auto nxt_off = infos[i + 0].sOffset;
          auto offset = cur_off + offsetof(vk::BaseInStructure, pNext);
          void* p_next = bytes.data() + nxt_off;
          std::memcpy(bytes.data() + offset, &p_next, sizeof(void*));
        }
      }
      void unlink() noexcept {
        if (infos.empty()) { return; }
        for (size_t i = 1; i < infos.size(); ++i) {
          auto cur_off = infos[i - 1].sOffset;
          auto nxt_off = infos[i + 0].sOffset;
          auto offset = cur_off + offsetof(vk::BaseInStructure, pNext);
          void* p_next = nullptr;
          std::memcpy(bytes.data() + offset, &p_next, sizeof(void*));
        }
      }
      void setPStruct(const void* p_data, vk::StructureType sType, size_t sSize)noexcept {
        auto iter = std::ranges::find_if(infos, [sType](const auto& info) { return info.sType == sType; });
        std::unique_ptr<std::byte[]> ptr(new std::byte[sSize]);
        std::memcpy(ptr.get(), p_data, sSize);
        void* null_p = nullptr;
        std::memcpy(ptr.get()+offsetof(vk::BaseInStructure,pNext), &null_p, sizeof(uintptr_t));
        if (iter == std::end(infos)) {
          auto sOffset = getSize();
          infos.emplace_back(sType, sSize, sOffset);
          if (bytes.size() +sSize> bytes.capacity()) {
            bytes.reserve(bytes.size() + sSize);
          }
          std::copy(ptr.get(),ptr.get() + sSize, std::back_inserter(bytes));
        }
        else {
          std::copy(ptr.get(), ptr.get() + sSize, bytes.begin()+iter->sOffset);
        }
      }
    private:
      friend struct VulkanPNextChain;
      using StructInfo = VulkanPNextChainStructInfo;
      std::vector<StructInfo> infos = {};
      std::vector<std::byte>  bytes = {};
    };
    struct VulkanPNextChain {
      VulkanPNextChain() noexcept {}
      VulkanPNextChain(const VulkanPNextChainBuilder& builder) noexcept
        :infos{ builder.infos }, bytes{ builder.bytes } {
        link();
      }
      VulkanPNextChain(VulkanPNextChainBuilder&& builder) noexcept
        :infos{ std::move(builder.infos)}, bytes{std::move(builder.bytes)} {
        link();
      }
      VulkanPNextChain(const VulkanPNextChain& lhs) noexcept
        :infos{ lhs.infos }, bytes{ lhs.bytes } {
        link();
      }
      VulkanPNextChain(VulkanPNextChain&& rhs) noexcept
        :infos{ std::move(rhs.infos) }, bytes{ std::move(rhs.bytes) }
      {}
      VulkanPNextChain& operator=(const VulkanPNextChain & lhs) noexcept{
        if (this != &lhs) {
          infos = lhs.infos;
          bytes = lhs.bytes;
          link();
        }
        return *this;
      }
      VulkanPNextChain& operator=(VulkanPNextChain&& rhs) noexcept
      {
        if (this != &rhs) {
          infos = std::move(rhs.infos);
          bytes = std::move(rhs.bytes);
        }
        return *this;
      }
      auto getInfos() const noexcept -> const std::vector<VulkanPNextChainStructInfo>& { return infos; }
      auto getCount() const noexcept -> size_t { return infos.size(); }
      auto getSize()  const noexcept -> size_t {
        return bytes.size();
      }
      auto getPHead() const noexcept -> const void* { return bytes.data(); }
      auto getPHead() noexcept -> void* { return bytes.data(); }
      template<VulkanStructure VulkanStructT>
      bool getStruct(VulkanStructT& vulkan_struct) const noexcept {
        auto iter = std::ranges::find_if(infos, [](const auto& info) { return info.sType == VulkanStructT::structureType; });
        if (iter == std::end(infos)) { return false; }
        // memcpy
        auto beg_struct = std::begin(bytes) + iter->sOffset;
        auto byte_storage = std::array<std::byte, sizeof(VulkanStructT)>{};
        std::ranges::copy(beg_struct, beg_struct + sizeof(VulkanStructT), std::begin(byte_storage));
        vulkan_struct = std::bit_cast<VulkanStructT>(byte_storage);
        return true;
      }
      template<VulkanStructure VulkanStructT>
      auto getStruct() const noexcept -> std::optional<VulkanStructT> {
        VulkanStructT vulkan_struct = {};
        if (getStruct(vulkan_struct)) { return std::optional(vulkan_struct); }
        else { return std::nullopt; }
      }
      template<VulkanStructure VulkanStructT>
      auto getStructTo() const -> VulkanStructT {
        VulkanStructT vulkan_struct = {};
        if (getStruct(vulkan_struct)) { return vulkan_struct; }
        else { throw std::invalid_argument("Failed To Find Type!"); }
      }
      template<VulkanStructure VulkanStructT>
      bool hasStruct() const noexcept {
        return hasStruct(VulkanStructT::structureType);
      }
      bool hasStruct(vk::StructureType sType) const noexcept {
        return std::ranges::find_if(infos, [sType](const auto& info) { return info.sType == sType; }) != std::end(infos);
      }
    private:
      friend struct VulkanPNextChainBuilder;
      void link() noexcept  {
        if (infos.empty()) { return; }
        for (size_t i = 1; i < infos.size(); ++i) {
          auto cur_off = infos[i - 1].sOffset;
          auto nxt_off = infos[i + 0].sOffset;
          auto offset = cur_off + offsetof(vk::BaseInStructure,pNext);
          void* p_next = bytes.data() + nxt_off;
          std::memcpy(bytes.data() + offset, &p_next, sizeof(void*));
        }
      }
      void unlink() noexcept {
        if (infos.empty()) { return; }
        for (size_t i = 1; i < infos.size(); ++i) {
          auto cur_off = infos[i - 1].sOffset;
          auto nxt_off = infos[i + 0].sOffset;
          auto offset = cur_off + offsetof(vk::BaseInStructure, pNext);
          void* p_next = nullptr;
          std::memcpy(bytes.data() + offset, &p_next, sizeof(void*));
        }
      }

    private:
      using StructInfo = VulkanPNextChainStructInfo;
      std::vector<StructInfo> infos = {};
      std::vector<std::byte>  bytes = {};
    };
    inline VulkanPNextChainBuilder::VulkanPNextChainBuilder(VulkanPNextChain&& chain) noexcept
      :infos{ std::move(chain.infos) }, bytes{ std::move(chain.bytes) } {
      unlink();
    }
    inline void VulkanPNextChainBuilder::setChain(const VulkanPNextChain& chain) noexcept
    {
      for (auto& info : chain.infos) {
        setPStruct(chain.bytes.data() + info.sOffset, info.sType, info.sSize);
      }
    }
    auto VulkanPNextChainBuilder::build() noexcept -> hikari::core::VulkanPNextChain { return VulkanPNextChain(std::move(*this)); }

    namespace vk_utils {
      inline auto findQueueFamily(const std::vector<::vk::QueueFamilyProperties>& props, vk::QueueFlags required_bits, vk::QueueFlags avoid_bits)
      {
        return std::ranges::find_if(props, [required_bits, avoid_bits](const auto& p) {
          if ((p.queueFlags & required_bits) == required_bits) { return false; }
          if ((p.queueFlags &    avoid_bits)) { return false; }
          return true;
        });
      }

      inline auto toNames(
        const std::vector<::vk::ExtensionProperties>& extension_properties
      ) -> std::vector<const char*> {
        auto ranges = extension_properties | std::ranges::views::transform([](const ::vk::ExtensionProperties& p) { return p.extensionName.data(); });
        return std::vector<const char*>(std::begin(ranges), std::end(ranges));
      }

      inline auto toNames(
        const std::vector<::vk::LayerProperties>& layer_properties
      ) -> std::vector<const char*> {
        auto ranges = layer_properties | std::ranges::views::transform([](const ::vk::LayerProperties& p) { return p.layerName.data(); });
        return std::vector<const char*>(std::begin(ranges), std::end(ranges));
      }

      inline auto findName(const std::vector<::vk::ExtensionProperties>& prop, const char* name) -> decltype(prop.begin()) {
        return std::ranges::find_if(prop, [name](const auto& p) { return p.extensionName.data() == std::string_view(name); });
      }
      inline auto findName(const std::vector<::vk::LayerProperties>&     prop, const char* name) -> decltype(prop.begin()) {
        return std::ranges::find_if(prop, [name](const auto& p) { return p.layerName.data() == std::string_view(name); });
      } 
    }

  }
}
