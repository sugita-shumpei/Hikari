#define HK_DLL_EXPORT
#define VK_NO_PROTOTYPES 
#include <memory>
#include <algorithm>
#include <string>
#include <vulkan/vulkan.hpp>
#include <hikari/graphics/vulkan/entry.h>
#include <hikari/graphics/entry.h>
#include <hikari/ref_cnt_object.h>

struct HK_DLL HKGraphicsVulkanEntryImpl : public HKGraphicsVulkanEntry, protected HKRefCntObject {
	HKGraphicsVulkanEntryImpl() noexcept : HKGraphicsVulkanEntry{}, HKRefCntObject{}, 
		m_dll{ new vk::DynamicLoader() }, 
		m_vkGetInstanceProcAddr{ nullptr }, 
		m_instanceVersion{ 0u },
		m_instanceExtensionProperties{ },
		m_instanceLayerProperties{ } {
		m_vkGetInstanceProcAddr = m_dll->getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
		if (m_vkGetInstanceProcAddr) {
			PFN_vkEnumerateInstanceVersion             vkEnumerateInstanceVersion             = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(m_vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
			PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(m_vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceExtensionProperties"));
			PFN_vkEnumerateInstanceLayerProperties     vkEnumerateInstanceLayerProperties     = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(m_vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceLayerProperties"));
			vkEnumerateInstanceVersion(&m_instanceVersion);
			{
				HKU32 count = 0;
				if (vkEnumerateInstanceLayerProperties(&count, nullptr) == VK_SUCCESS) {
					m_instanceLayerProperties.resize(count);
					vkEnumerateInstanceLayerProperties(&count, reinterpret_cast<VkLayerProperties*>(m_instanceLayerProperties.data()));
				}
			}

			{
				HKU32 count = 0;
				if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) == VK_SUCCESS) {
					m_instanceExtensionProperties.resize(count);
					vkEnumerateInstanceExtensionProperties(nullptr, &count, reinterpret_cast<VkExtensionProperties*>(m_instanceExtensionProperties.data()));
				}
			}

		}
	}
	virtual ~HKGraphicsVulkanEntryImpl() {}

	HKU32       HK_API addRef()  override
	{
		return HKRefCntObject::addRef();
	}
	HKU32       HK_API release() override
	{
		return HKRefCntObject::release();
	}
	HKBool      HK_API queryInterface(HKUUID iid, void** ppvInterface) override
	{
		if (iid == HK_OBJECT_TYPEID_Unknown || iid == HK_OBJECT_TYPEID_GraphicsEntry || iid == HK_OBJECT_TYPEID_GraphicsVulkanEntry)
		{
			addRef();
			*ppvInterface = this;
			return true;
		}
		return false;
	}
	void        HK_API destroyObject()     override
	{
		m_vkGetInstanceProcAddr = nullptr;
		m_dll = nullptr;
		return;
	}
	virtual Pfn_HKGraphicsVulkan_VoidFunction HK_API getProcAddress(const char* name) const override {
		if (m_vkGetInstanceProcAddr) {
			return m_vkGetInstanceProcAddr(nullptr, name);
		}
		else {
			return nullptr;
		}
	}
	virtual HKU32                             HK_API getVersion() const override {
		return m_instanceVersion;
	}
	virtual HKBool                            HK_API hasExtensionName(HKCStr name) const override {
		return std::find_if(std::begin(m_instanceExtensionProperties), std::end(m_instanceExtensionProperties),
			[name](const vk::ExtensionProperties& prop) {
				return std::string(name) == prop.extensionName.data();
			}) != std::begin(m_instanceExtensionProperties);
	}
	virtual HKCStr                            HK_API getExtensionName(HKU32 idx) const override {
		if (idx < m_instanceExtensionProperties.size()) {
		return m_instanceExtensionProperties.at(idx).extensionName.data();
	}
		else {
			return "";
		}
	}
	virtual HKU32                             HK_API getExtensionCount() const override {
		return m_instanceExtensionProperties.size();
	}
	virtual HKBool                            HK_API hasLayerName(HKCStr name) const override {
		return std::find_if(std::begin(m_instanceLayerProperties), std::end(m_instanceLayerProperties),
			[name](const vk::LayerProperties& prop) {
				return std::string(name) == prop.layerName.data();
			}) != std::begin(m_instanceLayerProperties);

	}
	virtual HKCStr                            HK_API getLayerName(HKU32 idx) const override {
		if (idx < m_instanceLayerProperties.size()) {
			return m_instanceLayerProperties.at(idx).layerName.data();
		}
		else {
			return "";
		}
	}
	virtual HKU32                             HK_API getLayerCount() const override {
		return m_instanceLayerProperties.size();
	}
	std::unique_ptr<vk::DynamicLoader>   m_dll;
	PFN_vkGetInstanceProcAddr            m_vkGetInstanceProcAddr;
	HKU32                                m_instanceVersion;
	std::vector<vk::ExtensionProperties> m_instanceExtensionProperties;
	std::vector<vk::LayerProperties>     m_instanceLayerProperties;
};

HK_EXTERN_C HK_DLL HKGraphicsEntry* HK_API HKGraphicsEntry_create() {
	HKGraphicsVulkanEntryImpl* res = new HKGraphicsVulkanEntryImpl();
	res->addRef();
	return res;
}