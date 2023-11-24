#ifndef HK_CORE_DYNAMIC_LOADER__H
#define HK_CORE_DYNAMIC_LOADER__H
#if !defined(__CUDACC__)

#include "platform.h"
#include "data_type.h"

/* Plugin Loader*/
#if defined(__cplusplus)
#define HK_DYNAMIC_LOADER_GET_PROC_ADDRESS(PLUGIN_LOADER, FUNCTION) \
	PLUGIN_LOADER.getProcAddress<Pfn##_##FUNCTION>(#FUNCTION)
#define HK_DYNAMIC_LOADER_INIT_FUNCTION(PLUGIN_LOADER, FUNCTION) \
	Pfn_##FUNCTION FUNCTION = HK_DYNAMIC_LOADER_GET_PROC_ADDRESS(PLUGIN_LOADER, FUNCTION)
#define HK_DYNAMIC_LOADER_INIT_FUNCTION_TABLE(PLUGIN_LOADER,TABLE,FUNCTION) \
	TABLE.pfn_##FUNCTION = HK_DYNAMIC_LOADER_GET_PROC_ADDRESS(PLUGIN_LOADER, FUNCTION)

struct     HKDynamicLoader {
	HKDynamicLoader() :
#if defined(_WIN32)
		module{ nullptr }
#endif
	{}
	HKDynamicLoader(HKCStr filename):HKDynamicLoader(){
#if defined(_WIN32)
		module = LoadLibraryA(filename);
#endif

	}

	~HKDynamicLoader() noexcept {
		reset();
	}

	HKDynamicLoader(const HKDynamicLoader&) = delete;
	HKDynamicLoader& operator=(const HKDynamicLoader&) = delete;
	HKDynamicLoader(HKDynamicLoader&& plugin) : module{ plugin.module } {
		plugin.module = nullptr;
	}
	HKDynamicLoader& operator=(HKDynamicLoader&& plugin) {
		if (this != &plugin) {
			module = plugin.module;
			plugin.module = nullptr;
		}
		return *this;
	}

	HK_PFN_PROC internal_getProcAddress(HKCStr func_name) {
#if defined(_WIN32)
 		return GetProcAddress(module, func_name);
#endif
		return nullptr;
	}
	template<typename PFN_Type>
	PFN_Type getProcAddress(HKCStr func_name) {
#if defined(_WIN32)
 		return reinterpret_cast<PFN_Type>(GetProcAddress(module, func_name));
#endif
	}


	void reset() {
#if defined(_WIN32)
		if (module) {
			FreeLibrary(module);
			module = nullptr;
		}
#endif
	}

#if defined(_WIN32)
	HMODULE module;
#endif
};

HK_INLINE  HKDynamicLoader HKDynamicLoader_load(HKCStr filename) {
	return HKDynamicLoader(filename);
}
HK_INLINE void             HKDynamicLoader_unload(HKDynamicLoader* pl) {
	if (pl) pl->reset();
}


#else

#if defined(_WIN32)
#define HK_DYNAMIC_LOADER_GET_PROC_ADDRESS(PLUGIN_LOADER, FUNCTION) \
	(Pfn_##FUNCTION)GetProcAddress(PLUGIN_LOADER.module,#FUNCTION)
#else
#define HK_DYNAMIC_LOADER_GET_PROC_ADDRESS(PLUGIN_LOADER, FUNCTION) NULL
#endif
#define HK_DYNAMIC_LOADER_INIT_FUNCTION(PLUGIN_LOADER, FUNCTION) \
	Pfn_##FUNCTION FUNCTION = HK_DYNAMIC_LOADER_GET_PROC_ADDRESS(PLUGIN_LOADER, FUNCTION)

#define HK_DYNAMIC_LOADER_INIT_FUNCTION_TABLE(PLUGIN_LOADER,TABLE,FUNCTION) \
	TABLE.pfn_##FUNCTION = HK_DYNAMIC_LOADER_GET_PROC_ADDRESS(PLUGIN_LOADER, FUNCTION)

typedef struct HKDynamicLoader {
#if defined(_WIN32)
	HMODULE module;
#endif
}HKDynamicLoader;

HK_INLINE HKDynamicLoader HKDynamicLoader_load(HKCStr filename) {
	HKDynamicLoader pl;
#if defined(_WIN32)
	pl.module = LoadLibraryA(filename);
#endif
	return pl;
}
HK_INLINE void           HKDynamicLoader_unload(HKDynamicLoader* pl) {
	if (pl) {
#if defined(_WIN32)
		if (pl->module) {
			FreeLibrary(pl->module);
			pl->module = NULL;
		}
#endif
	}
}

#endif
HK_NAMESPACE_TYPE_ALIAS(DynamicLoader);
#endif
#endif
