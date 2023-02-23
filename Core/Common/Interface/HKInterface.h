#ifndef HK_CORE_COMMON_INTERFACE_H
#define HK_CORE_COMMON_INTERFACE_H
#include <atomic>
#include <string>
#include <array>
#include <iostream>
#include <type_traits>
#include <HKUUID.h>
#include <boost/smart_ptr/intrusive_ptr.hpp>
#define HK_INTERFACE_PROPERTY_DEFAULT_DECL(TYPE,NAME) \
		TYPE NAME; \
		auto Get##NAME()const -> TYPE { return NAME; } \
		void Set##NAME(TYPE v){ NAME = v; } 

#define HK_INTERFACE_PROPERTY_DEFAULT_INIT(NAME) NAME{}
#define HK_INTERFACE_PROPERTY_CUSTOM_DECL(TYPE,NAME) HKInterfaceProperty<this_type,TYPE,&this_type::Get##NAME,&this_type::Set##NAME> NAME
#define HK_INTERFACE_PROPERTY_CUSTOM_INIT(NAME) NAME{*this}

#define HK_INTERFACE_METHOD_DECL_QUERY_INTERFACE() \
	virtual bool QueryInterface(const HKInterfaceID& id, void** ppv) override \
	{ *ppv = nullptr; for(auto& elm: HKInterfaceTraits<this_type>::this_id_list){ if(id == elm){ *ppv = static_cast<void*>(this); AddRef();return true; }} return false; }

#define HK_INTERFACE_DECLARE(TYPE, BASE, ID) \
struct TYPE; \
static inline constexpr HKInterfaceID HKInterfaceID_##TYPE = HKUUID::FromCStr(ID); \
template<> struct HKInterfaceTraits<TYPE> { \
using base_type = BASE; \
static inline constexpr auto base_id = HKInterfaceTraits<BASE>::this_id; \
static inline constexpr auto base_id_list = HKInterfaceTraits<BASE>::this_id_list; \
using this_type = TYPE; \
static inline constexpr auto this_id = HKInterfaceID_##TYPE; \
static inline constexpr auto this_id_list = HKInterfaceUtils::AppendID(base_id_list,this_id); \
}; \
struct TYPE:public BASE 

using HKInterfaceID = HKUUID;

template<typename HKInterfaceT, typename VariableT, VariableT(HKInterfaceT::* Getter)()const, void(HKInterfaceT::* Setter)(VariableT)>
struct HKInterfaceProperty
{
	HKInterfaceProperty(HKInterfaceT& interface)
		:m_HKInterface{ interface } {}

	HKInterfaceProperty(VariableT defVal) = delete;

	HKInterfaceProperty(const HKInterfaceProperty&) = delete;
	HKInterfaceProperty(HKInterfaceProperty&&) = delete;

	HKInterfaceProperty& operator=(const HKInterfaceProperty& property) = delete;
	HKInterfaceProperty& operator=(HKInterfaceProperty&&) = delete;

	HKInterfaceProperty& operator=(VariableT v)
	{
		Set(v);
		return *this;
	}

	operator VariableT()const { return Get(); }

	auto Get()const -> VariableT { return (m_HKInterface.*Getter)(); }
	void Set(VariableT v) { return (m_HKInterface.*Setter)(v); }

	HKInterfaceT& m_HKInterface;
};

template<typename HKInterfaceT>
struct HKInterfaceTraits;

static inline constexpr auto HKInterfaceID_Unknown = HKInterfaceID{};

struct HKInterface;
template<>
struct HKInterfaceTraits<HKInterface>
{
	using this_type = HKInterface;
	static inline constexpr auto this_id = HKInterfaceID_Unknown;
	static inline constexpr auto this_id_list = std::array< HKInterfaceID, 1> { this_id };
};

struct HKInterface
{
	using this_type = HKInterface;

	HKInterface()noexcept :
		m_Counter{ 1 },
		HK_INTERFACE_PROPERTY_DEFAULT_INIT(Name)
	{}
	HKInterface(std::string name) :
		m_Counter{ 1 },
		HK_INTERFACE_PROPERTY_DEFAULT_INIT(Name)
	{
		Name = name;
	}

	virtual ~HKInterface() noexcept
	{}

	virtual bool QueryInterface(const HKInterfaceID& id, void** ppv) = 0;
	template<typename Derived>
	bool QueryInterface(Derived** ppv)
	{
		*ppv = nullptr;
		void* pTmp = nullptr;
		if (QueryInterface(HKInterfaceTraits<Derived>::this_id, &pTmp))
		{
			*ppv = static_cast<Derived*>(pTmp);
			return true;
		}
		return false;
	}

	auto UseCount() -> unsigned int
	{
		return m_Counter.load();
	}
	auto AddRef() -> unsigned int
	{
		return m_Counter.fetch_add(1) + 1;
	}
	auto Release() -> unsigned int
	{
		auto cnt = m_Counter.fetch_sub(1) - 1;
		if (cnt == 0) {
			delete this;
		}
		return cnt;
	}

	HK_INTERFACE_PROPERTY_DEFAULT_DECL(std::string, Name);
private:
	std::atomic_uint m_Counter;
};

struct HKInterfaceUtils
{
	template<typename T, size_t N>
	static inline constexpr auto AppendID(const std::array<T, N>& arr, const T& val) -> std::array<T, N + 1> { std::array<T, N + 1> res = {};  for (auto i = 0; i < N; ++i) { res[i] = arr[i]; } res[N] = val; return res; }
};

template<typename T, bool Cond = std::is_base_of_v<HKInterface, T>>
void intrusive_ptr_add_ref(T* ptr)
{
	ptr->AddRef();
}
template<typename T, bool Cond = std::is_base_of_v<HKInterface, T>>
void intrusive_ptr_release(T* ptr)
{
	ptr->Release();
}

template<typename T, bool Cond = std::is_base_of_v<HKInterface, T>>
struct HKInterfacePtr {
	HKInterfacePtr() noexcept :m_Handle{} {}
	HKInterfacePtr(T* ptr, bool add_ref = false) noexcept :m_Handle{ ptr, add_ref } {}

	template<typename S, bool Cond2 = std::is_base_of_v<HKInterface, S>>
	bool QueryInterface(HKInterfacePtr<S>& ptr)
	{
		S* pTmp = nullptr;
		if (m_Handle->HKInterface::QueryInterface(&pTmp)) {
			ptr = pTmp;
			return true;
		}
		return false;
	}

	void Reset() { m_Handle.reset(); }

	void Reset(T* rhs) { m_Handle.reset(rhs); }

	void Reset(T* rhs, bool add_ref) { m_Handle.reset(rhs, add_ref); }

	T* Get() const { return m_Handle.get(); }

	T* Detach() noexcept
	{
		return m_Handle.detach();
	}

	T& operator*() const noexcept
	{
		return *m_Handle;
	}

	T* operator->() const noexcept
	{
		return m_Handle.operator->();
	}

	void Swap(HKInterfacePtr<T>&& rhs)
	{
		m_Handle.swap(rhs.m_Handle);
	}
private:
	boost::intrusive_ptr<T> m_Handle;
};

namespace HK
{
	using InterfaceID = HKInterfaceID;
	using Interface = HKInterface;

	template<typename T>
	using InterfaceTraits = HKInterfaceTraits<T>;

	template<typename HKInterfaceT, typename VariableT, VariableT(HKInterfaceT::* Getter)()const, void(HKInterfaceT::* Setter)(VariableT)>
	using InterfaceProperty = HKInterfaceProperty<HKInterfaceT, VariableT, Getter, Setter>;

	template<typename T>
	using InterfacePtr = HKInterfacePtr<T>;
}
#endif
