#include <Test0.h>

HK_INTERFACE_DECLARE(Vector2, HKInterface, "03CCE43E-D743-40EF-B14B-C57383F82164")
{
public:
	using this_type = Vector2;

	Vector2() noexcept :HKInterface(),
		m_X{ 0 }, m_Y{ 0 },
		HK_INTERFACE_PROPERTY_CUSTOM_INIT(X),
		HK_INTERFACE_PROPERTY_CUSTOM_INIT(Y)
	{}
	virtual ~Vector2()noexcept {}

	auto GetX()const -> int { return m_X; }
	void SetX(int x) { m_X = x; Show(); }
	auto GetY()const -> int { return m_Y; }
	void SetY(int x) { m_Y = x; Show(); }

	HK_INTERFACE_METHOD_DECL_QUERY_INTERFACE();
	HK_INTERFACE_PROPERTY_CUSTOM_DECL(int, X) ;
	HK_INTERFACE_PROPERTY_CUSTOM_DECL(int, Y) ;
private:
	void Show()const {
		std::cout << "[" << m_X << "," << m_Y << "]" << std::endl;
	}
private:
	int m_X;
	int m_Y;
};
HK_INTERFACE_DECLARE(Vector3, HKInterface, "6A8180D8-5B53-43D7-99A3-7E772AA56424")
{
public:
	using this_type = Vector3;

	Vector3() noexcept :HKInterface(),
		m_X{ 0 }, m_Y{ 0 }, m_Z{ 0 },
		HK_INTERFACE_PROPERTY_CUSTOM_INIT(X),
		HK_INTERFACE_PROPERTY_CUSTOM_INIT(Y),
		HK_INTERFACE_PROPERTY_CUSTOM_INIT(Z)
	{}

	virtual ~Vector3()noexcept {}

	auto GetX()const -> int { return m_X; }
	void SetX(int x) { m_X = x; Show(); }
	auto GetY()const -> int { return m_Y; }
	void SetY(int x) { m_Y = x; Show(); }
	auto GetZ()const -> int { return m_Z; }
	void SetZ(int x) { m_Z = x; Show(); }

	HK_INTERFACE_METHOD_DECL_QUERY_INTERFACE();
	HK_INTERFACE_PROPERTY_CUSTOM_DECL(int, X);
	HK_INTERFACE_PROPERTY_CUSTOM_DECL(int, Y);
	HK_INTERFACE_PROPERTY_CUSTOM_DECL(int, Z);
private:
	void Show()const {
		std::cout << "[" << m_X << "," << m_Y << "," << m_Z << "]" << std::endl;
	}
private:
	int m_X;
	int m_Y;
	int m_Z;
};
constexpr auto f() -> std::array<char,4> {
	std::string str("s10");
	std::array<char, 4> res = {};
	for (size_t i = 0; i < 3; ++i) {
		res[i] = str[i];
	}
	{
		constexpr auto arr = HK::SplitStrStorage<6, 6>(HK::ToArray("Hello, World"));
		constexpr auto str0 = std::get<0>(arr);
		constexpr auto str1 = std::get<1>(arr);
	}
	return res;
}
int main(int argc, const char* argv[])
{
	HK_CXX17_CONSTEXPR int v = 0;
	static_assert(v == 0);
	{
		auto ptr1 = HKInterfacePtr(new Vector2());
		auto ptr2 = HKInterfacePtr(new Vector3());
		auto ptr3 = HKInterfacePtr(new Vector3());
		// QUERY
		std::cout << ptr1->UseCount() << std::endl;
		std::cout << ptr2->UseCount() << std::endl;
		std::cout << ptr3->UseCount() << std::endl;
		{
			auto v2_1 = HKInterfacePtr<Vector2>();
			auto v3_1 = HKInterfacePtr<Vector3>();
			auto v3_2 = HKInterfacePtr<Vector3>();

			if (ptr1.QueryInterface(v2_1)) {
				std::cout << ptr1->UseCount() << std::endl;
				v2_1->X = 1; v2_1->Y = 2;
			}
			if (ptr2.QueryInterface(v3_1)) {
				std::cout << ptr2->UseCount() << std::endl;
			}
			if (ptr3.QueryInterface(v3_2)) {
				std::cout << ptr3->UseCount() << std::endl;
			}
		}
		std::cout << ptr1->UseCount() << std::endl;
		std::cout << ptr2->UseCount() << std::endl;
		std::cout << ptr3->UseCount() << std::endl;
	}
	{ 
		constexpr HKUUID uuid1 = // {03CCE43E-D743-40EF-B14B-C57383F82164}
		{ 0x3cce43e, 0xd743, 0x40ef, { 0xb1, 0x4b, 0xc5, 0x73, 0x83, 0xf8, 0x21, 0x64 } };

		constexpr HKUUID uuid2 = HKUUID::FromCStr("03CCE43E-D743-40EF-B14B-C57383F82164");
		static_assert(uuid1 == uuid2);

		std::unordered_set<HK::UUID> sets = {
			 HKUUID::FromCStr("03CCE43E-D743-40EF-B14B-C57383F82164"),
			 HKUUID::FromCStr("03CCE43E-D743-40EF-B14B-C57383F82164")
		};
		assert(sets.size() == 1);
	}
	return 0;
}