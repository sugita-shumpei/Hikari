#pragma once
#include <hikari/core/object.h>
namespace hikari {
  inline namespace core {
    // DataBlob
    // 読み込んだ不定データ の抽象クラス
    // 
    struct DataBlobObject : public Object {
      using base_type = Object;
      static inline constexpr const char* TypeString() { return "DataBlob"; }
      static inline bool Convertible(const Str& type) noexcept {
        if (base_type::Convertible(type)) { return true; }
        return type == TypeString();
      }
      virtual ~DataBlobObject() noexcept {}
      virtual auto getBufferPointer() const -> const Byte* = 0;
      virtual auto getBufferSize()    const -> const U64   = 0;
    };
    struct DataBlob : protected ObjectWrapperImpl<impl::ObjectWrapperHolderSharedRef,DataBlobObject>{
      using impl_type = ObjectWrapperImpl<impl::ObjectWrapperHolderSharedRef, DataBlobObject>;

      using property_ref_type = typename impl_type::property_ref_type;
      using property_type = typename impl_type::property_type;
      using type = typename impl_type::type;

      DataBlob() noexcept : impl_type() {}
      DataBlob(nullptr_t) noexcept : impl_type(nullptr) {}
      DataBlob(const std::shared_ptr<type>& object) noexcept : impl_type(object) {}
      DataBlob(const DataBlob& opb) noexcept : impl_type(opb.getObject()) {}
      DataBlob(DataBlob&& opb) noexcept : impl_type(opb.getObject()) { opb.setObject({}); }

      DataBlob& operator=(const DataBlob& opb) noexcept
      {
        auto old_object = getObject();
        auto new_object = opb.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }
      DataBlob& operator=(DataBlob&& opb) noexcept
      {
        auto old_object = getObject();
        auto new_object = opb.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
          opb.setObject({});
        }
        return *this;
      }
      DataBlob& operator=(const std::shared_ptr<type>& obj) noexcept
      {
        auto old_object = getObject();
        auto& new_object = obj;
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }

      template <typename ObjectWrapperLike, std::enable_if_t<std::is_base_of_v<type, typename ObjectWrapperLike::type>, nullptr_t> = nullptr>
      DataBlob(const ObjectWrapperLike& wrapper) noexcept : impl_type(wrapper.getObject()) {}
      template <typename ObjectWrapperLike, std::enable_if_t<std::is_base_of_v<type, typename ObjectWrapperLike::type>, nullptr_t> = nullptr>
      DataBlob& operator=(const ObjectWrapperLike& wrapper) noexcept
      {
        auto old_object = getObject();
        auto new_object = wrapper.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }

      auto getBufferPointer() const -> const Byte*;
      auto getBufferSize() const -> const U64;

      using impl_type::operator!;
      using impl_type::operator bool;
      using impl_type::operator[];
      using impl_type::isConvertible;
      using impl_type::getName;
      using impl_type::getPropertyNames;
      using impl_type::getObject;
      using impl_type::getPropertyBlock;
      using impl_type::setPropertyBlock;
      using impl_type::getValue;
      using impl_type::hasValue;
      using impl_type::setValue;
    };
    // FileBlob
    // 読み込んだ不定ファイルの抽象クラス
    // 
    struct FileBlobObject : public DataBlobObject {
      using base_type = DataBlobObject;
      static inline constexpr const char* TypeString() { return "FileBlob"; }
      static inline bool Convertible(const Str& type) noexcept {
        if (base_type::Convertible(type)) { return true; }
        return type == TypeString();
      }
      virtual ~FileBlobObject() noexcept {}
      virtual auto getFilePath() const->Str = 0;
    };
    struct FileBlob : protected ObjectWrapperImpl<impl::ObjectWrapperHolderSharedRef, FileBlobObject> {
      using impl_type = ObjectWrapperImpl<impl::ObjectWrapperHolderSharedRef, FileBlobObject>;

      using property_ref_type = typename impl_type::property_ref_type;
      using property_type = typename impl_type::property_type;
      using type = typename impl_type::type;

      FileBlob() noexcept : impl_type() {}
      FileBlob(nullptr_t) noexcept : impl_type(nullptr) {}
      FileBlob(const std::shared_ptr<type>& object) noexcept : impl_type(object) {}
      FileBlob(const FileBlob& opb) noexcept : impl_type(opb.getObject()) {}
      FileBlob(FileBlob&& opb) noexcept : impl_type(opb.getObject()) { opb.setObject({}); }

      FileBlob& operator=(const FileBlob& opb) noexcept
      {
        auto old_object = getObject();
        auto new_object = opb.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }
      FileBlob& operator=(FileBlob&& opb) noexcept
      {
        auto old_object = getObject();
        auto new_object = opb.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
          opb.setObject({});
        }
        return *this;
      }
      FileBlob& operator=(const std::shared_ptr<type>& obj) noexcept
      {
        auto old_object = getObject();
        auto& new_object = obj;
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }

      template <typename ObjectWrapperLike, std::enable_if_t<std::is_base_of_v<type, typename ObjectWrapperLike::type>, nullptr_t> = nullptr>
      FileBlob(const ObjectWrapperLike& wrapper) noexcept : impl_type(wrapper.getObject()) {}
      template <typename ObjectWrapperLike, std::enable_if_t<std::is_base_of_v<type, typename ObjectWrapperLike::type>, nullptr_t> = nullptr>
      FileBlob& operator=(const ObjectWrapperLike& wrapper) noexcept
      {
        auto old_object = getObject();
        auto new_object = wrapper.getObject();
        if (old_object != new_object)
        {
          impl_type::setObject(new_object);
        }
        return *this;
      }

      auto getBufferPointer() const -> const Byte*;
      auto getBufferSize() const -> const U64;
      auto getFilePath() const->Str;

      using impl_type::operator!;
      using impl_type::operator bool;
      using impl_type::operator[];
      using impl_type::isConvertible;
      using impl_type::getName;
      using impl_type::getPropertyNames;
      using impl_type::getObject;
      using impl_type::getPropertyBlock;
      using impl_type::setPropertyBlock;
      using impl_type::getValue;
      using impl_type::hasValue;
      using impl_type::setValue;
    };
  }
}
