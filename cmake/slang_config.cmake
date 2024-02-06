# include(FindPackageHandleStandardArgs)

# find_path(_VCPKG_INSTALL_DIR PATHS ${CMAKE_BINARY_DIR} PATH_SUFFIXES vcpkg_installed)
# find_path(_SLANG_INCLUDE_DIR
# NAMES 
#     slang.h
#     slang-tag-version.h
#     slang-gfx.h
#     slang-com-ptr.h
#     slang-com-helper.h
# PATHS
#     ${_VCPKG_INSTALL_DIR}/x64-windows/include
# )

# find_library( _SLANG_LIBRARIES_DEBUG          NAMES slang.lib         PATHS ${_VCPKG_INSTALL_DIR}/x64-windows/debug PATH_PREFFIXES lib)
# find_file   ( _SLANG_DLL_DEBUG                NAMES slang.dll         PATHS ${_VCPKG_INSTALL_DIR}/x64-windows/debug PATH_PREFFIXES bin)
# find_file   ( _SLANG_LLVM_DLL_DEBUG           NAMES slang-llvm.dll    PATHS ${_VCPKG_INSTALL_DIR}/x64-windows/debug PATH_PREFFIXES bin)
# find_file   ( _SLANG_GLSLANG_DLL_DEBUG        NAMES slang-glslang.dll PATHS ${_VCPKG_INSTALL_DIR}/x64-windows/debug PATH_PREFFIXES bin)
# find_file   ( _SLANG_GFX_DLL_DEBUG            NAMES gfx.dll           PATHS ${_VCPKG_INSTALL_DIR}/x64-windows/debug PATH_PREFFIXES bin)

# find_library( _SLANG_LIBRARIES_RELEASE        NAMES slang.lib         PATHS ${_VCPKG_INSTALL_DIR}/x64-windows PATH_PREFFIXES lib)
# find_file   ( _SLANG_DLL_RELEASE              NAMES slang.dll         PATHS ${_VCPKG_INSTALL_DIR}/x64-windows PATH_PREFFIXES bin)
# find_file   ( _SLANG_LLVM_DLL_RELEASE         NAMES slang-llvm.dll    PATHS ${_VCPKG_INSTALL_DIR}/x64-windows PATH_PREFFIXES bin)
# find_file   ( _SLANG_GLSLANG_DLL_RELEASE      NAMES slang-glslang.dll PATHS ${_VCPKG_INSTALL_DIR}/x64-windows PATH_PREFFIXES bin)
# find_file   ( _SLANG_GFX_DLL_RELEASE          NAMES gfx.dll           PATHS ${_VCPKG_INSTALL_DIR}/x64-windows PATH_PREFFIXES bin)


