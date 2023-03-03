add_library(Hikari-Base STATIC
Include/Hikari/Math.h
Include/Hikari/Preprocessor.h
Include/Hikari/SpinLock.h
Include/Hikari/TypeTraits.h
Include/Hikari/Math/Vec.h
Include/Hikari/Math/Mat.h
Include/Hikari/Math/Qua.h
Include/Hikari/Math/Table.h
Include/Hikari/Math/Utils.h

Src/SpinLock.cpp 
)
target_include_directories(Hikari-Base PUBLIC Include)