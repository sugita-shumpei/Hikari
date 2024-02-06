#include <cstdio>
#include <slang.h>
#include <slang-com-helper.h>
#include <slang-com-ptr.h>
#include "test1_config.h"
int main() {
  Slang::ComPtr<slang::IGlobalSession>             global_session;

  slang::createGlobalSession(global_session.writeRef());
  slang::SessionDesc                                 session_desc;
  slang::TargetDesc                                   target_desc;
  target_desc.format  =                                SLANG_GLSL;
  target_desc.profile =   global_session->findProfile("glsl_450");
  session_desc.targetCount                                    = 1;
  session_desc.targets                             = &target_desc;
  const char* searchPaths[]    = { HK_DATA_ROOT"/../test/test1" };

  session_desc.searchPathCount = 1;
  session_desc.searchPaths     = searchPaths;

  Slang::ComPtr<slang::ISession> session;
  global_session->createSession(session_desc, session.writeRef());

  Slang::ComPtr<slang::IBlob>   diagnostics;
  Slang::ComPtr<slang::IModule> module(session->loadModule("test1", diagnostics.writeRef()));
  if (diagnostics) { fprintf(stderr, "%s\n", (const char*)diagnostics->getBufferPointer()); }
  Slang::ComPtr<slang::IEntryPoint> computeEntryPoint;
  module->findEntryPointByName("computeMain", computeEntryPoint.writeRef());

  slang::ProgramLayout * layout    = module->getLayout();
  slang::TypeReflection* ibinary   = layout->findTypeByName("IBinaryOp");
  slang::TypeReflection* icoord    = layout->findTypeByName("ICoord");
  slang::TypeReflection* add_type  = layout->findTypeByName("Add");
  slang::TypeReflection* mul_type  = layout->findTypeByName("Mul");
  slang::TypeReflection* crd1_type = layout->findTypeByName("Coord1");
  slang::TypeReflection* crd2_type = layout->findTypeByName("Coord2");

  Slang::ComPtr<slang::ITypeConformance> add_comformance;
  Slang::ComPtr<slang::ITypeConformance> mul_comformance;
  Slang::ComPtr<slang::ITypeConformance> crd1_comformance;
  Slang::ComPtr<slang::ITypeConformance> crd2_comformance;
  // 使用する可能性のあるType派生関係はすべて記述しておく
  if (session->createTypeConformanceComponentType(add_type , ibinary,  add_comformance.writeRef(), 0, diagnostics.writeRef())) {
    if (diagnostics) {
      fprintf(stderr, "%s\n", (const char*)diagnostics->getBufferPointer());
    }
    else {
      fprintf(stdout, "compformance success!\n");
    }
  }
  if (session->createTypeConformanceComponentType(mul_type , ibinary,  mul_comformance.writeRef(), 1, diagnostics.writeRef())){
    if (diagnostics) {
      fprintf(stderr, "%s\n", (const char*)diagnostics->getBufferPointer());
    }
    else {
      fprintf(stdout, "compformance success!\n");
    }
  }
  if (session->createTypeConformanceComponentType(crd1_type, icoord , crd1_comformance.writeRef(), 0, diagnostics.writeRef())) {
    if (diagnostics) {
      fprintf(stderr, "%s\n", (const char*)diagnostics->getBufferPointer());
    }
    else {
      fprintf(stdout, "compformance success!\n");
    }
  }
  if (session->createTypeConformanceComponentType(crd2_type, icoord , crd2_comformance.writeRef(), 1, diagnostics.writeRef())) {
    if (diagnostics) {
      fprintf(stderr, "%s\n", (const char*)diagnostics->getBufferPointer());
    }
    else {
      fprintf(stdout, "compformance success!\n");
    }
  }

  slang::IComponentType* components[] = { module,computeEntryPoint,add_comformance,mul_comformance,crd1_comformance,crd2_comformance };
  Slang::ComPtr<slang::IComponentType>   program;
  session->createCompositeComponentType(components, 6, program.writeRef());

  int entryPointIndex = 0; // only one entry point
  int targetIndex     = 0; // only one target
  Slang::ComPtr<slang::IBlob> kernelBlob;
  // 設定したTypeのうち実際に使用していないType派生関係を除去することが可能
  program->getEntryPointCode(entryPointIndex,targetIndex,kernelBlob.writeRef(),diagnostics.writeRef());

  if (kernelBlob) {
    fprintf(stdout, "%s\n", (const char*)kernelBlob->getBufferPointer());
  }
  if (diagnostics) {
    fprintf(stderr, "%s\n", (const char*)diagnostics->getBufferPointer());
  }


  return 0;
}
