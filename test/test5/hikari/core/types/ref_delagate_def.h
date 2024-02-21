#pragma once

#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD(NUM_ARG, NUM_RET, CST_TF, MODE) \
HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_IMPL(NUM_ARG, NUM_RET, CST_TF, MODE)
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_IMPL(NUM_ARG, NUM_RET, CST_TF, MODE) \
HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_##NUM_ARG##_RET_##NUM_RET##_CST_##CST_TF##_WITH_##MODE

#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_0_RET_0_CST_0_WITH_RES(OWNER,RET, METHOD, SUC, FAL) \
  inline RET METHOD()       { auto ref = OWNER; if (!ref){ return FAL;} ref->METHOD(); return SUC; }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_0_RET_0_CST_1_WITH_RES(OWNER,RET, METHOD, SUC, FAL) \
  inline RET METHOD() const { auto ref = OWNER; if (!ref){ return FAL;} ref->METHOD(); return SUC; }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_0_RET_1_CST_0_WITH_DEF(OWNER,RET, METHOD, DEF) \
  inline RET METHOD()       { auto ref = OWNER; if (!ref){ return DEF;} return ref->METHOD(); }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_0_RET_1_CST_1_WITH_DEF(OWNER,RET, METHOD, DEF) \
  inline RET METHOD() const { auto ref = OWNER; if (!ref){ return DEF;} return ref->METHOD(); }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_0_RET_1_CST_0_WITH_OPT(OWNER,RET, METHOD, DEF) \
  inline RET METHOD()       { auto ref = OWNER; if (!ref){ return std::nullopt;} return ref->METHOD(); }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_0_RET_1_CST_1_WITH_OPT(OWNER,RET, METHOD, DEF) \
  inline RET METHOD() const { auto ref = OWNER; if (!ref){ return std::nullopt;} return ref->METHOD(); }

#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_1_RET_0_CST_0_WITH_RES(OWNER,RET, METHOD, ARGT1, ARGV1, SUC, FAL) \
  inline RET METHOD(ARGT1 ARGV1)       { auto ref = OWNER; if (!ref){ return FAL;} ref->METHOD(ARGV1); return SUC; }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_1_RET_0_CST_1_WITH_RES(OWNER,RET, METHOD, ARGT1, ARGV1, SUC, FAL) \
  inline RET METHOD(ARGT1 ARGV1) const { auto ref = OWNER; if (!ref){ return FAL;} ref->METHOD(ARGV1); return SUC; }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_1_RET_1_CST_0_WITH_DEF(OWNER,RET, METHOD, ARGT1, ARGV1, DEF) \
  inline RET METHOD(ARGT1 ARGV1)       { auto ref = OWNER; if (!ref){ return DEF;} return ref->METHOD(ARGV1); }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_1_RET_1_CST_1_WITH_DEF(OWNER,RET, METHOD, ARGT1, ARGV1, DEF) \
  inline RET METHOD(ARGT1 ARGV1) const { auto ref = OWNER; if (!ref){ return DEF;} return ref->METHOD(ARGV1); }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_1_RET_1_CST_0_WITH_OPT(OWNER,RET, METHOD, ARGT1, ARGV1, DEF) \
  inline RET METHOD(ARGT1 ARGV1)       { auto ref = OWNER; if (!ref){ return std::nullopt;} return ref->METHOD(ARGV1); }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_1_RET_1_CST_1_WITH_OPT(OWNER,RET, METHOD, ARGT1, ARGV1, DEF) \
  inline RET METHOD(ARGT1 ARGV1) const { auto ref = OWNER; if (!ref){ return std::nullopt;} return ref->METHOD(ARGV1); }

#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_2_RET_0_CST_0_WITH_RES(OWNER,RET, METHOD, ARGT1, ARGV1, ARGT2, ARGV2, SUC, FAL) \
  inline RET METHOD(ARGT1 ARGV1, ARGT2 ARGV2)       { auto ref = OWNER; if (!ref){ return FAL;} ref->METHOD(ARGV1,ARGV2); return SUC; }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_2_RET_0_CST_1_WITH_RES(OWNER,RET, METHOD, ARGT1, ARGV1, ARGT2, ARGV2, SUC, FAL) \
  inline RET METHOD(ARGT1 ARGV1, ARGT2 ARGV2) const { auto ref = OWNER; if (!ref){ return FAL;} ref->METHOD(ARGV1,ARGV2); return SUC; }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_2_RET_1_CST_0_WITH_DEF(OWNER,RET, METHOD, ARGT1, ARGV1, ARGT2, ARGV2, DEF) \
  inline RET METHOD(ARGT1 ARGV1, ARGT2 ARGV2)       { auto ref = OWNER; if (!ref){ return DEF;} return ref->METHOD(ARGV1,ARGV2); }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_2_RET_1_CST_1_WITH_DEF(OWNER,RET, METHOD, ARGT1, ARGV1, ARGT2, ARGV2, DEF) \
  inline RET METHOD(ARGT1 ARGV1, ARGT2 ARGV2) const { auto ref = OWNER; if (!ref){ return DEF;} return ref->METHOD(ARGV1,ARGV2); }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_2_RET_1_CST_0_WITH_OPT(OWNER,RET, METHOD, ARGT1, ARGV1, ARGT2, ARGV2, DEF) \
  inline RET METHOD(ARGT1 ARGV1, ARGT2 ARGV2)       { auto ref = OWNER; if (!ref){ return std::nullopt;} return ref->METHOD(ARGV1,ARGV2); }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_2_RET_1_CST_1_WITH_OPT(OWNER,RET, METHOD, ARGT1, ARGV1, ARGT2, ARGV2, DEF) \
  inline RET METHOD(ARGT1 ARGV1, ARGT2 ARGV2) const { auto ref = OWNER; if (!ref){ return std::nullopt;} return ref->METHOD(ARGV1,ARGV2); }

#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_3_RET_0_CST_0_WITH_RES(OWNER,RET, METHOD, ARGT1, ARGV1, ARGT2, ARGV2, ARGT3, ARGV3, SUC, FAL) \
  inline RET METHOD(ARGT1 ARGV1, ARGT2 ARGV2, ARGT3 ARGV3)       { auto ref = OWNER; if (!ref){ return FAL;} ref->METHOD(ARGV1,ARGV2,ARGV3); return SUC; }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_3_RET_0_CST_1_WITH_RES(OWNER,RET, METHOD, ARGT1, ARGV1, ARGT2, ARGV2, ARGT3, ARGV3, SUC, FAL) \
  inline RET METHOD(ARGT1 ARGV1, ARGT2 ARGV2, ARGT3 ARGV3) const { auto ref = OWNER; if (!ref){ return FAL;} ref->METHOD(ARGV1,ARGV2,ARGV3); return SUC; }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_3_RET_1_CST_0_WITH_DEF(OWNER,RET, METHOD, ARGT1, ARGV1, ARGT2, ARGV2, ARGT3, ARGV3, DEF) \
  inline RET METHOD(ARGT1 ARGV1, ARGT2 ARGV2, ARGT3 ARGV3)       { auto ref = OWNER; if (!ref){ return DEF;} return ref->METHOD(ARGV1,ARGV2,ARGV3); }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_3_RET_1_CST_1_WITH_DEF(OWNER,RET, METHOD, ARGT1, ARGV1, ARGT2, ARGV2, ARGT3, ARGV3, DEF) \
  inline RET METHOD(ARGT1 ARGV1, ARGT2 ARGV2, ARGT3 ARGV3) const { auto ref = OWNER; if (!ref){ return DEF;} return ref->METHOD(ARGV1,ARGV2,ARGV3); }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_3_RET_1_CST_0_WITH_OPT(OWNER,RET, METHOD, ARGT1, ARGV1, ARGT2, ARGV2, ARGT3, ARGV3, DEF) \
  inline RET METHOD(ARGT1 ARGV1, ARGT2 ARGV2, ARGT3 ARGV3)       { auto ref = OWNER; if (!ref){ return std::nullopt;} return ref->METHOD(ARGV1,ARGV2,ARGV3); }
#define HK_CORE_TYPES_REF_DELEGATE_DEFINE_METHOD_ARG_3_RET_1_CST_1_WITH_OPT(OWNER,RET, METHOD, ARGT1, ARGV1, ARGT2, ARGV2, ARGT3, ARGV3, DEF) \
  inline RET METHOD(ARGT1 ARGV1, ARGT2 ARGV2, ARGT3 ARGV3) const { auto ref = OWNER; if (!ref){ return std::nullopt;} return ref->METHOD(ARGV1,ARGV2,ARGV3); }




