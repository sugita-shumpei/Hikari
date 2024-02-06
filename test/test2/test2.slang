
StructuredBuffer<float>   buffer0;
StructuredBuffer<float>   buffer1;
RWStructuredBuffer<float> result;

interface    IBinaryOp {
  static float eval(float x, float y);
};

struct Add : IBinaryOp{
  __init(int){}
  static float eval(float x, float y) { return x+y;}
};

struct Mul : IBinaryOp{
  __init(int){}
  static float eval(float x, float y) { return x*y;}
};

interface    ICoord {
  float length();
};

struct Coord1 :ICoord{
  __init(float v){x=v;}
  float length() { return abs(x);}
  float x;
};

struct Coord2 :ICoord{
  __init(float v){x=v;y=v;}
  float length() { return sqrt(x*x+y*y);}
  float x;
  float y;
};

[shader("compute")]
[numthreads(1,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    uint index    = threadId.x;
    let op        = createDynamicObject<IBinaryOp, int  >(0,0);
    let coord0    = createDynamicObject<ICoord   , float>(0,buffer0[index]);
    let coord1    = createDynamicObject<ICoord   , float>(1,buffer1[index]);
    result[index] = op.eval(coord0.length(),coord1.length());
}
