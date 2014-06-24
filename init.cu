#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "SpatialLogSoftMax.cu"
#include "SpatialSoftMax.cu"
#include "SpatialConvolution1.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libeladtools(lua_State *L);

int luaopen_libeladtools(lua_State *L)
{
  lua_newtable(L);


  cunn_SpatialLogSoftMax_init(L);
  cunn_SpatialSoftMax_init(L);
  cunn_SpatialConvolution1_init(L);

  return 1;
}
