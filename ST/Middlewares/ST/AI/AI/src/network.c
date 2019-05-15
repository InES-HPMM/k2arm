/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Tue Apr 16 17:38:05 2019
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2018 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */



#include "network.h"

#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "layers.h"

#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#define AI_TOOLS_VERSION_MAJOR 3
#define AI_TOOLS_VERSION_MINOR 3
#define AI_TOOLS_VERSION_MICRO 0

#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#define AI_TOOLS_API_VERSION_MAJOR 1
#define AI_TOOLS_API_VERSION_MINOR 1
#define AI_TOOLS_API_VERSION_MICRO 0

#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "6d6ccddcd8db9bb2906bd391c0e6d6dc"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     "(rev-)"
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Tue Apr 16 17:38:05 2019"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

/**  Forward network declaration section  *************************************/
AI_STATIC ai_network AI_NET_OBJ_INSTANCE;


/**  Forward network arrays declarations  *************************************/
AI_STATIC ai_array input_0_output_array;   /* Array #0 */
AI_STATIC ai_array dense_22_output_array;   /* Array #1 */
AI_STATIC ai_array activation_22_output_array;   /* Array #2 */
AI_STATIC ai_array dense_23_output_array;   /* Array #3 */
AI_STATIC ai_array activation_23_output_array;   /* Array #4 */
AI_STATIC ai_array dense_24_output_array;   /* Array #5 */
AI_STATIC ai_array activation_24_output_array;   /* Array #6 */
AI_STATIC ai_array dense_25_output_array;   /* Array #7 */
AI_STATIC ai_array activation_25_output_array;   /* Array #8 */


/**  Forward network tensors declarations  ************************************/
AI_STATIC ai_tensor input_0_output;   /* Tensor #0 */
AI_STATIC ai_tensor dense_22_output;   /* Tensor #1 */
AI_STATIC ai_tensor activation_22_output;   /* Tensor #2 */
AI_STATIC ai_tensor dense_23_output;   /* Tensor #3 */
AI_STATIC ai_tensor activation_23_output;   /* Tensor #4 */
AI_STATIC ai_tensor dense_24_output;   /* Tensor #5 */
AI_STATIC ai_tensor activation_24_output;   /* Tensor #6 */
AI_STATIC ai_tensor dense_25_output;   /* Tensor #7 */
AI_STATIC ai_tensor activation_25_output;   /* Tensor #8 */


/**  Forward network tensor chain declarations  *******************************/
AI_STATIC_CONST ai_tensor_chain dense_22_chain;   /* Chain #0 */
AI_STATIC_CONST ai_tensor_chain activation_22_chain;   /* Chain #1 */
AI_STATIC_CONST ai_tensor_chain dense_23_chain;   /* Chain #2 */
AI_STATIC_CONST ai_tensor_chain activation_23_chain;   /* Chain #3 */
AI_STATIC_CONST ai_tensor_chain dense_24_chain;   /* Chain #4 */
AI_STATIC_CONST ai_tensor_chain activation_24_chain;   /* Chain #5 */
AI_STATIC_CONST ai_tensor_chain dense_25_chain;   /* Chain #6 */
AI_STATIC_CONST ai_tensor_chain activation_25_chain;   /* Chain #7 */


/**  Subgraph network operators tensor chain declarations  *********************************/


/**  Subgraph network operators declarations  *********************************/


/**  Forward network layers declarations  *************************************/
AI_STATIC ai_layer_dense dense_22_layer; /* Layer #0 */
AI_STATIC ai_layer_nl activation_22_layer; /* Layer #1 */
AI_STATIC ai_layer_dense dense_23_layer; /* Layer #2 */
AI_STATIC ai_layer_nl activation_23_layer; /* Layer #3 */
AI_STATIC ai_layer_dense dense_24_layer; /* Layer #4 */
AI_STATIC ai_layer_nl activation_24_layer; /* Layer #5 */
AI_STATIC ai_layer_dense dense_25_layer; /* Layer #6 */
AI_STATIC ai_layer_sm activation_25_layer; /* Layer #7 */


/**  Arrays declarations section  *********************************************/
AI_ARRAY_OBJ_DECLARE(
  input_0_output_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 784,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  dense_22_output_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 64,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  activation_22_output_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 64,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  dense_23_output_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 64,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  activation_23_output_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 64,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  dense_24_output_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 64,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  activation_24_output_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 64,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  dense_25_output_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 10,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  activation_25_output_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 10,
  AI_STATIC)


/**  Activations tensors declaration section  *********************************/
AI_TENSOR_OBJ_DECLARE(
  input_0_output,
  AI_SHAPE_INIT(1, 1, 784, 1),
  AI_STRIDE_INIT(3136, 3136, 4, 4),
  &input_0_output_array,
  AI_STATIC)
AI_TENSOR_OBJ_DECLARE(
  dense_22_output,
  AI_SHAPE_INIT(1, 1, 64, 1),
  AI_STRIDE_INIT(256, 256, 4, 4),
  &dense_22_output_array,
  AI_STATIC)
AI_TENSOR_OBJ_DECLARE(
  activation_22_output,
  AI_SHAPE_INIT(1, 1, 64, 1),
  AI_STRIDE_INIT(256, 256, 4, 4),
  &activation_22_output_array,
  AI_STATIC)
AI_TENSOR_OBJ_DECLARE(
  dense_23_output,
  AI_SHAPE_INIT(1, 1, 64, 1),
  AI_STRIDE_INIT(256, 256, 4, 4),
  &dense_23_output_array,
  AI_STATIC)
AI_TENSOR_OBJ_DECLARE(
  activation_23_output,
  AI_SHAPE_INIT(1, 1, 64, 1),
  AI_STRIDE_INIT(256, 256, 4, 4),
  &activation_23_output_array,
  AI_STATIC)
AI_TENSOR_OBJ_DECLARE(
  dense_24_output,
  AI_SHAPE_INIT(1, 1, 64, 1),
  AI_STRIDE_INIT(256, 256, 4, 4),
  &dense_24_output_array,
  AI_STATIC)
AI_TENSOR_OBJ_DECLARE(
  activation_24_output,
  AI_SHAPE_INIT(1, 1, 64, 1),
  AI_STRIDE_INIT(256, 256, 4, 4),
  &activation_24_output_array,
  AI_STATIC)
AI_TENSOR_OBJ_DECLARE(
  dense_25_output,
  AI_SHAPE_INIT(1, 1, 10, 1),
  AI_STRIDE_INIT(40, 40, 4, 4),
  &dense_25_output_array,
  AI_STATIC)
AI_TENSOR_OBJ_DECLARE(
  activation_25_output,
  AI_SHAPE_INIT(1, 1, 10, 1),
  AI_STRIDE_INIT(40, 40, 4, 4),
  &activation_25_output_array,
  AI_STATIC)





/* Layer #0: "dense_22" (Dense) */
  

/* Weight tensor #1 */
AI_ARRAY_OBJ_DECLARE(
  dense_22_weights_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 50176,
  AI_STATIC)

AI_TENSOR_OBJ_DECLARE(
  dense_22_weights,
  AI_SHAPE_INIT(1, 1, 64, 784),
  AI_STRIDE_INIT(200704, 200704, 3136, 4),
  &dense_22_weights_array,
  AI_STATIC)

/* Weight tensor #2 */
AI_ARRAY_OBJ_DECLARE(
  dense_22_bias_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 64,
  AI_STATIC)

AI_TENSOR_OBJ_DECLARE(
  dense_22_bias,
  AI_SHAPE_INIT(1, 1, 64, 1),
  AI_STRIDE_INIT(256, 256, 4, 4),
  &dense_22_bias_array,
  AI_STATIC)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_22_chain, AI_STATIC_CONST, 
  AI_TENSOR_LIST_ENTRY(&input_0_output),
  AI_TENSOR_LIST_ENTRY(&dense_22_output),
  AI_TENSOR_LIST_ENTRY(&dense_22_weights, &dense_22_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_22_layer, 0,
  DENSE_TYPE,
  dense, forward_dense,
  &AI_NET_OBJ_INSTANCE, &activation_22_layer, AI_STATIC,
  .tensors = &dense_22_chain, 
)

/* Layer #1: "activation_22" (Nonlinearity) */
  


AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_22_chain, AI_STATIC_CONST, 
  AI_TENSOR_LIST_ENTRY(&dense_22_output),
  AI_TENSOR_LIST_ENTRY(&activation_22_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_22_layer, 1,
  NL_TYPE,
  nl, forward_relu,
  &AI_NET_OBJ_INSTANCE, &dense_23_layer, AI_STATIC,
  .tensors = &activation_22_chain, 
)

/* Layer #2: "dense_23" (Dense) */
  

/* Weight tensor #1 */
AI_ARRAY_OBJ_DECLARE(
  dense_23_weights_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 4096,
  AI_STATIC)

AI_TENSOR_OBJ_DECLARE(
  dense_23_weights,
  AI_SHAPE_INIT(1, 1, 64, 64),
  AI_STRIDE_INIT(16384, 16384, 256, 4),
  &dense_23_weights_array,
  AI_STATIC)

/* Weight tensor #2 */
AI_ARRAY_OBJ_DECLARE(
  dense_23_bias_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 64,
  AI_STATIC)

AI_TENSOR_OBJ_DECLARE(
  dense_23_bias,
  AI_SHAPE_INIT(1, 1, 64, 1),
  AI_STRIDE_INIT(256, 256, 4, 4),
  &dense_23_bias_array,
  AI_STATIC)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_23_chain, AI_STATIC_CONST, 
  AI_TENSOR_LIST_ENTRY(&activation_22_output),
  AI_TENSOR_LIST_ENTRY(&dense_23_output),
  AI_TENSOR_LIST_ENTRY(&dense_23_weights, &dense_23_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_23_layer, 2,
  DENSE_TYPE,
  dense, forward_dense,
  &AI_NET_OBJ_INSTANCE, &activation_23_layer, AI_STATIC,
  .tensors = &dense_23_chain, 
)

/* Layer #3: "activation_23" (Nonlinearity) */
  


AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_23_chain, AI_STATIC_CONST, 
  AI_TENSOR_LIST_ENTRY(&dense_23_output),
  AI_TENSOR_LIST_ENTRY(&activation_23_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_23_layer, 3,
  NL_TYPE,
  nl, forward_relu,
  &AI_NET_OBJ_INSTANCE, &dense_24_layer, AI_STATIC,
  .tensors = &activation_23_chain, 
)

/* Layer #4: "dense_24" (Dense) */
  

/* Weight tensor #1 */
AI_ARRAY_OBJ_DECLARE(
  dense_24_weights_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 4096,
  AI_STATIC)

AI_TENSOR_OBJ_DECLARE(
  dense_24_weights,
  AI_SHAPE_INIT(1, 1, 64, 64),
  AI_STRIDE_INIT(16384, 16384, 256, 4),
  &dense_24_weights_array,
  AI_STATIC)

/* Weight tensor #2 */
AI_ARRAY_OBJ_DECLARE(
  dense_24_bias_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 64,
  AI_STATIC)

AI_TENSOR_OBJ_DECLARE(
  dense_24_bias,
  AI_SHAPE_INIT(1, 1, 64, 1),
  AI_STRIDE_INIT(256, 256, 4, 4),
  &dense_24_bias_array,
  AI_STATIC)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_24_chain, AI_STATIC_CONST, 
  AI_TENSOR_LIST_ENTRY(&activation_23_output),
  AI_TENSOR_LIST_ENTRY(&dense_24_output),
  AI_TENSOR_LIST_ENTRY(&dense_24_weights, &dense_24_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_24_layer, 4,
  DENSE_TYPE,
  dense, forward_dense,
  &AI_NET_OBJ_INSTANCE, &activation_24_layer, AI_STATIC,
  .tensors = &dense_24_chain, 
)

/* Layer #5: "activation_24" (Nonlinearity) */
  


AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_24_chain, AI_STATIC_CONST, 
  AI_TENSOR_LIST_ENTRY(&dense_24_output),
  AI_TENSOR_LIST_ENTRY(&activation_24_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_24_layer, 5,
  NL_TYPE,
  nl, forward_relu,
  &AI_NET_OBJ_INSTANCE, &dense_25_layer, AI_STATIC,
  .tensors = &activation_24_chain, 
)

/* Layer #6: "dense_25" (Dense) */
  

/* Weight tensor #1 */
AI_ARRAY_OBJ_DECLARE(
  dense_25_weights_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 640,
  AI_STATIC)

AI_TENSOR_OBJ_DECLARE(
  dense_25_weights,
  AI_SHAPE_INIT(1, 1, 10, 64),
  AI_STRIDE_INIT(2560, 2560, 256, 4),
  &dense_25_weights_array,
  AI_STATIC)

/* Weight tensor #2 */
AI_ARRAY_OBJ_DECLARE(
  dense_25_bias_array, AI_DATA_FORMAT_FLOAT, 
  NULL, NULL, 10,
  AI_STATIC)

AI_TENSOR_OBJ_DECLARE(
  dense_25_bias,
  AI_SHAPE_INIT(1, 1, 10, 1),
  AI_STRIDE_INIT(40, 40, 4, 4),
  &dense_25_bias_array,
  AI_STATIC)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_25_chain, AI_STATIC_CONST, 
  AI_TENSOR_LIST_ENTRY(&activation_24_output),
  AI_TENSOR_LIST_ENTRY(&dense_25_output),
  AI_TENSOR_LIST_ENTRY(&dense_25_weights, &dense_25_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_25_layer, 6,
  DENSE_TYPE,
  dense, forward_dense,
  &AI_NET_OBJ_INSTANCE, &activation_25_layer, AI_STATIC,
  .tensors = &dense_25_chain, 
)

/* Layer #7: "activation_25" (Nonlinearity) */
  


AI_TENSOR_CHAIN_OBJ_DECLARE(
  activation_25_chain, AI_STATIC_CONST, 
  AI_TENSOR_LIST_ENTRY(&dense_25_output),
  AI_TENSOR_LIST_ENTRY(&activation_25_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  activation_25_layer, 7,
  SM_TYPE,
  sm, forward_sm,
  &AI_NET_OBJ_INSTANCE, &activation_25_layer, AI_STATIC,
  .tensors = &activation_25_chain, 
)


AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE,
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 236840, 1,
                     NULL),
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 516, 1,
                     NULL),
  &input_0_output, &activation_25_output,
  &dense_22_layer, 0)


AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_buffer* activation_buffer)
{
  AI_ASSERT(net_ctx &&  activation_buffer && activation_buffer->data)

  ai_ptr activations = AI_PTR(AI_PTR_ALIGN(activation_buffer->data, 4));
  AI_ASSERT( activations )
  AI_FLAG_SET(net_ctx->flags, AI_NETWORK_FLAG_OUT_COPY);

  {
    /* Updating activations (byte) offsets */
    input_0_output_array.data = NULL;
  input_0_output_array.data_start = NULL;
  dense_22_output_array.data = activations + 0;
  dense_22_output_array.data_start = activations + 0;
  activation_22_output_array.data = activations + 0;
  activation_22_output_array.data_start = activations + 0;
  dense_23_output_array.data = activations + 256;
  dense_23_output_array.data_start = activations + 256;
  activation_23_output_array.data = activations + 256;
  activation_23_output_array.data_start = activations + 256;
  dense_24_output_array.data = activations + 0;
  dense_24_output_array.data_start = activations + 0;
  activation_24_output_array.data = activations + 0;
  activation_24_output_array.data_start = activations + 0;
  dense_25_output_array.data = activations + 256;
  dense_25_output_array.data_start = activations + 256;
  activation_25_output_array.data = activations + 256;
  activation_25_output_array.data_start = activations + 256;
  
  }
  return true;
}

AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_buffer* weights_buffer)
{
  AI_ASSERT(net_ctx &&  weights_buffer && weights_buffer->data)

  ai_ptr weights = AI_PTR(weights_buffer->data);
  AI_ASSERT( weights )

  {
    /* Updating weights (byte) offsets */
    dense_22_weights_array.format |= AI_FMT_FLAG_CONST;
  dense_22_weights_array.data = weights + 0;
  dense_22_weights_array.data_start = weights + 0;
  dense_22_bias_array.format |= AI_FMT_FLAG_CONST;
  dense_22_bias_array.data = weights + 200704;
  dense_22_bias_array.data_start = weights + 200704;
  dense_23_weights_array.format |= AI_FMT_FLAG_CONST;
  dense_23_weights_array.data = weights + 200960;
  dense_23_weights_array.data_start = weights + 200960;
  dense_23_bias_array.format |= AI_FMT_FLAG_CONST;
  dense_23_bias_array.data = weights + 217344;
  dense_23_bias_array.data_start = weights + 217344;
  dense_24_weights_array.format |= AI_FMT_FLAG_CONST;
  dense_24_weights_array.data = weights + 217600;
  dense_24_weights_array.data_start = weights + 217600;
  dense_24_bias_array.format |= AI_FMT_FLAG_CONST;
  dense_24_bias_array.data = weights + 233984;
  dense_24_bias_array.data_start = weights + 233984;
  dense_25_weights_array.format |= AI_FMT_FLAG_CONST;
  dense_25_weights_array.data = weights + 234240;
  dense_25_weights_array.data_start = weights + 234240;
  dense_25_bias_array.format |= AI_FMT_FLAG_CONST;
  dense_25_bias_array.data = weights + 236800;
  dense_25_bias_array.data_start = weights + 236800;
  
  }

  return true;
}

/**  PUBLIC APIs SECTION  *****************************************************/

AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if ( report && net_ctx )
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = {AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR,
                            AI_TOOLS_API_VERSION_MICRO, 0x0},

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 59350,
      .n_inputs          = AI_NETWORK_IN_NUM,
      .inputs            = AI_BUFFER_OBJ_INIT(
                              AI_BUFFER_FORMAT_FLOAT,
                              1,
                              1,
                              784,
                              1, NULL),
      .n_outputs         = AI_NETWORK_OUT_NUM,
      .outputs           = AI_BUFFER_OBJ_INIT(
                              AI_BUFFER_FORMAT_FLOAT,
                              1,
                              1,
                              10,
                              1, NULL),
      .activations       = net_ctx->activations,
      .weights           = net_ctx->params,
      .n_nodes           = 0,
      .signature         = net_ctx->signature,
    };

    AI_FOR_EACH_NODE_DO(node, net_ctx->input_node)
    {
      r.n_nodes++;
    }

    *report = r;

    return ( r.n_nodes>0 ) ? true : false;
  }
  
  return false;
}

AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if ( !net_ctx ) return false;

  ai_bool ok = true;
  ok &= network_configure_weights(net_ctx, &params->params);
  ok &= network_configure_activations(net_ctx, &params->activations);
  
  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}

#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

