/**
  ******************************************************************************
  * @file    ai_memory_manager.h
  * @author  AST Embedded Analytics Research Platform
  * @date    18-Jun-2018
  * @brief   AI Library Memory Management Wrappers
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

#ifndef __AI_MEMORY_MANAGER_H__
#define __AI_MEMORY_MANAGER_H__
#pragma once

#include <string.h>  /* memcpy */
#include <stdlib.h>

#include "ai_datatypes_defines.h"

/*!
 * @section MemoryManager
 * @ingroup ai_memory_manager
 * Macros to handle memory allocation and management as generic wrappers.
 * Dynamic allocations, freeing, clearing and copy are provided.
 * @{
 */

#define AI_MEM_ALLOC(size, type) \
          ((type*)malloc((size)*sizeof(type)))          

#define AI_MEM_FREE(ptr) \
          { free((void*)(ptr)); }

#define AI_MEM_CLEAR(ptr, size) \
          { memset((void*)(ptr), 0, (size)); }

#define AI_MEM_COPY(dst, src, size) \
          { memcpy((void*)(dst), (const void*)(src), (size)); }

#define AI_MEM_MOVE(dst, src, size) \
          { memmove((void*)(dst), (const void*)(src), (size)); }

/*!
 * @brief Copy an array into another.
 * @ingroup ai_memory_manager
 * @param src the source array handle
 * @param dst the destination array handle
 * @param size the size in byte of the two arrays
 * @return a pointer to the destination buffer
 */
AI_DECLARE_STATIC
ai_handle ai_mem_copy_buffer(
  ai_handle dst, const ai_handle src, const ai_size byte_size)
{
  AI_ASSERT(src && dst && byte_size>0)
  AI_MEM_COPY(dst, src, byte_size)

  return dst;
}

/*! @} */
          
#endif    /*__AI_MEMORY_MANAGER_H__*/
