# Install script for directory: /home/ehoffer/Torch/eladtools

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/home/ehoffer/torch/install")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "Release")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/eladtools/scm-1/lua/eladtools" TYPE FILE FILES
    "/home/ehoffer/Torch/eladtools/ODCT.lua"
    "/home/ehoffer/Torch/eladtools/testSwallowBN.lua"
    "/home/ehoffer/Torch/eladtools/RecurrentLayer.lua"
    "/home/ehoffer/Torch/eladtools/SpatialConvolutionDCT.lua"
    "/home/ehoffer/Torch/eladtools/SpatialBottleNeck.lua"
    "/home/ehoffer/Torch/eladtools/utils.lua"
    "/home/ehoffer/Torch/eladtools/Optimizer.lua"
    "/home/ehoffer/Torch/eladtools/SelectPoint.lua"
    "/home/ehoffer/Torch/eladtools/init.lua"
    "/home/ehoffer/Torch/eladtools/SpatialNMS.lua"
    "/home/ehoffer/Torch/eladtools/EarlyStop.lua"
    "/home/ehoffer/Torch/eladtools/NetConversion.lua"
    "/home/ehoffer/Torch/eladtools/test/test.lua"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
ELSE(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
ENDIF(CMAKE_INSTALL_COMPONENT)

FILE(WRITE "/home/ehoffer/Torch/eladtools/build/${CMAKE_INSTALL_MANIFEST}" "")
FOREACH(file ${CMAKE_INSTALL_MANIFEST_FILES})
  FILE(APPEND "/home/ehoffer/Torch/eladtools/build/${CMAKE_INSTALL_MANIFEST}" "${file}\n")
ENDFOREACH(file)
