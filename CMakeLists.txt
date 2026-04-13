cmake_minimum_required(VERSION 3.16)

project(DeepSORT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DEEPSORT_ENABLE_AXCL "Enable AXCL backend for detector" ON)
option(DEEPSORT_ENABLE_AX650 "Enable AX650/MSP backend for detector" OFF)
option(DEEPSORT_ENABLE_AX_VIDEO_SDK "Enable ax-video-sdk integration" ON)
option(DEEPSORT_AXCL_INIT_IN_RUNNER "Initialize AXCL runtime/context inside runner (standalone mode)" OFF)
option(DEEPSORT_BUILD_TEST_DETECTOR "Build test_detector executable" ON)
option(DEEPSORT_BUILD_TEST_REID "Build test_reid executable" ON)
option(DEEPSORT_BUILD_RUN_DEEPSORT "Build run_deepsort executable" ON)
option(DEEPSORT_BUILD_DEMO_AXVSDK "Build ax-video-sdk tracking demo" ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

# SimpleCV (submodule)
set(SIMPLECV_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/SimpleCV)

if(DEEPSORT_ENABLE_AX_VIDEO_SDK)
    # ax-video-sdk (submodule)
    if(DEEPSORT_ENABLE_AXCL)
        set(DEEPSORT_AXCL_DIR "/usr" CACHE PATH "Path to AXCL SDK root (contains include/axcl and lib/axcl)")
        set(AXSDK_CHIP_TYPE "axcl" CACHE STRING "Target chip type: ax630c, ax620q, ax620qp, ax650, axcl" FORCE)
        set(AXSDK_AXCL_DIR "${DEEPSORT_AXCL_DIR}" CACHE PATH "Path to AXCL SDK root" FORCE)
    elseif(DEEPSORT_ENABLE_AX650)
        set(AXSDK_CHIP_TYPE "ax650" CACHE STRING "Target chip type: ax630c, ax620q, ax620qp, ax650, axcl" FORCE)
        # Reuse the same MSP dir as the AX650 runner.
        if(NOT DEEPSORT_AX650_MSP_DIR)
            set(DEEPSORT_AX650_MSP_DIR "/home/axera/msp_dir/ax650/msp" CACHE PATH "Path to AX650 MSP root (contains out/include and out/lib)")
        endif()
        set(AXSDK_MSP_DIR "${DEEPSORT_AX650_MSP_DIR}" CACHE PATH "Path to target MSP root" FORCE)
    else()
        message(WARNING "DEEPSORT_ENABLE_AX_VIDEO_SDK=ON but no AX backend enabled; disabling SDK integration.")
        set(DEEPSORT_BUILD_DEMO_AXVSDK OFF CACHE BOOL "" FORCE)
    endif()

    if(DEEPSORT_BUILD_DEMO_AXVSDK)
        # Build as shared so downstream executables do not need to link AXCL/MSP flags explicitly.
        set(AXSDK_BUILD_SHARED ON CACHE BOOL "" FORCE)
        set(AXSDK_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        set(AXSDK_BUILD_SMOKE_TESTS OFF CACHE BOOL "" FORCE)
        add_subdirectory(third_party/ax-video-sdk EXCLUDE_FROM_ALL)
    endif()
endif()

add_library(deepsort_npu_runner STATIC
    npu/runner/ax650/ax_model_runner_ax650.cpp
)

target_include_directories(deepsort_npu_runner
    PUBLIC
        ${CMAKE_SOURCE_DIR}
)

if(DEEPSORT_ENABLE_AXCL)
    set(DEEPSORT_AXCL_DIR "/usr" CACHE PATH "Path to AXCL SDK root (contains include/axcl and lib/axcl)")
    target_sources(deepsort_npu_runner PRIVATE
        npu/runner/axcl/ax_model_runner_axcl.cpp
    )
    target_compile_definitions(deepsort_npu_runner PUBLIC DEEPSORT_HAVE_AXCL=1)
    if(DEEPSORT_AXCL_INIT_IN_RUNNER)
        target_compile_definitions(deepsort_npu_runner PUBLIC DEEPSORT_AXCL_INIT_IN_RUNNER=1)
    endif()
    target_include_directories(deepsort_npu_runner PUBLIC "${DEEPSORT_AXCL_DIR}/include" "${DEEPSORT_AXCL_DIR}/include/axcl")
    find_library(DEEPSORT_AXCL_RT_LIB axcl_rt HINTS "${DEEPSORT_AXCL_DIR}/lib/axcl" "${DEEPSORT_AXCL_DIR}/lib" "${DEEPSORT_AXCL_DIR}/lib64" REQUIRED)
    target_link_libraries(deepsort_npu_runner PUBLIC "${DEEPSORT_AXCL_RT_LIB}")
    foreach(_axcl_libdir "${DEEPSORT_AXCL_DIR}/lib/axcl" "${DEEPSORT_AXCL_DIR}/lib" "${DEEPSORT_AXCL_DIR}/lib64")
        if(EXISTS "${_axcl_libdir}")
            # When DEEPSORT_AXCL_DIR is an extracted SDK (esp. cross-build), make sure the linker can
            # locate libaxcl_rt's transitive shared-library dependencies (libaxcl_pkg, libaxcl_comm,
            # libspdlog, ...).
            target_link_options(deepsort_npu_runner PUBLIC "-Wl,-rpath-link=${_axcl_libdir}")
        endif()
    endforeach()
endif()

if(DEEPSORT_ENABLE_AX650)
    target_compile_definitions(deepsort_npu_runner PUBLIC DEEPSORT_HAVE_AX650=1)
    set(DEEPSORT_AX650_MSP_DIR "/home/axera/msp_dir/ax650/msp" CACHE PATH "Path to AX650 MSP root (contains out/include and out/lib)")
    if(NOT EXISTS "${DEEPSORT_AX650_MSP_DIR}/out/include" OR NOT EXISTS "${DEEPSORT_AX650_MSP_DIR}/out/lib")
        message(FATAL_ERROR "DEEPSORT_ENABLE_AX650=ON requires DEEPSORT_AX650_MSP_DIR pointing to AX650 MSP root (out/include, out/lib). Current: ${DEEPSORT_AX650_MSP_DIR}")
    endif()
    target_include_directories(deepsort_npu_runner PUBLIC "${DEEPSORT_AX650_MSP_DIR}/out/include")
    target_link_directories(deepsort_npu_runner PUBLIC "${DEEPSORT_AX650_MSP_DIR}/out/lib")
    target_link_libraries(deepsort_npu_runner PUBLIC ax_engine ax_sys ax_interpreter Threads::Threads)
    target_link_options(deepsort_npu_runner PUBLIC "-Wl,-rpath-link=${DEEPSORT_AX650_MSP_DIR}/out/lib")
endif()

add_library(deepsort_detector STATIC
    detector/yolo_detector.cpp
)

target_include_directories(deepsort_detector
    PUBLIC
        ${CMAKE_SOURCE_DIR}
)

target_link_libraries(deepsort_detector
    PUBLIC
        SimpleCV::simplecv
        deepsort_npu_runner
)

if(DEEPSORT_BUILD_TEST_DETECTOR)
    add_executable(test_detector
        test_detector.cpp
    )
    target_link_libraries(test_detector PRIVATE deepsort_detector)
endif()

add_library(deepsort_tracker STATIC
    tracker/deepsort/src/FeatureTensor.cpp
    tracker/deepsort/src/hungarianoper.cpp
    tracker/deepsort/src/kalmanfilter.cpp
    tracker/deepsort/src/linear_assignment.cpp
    tracker/deepsort/src/model.cpp
    tracker/deepsort/src/munkres.cpp
    tracker/deepsort/src/nn_matching.cpp
    tracker/deepsort/src/reid_feature_extractor.cpp
    tracker/deepsort/src/track.cpp
    tracker/deepsort/src/tracker.cpp
)

target_include_directories(deepsort_tracker
    PUBLIC
        ${CMAKE_SOURCE_DIR}/tracker/deepsort/include
)

target_link_libraries(deepsort_tracker
    PUBLIC
        Eigen3::Eigen
        SimpleCV::simplecv
        deepsort_npu_runner
)

if(DEEPSORT_BUILD_TEST_REID)
    add_executable(test_reid
        test_reid.cpp
    )
    target_link_libraries(test_reid PRIVATE deepsort_tracker)
endif()

if(DEEPSORT_BUILD_RUN_DEEPSORT)
    add_executable(run_deepsort
        run_deepsort.cpp
    )
    target_link_libraries(run_deepsort PRIVATE deepsort_detector deepsort_tracker)
endif()

if(DEEPSORT_ENABLE_AX_VIDEO_SDK AND DEEPSORT_BUILD_DEMO_AXVSDK AND (DEEPSORT_ENABLE_AXCL OR DEEPSORT_ENABLE_AX650))
    add_executable(demo_axvsdk_deepsort
        demo_axvsdk_deepsort.cpp
    )
    target_link_libraries(demo_axvsdk_deepsort
        PRIVATE
            deepsort_detector
            deepsort_tracker
            ax_video_sdk::ax_video_sdk
    )
    # The demo optionally uses a few ax-video-sdk internal helpers for device-side frame copy + metadata.
    target_include_directories(demo_axvsdk_deepsort PRIVATE ${CMAKE_SOURCE_DIR}/third_party/ax-video-sdk/src)
    if(DEEPSORT_ENABLE_AX650 AND UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        # AX650 MSP libraries (e.g. libax_venc.so) may contain unresolved libexif symbols.
        # Keep the demo link step aligned with ax-video-sdk internal tools.
        target_link_options(demo_axvsdk_deepsort PRIVATE -Wl,--allow-shlib-undefined)
    endif()
endif()
