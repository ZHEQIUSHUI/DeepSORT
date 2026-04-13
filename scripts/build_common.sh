#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <chip>" >&2
    echo "  chip: ax650 | axcl-x86_64 | axcl-aarch64" >&2
    exit 1
fi

CHIP="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_ARCH="$(uname -m)"

case "${CHIP}" in
    ax650)
        MSP_ZIP_NAME="msp_50_3.10.2.zip"
        MSP_ZIP_DEFAULT="${ROOT_DIR}/.ci/downloads/${MSP_ZIP_NAME}"
        MSP_EXTRACT_DIR="${ROOT_DIR}/.ci/msp/ax650"
        MSP_ROOT="${MSP_EXTRACT_DIR}/msp"
        TOOLCHAIN_FILE="${ROOT_DIR}/toolchains/aarch64-none-linux-gnu.toolchain.cmake"
        DEFAULT_TOOLCHAIN_BIN="${ROOT_DIR}/.ci/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin"
        COMPILER_CHECK="aarch64-none-linux-gnu-g++"
        ;;
    axcl-x86_64)
        MSP_ZIP_NAME="axcl_linux_3.10.2.zip"
        MSP_ZIP_DEFAULT="${ROOT_DIR}/.ci/downloads/${MSP_ZIP_NAME}"
        MSP_EXTRACT_DIR="${ROOT_DIR}/.ci/axcl/axcl_linux_3.10.2_x86_64"
        MSP_ROOT=""
        TOOLCHAIN_FILE=""
        DEFAULT_TOOLCHAIN_BIN=""
        COMPILER_CHECK="g++"
        AXCL_SUBDIR_NAME="axcl_linux_x86"
        ;;
    axcl-aarch64)
        MSP_ZIP_NAME="axcl_linux_3.10.2.zip"
        MSP_ZIP_DEFAULT="${ROOT_DIR}/.ci/downloads/${MSP_ZIP_NAME}"
        MSP_EXTRACT_DIR="${ROOT_DIR}/.ci/axcl/axcl_linux_3.10.2_aarch64"
        MSP_ROOT=""
        TOOLCHAIN_FILE="${ROOT_DIR}/toolchains/aarch64-none-linux-gnu.toolchain.cmake"
        DEFAULT_TOOLCHAIN_BIN="${ROOT_DIR}/.ci/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin"
        COMPILER_CHECK="aarch64-none-linux-gnu-g++"
        AXCL_SUBDIR_NAME="axcl_linux_arm64"

        # Native build on an aarch64 host (e.g. Raspberry Pi):
        # do NOT use the x86_64 cross toolchain path by default.
        if [[ "${HOST_ARCH}" == "aarch64" || "${HOST_ARCH}" == "arm64" ]]; then
            TOOLCHAIN_FILE=""
            DEFAULT_TOOLCHAIN_BIN=""
            COMPILER_CHECK="g++"
        fi
        ;;
    *)
        echo "unsupported chip: ${CHIP}" >&2
        exit 1
        ;;
esac

MSP_ZIP_PATH="${DEEPSORT_MSP_ZIP_PATH:-${MSP_ZIP_DEFAULT}}"
TOOLCHAIN_BIN="${DEEPSORT_TOOLCHAIN_BIN:-${DEFAULT_TOOLCHAIN_BIN}}"
BUILD_DIR="${DEEPSORT_BUILD_DIR:-${ROOT_DIR}/build_${CHIP}_ci}"
STAGE_DIR="${DEEPSORT_STAGE_DIR:-${ROOT_DIR}/artifacts/${CHIP}}"
PACKAGE_BASENAME="deepsort_${CHIP}"
PACKAGE_DIR="${STAGE_DIR}/${PACKAGE_BASENAME}"

if [[ -n "${TOOLCHAIN_BIN}" && -d "${TOOLCHAIN_BIN}" ]]; then
    export PATH="${TOOLCHAIN_BIN}:${PATH}"
fi

if ! command -v "${COMPILER_CHECK}" >/dev/null 2>&1; then
    echo "missing compiler ${COMPILER_CHECK}; expected under ${TOOLCHAIN_BIN} or available in PATH" >&2
    exit 1
fi

"${COMPILER_CHECK}" -v >/dev/null 2>&1 || {
    echo "compiler probe failed: ${COMPILER_CHECK} -v" >&2
    exit 1
}

if [[ "${CHIP}" == axcl-* ]]; then
    AXCL_ROOT="${DEEPSORT_AXCL_DIR:-}"

    if [[ -z "${AXCL_ROOT}" ]]; then
        if [[ -f "${MSP_ZIP_PATH}" ]]; then
            mkdir -p "${MSP_EXTRACT_DIR}"
            if [[ ! -d "${MSP_EXTRACT_DIR}/${AXCL_SUBDIR_NAME}" ]]; then
                rm -rf "${MSP_EXTRACT_DIR}"
                mkdir -p "${MSP_EXTRACT_DIR}"
                unzip -q "${MSP_ZIP_PATH}" -d "${MSP_EXTRACT_DIR}"
            fi

            if [[ -d "${MSP_EXTRACT_DIR}/${AXCL_SUBDIR_NAME}" ]]; then
                AXCL_ROOT="${MSP_EXTRACT_DIR}/${AXCL_SUBDIR_NAME}"
            else
                DETECTED_AXCL_ROOT="$(find "${MSP_EXTRACT_DIR}" -maxdepth 5 \( -path "*/${AXCL_SUBDIR_NAME}/include/axcl.h" -o -path "*/${AXCL_SUBDIR_NAME}/include/axcl/axcl.h" \) | head -n 1 || true)"
                if [[ "${DETECTED_AXCL_ROOT}" == */include/axcl/axcl.h ]]; then
                    AXCL_ROOT="${DETECTED_AXCL_ROOT%/include/axcl/axcl.h}"
                elif [[ "${DETECTED_AXCL_ROOT}" == */include/axcl.h ]]; then
                    AXCL_ROOT="${DETECTED_AXCL_ROOT%/include/axcl.h}"
                fi
            fi
        fi

        if [[ -z "${AXCL_ROOT}" && -f "/usr/include/axcl/axcl.h" && -d "/usr/lib/axcl" ]]; then
            AXCL_ROOT="/usr"
        elif [[ -z "${AXCL_ROOT}" && -f "/usr/include/axcl.h" && ( -d "/usr/lib" || -d "/usr/lib64" ) ]]; then
            AXCL_ROOT="/usr"
        fi
    fi

    if [[ -z "${AXCL_ROOT}" || ! -d "${AXCL_ROOT}" ]]; then
        echo "AXCL root not found for ${CHIP}; set DEEPSORT_AXCL_DIR, provide ${MSP_ZIP_PATH}, or install AXCL under /usr" >&2
        exit 1
    fi
else
    if [[ ! -f "${MSP_ZIP_PATH}" ]]; then
        echo "missing MSP zip: ${MSP_ZIP_PATH}" >&2
        exit 1
    fi

    mkdir -p "${MSP_EXTRACT_DIR}"
    DETECTED_EXTRACT_MARKER=""
    if [[ -d "${MSP_ROOT}" ]]; then
        DETECTED_EXTRACT_MARKER="$(find "${MSP_ROOT}" -maxdepth 3 -type d -name msp 2>/dev/null | head -n 1 || true)"
    fi
    if [[ ! -d "${MSP_ROOT}" || -z "${DETECTED_EXTRACT_MARKER}" ]]; then
        rm -rf "${MSP_EXTRACT_DIR}"
        mkdir -p "${MSP_EXTRACT_DIR}"
        unzip -q "${MSP_ZIP_PATH}" -d "${MSP_EXTRACT_DIR}"
    fi

    DETECTED_MSP_ROOT="$(find "${MSP_EXTRACT_DIR}" -maxdepth 3 -type d -name msp | head -n 1 || true)"
    if [[ -n "${DETECTED_MSP_ROOT}" ]]; then
        MSP_ROOT="${DETECTED_MSP_ROOT}"
    else
        echo "extracted MSP root not found: ${MSP_ROOT}" >&2
        exit 1
    fi
fi

rm -rf "${BUILD_DIR}"

if [[ "${CHIP}" == axcl-* ]]; then
    CMAKE_ARGS=(
        -S "${ROOT_DIR}"
        -B "${BUILD_DIR}"
        -DDEEPSORT_ENABLE_AXCL=ON
        -DDEEPSORT_ENABLE_AX650=OFF
        -DDEEPSORT_ENABLE_AX_VIDEO_SDK=ON
        -DDEEPSORT_BUILD_DEMO_AXVSDK=ON
        -DDEEPSORT_AXCL_DIR="${AXCL_ROOT}"
        -DDEEPSORT_AXCL_INIT_IN_RUNNER=OFF
    )
    if [[ -n "${TOOLCHAIN_FILE}" ]]; then
        CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}")
    fi
else
    CMAKE_ARGS=(
        -S "${ROOT_DIR}"
        -B "${BUILD_DIR}"
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}"
        -DDEEPSORT_ENABLE_AXCL=OFF
        -DDEEPSORT_ENABLE_AX650=ON
        -DDEEPSORT_ENABLE_AX_VIDEO_SDK=ON
        -DDEEPSORT_BUILD_DEMO_AXVSDK=ON
        -DDEEPSORT_AX650_MSP_DIR="${MSP_ROOT}"
    )
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" -j"$(getconf _NPROCESSORS_ONLN)"

rm -rf "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}/bin" "${PACKAGE_DIR}/lib"

for bin in test_detector test_reid run_deepsort demo_axvsdk_deepsort; do
    if [[ -f "${BUILD_DIR}/${bin}" ]]; then
        cp -a "${BUILD_DIR}/${bin}" "${PACKAGE_DIR}/bin/"
    fi
done

if [[ -f "${BUILD_DIR}/third_party/ax-video-sdk/libax_video_sdk.so" ]]; then
    cp -a "${BUILD_DIR}/third_party/ax-video-sdk/libax_video_sdk.so" "${PACKAGE_DIR}/lib/"
fi

cp -a "${ROOT_DIR}/README.md" "${PACKAGE_DIR}/"

cat > "${PACKAGE_DIR}/BUILD_INFO.txt" <<EOF
chip=${CHIP}
build_dir=${BUILD_DIR}
msp_zip=${MSP_ZIP_PATH}
msp_root=${MSP_ROOT:-}
axcl_root=${AXCL_ROOT:-}
toolchain_file=${TOOLCHAIN_FILE}
toolchain_bin=${TOOLCHAIN_BIN}
compiler=$("${COMPILER_CHECK}" --version | head -n 1)
EOF

(
    cd "${STAGE_DIR}"
    tar -czf "${PACKAGE_BASENAME}.tar.gz" "${PACKAGE_BASENAME}"
)

echo "package=${STAGE_DIR}/${PACKAGE_BASENAME}.tar.gz"
