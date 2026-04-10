#pragma once

#include <cstdio>

#ifndef ALOGE
#define ALOGE(fmt, ...) std::fprintf(stderr, "[E] " fmt "\n", ##__VA_ARGS__)
#endif

#ifndef ALOGW
#define ALOGW(fmt, ...) std::fprintf(stderr, "[W] " fmt "\n", ##__VA_ARGS__)
#endif

#ifndef ALOGI
#define ALOGI(fmt, ...) std::fprintf(stdout, "[I] " fmt "\n", ##__VA_ARGS__)
#endif
