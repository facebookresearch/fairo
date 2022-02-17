#pragma once

#include <spdlog/spdlog.h>

#define VALIDATE(x) (!(x) && (spdlog::error("VALIDATE(" #x ") failed."), exit(1), true))
#define VALIDATE_EQ(x, y) (((x) != (y)) && (spdlog::error("VALIDATE_EQ(" #x ", " #y ") failed (%s != %s).", \
        std::to_string((x)).c_str(), std::to_string((y)).c_str()), exit(1), true))
