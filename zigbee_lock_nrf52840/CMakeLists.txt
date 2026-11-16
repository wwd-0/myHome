# ==============================================================================
# MyHome Zigbee 门锁光耦开锁器固件 (Pro Micro nRF52840 + PC817)
# 适用环境: VS Code + nRF Connect SDK (Zephyr RTOS v2.5.x / v2.6.x / v2.7.x+)
# ==============================================================================
cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(myhome_zigbee_lock_nrf52840 LANGUAGES C)

target_include_directories(app PRIVATE include)

target_sources(app PRIVATE
    src/main.c
)
