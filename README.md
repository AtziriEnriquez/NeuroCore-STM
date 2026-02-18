# NeuroCore-STM
# Quantized XOR Neural Network on STM32F042K6

Bare-metal C implementation of a fixed-point neural network running on an STM32F042K6 (ARM Cortex-M0).

This project demonstrates deploying a quantized ML model to resource-constrained embedded hardware using direct register control and deterministic timing.

---

## What This Project Shows

- Fixed-point (Q-format) neural network inference
- Integer-only math (no FPU required)
- Register-level peripheral configuration (no HAL)
- Modular embedded driver design
- Interrupt-driven timing with SysTick
- USART-based runtime debugging

---

## Hardware

- STM32F042K6 (Cortex-M0)
- GPIO, ADC, USART, SysTick
- Custom startup + linker configuration

---

## Key Files
```
nn.c / nn.h → Quantized neural network inference
main.c → Application control flow
adc.c → ADC interface
gpio.c / led.c → GPIO + LED control
usart.c → Serial debugging
systick.c → Timing
startup_stm32f042x6.s
STM32F042K6Tx_FLASH.ld
assignment.ipynb → Python validation + quantization
```

---

## Build

```bash
make
Toolchain: arm-none-eabi-gcc
