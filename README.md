# NeuroCore-STM  
Fixed-Point Neural Network Inference on STM32F042K6

NeuroCore-STM is a bare-metal C implementation of a quantized neural network running on an STM32F042K6 (ARM Cortex-M0).  
The project demonstrates how floating-point models can be translated into deterministic, fixed-point embedded firmware without using vendor HAL libraries.

---

## Model Training & Quantization

The neural network was validated and quantized in Python before being deployed to the microcontroller.

See `assignment.ipynb` for:
- Floating-point model validation
- Quantization scaling analysis
- Fixed-point comparison results
- Input/output verification

The notebook demonstrates how the embedded fixed-point implementation was derived and tested.

---

## Technical Highlights

- Q-format fixed-point neural network inference
- Integer-only math (Cortex-M0 has no FPU)
- Register-level peripheral configuration (no STM32 HAL)
- Interrupt-driven timing using SysTick
- Modular embedded driver architecture
- Python-based quantization validation

---

## System Architecture

1. Python notebook validates quantization and scaling
2. Weights and biases converted to fixed-point
3. Embedded inference implemented in `nn.c`
4. SysTick ensures deterministic execution
5. USART used for debugging and output validation

All peripherals are configured via direct register access.

---

## Hardware

- STM32F042K6 (ARM Cortex-M0)
- GPIO, ADC, USART, SysTick
- Custom startup file and linker script

---

## Build

```bash
make
```
Toolchain: arm-none-eabi-gcc
