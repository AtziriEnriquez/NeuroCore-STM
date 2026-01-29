#include "systick.h"

static void(*systick_cbfn)(void) = 0;

// Implement a 2 Hz SysTick timer event
void systick_init(void(*cbfn)(void)) {
    // register the callback function
    systick_cbfn = cbfn;

    // SysTick clock = HCLK/8 = 48 MHz / 8 = 6 MHz
    // 100 ms period -> 0.1 s * 6,000,000 = 600,000 counts
    // RVR must be set to (count − 1)
    // per the STM32F042K6 programming manual section 4.4.2 
    // register description
    SYSTICK->RVR = 600000 - 1;
    // Switch to the "external clock source" (HCLK/8 = 6 MHz), 
    // enable the counter,
    // and enable an exception request when the counter reaches 0
    SYSTICK->CSR = (SYSTICK_CSR_ENABLE | SYSTICK_CSR_TICKINT);
}

void SysTick_Handler(void) {
    if( systick_cbfn) 
        systick_cbfn();
}