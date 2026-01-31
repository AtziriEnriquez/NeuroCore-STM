#include "adc.h"
#include "nvic.h"

#include "stm32f042k6.h"

// Registered callback function for ADC conversions
static void (*adc_cbfn_ptr)(ADC_CHANNEL_t channel, uint16_t data) = 0;

// Remember which ADC channel is currently converting
static ADC_CHANNEL_t active_channel;

// ADC end-of-conversion interrupt handler
// Reads ADC data register and forwards result to registered callback
void ADC1_IRQHandler(void) {

    // Check if interrupt was caused by End Of Conversion (EOC)
    if (ADC->ISR & ADC_ISR_EOC) {

        // Read the converted sample data
        // (reading DR clears the EOC flag)
        uint16_t data = (uint16_t)ADC->DR;

        // If a valid callback function is registered, call it
        if (adc_cbfn_ptr) {
            adc_cbfn_ptr(active_channel, data);
        }

    }
}


// enable the ADC and any IO pins used by the ADC using the bus clock (8 Mhz)
// configure for 12-bit sampling mode
void adc_init(void (*adc_cbfn)(ADC_CHANNEL_t channel, uint16_t data)) {

    // Set the correct sampling time for the input signal's impedance
    ADC->SMPR = ADC_SMPR_13_5;

    // Calibrate the ADC
    ADC->CR |= ADC_CR_ADCAL;
    // hardware clears the bit when calibration completes
    // if your program hangs and you hit break & are sitting on this 
    // line something is wrong with your request to calibrate... 
    // check the requirements in the user manual!
    while( ADC->CR & ADC_CR_ADCAL);

    // Enable the ADC
    ADC->CR |= ADC_CR_ADEN;

    // Wait for the ADC to become ready
    while( !(ADC->ISR & ADC_ISR_ADRDY) );

    // Register the callback function pointer
    adc_cbfn_ptr = adc_cbfn;

    // Enable ADC end of conversion interrupts in the ADC peripheral
    ADC->IER |= ADC_IER_EOCIE;

    // Enable any ADC interrupts through the NVIC
    NVIC_ISER |= NVIC_ISER_SETENA_ADC_COMP;

} 

// initiate a conversion from the selected channel and wait for the result
ret_val_t adc_convert(ADC_CHANNEL_t channel) {

    // Make sure the ADC is ready
    if( !(ADC->ISR & ADC_ISR_ADRDY ) )
        return RET_ERROR;

    // Remember which channel we are converting
    active_channel = channel;

    // Configure the channel selection register 
    ADC->CHSELR = channel;

    // Start the conversion
    ADC->CR |= ADC_CR_ADSTART;

    return RET_SUCCESS;
}