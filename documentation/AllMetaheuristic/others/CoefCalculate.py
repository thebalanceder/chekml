import numpy as np

def CoefCalculate(F, T):
    """
    Calculate the atmospheric absorption coefficient.

    Parameters:
    - F: Frequency (Hz)
    - T: Temperature (°C)

    Returns:
    - AbsorbCoef: Rounded absorption coefficient (dB)
    """
    pres = 1  # Atmospheric pressure (atm)
    relh = 50  # Relative humidity (%)
    freq_hum = F
    temp = T + 273  # Convert to Kelvin

    # Calculate humidity
    C_humid = 4.6151 - 6.8346 * ((273.15 / temp) ** 1.261)
    hum = relh * (10 ** C_humid) * pres

    # Temperature ratio
    tempr = temp / 293.15

    # Oxygen and nitrogen relaxation frequencies
    frO = pres * (24 + 4.04e4 * hum * (0.02 + hum) / (0.391 + hum))
    frN = pres * (tempr ** -0.5) * (9 + 280 * hum * np.exp(-4.17 * ((tempr ** (-1/3)) - 1)))

    # Absorption coefficient calculation
    alpha = 8.686 * freq_hum * freq_hum * (
        1.84e-11 * (1 / pres) * np.sqrt(tempr) +
        (tempr ** -2.5) * (
            0.01275 * (np.exp(-2239.1 / temp) * 1 / (frO + freq_hum * freq_hum / frO)) +
            0.1068 * (np.exp(-3352 / temp) * 1 / (frN + freq_hum * freq_hum / frN))
        )
    )

    # Round to 3 decimal places
    db_humi = np.round(alpha * 1000) / 1000

    return db_humi
