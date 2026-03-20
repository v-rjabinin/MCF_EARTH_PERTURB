from classes import Orbit, AccelerComputer, CoordTransformer

import matplotlib.pyplot as plt
import numpy as np
import data

if __name__ == "__main__":
    # Input data
    a = 12_000                                  # km
    e = 0.1
    i = np.radians(77.6)                        # rad
    omega = np.radians(60)                      # rad
    arg_per = np.radians(0)                     # rad
    M = np.arange(0, np.pi + 1e-10, np.pi / 48)  # rad


    orbit = Orbit(a, e, i, M, omega, arg_per)

    transformer = CoordTransformer(orbit)
    u = transformer.transform_to_eci()
    transformer.transform_to_gcs()
    transformer.transform_to_gscs()

    computer = AccelerComputer(orbit, transformer, n_max=6)
    computer.calculate_associated_legendre()
    computer.calculate_associated_legendre_derivative()
    computer.calculate_acceleration()
    computer.calculate_components(u)

    acceler = computer.comp # (S, T, W)
    total_a = np.sqrt(np.sum(acceler ** 2, axis=0))

    acceler = np.vstack([acceler, total_a])
    heights = transformer.gscs_coords[0, :] - data.r0

    g_central = data.mu / transformer.gscs_coords[0, :] ** 2

    ratio = acceler[3, :] / g_central * 100

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(heights, acceler[3, :], 'b-', linewidth=2)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=12))
    plt.ylabel(r'$|\bar{a}_{возм}|$, km/s²')
    plt.title('Возмущающее ускорение от нецентральности поля')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(heights, g_central, 'r-', linewidth=2)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=12))
    plt.ylabel(r'$\bar{g}$, km/s²')
    plt.title('Центральное гравитационное ускорение')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(heights, ratio, 'g-', linewidth=2)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=12))
    plt.xlabel('Высота, км')
    plt.ylabel(r'$\frac{|a_{возм}|}{g}$, %')
    plt.title('Относительная величина возмущения')
    plt.grid(True)

    plt.tight_layout()
    plt.show()