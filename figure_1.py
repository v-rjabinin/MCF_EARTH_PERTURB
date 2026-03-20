from classes import Orbit, CoordTransformer, AccelerComputer

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Input data
    a = 12_000                                  # km
    e = 0.1
    i = np.radians(77.6)                        # rad
    omega = np.radians(60)                      # rad
    arg_per = np.radians(0)                     # rad
    M = np.linspace(0, 2 * np.pi, 9)  # rad


    orbit = Orbit(a, e, i, M, omega, arg_per)

    transformer = CoordTransformer(orbit)
    u = transformer.transform_to_eci()
    transformer.transform_to_gcs()
    transformer.transform_to_gscs()

    all_acceler = np.empty((5, 3, 9), dtype=np.float64)

    for i in range(2, 6 + 1):
        computer = AccelerComputer(orbit, transformer, n_max=i)
        computer.calculate_associated_legendre()
        computer.calculate_associated_legendre_derivative()
        computer.calculate_acceleration()
        computer.calculate_components(u)

        all_acceler[i - 2, :, :] = computer.comp

    reference = all_acceler[-1, :, :]
    reference_len = np.sqrt(np.sum(reference ** 2, axis=0))

    fig, axis = plt.subplots(4, 4, figsize=(20, 14), sharex="col")
    n_max = np.arange(2, 6 + 1, 1, dtype=np.int64)
    labels = [r"$|Δ\bar{a}_{рад}|$", r"$|Δ\bar{a}_{тр}|$", r"$|Δ\bar{a}_{бин}|$", r"$|Δ\bar{a}_{возм}|$"]
    nu_labels = [r"$\nu = 0$", r"$\nu = \frac{\pi}{2}$", r"$\nu = \pi$", r"$\nu = \frac{3\pi}{2}$"]

    for i in range(4):
        for j in range(3):
            diff = np.abs(all_acceler[:, j, 2 * i] - reference[j, 2 * i])
            axis[i, j].plot(n_max, diff, "-.or", label=labels[j])
            axis[i, j].legend(fontsize=12)

            if j == 0:
                axis[i, j].set_ylabel(r"$km/s^2$", fontsize=10)

            if i == 3:
                axis[i, j].set_xlabel(r"$n_{max}$", fontsize=10)

        diff = np.abs(np.sqrt(np.sum(all_acceler[:, :, 2 * i] ** 2, axis=1)) - reference_len[2 * i])
        axis[i, 3].plot(n_max, diff, "-.ob", label=labels[3])
        axis[i, 3].legend(fontsize=12)

    axis[3, 3].set_xlabel(r"$n_{max}$", fontsize=10)

    y_heights = [0.8668, 0.625, 0.386, 0.145]

    for i in range(4):
        fig.text(0.02, y_heights[i], nu_labels[i],
                 va='center', ha='center', fontsize=18, fontweight='bold',
                 transform=fig.transFigure)

    plt.tight_layout(rect=[0.05, 0, 1, 1])
    plt.show()