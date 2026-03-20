from classes import Orbit, CoordTransformer, AccelerComputer
from data import r0

import numpy as np
import matplotlib.pyplot as plt

def characteristic(array: np.ndarray, h: np.ndarray, u: np.ndarray) -> None:
    ### if array is cut, other must be cut as well
    n = 20
    arg_max, arg_min = np.argpartition(array, -n)[-n:], np.argpartition(array, n)[:n]
    modules = np.abs(array)
    indx = np.argpartition(modules, 3)[:3]

    print(f"max = {array[arg_max]}, u_rad = {u[arg_max]}, u_deg = {np.degrees(u[arg_max])}, h = {h[arg_max]}")
    print(f"min = {array[arg_min]}, u_rad = {u[arg_min]}, u_deg = {np.degrees(u[arg_min])}, h = {h[arg_min]}")
    # print(f"zeros = {array[indx]}, u_rad = {u[indx]}, u_deg = {np.degrees(u[indx])}, h = {h[indx]}")
    print("\n\n")

if __name__ == "__main__":
    # Input data
    a = 12_000                                  # km
    e = 0.1
    i = np.radians(77.6)                        # rad
    omega = np.radians(60)                      # rad
    arg_per = np.radians(0)                     # rad
    M = np.arange(0, 2 * np.pi + 1e-10, np.pi / 48)  # rad


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
    heights = transformer.gscs_coords[0, :] - r0

    fig, ax = plt.subplots(2, 4, figsize=(20, 14), sharex="col")
    colors = ["c", "g", "b", "r"]
    labels = [r"$\bar{a}_{рад}$", r"$\bar{a}_{тр}$", r"$\bar{a}_{бин}$", r"|$\bar{a}_{возм}$|"]
    range_labels = [r"$\nu \in [0, \pi]$", r"$\nu \in [\pi, 2\pi]$"]
    y_heights = [0.7545, 0.262]
    extra_point_texts = (("Pericenter", r"$\nu=\frac{\pi}{2}$", "Apocenter"), ("Apocenter", r"$\nu=\frac{3\pi}{2}$", "Pericenter"))
    coords_for_extra_points = ((((-40, -40), (20, 10), (-30, -30)), ((15, -20), (20, 10), (-30, 30)), ((15, -20), (-40, -40), (-90, -10)), ((-25, 30), (20, 0), (-20, 30))),
                               (((-20, -40), (20, 10), (30, 0)), ((-50, -30), (20, 0), (-5, 30)), ((-90, -10), (-20, 30), (30, 0)), ((-20, 30), (20, 0), (20, 10))))

    mask_1, mask_2 = u < np.pi, u > np.pi

    # characteristic(acceler[0][mask_1], heights[mask_1], u[mask_1])
    # characteristic(acceler[0][mask_2], heights[mask_2], u[mask_2])
    # characteristic(acceler[1][mask_1], heights[mask_1], u[mask_1])
    # characteristic(acceler[1][mask_2], heights[mask_2], u[mask_2])
    # characteristic(acceler[2][mask_1], heights[mask_1], u[mask_1])
    # characteristic(acceler[2][mask_2], heights[mask_2], u[mask_2])
    characteristic(acceler[3][mask_1], heights[mask_1], u[mask_1])
    characteristic(acceler[3][mask_2], heights[mask_2], u[mask_2])

    for i in range(2):
        mask = u >= np.pi if i else u <= np.pi
        curr_heights = heights[mask]
        key_elem = np.array([np.pi, 1.5 * np.pi, 2 * np.pi], dtype=np.float64) if i else np.array([0, 0.5 * np.pi, np.pi], dtype=np.float64)

        for j in range(4):
            curr_points = acceler[j][mask]

            ax[i, j].plot(curr_heights, curr_points, f"--.{colors[j]}", label=labels[j])
            ax[i, j].legend()

            key_indx = np.argmin(np.abs(u - key_elem[:, np.newaxis]), axis=1)

            ax[i, j].annotate(extra_point_texts[i][0], xy=(heights[key_indx[0]], acceler[j][key_indx[0]]), fontsize=10, fontweight="bold", xycoords='data',
                              xytext=coords_for_extra_points[i][j][0], textcoords="offset points", arrowprops=dict(mutation_scale=12, arrowstyle="fancy", fc="0.4", ec="none", connectionstyle="angle3,angleA=0,angleB=-90"))

            ax[i, j].annotate(extra_point_texts[i][1], xy=(heights[key_indx[1]], acceler[j][key_indx[1]]), fontsize=12, fontweight="bold", xycoords='data',
                              xytext=coords_for_extra_points[i][j][1], textcoords="offset points", arrowprops=dict(mutation_scale=12, arrowstyle="fancy", fc="0.4", ec="none", connectionstyle="angle3,angleA=0,angleB=-90"))

            ax[i, j].annotate(extra_point_texts[i][2], xy=(heights[key_indx[2]], acceler[j][key_indx[2]]), fontsize=10, fontweight="bold", xycoords='data',
                              xytext=coords_for_extra_points[i][j][2], textcoords="offset points", arrowprops=dict(mutation_scale=12, arrowstyle="fancy", fc="0.4", ec="none", connectionstyle="angle3,angleA=0,angleB=-90"))

            ax[i, j].plot(heights[key_indx], acceler[j][key_indx], '.', color='purple')
            ax[i, j].yaxis.set_major_locator(plt.MaxNLocator(12))
            ax[i, j].grid(True)

            if i:
                ax[i, j].xaxis.set_major_locator(plt.MaxNLocator(6))
                ax[i, j].tick_params(axis='x', labelsize=8)
                ax[i, j].set(xlim=(4150, 7250))
                ax[i, j].set_xlabel(r"$h, km$", fontsize=10)

            if not j:
                ax[i, j].set_ylabel(r"$km/s^2$", fontsize=10)

        fig.text(0.03, y_heights[i], range_labels[i],
                 va='center', ha='center', fontsize=16, fontweight='bold',
                 transform=fig.transFigure)

    plt.tight_layout(rect=[0.05, 0, 1, 1])

    # plt.show()