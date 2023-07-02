from PySide6.QtGui import QColor
from qtpex.qt_utility.qt_color_tables import get_color_table

# y-axis
time_series_namings = {
    "p_1": "pressure [N/m^2] (sensor 1)",
    "p_2": "pressure [N/m^2] (sensor 2)",
    "mob_A": "mobile CO2 [g] in Box A",
    "imm_A": "immobile CO2 [g] in Box A",
    "diss_A": "dissolved CO2 [g] in Box A",
    "seal_A": "sealed CO2 [g] in Box A",
    "mob_B": "mobile CO2 [g] in Box B",
    "imm_B": "immobile CO2 [g] in Box B",
    "diss_B": "dissolved CO2 [g] in Box B",
    "seal_B": "sealed CO2 [g] in Box B",
    "M_C": "convection M [m] in Box C",
    "total_CO2_mass": "total CO2 [g] mass",
}

time_series_namings_keys = ["p_1", "p_2", "mob_A", "imm_A", "diss_A", "seal_A", "mob_B", "imm_B", "diss_B", "seal_B",
                            "M_C", "total_CO2_mass"]  # sorted(time_series_namings.keys())

# x-axis
time_series_naming_time = ("t", "time in seconds")

groups = ['austin', 'csiro', 'delft-DARSim', 'delft-DARTS', 'heriot-watt', 'lanl', 'melbourne', 'stanford', 'stuttgart',
          'experiment_run1', 'experiment_run2', 'experiment_run4', 'experiment_run5']

category_10_colors = get_color_table("category10")

group_to_color = {'austin': category_10_colors[0],
                  'csiro': category_10_colors[1],
                  'delft-DARSim': category_10_colors[2],
                  'delft-DARTS': category_10_colors[3],
                  'heriot-watt': category_10_colors[4],
                  'lanl': category_10_colors[5],
                  'melbourne': category_10_colors[6],
                  'stanford': category_10_colors[7],
                  'stuttgart': category_10_colors[8],
                  'experiment_run1': category_10_colors[9],
                  'experiment_run2': QColor("#000075"),  # navy blue
                  'experiment_run4': QColor("#ffe119"),  # yellow
                  'experiment_run5': QColor("#bfef45"),  # lime
                  }

screenshot_locations = "./screenshots"
screenshot_randint_max = 10000
