import math
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt

"""
Parameters
"""
# Larnaca, Cyprus
winter_solstice = 31 # Dec 21 (Day 355 or -10)
summer_solstice = 78 # Jun 21 (Day 172)
ang_equinox = 55 # March 21 & Sept 21 (Days 80 & 264)
year_length = 365
ocean_examined = 1.33 # the refractive index of the body of water through which the light is travelling

longest_day = 14.33 # hours = 14h 20m
shortest_day = 9.64 # hours = 9h 39m

winter_sol = [21, 12]
summer_sol = [21, 6]
eq_1 = [21, 3]
eq_2 = [21, 9]

"""
IMPORTANT: The elevations outputted are the maximum elevations on that calendar day
"""

def sun_elevation(day_of_year):
    # model elevation as sinusoidal function
    amplitude = (summer_solstice - winter_solstice) / 2
    mean_elevation = (summer_solstice + winter_solstice) / 2

    # align
    phase_shift = -80  # 21 march is quarter-phase shift

    elevation = mean_elevation + amplitude * np.sin(2 * np.pi * (day_of_year + phase_shift) / year_length)

    return elevation

def sun_hours(day_of_year):
    # models hours as sinusoidal function
    amplitude = (longest_day - shortest_day) / 2
    mean_hours = (longest_day + shortest_day) / 2

    # align
    phase_shift = -80  # 21 march is quarter-phase shift

    hours = mean_hours + amplitude * np.sin(2 * np.pi * (day_of_year + phase_shift) / year_length)

    return hours

def get_day_of_year(date):
    return date.timetuple().tm_yday


def get_date_from_day_of_year(day_of_year, year=2024):
    try:
        return datetime.date(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
    except ValueError:
        return None  # for invalid inputs

def stage1(day=1,month=1,year=2024, day_of_year=None):
    if day_of_year is None:
        given_date = datetime.date(year, month, day)
        doy = get_day_of_year(given_date)
    else:
        doy = day_of_year

    elevation = sun_elevation(doy)
    #print(f"On {today}, the estimated maximum solar elevation is {elevation:.2f} degrees.")
    return elevation


def stage_2(incident_angle_air, n1=1.0, n2=1.333):
  """
    n1: Refractive index of air (default is 1.0).
    n2: Refractive index of seawater (default is 1.333).
  """
  incident_angle_rad = np.radians(incident_angle_air)
  #snell's Law: n1 * sin(theta1) = n2 * sin(theta2)
  sin_refracted_angle = (n1 * np.sin(incident_angle_rad)) / n2

  if sin_refracted_angle > 1:
    return None  # total internal reflection occurs

  refracted_angle_rad = np.arcsin(sin_refracted_angle)
  refracted_angle_deg = np.degrees(refracted_angle_rad)

  return refracted_angle_deg


sunlight_angles_in_water = []
sunlight_hours = []
days = [i for i in range(year_length)]
actual_dates = [get_date_from_day_of_year(i) for i in range(year_length)]

for i in days:
    sunlight_angles_in_water.append(stage_2(stage1(day_of_year=i),n2=ocean_examined))
    sunlight_hours.append(sun_hours(day_of_year=i))

plt.plot(actual_dates,sunlight_angles_in_water,label='Angle of sunlight in water')
plt.plot(actual_dates,sunlight_hours,label='Hours of sunlight')
plt.legend()
plt.show()

""" not considering TIR to be sufficient for cut-off calc """