
from typing_extensions import OrderedDict
import  pandas as pd
from IPython.display import display
import numpy  as np

def helper_step1(df_raw):
    column_descriptions = OrderedDict([
    ('name', 'Human-readable location name returned by the weather provider.'),
    ('address', 'Original query string for the location.'),
    ('resolvedAddress', 'Provider-normalised address string.'),
    ('latitude', 'Latitude of the observed location (decimal degrees).'),
    ('longitude', 'Longitude of the observed location (decimal degrees).'),
    ('datetime', 'Local calendar date of the aggregated daily record (YYYY-MM-DD).'),
    ('tempmax', 'Daily maximum temperature in °C.'),
    ('tempmin', 'Daily minimum temperature in °C.'),
    ('temp', 'Daily average temperature in °C — this is the forecasting target.'),
    ('feelslikemax', 'Maximum “feels like” (apparent) temperature in °C.'),
    ('feelslikemin', 'Minimum “feels like” (apparent) temperature in °C.'),
    ('feelslike', 'Mean “feels like” (apparent) temperature in °C.'),
    ('dew', 'Average dew point temperature in °C.'),
    ('humidity', 'Mean relative humidity (%).'),
    ('precip', 'Total precipitation depth in mm (liquid equivalent).'),
    ('precipprob', 'Probability of precipitation during the day (%).'),
    ('precipcover', 'Portion of the day with measurable precipitation (%).'),
    ('preciptype', 'Dominant precipitation type (rain, snow, etc.).'),
    ('windgust', 'Peak wind gust speed (km/h).'),
    ('windspeed', 'Headline sustained wind speed summary (km/h).'),
    ('windspeedmax', 'Maximum sustained wind speed (km/h).'),
    ('windspeedmean', 'Mean sustained wind speed (km/h).'),
    ('windspeedmin', 'Minimum sustained wind speed (km/h).'),
    ('winddir', 'Prevailing wind direction (degrees from north).'),
    ('sealevelpressure', 'Average sea-level pressure (hPa).'),
    ('cloudcover', 'Average cloud cover (%).'),
    ('visibility', 'Average horizontal visibility (km).'),
    ('solarradiation', 'Mean solar radiation (W/m²).'),
    ('solarenergy', 'Total daily solar energy (MJ/m²).'),
    ('uvindex', 'Maximum UV index (unitless).'),
    ('severerisk', 'Severe weather risk index (0–100, higher means larger risk).'),
    ('sunrise', 'Local sunrise timestamp (ISO 8601).'),
    ('sunset', 'Local sunset timestamp (ISO 8601).'),
    ('moonphase', 'Lunar phase fraction: 0=new moon, 0.5=full moon, 1=next new moon.'),
    ('conditions', 'Short text summary of dominant weather conditions.'),
    ('description', 'Narrative daily weather description.'),
    ('icon', 'Icon category matching the conditions (e.g., rain, partly-cloudy-day).'),
    ('source', 'Data source flag (observations vs forecasts).')
    ])

    column_summary = []
    for col in df_raw.columns:
        series = df_raw[col]
        entry = {
            'column': col,
            'dtype': series.dtype.name,
            'missing': int(series.isna().sum()),
            'unique': int(series.nunique(dropna=True)),
            'sample_values': series.dropna().unique()[:5].tolist(),
            'description': column_descriptions.get(col, '')
        }
        if np.issubdtype(series.dtype, np.number) and not series.dropna().empty:
            entry['min'] = float(series.min())
            entry['max'] = float(series.max())
            entry['mean'] = float(series.mean())
        else:
            entry['min'] = entry['max'] = entry['mean'] = None
        column_summary.append(entry)

    column_info = pd.DataFrame(column_summary)
    display(column_info)
    print(f"Total columns documented: {len(column_info)}")
