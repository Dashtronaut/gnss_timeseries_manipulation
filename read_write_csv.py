def read_csv_timeseries(
    filepath,
    name,
    latitude,
    longitude,
    reference_frame,
    eqtimes=None,
    offsets=None
):

    df = pd.read_csv(
        filepath,
        parse_dates=["date"]
    )

    df = df.set_index("date")

    return TimeSeries(
        name=name,
        data=df,

        latitude=latitude,
        longitude=longitude,

        reference_frame=reference_frame,

        eqtimes=eqtimes or [],
        offsets=offsets or []
    )

def write_csv_timeseries(ts, filepath):

    df = ts.data.copy()

    df.index.name = "date"

    df.to_csv(filepath)
