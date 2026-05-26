# Time Series Aggregation Tutorial

This is the time-series aggregation tutorial presented by Prof. Sonja Wogrin
at the DTU PES Summer School 2026, converted to marimo notebooks.

The original Jupyter notebook is available on Google colab at [https://colab.research.google.com/drive/1b5bc4P-H-8XP_VyYKzUwbkfFY2kppMME](https://colab.research.google.com/drive/1b5bc4P-H-8XP_VyYKzUwbkfFY2kppMME)

To create virtual environment and install dependencies:

```sh
uv sync
```

To open the notebook editor in a browser:

```sh
poe notebook
```

## Run as web app in a container

Build the image:

```bash
docker build -t time-series-aggregation-tutorial .
```

Run it locally:

```bash
docker run --rm -p 8080:8080 -e PORT=8080 time-series-aggregation-tutorial
```

Open `http://localhost:8080`.

## Deploy to Google Cloud Run as web app

Set your project and region in `.env`:

```
PROJECT_ID=your-gcp-project-id
SERVICE_NAME=time-series-aggregation-tutorial
REGION=europe-west1
```

Then run:

```bash
poe deploy
```
