# 🏀 CBB Shot Chart Explorer

An interactive web application for exploring college basketball shot charts with filtering by team, player, or game.

## Features

- **Filter Options:**
  - Team: View all shots for a specific team
  - Player: View shots for individual players
  - Game: View shots from specific games (with option to filter by team)
  - All Games: View aggregate data across all games

- **Chart Types:**
  - **Scatter Plot**: Individual shot locations (green = made, red = missed)
  - **Heatmap**: Expected points per shot location with statistics breakdown (layup, paint, midrange, 3PT)

- **Statistics Dashboard:**
  - Total shots, makes, field goal percentage
  - Expected points per shot
  - Detailed 2PT/3PT breakdowns
  - Total points scored

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the App

Run the Streamlit app from your terminal:

```bash
streamlit run shotchart_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Usage

1. **Select Filter Mode** (sidebar):
   - Choose between Team, Player, Game, or All Games

2. **Choose Your Selection**:
   - Pick from dropdown menus based on your filter mode

3. **Select Chart Type**:
   - Scatter: See individual shot attempts
   - Heatmap: See expected points by court location

4. **View Statistics**:
   - Sidebar shows quick stats
   - Bottom of page shows detailed breakdowns

## Data Requirements

- The app now loads data in this order:
   1. `CBB_DATA_URL` environment variable (hosted CSV or Parquet URL)
   2. Local `cbb_pbp.csv`
   3. `CBB_DEFAULT_DATA_URL` (defaults to hosted `cbb_pbp_shots.parquet` release asset)
   4. `CBB_DATA_FILE` environment variable (local file path)
   5. Local `filtered_shots.csv`
- Dataset must contain: game_id, coord_x, coord_y, play_type, team, score, miss, text

## Deployment Options

### Streamlit Community Cloud (Free)
1. Push this project to GitHub (the included `.gitignore` keeps `cbb_pbp.csv` out of git).
2. Name the repo something like `cbb-shot-chart` so your URL includes that wording.
3. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
4. Deploy `shotchart_app.py`.
5. Optional for full data: set `CBB_DATA_URL` in app settings/secrets to a hosted CSV/Parquet file.

### Render (Custom Slug URL)
1. Push this project to GitHub.
2. In Render, create a new Web Service from the repo.
3. Keep service name as `cbb-shot-chart` (defined in `render.yaml`) for a URL like:
    `https://cbb-shot-chart.onrender.com`
4. Full shot-level dataset from `cbb_pbp.csv` is already configured via `CBB_DATA_URL` in `render.yaml` using an optimized parquet asset.

### Local Network
```bash
streamlit run shotchart_app.py --server.address 0.0.0.0
```
Access from other devices on your network using your computer's IP address.

## Customization

Edit `shotchart_app.py` to customize:
- Colors and styling (CSS in markdown section)
- Court dimensions (in `draw_court` function)
- Statistics calculations
- Chart appearance
