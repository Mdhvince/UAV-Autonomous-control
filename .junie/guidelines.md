# UAV-Autonomous-control — Development Guidelines

This document captures project-specific practices to set up, test, and extend the codebase efficiently.

## Environment and Build
- Runtime: Python 3.10 (CI uses 3.10).
- Dependency manager: Poetry.
  - Install: `python -m pip install poetry`
  - Install project deps: `poetry install`
  - Use Poetry-run for commands to ensure the venv is used: `poetry run <cmd>`
- Core runtime deps used by the code and/or tests:
  - numpy, matplotlib, plotly, pytest, pytest-cov, coverage-badge.
- Package name in pyproject: `quad3d-sim`; module path is `uav_ac`.
- Poetry packaging: `[tool.poetry].packages = [{ include = "uav_ac" }]` to ensure CI can install the project.

## Configuration Notes
- Project runtime parameters are in `uav_ac/config.ini`.
  - Sections: [RRT], [SIM_FLIGHT], [VEHICLE], [CONTROLLER]. Units are SI (m, s, N, etc.).
  - Obstacles: `coord_obstacles` expects rows of `[x_min, x_max, y_min, y_max, z_min, z_max]`.
  - Space limits: `space_limits = [[x_min, y_min, z_min], [x_max, y_max, z_max]]`.
- `uav_ac/main.py` reads the config via `utils.get_config()` and uses Plotly for visualization. Plotting is excluded from coverage and not required for tests.

## Testing
- Test runner: pytest, configured by `pytest.ini`:
  - `pythonpath = .` (import from repo root)
  - `testpaths = tests`
  - Coverage enabled: `--cov=uav_ac --cov-config=.coveragerc --cov-fail-under=0`
- Coverage config: `.coveragerc` excludes `__init__.py`, `main.py`, and plotting helpers from coverage.
- Running tests (verified):
  - With Poetry: `poetry run pytest`
  - Directly: `python -m pytest`
- Typical invocations:
  - Single test file: `poetry run pytest tests/unit/planning/test_rrt.py -q`
  - Filter by expression: `poetry run pytest -k minimum_snap -q`
  - Generate coverage XML for CI tooling if needed: `poetry run pytest --cov-report=xml`
  - Update badge (after a coverage run): `poetry run coverage-badge -o docs/coverage.svg -f`

### Existing tests and fixtures
- Planning layer is covered by unit tests:
  - `tests/unit/planning/test_minimum_snap.py` covers `MinimumSnap.polynom`, `insert_midpoints_at_indexes`, and `is_collision_cuboid`.
  - `tests/unit/planning/test_rrt.py` covers helper methods of `RRTStar` (`path_cost`, random sampling bounds, nearest/neighbor logic, step adaptation).
- Common fixtures:
  - `tests/conftest.py` defines `rrt_object()` with default `space_limits`, `start`, `goal`, `max_distance`, and `max_iterations` for deterministic-shaped tests.

### Adding new tests
- Place tests under `tests/` (pytest auto-discovers via `pytest.ini`).
- Name files `test_*.py` and functions `test_*`.
- For tests touching RRT randomness, seed NumPy to achieve determinism:
  ```python
  import numpy as np
  def test_rrt_sampling_deterministic(rrt_object):
      np.random.seed(0)
      samples = [rrt_object._generate_random_node() for _ in range(3)]
      assert all(np.all(s >= rrt_object.space_limits_lw) for s in samples)
  ```
- For `MinimumSnap`, note `n_coeffs = 8` (snap order). Use `MinimumSnap.polynom(n_coeffs, order, t)` as in the current tests.

### Demonstrated test creation (verified locally)
- Example minimal test that was created and executed during guideline authoring:
  ```python
  # tests/unit/test_demo_guideline.py
  def test_demo_guideline():
      assert 2 + 2 == 4
  ```
  - Collected and passed with the suite. Remove temporary demonstration tests before committing unless they provide value.

## CI
- GitHub Actions workflow at `.github/workflows/ci.yml`:
  - Runs on ubuntu-latest with Python 3.10
  - Installs Poetry then runs `poetry install`
  - Executes tests via `poetry run pytest` with coverage flags (see pytest.ini)
- `--cov-fail-under=0` prevents CI failures due to low coverage.

## Development Tips and Project Internals
- Planning
  - `RRTStar`:
    - Rounds sampled nodes to 2 decimals; `epsilon=0.15` goal bias.
    - Asserts: `neighborhood_radius > step_size`, and z-upper-limit > z of start/goal.
    - Neighborhood radius defaults to `1.5 * max_distance`.
    - When writing new unit tests for internals, prefer pure helpers (`path_cost`, `_adapt_random_node_position`, `_find_valid_neighbors`) and seed RNG if outcomes depend on sampling.
  - `MinimumSnap`:
    - Uses 8 coefficients per spline to satisfy boundary conditions up to jerk; trajectory arrays include position, velocity, acceleration, yaw (0), and spline id.
    - `insert_midpoints_at_indexes(points, indexes)` and `is_collision_cuboid` are stable unit-test surfaces.
- Runtime / Simulation
  - `uav_ac/main.py` performs an RRT-based plan and minimum snap trajectory generation, then simulates a controller/quadrotor loop. Visualization requires Plotly; not exercised in tests.
  - Adjust `uav_ac/config.ini` to change environment, obstacles, and controller gains.
- Style/tooling
  - No linters/formatters are enforced in-repo; follow PEP 8 and keep numerical routines vectorized with NumPy.
  - If adding linting (e.g., ruff/black), prefer CI alignment and avoid imposing failures until configured across the repo.

## Common Pitfalls
- Import paths: tests assume `pythonpath=.` so `from uav_ac...` works when running at repo root.
- Plotly/matplotlib are installed but not required for tests. Avoid importing plotting modules in new tests to keep them headless.
- Coverage badge: run `coverage-badge` after a coverage-producing pytest run; otherwise the SVG may not reflect the latest state.

## Quick Start (verified)
1) Install deps
   - `python -m pip install poetry`
   - `poetry install`
2) Run tests
   - `poetry run pytest -q`
3) Optional: update coverage badge
   - `poetry run coverage-badge -o docs/coverage.svg -f`
