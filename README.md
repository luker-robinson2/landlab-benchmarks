# landlab-benchmarks

Performance benchmarks for [Landlab](https://github.com/landlab/landlab), written for
[airspeed velocity (asv)](https://asv.readthedocs.io/).

## What's here

| File | What it times |
| --- | --- |
| `benchmarks/bench_raster_gradients.py` | Raster gradient, difference and flux divergence, by grid size |
| `benchmarks/bench_raster_fill.py` | Building raster graph connectivity (nodes, links, patches, faces) |
| `benchmarks/bench_flexure.py` | `Flexure.subside_loads`, by grid size and number of loads |

## Running

Install asv into an environment that has Landlab:

```bash
pip install asv
```

**Against the Landlab you have installed** (fast; good while writing benchmarks):

```bash
asv machine --yes                 # once per machine
asv run --python=same             # all benchmarks
asv run --python=same --bench flexure  # just the ones matching a regex
asv run --python=same --quick     # each benchmark once, no statistics
```

or with nox: `nox -s bench` (pass asv options after `--`) and `nox -s bench-check`.

**Across Landlab's history** (builds Landlab from git in its own environment):

```bash
asv run master^!                  # the tip of landlab's master branch
asv run v2.10.0..master --steps 10
asv publish && asv preview        # browse the results as a website
asv compare v2.11.0 master        # what got faster or slower
```

## Threads and timing

Landlab's compiled kernels use OpenMP, so timings depend on `OMP_NUM_THREADS`.
Set it explicitly when you compare runs.

On laptops with performance and efficiency cores, or with turbo boost, timings
move around. Pin the process (`taskset -c 0-3 asv run ...` on Linux) and
compare runs from the same machine only. The macOS wheels are built without
OpenMP, so there every thread count runs serially.

## Writing a benchmark

asv runs every method whose name starts with `time_`, once for each combination
of `params`. Do the setup in `setup`, so only the call you care about is timed:

```python
class TimeLinearDiffuser:
    params = [[64, 128, 256, 512]]
    param_names = ["n"]

    def setup(self, n):
        grid = RasterModelGrid((n, n))
        grid.add_zeros("topographic__elevation", at="node")
        self.diffuser = LinearDiffuser(grid, linear_diffusivity=0.01)

    def time_run_one_step(self, n):
        self.diffuser.run_one_step(100.0)
```

Classes whose names start with `_` are not collected, so shared setup can live
in a base class. CI runs `nox -s bench-check`, which runs each benchmark once
against Landlab's master branch to catch benchmarks that break. It doesn't
compare timings.
