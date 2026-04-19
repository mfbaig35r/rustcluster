# Source-to-Test Pairing in Python, Rust, and Hybrid Repositories

## What the evidence points to

The single most important implementation choice is to treat source-to-test pairing as a **ranked candidate search problem**, not as a single naming-rule lookup. In the Python repositories I could directly verify, three patterns dominate: a dedicated `tests/` tree with flat `test_*.py` files keyed to concepts rather than exact paths; a dedicated `tests/` tree with mostly basename matches; and a package-area mirror that maps at directory granularity rather than file granularity. The Rust tooling story is even more bifurcated: Cargo natively distinguishes inline unit tests in `src/**` from integration tests in `tests/`, and workspaces frequently add crate-level or workspace-level test crates rather than one test file per source file. Coverage contexts are the best refinement layer after static heuristics, because `coverage.py` can record per-test execution context, which is the closest thing to a real source↔test linkage produced during a run. citeturn42search0turn42search1turn42search2turn43search3turn43search6

A practical priority order follows from that evidence. First, parse config and locate roots. Second, detect whether the repository matches one of a small number of structural families. Third, generate and score multiple candidate test files, not just one. Fourth, upgrade or downgrade those candidates with dynamic evidence from coverage or mutation runs. If a tool guesses a single pair too early, it will fail badly on projects like Django and SQLAlchemy, where the mapping is often between a source **area** and a test **area**, not between one source file and one test file. citeturn15view0turn16view2turn24view0turn27view0

## Python layouts in real repositories

The high-confidence findings below are from repository trees and config files I directly verified. For a few projects later in the section, I could verify only the repository identity or CI stack in this pass, so I mark those as lower-confidence.

**FastAPI** keeps source in `fastapi/` and tests in `tests/`. The verified pattern is a flat `tests/test_*.py` layout, with names that are often close to the source basename but not always exact. This is **not** a strong path mirror. I did not verify a `tests/conftest.py` at the tests root in the visible tree page I inspected, and I verified that the repository root includes `pyproject.toml`. The important pairing lesson is that singular/plural and concept renaming happen: `applications.py` maps to `test_application.py`, not necessarily to `test_applications.py`. citeturn2view0turn3view0turn8view0turn9view0

```text
fastapi/applications.py
tests/test_application.py

fastapi/datastructures.py
tests/test_datastructures.py

fastapi/exception_handlers.py
tests/test_exception_handlers.py
```

Those representative paths come from the verified `fastapi/` source tree and the `tests/` tree pages. citeturn8view0turn9view0

**Django** is the strongest Python outlier in your list. The framework itself keeps source in `django/` but tests in a very large top-level `tests/` tree organized by feature areas such as `forms_tests`, `contenttypes_tests`, `dispatch`, `flatpages_tests`, and many more. It does **not** use a verified pytest config in the files I inspected; instead, the `tests/README.rst` and `tox.ini` show a custom `runtests.py` workflow, with tox changing into `tests/` and invoking that runner. I also did not find `conftest.py` in the verified `tests/` root page. For pairing, Django means you must support **directory-area pairing**, not only file pairing. citeturn15view0turn15view1turn15view3turn16view1turn16view2

```text
django/forms/
tests/forms_tests/

django/dispatch/
tests/dispatch/

django/contrib/contenttypes/
tests/contenttypes_tests/

django/contrib/flatpages/
tests/flatpages_tests/
```

These are representative verified source/test area mappings from the `django/` and `tests/` trees. citeturn16view1turn16view2

**Pydantic** is a much cleaner pytest-style repository. Source lives in `pydantic/`, tests live in `tests/`, the tests root includes `tests/conftest.py`, and the repository configures pytest through `[tool.pytest.ini_options]` in `pyproject.toml` with `testpaths = 'tests'` and no verified custom `python_files`/`python_classes` override. Its dominant pattern is flat `tests/test_<module_or_feature>.py` naming against a flat package root. citeturn18view0turn19view0turn19view1turn21view0turn21view2turn21view4

```text
pydantic/config.py
tests/test_config.py

pydantic/fields.py
tests/test_fields.py

pydantic/json_schema.py
tests/test_json_schema.py

pydantic/networks.py
tests/test_networks.py
```

Those paths are directly visible in the package tree and tests tree. citeturn19view0turn20view1turn20view2turn20view3turn20view4

**SQLAlchemy** uses `lib/sqlalchemy/` for source and a singular top-level `test/` directory for tests. The `test/` tree includes `conftest.py`, and the official unit-test README explicitly says the suite is run with pytest, but with substantial SQLAlchemy-specific pytest plugin behavior and database-selection options layered on top. Structurally, the mapping is mostly **package-area mirror** rather than file-basis mirror: `lib/sqlalchemy/engine/` aligns with `test/engine/`, `lib/sqlalchemy/orm/` with `test/orm/`, `lib/sqlalchemy/sql/` with `test/sql/`, and `lib/sqlalchemy/ext/` with `test/ext/`. I did not verify a standard `[tool:pytest]` block in `setup.cfg`. citeturn23view0turn24view0turn24view1turn26view0turn27view0

```text
lib/sqlalchemy/engine/
test/engine/

lib/sqlalchemy/orm/
test/orm/

lib/sqlalchemy/sql/
test/sql/test_query.py

lib/sqlalchemy/ext/
test/ext/
```

The final line shows the kind of mixed directory/file-area pairing this repository uses. citeturn24view0turn24view1turn27view0

**Requests** uses a modern `src/` layout: source in `src/requests/`, tests in top-level `tests/`, `tests/conftest.py` at the test root, and pytest configured in `pyproject.toml` with `[tool.pytest.ini_options]`, `testpaths = ["tests"]`, and `addopts = "--doctest-modules"`. The dominant convention is flat `tests/test_<module>.py` against flat source module filenames. citeturn29view0turn30view0turn30view1turn32view0turn32view2

```text
src/requests/adapters.py
tests/test_adapters.py

src/requests/hooks.py
tests/test_hooks.py

src/requests/structures.py
tests/test_structures.py

src/requests/utils.py
tests/test_utils.py
```

These are clean basename matches, and they are exactly the sort of case a scanner should solve with a high-confidence direct rule before trying looser heuristics. citeturn30view0turn31view0turn31view1turn31view2turn31view3

**Flask** is also very regular: source in `src/flask/`, tests in `tests/`, `tests/conftest.py` at the root, and pytest configured via `[tool.pytest.ini_options]` in `pyproject.toml` with `testpaths = ["tests"]` and no verified custom file/class naming override. The test layout is flat `tests/test_*.py`, keyed to features more than to a mirrored directory tree. citeturn34view0turn35view0turn35view1turn37view0turn37view1turn37view2turn37view3

```text
src/flask/app.py
tests/test_basic.py

src/flask/blueprints.py
tests/test_blueprints.py

src/flask/config.py
tests/test_config.py

src/flask/views.py
tests/test_views.py
```

The first pair is deliberately instructive: `app.py` does **not** map to `test_app.py`; it maps to `test_basic.py` in the verified test tree, which means conceptual naming beats exact basename matching here. citeturn35view0turn36view0turn36view1turn36view3turn36view5

**Black** keeps source under `src/` with subpackages `black`, `blackd`, and `blib2to3`, while keeping tests in a dedicated top-level `tests/` directory with `tests/conftest.py`. The verified tests are flat files such as `test_black.py`, `test_blackd.py`, `test_ipynb.py`, and `test_trans.py`. This repository is another reminder that test files often target features or subpackages, not a file-by-file mirror. citeturn39view0turn40view0turn40view1turn41view0turn41view1turn41view2turn41view3

```text
src/black/
tests/test_black.py

src/blackd/
tests/test_blackd.py

src/black/
tests/test_ipynb.py

src/blib2to3/
tests/test_trans.py
```

Those are representative tree excerpts from the verified source and test roots. citeturn40view0turn40view1turn41view0turn41view1turn41view2turn41view3

**Ruff** is not a pure pytest-layout Python package. The verified repository identity and README establish that it is a Rust implementation, installable from Python, configured through `pyproject.toml`, and intended as a Python tool, but its core is Rust. For your scanner, Ruff belongs in the **hybrid/outlier** bucket, not in the same bucket as Requests or Flask. A naive Python-only pairing algorithm will misclassify the repository. citeturn44search0turn44search1

**HTTPX** is clearly a Python project with a dedicated “Test Suite” workflow in CI, but I did not directly verify the current directory tree in this pass, so I would not hard-code a repository-specific mapping rule for it yet. The safe implementation choice is to treat it as a likely `tests/`-root pytest project until a targeted tree inspection confirms the exact layout. citeturn47search0turn47search4

**Uvicorn** likewise has a verified “Test Suite” workflow using pytest-related dependencies in CI, but I did not directly verify its current repository tree in this pass. The scanner should therefore treat any Uvicorn-specific pairing rule as provisional until a follow-up tree inspection confirms the exact file layout. citeturn47search3

From those verified repositories, the implementable Python heuristics are straightforward. First, check for `tests/` and `test/` roots. Second, parse any pytest config and respect `testpaths` if present. Third, try exact basename matches such as `foo.py → test_foo.py` and `foo.py → foo_test.py`. Fourth, try source-root stripping plus package-area mirroring, such as `src/pkg/sub/foo.py → tests/sub/test_foo.py` and `src/pkg/sub/foo.py → tests/pkg/sub/test_foo.py`. Fifth, add a lower-confidence conceptual layer for singular/plural and feature renames, because real projects use `applications.py → test_application.py` and `app.py → test_basic.py`. Sixth, if the repository is structurally hybrid or workspace-like, stop looking for 1:1 pairs too early and allow area-level matches. citeturn9view0turn21view0turn24view0turn32view0turn37view0turn44search1

## Pytest discovery and Python pairing heuristics

Pytest’s actual default discovery rules are more concrete than many scanners assume. In pytest’s own implementation, the default `python_files` patterns are `test_*.py` and `*_test.py`; the default `python_classes` pattern is `Test`; and the default `python_functions` pattern is `test`. That means a scanner that only looks for `test_*.py` misses valid default pytest modules named `foo_test.py`. citeturn42search0turn42search4

Pytest’s collection docs also show exactly how custom naming changes discovery. Repositories can redefine `python_files`, `python_classes`, and `python_functions`, and can set `testpaths` to constrain recursion to selected roots. The docs are explicit that `python_classes` and `python_functions` **do not affect** `unittest.TestCase` discovery, because pytest delegates that to `unittest`. `conftest.py` can also influence collection directly, for example with `collect_ignore`. citeturn42search1turn42search4

That leads to a clean implementation order for Python pairing. Parse config first, in this order: `pytest.toml`/`pyproject.toml` pytest section, `pytest.ini`, `setup.cfg`, then `tox.ini` only for indirect hints such as `pytest {posargs:tests}`. Build the effective test roots from `testpaths` if present; otherwise, score `tests/` and `test/` highest if they exist. Only then generate filename candidates. If you skip config-first discovery, you will miss repositories like Pydantic, Requests, and Flask that explicitly pin `testpaths`, and you will mis-handle non-pytest projects like Django. citeturn21view0turn32view0turn37view0turn15view3

The easiest edge cases that break naive pairing are visible in the verified repositories. Exact basename matching fails on `fastapi/applications.py → tests/test_application.py`. Exact path mirroring fails on Flask, where a central source file like `src/flask/app.py` is not paired with `tests/test_app.py` in the verified tree. File-based pairing itself fails on Django and SQLAlchemy, where tests are often organized by area or subsystem rather than by one source file per test file. Those cases argue for a **scored candidate set** with confidences instead of a forced single match. citeturn9view0turn35view0turn36view0turn24view0turn16view2

## Rust discovery and Rust pairing heuristics

Cargo’s documented discovery model is simpler than Python’s, but it creates a different pairing problem. The Cargo Book says `cargo test` looks in two places: tests in your `src` files and tests in `tests/`. Tests inside `src` are unit tests and doc tests; `tests/` contains integration tests. Cargo also compiles examples and runs documentation tests, and `cargo test` builds tests by invoking `rustc --test`, which enables the test harness. citeturn43search3turn43search6

The Rust testing attribute rules matter because they define what a scanner should count as an inline test. The Rust Reference says `#[test]` marks a function to execute as a test; those functions are compiled only in test mode; test mode also enables the `test` conditional compilation option. The same reference family also documents `#[should_panic]`, which matters if you want to exempt expected-panics from “unsafe panic path” warnings. citeturn43search0turn43search4turn43search5

For layout, the directly verified Rust example in this pass is **Serde**. Its repository root is a workspace-style layout with `serde`, `serde_core`, `serde_derive`, `serde_derive_internals`, and a top-level `test_suite`, not a single flat crate with one `tests/` directory. That is exactly the kind of structure that breaks a scanner that assumes “repository root `tests/` equals all tests for all code.” citeturn45search3

```text
serde/
serde_core/
serde_derive/
serde_derive_internals/
test_suite/
```

That verified root tree is enough to justify a workspace-aware branch in your algorithm. citeturn45search3

A second directly verified example is **pydantic-core**, which doubles as both a Rust and a hybrid case. Its root contains `src`, `tests`, `benches`, `python/pydantic_core`, `Cargo.toml`, and `pyproject.toml`. In other words, the repository simultaneously advertises Rust unit/integration/benchmark structure **and** a Python packaging layer. That combination is a strong signal that the scanner must model more than one test universe in the same repository. citeturn44search3

```text
src/
tests/
benches/
python/pydantic_core/
Cargo.toml
pyproject.toml
```

Those paths are directly visible in the repository root. citeturn44search3

From the official Cargo docs and the verified roots above, the Rust pairing rules should be: treat inline tests in `src/**` as first-class “same-file” tests; treat each file in `tests/` as an integration-test binary rather than a direct sibling of one module; and detect workspaces by multiple crates or a dedicated test crate such as Serde’s `test_suite`. If you need a 1:1 pairing anyway, the best Rust fallback is **same-file inline first, same-crate integration second, workspace test-crate last**. citeturn43search3turn43search9turn45search3turn44search3

## Hybrid Rust and Python projects

The most useful directly verified hybrid repository in this pass is **pydantic-core**. Its root structure proves a split between Rust internals (`src`, `tests`, `benches`) and Python package content (`python/pydantic_core`) under one repository, with both `Cargo.toml` and `pyproject.toml` present. For pairing, that means the scanner should create two parallel graphs: Rust source↔Rust tests, and Python package/wrapper↔Python tests or import-level checks. Do not collapse them into one namespace. citeturn44search3

The best general documentation for hybrid organization in the sources I reviewed is **maturin**’s user guide excerpt. It explicitly documents two canonical mixed-project layouts: one where the Python package directory sits next to `Cargo.toml`, and another where Python sources live under a configurable `python/` directory. The same docs distinguish `maturin develop` from `pip install .`, note that `maturin develop` installs directly into the active virtualenv, and show how mixed Rust/Python projects are laid out around `Cargo.toml`, `pyproject.toml`, and `src/lib.rs`. citeturn47search1

The practical inference for FFI-boundary testing is that Python-facing boundary behavior should usually be paired to the Python layer first, not to the inner Rust file that implements it. That is because install, import, exception translation, and Python type-surface behavior all present at the Python boundary even when the underlying logic lives in Rust. Rust inline and integration tests still belong to the inner library, but cross-language error propagation, conversion failures, and wrapper behavior are more naturally discovered in Python tests. That inference is supported by the split layout documented by maturin and the dual root structure visible in pydantic-core. citeturn47search1turn44search3

**Ruff** is another useful hybrid signal. The verified repository identity says it is a Rust implementation that is pip-installable and configured with `pyproject.toml`. For your scanner, that means “Python project” and “Python test layout” are not synonyms. A Rust-first repository can still participate in Python workflows and Python packaging, and should be detected as hybrid from root signals before pairing begins. citeturn44search0turn44search1

CI evidence also supports a split-toolchain mental model. **HTTPX** and **Uvicorn** both expose a dedicated “Test Suite” workflow in CI, and the Uvicorn snippet explicitly shows pytest, pytest-mock, pytest-xdist, coverage, and related tooling in the dependency update footprint. For hybrid or multi-tool repositories, the most robust scanner architecture is to let CI/tooling clues reinforce what the filesystem already suggests, rather than treating the tree alone as the entire truth. citeturn47search3turn47search4

## Existing tools, anti-pattern calibration, and quality metrics

Pytest’s documented collection APIs give you naming and collection controls, but they do **not** give you a documented source-file→test-file mapping service. The docs focus on collection rules such as `python_files`, `python_classes`, `python_functions`, `testpaths`, and `collect_ignore`. That means pairing is an inference problem you have to solve yourself. The good news is that pytest’s collection model is regular enough that a small number of heuristics covers most verified Python repositories. citeturn42search0turn42search1turn42search4

`coverage.py` is the strongest off-the-shelf refinement tool for Python pairing. Its measurement-context feature can record dynamic context per test function, and the docs explicitly say this is commonly used to answer “what test ran this line?” The docs also note that pytest-cov can set dynamic context for each test. For your scanner, this means you can first generate static candidate pairs, then re-rank them with actual per-test execution evidence. That is a much more reliable second pass than trying to divine perfect pairs from filenames alone. citeturn42search2

On the mutation-testing side, `mutmut` is directly relevant because its own docs say it “knows which tests to execute,” defaults to running pytest on `tests` or `test`, can infer what code to mutate, and can optionally use coverage data to mutate only covered lines. That combination makes mutation tooling a valuable **quality overlay** for a pairing scanner: it will not replace pairing, but it can confirm whether the chosen tests actually exercise the paired source. citeturn46search0turn46search1turn46search6

Property-based testing is the clearest literature-backed metric-adjacent recommendation in the sources I reviewed. The Hypothesis docs describe property-based testing as writing tests that should pass across generated inputs, especially edge cases, while shrinking failures to the simplest counterexample. The proptest docs describe essentially the same model on the Rust side, including automatic failure minimization. For your scanner, the right recommendation trigger is **high-branch, high-input-space code with low example diversity**, not a generic “add more tests” nudge. citeturn46search2turn46search4turn46search5

On anti-pattern calibration, the strongest conclusion from this pass is negative: do **not** attach hard prevalence claims without a dedicated code-search sweep. The verified project trees were enough to study structure, but not enough to count empty tests, unreasoned skips, `time.sleep`, or weak-assertion patterns across all requested repositories. The safer implementation policy is therefore severity-based rather than absolute. Flag `pass`-only tests, no-reason skips, sleep-driven timing, and call-without-assert patterns as smells, but down-rank them when the test name or surrounding code clearly indicates a smoke test, timeout test, or panic/exception expectation. Rust’s explicit `#[should_panic]` attribute is one concrete example of context that should reduce severity instead of increasing it. citeturn43search4turn43search5

## Implementation priority and the pairing algorithm

The implementation priority that best matches the verified evidence is this:

1. **Detect repository mode**: pure Python, pure Rust, or hybrid/workspace. Signals include `src/`, `tests/`, `test/`, `Cargo.toml`, `pyproject.toml`, language-specific subtrees like `python/`, and multi-crate roots. Ruff and pydantic-core show why this step must happen before any pairing attempt. citeturn44search1turn44search3

2. **Parse test discovery config before generating pairs**. In Python, honor pytest’s effective `testpaths`, `python_files`, `python_classes`, and `python_functions`. In Rust, trust Cargo’s unit/integration/doc-test boundaries rather than inventing your own discovery model. citeturn42search0turn42search1turn42search4turn43search3turn43search6

3. **Generate multiple candidates with scores**. For Python, score exact basename matches highest, then `src`-stripped mirrors, then package-area mirrors, then conceptual/singular-plural variants. For Rust, score same-file inline tests highest, then same-crate integration tests, then workspace-level test crates. Django, Flask, FastAPI, and SQLAlchemy all show why a single-rule system is too brittle. citeturn9view0turn16view2turn24view0turn35view0

4. **Upgrade with runtime evidence**. If coverage contexts are available, use them to map source lines to actual tests. If mutation testing output is available, use surviving mutants to identify weak or misleading pairs. `coverage.py` is the most concrete tool support for this today in the sources I reviewed. citeturn42search2turn46search0turn46search6

5. **Allow many-to-many and area-level pairing**. A scanner should be allowed to say “this source package is covered by these test areas” when repository structure supports that conclusion better than a fake 1:1 file pair. Django and SQLAlchemy are the clearest reasons to support that representation from day one. citeturn16view2turn24view0turn27view0

If the goal is “handle the most projects without custom configuration,” the best default is: **config-first, then filesystem family detection, then scored candidates, then coverage refinement**. In practice, that will correctly handle the dominant Python layouts I verified, will respect Cargo’s native structure instead of fighting it, and will keep hybrid repositories from poisoning the model with false assumptions. Everything else in your scanner—coverage gaps, edge-case analysis, mutation guidance, and LLM-based review—will improve if that pairing layer is probabilistic, structure-aware, and coverage-assisted instead of regex-only. citeturn21view0turn32view0turn37view0turn43search3turn44search3turn42search2