# AI Team Training '27 — Task 2: Football Analytics & Machine Learning from Scratch

This repository contains three data science tasks analyzing football data — from
150 years of international match results, to building Principal Component
Analysis (PCA) and similarity search algorithms entirely from scratch on FIFA 19
player data.

## 📁 Project Structure

```
├── task_2.1_world_cup_analysis.ipynb      # Historical international football analysis
├── task_2.2_pca_from_scratch.ipynb        # PCA implementation (NumPy + Pure Python)
├── task_2.3_scouting_engine.ipynb         # Player similarity search engine
└── README.md
```

---

## Task 2.1 — A Century of Football: Decoding National Team Performance

**Goal:** Analyze 150 years of international football (1872–2026) to uncover
patterns in team dominance, efficiency, drama, and the evolution of the game.

**Data sources:** `results.csv`, `goalscorers.csv`, `shootouts.csv`
([International Football Results 1872–2024, Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017))

### What was done
- Merged three datasets into a unified team-level view (99,040 team-match rows)
- Cleaned and validated data: recovered two "missing" fixtures that turned out
  to be real, already-played 2026 World Cup matches, and imputed their true
  results
- Identified and filtered ~128 non-FIFA entities (defunct states, disputed
  territories, and small island competitions like the Island Games) that were
  skewing team rankings
- Cleaned `goalscorers.csv` (excluded own goals and undocumented rows)
- Engineered features: `decade`, `total_goals`, `goal_margin`, `points` (3/1/0),
  match outcome

### Key Questions Answered
| Question | Answer |
|---|---|
| Top scoring teams (all-time) | England, Germany, Brazil |
| Top individual goalscorers | Cristiano Ronaldo (124), Harry Kane (75), Messi (71) |
| Most efficient team (min. 100 matches, FIFA members) | Brazil (2.11 pts/match) |
| Decade with most penalty shootouts (rate/year) | 2020s (17.7/year) — a sustained, continuous rise since the 1970s |
| Team most likely to fail after scoring first | Malta (63.9% failure rate) |
| Most lopsided match in history | Australia 31–0 American Samoa (2001 WC qualifying) |
| Goals per match, 1880s → 2020s | ~5.6 → ~2.7 (football has become significantly more defensive) |

### Notable Data Quality Findings
- `goalscorers.csv` under-represents historical non-European scorers — e.g. Ali
  Daei (Iran), the actual record holder before Ronaldo (109 goals), only has
  ~49 goals logged in this dataset
- Outlier detection comparison: IQR flagged an impractical 12.16% of matches as
  outliers due to the right-skewed nature of goal margins, while Z-score
  (|Z|>3) gave a more usable 1.30% — though neither substitutes for
  domain-aware manual review

### Visualizations
Bar charts (top teams/scorers), trend lines (goals/match, draw rate, goal
margin, shootout frequency by decade), and a decade × tournament heatmap of
scoring rates.

---

## Task 2.2 — Finding the Key Factors Behind Player Performance with PCA

**Goal:** Implement Principal Component Analysis **from scratch** (no
`sklearn.decomposition.PCA`) to reduce FIFA 19 player statistics into 2
interpretable performance archetypes.

**Data source:** `FIFA 19 Complete Player Dataset`
([Kaggle](https://www.kaggle.com/datasets/javagarm/fifa-19-complete-player-dataset), CP1252-encoded)

**Features used:** Finishing, ShortPassing, Dribbling, SprintSpeed, Strength,
Stamina, Interceptions, StandingTackle, and Value (parsed from strings like
`€110.5M`/`€425K` and log-transformed to correct for extreme skew)

### Pipeline (NumPy version)
1. Load & clean data (18,207 → 18,147 players after removing incomplete profiles)
2. Standardize features (Z-score)
3. Compute the covariance matrix
4. Eigen-decomposition (`numpy.linalg.eig`)
5. Sort eigenvectors by eigenvalue (descending)
6. Project data onto the top 2 principal components

### Results
- **PC1 (49.4% variance) — "Tactical/Technical Player" axis:** dominated by
  ShortPassing, Dribbling, and Stamina; Strength contributes almost nothing
- **PC2 (22.2% variance) — "Defense vs. Attack" axis:** Interceptions/
  StandingTackle/Strength on one end, Finishing/SprintSpeed/Dribbling on the
  other
- Together, **71.6% of total variance** is captured by just 2 components
- The resulting 2D scatter plot, colored by position, shows goalkeepers
  separating almost perfectly, and defenders/midfielders/attackers arranging
  along a clear defensive-to-attacking spectrum — recovered entirely from
  numerical stats, with no position label used in the computation

### Bonus: Pure Python Implementation
All core steps (transpose, matrix multiplication, standardization, covariance,
projection) were re-implemented using only Python lists — no NumPy matrix ops.

| | NumPy | Pure Python |
|---|---|---|
| Runtime (full dataset) | 0.0037s | 0.2712s |
| Speedup | — | **73.7× slower** |
| Max output difference | — | 2.28×10⁻¹³ (numerically identical) |

NumPy's speed advantage comes from vectorized C-level operations, contiguous
memory layout, and SIMD instructions — versus Python's interpreted, per-element
loop overhead in the pure Python version.

---

## Task 2.3 — The Scouting Engine: Finding a Successor to M. Salah

**Goal:** Build a similarity search engine (from scratch, fully vectorized) to
find players most similar to Mohamed Salah using the same FIFA 19 dataset and
features as Task 2.2.

### Metrics implemented (all vectorized, no per-player loops)
- **Euclidean Distance** — absolute skill-level difference (lower = more similar)
- **Manhattan Distance** — sum of absolute differences, less sensitive to single large gaps
- **Cosine Similarity** — shape of the skill profile, ignoring magnitude (higher = more similar)
- **Pearson Correlation** — pattern of relative strengths/weaknesses (higher = more similar)

### Results
| Rank | Euclidean | Cosine | Manhattan | Pearson |
|---|---|---|---|---|
| 1 | K. Mbappé | A. Lacazette | K. Mbappé | K. Mbappé |
| 2 | A. Lacazette | A. Sánchez | S. Mané | Marco Asensio |
| 3 | S. Mané | K. Mbappé | A. Griezmann | E. Hazard |
| 4 | M. Reus | M. Reus | A. Lacazette | A. Lacazette |
| 5 | G. Bale | Santi Mina | M. Reus | M. Reus |

**Final shortlist:** Kylian Mbappé, Alexandre Lacazette, and Marco Reus each
appeared across **all four** metrics — the strongest possible signal. Sadio
Mané appeared in 2 of 4 and is included as a well-known real-world stylistic
comparison.

### Standardized vs. Unstandardized Experiment
Running the same search on raw, unstandardized data (including Salah's true
€69.5M market value with no scaling) shifted the shortlist toward players with
*similar market value* (e.g., Cavani, Bale) rather than similar playing style —
demonstrating why standardization is essential before any distance-based
similarity search.

### Bonus: PCA Validation
Salah's shortlist was projected onto the PCA space built in Task 2.2, confirming
they occupy a tight, coherent region close to Salah — separate from goalkeepers
and the broader outfield player cloud.

---

## 🛠️ Tech Stack
`pandas` · `numpy` · `matplotlib` · `seaborn` · Python standard library (for
the from-scratch pure-Python PCA implementation)

## ⚠️ Key Limitations (across all tasks)
- `goalscorers.csv` (Task 2.1) has uneven historical/geographic coverage
- The ~128-entity FIFA-member exclusion list (Task 2.1) was manually curated
  and may not be 100% exhaustive
- FIFA 19 attribute ratings (Tasks 2.2/2.3) are subjective, game-assigned
  values rather than raw match-tracking data
- The 2020s decade (Task 2.1) is incomplete (2020–2026), so decade-level
  comparisons involving it are directional, not final
