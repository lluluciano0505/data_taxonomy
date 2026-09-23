# Biljmermeer demo

This is a self-contained demonstration project for DataTaxonomy, based on a **synthetic Amsterdam Bijlmermeer urban-renewal case**. It is intended for product demonstrations and first-run testing, not as a record of a real client project.

## What is included

- `demo_data/Amsterdam_Bijlmermeer_Urban_Renewal_Case/` — 55 synthetic project files across strategy, surveys, design, delivery, governance, and coordination folders.
- `demo_data/bijlmermeer_demo.yaml` — portable project configuration with paths relative to the repository root.
- `taxonomies/bijlmermeer_demo.yaml` — the demo taxonomy snapshot.
- `demo_data/bijlmermeer_demo.csv` — a small reference result from a previous run, included only as an optional preview.

The reference result is not required to run the demo. Running the project again with the user's own provider and API key produces a fresh result locally.

## Run it

1. Start DataTaxonomy by double-clicking `start_dashboard.bat`.
2. Open the Configuration Wizard and choose **Load / import project configuration** if available, or create a project manually.
3. Use the project folder:

   `demo_data/Amsterdam_Bijlmermeer_Urban_Renewal_Case`

4. Choose an AI provider, model, and the user's own API key.
5. For a quick demonstration, process a small sample first. Use all files for the full demo run.
6. Save the project and start processing. The Dashboard opens when the run is complete.

On Windows, use forward slashes or the native **Choose Folder…** button. No API key or client data is included in this demo.

## Reference result

`demo_data/bijlmermeer_demo.csv` contains 53 classified rows from the synthetic input set. Its paths are relative so it can be inspected on another computer. It is a reference snapshot, not a guarantee of the result produced by every provider/model combination.
