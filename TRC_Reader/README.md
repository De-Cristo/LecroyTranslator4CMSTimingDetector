# LecroyTranslator

Small C++ utility to read LeCroy oscilloscope binary waveform files (template LECROY_2_3) and export two CSV files:

- `<basename>_data.csv` — flattened waveform data as `Segment,Time_s,Voltage_V`.
- `<basename>_meta.csv` — waveform metadata as `Field,Value`.

The program source is `LecroyTranslator.cpp` (requires a C++17 compiler).

## Build

From the `LecroyTranslator/` directory run:

    g++ -std=c++17 -O2 LecroyTranslator.cpp -o read_lecroy

(Use your preferred compiler flags or toolchain.)

## Usage

    ./read_lecroy <waveform.trc> [out_dir]

- `<waveform.trc>`: path to the LeCroy `.trc` file.
- `[out_dir]` (optional): directory where the two CSV files will be written. If the directory does not exist it will be created. If omitted, the CSV files are written to the current working directory.

You can run it in background if desired:

    ./read_lecroy ../LecroyBin2ascii/raw_C1_0004522_0000001_13675.trc ./outdir &

## Output files

For an input file named `raw_C1_0004522_0000001_13675.trc` the program will create:

- `raw_C1_0004522_0000001_13675_data.csv`
- `raw_C1_0004522_0000001_13675_meta.csv`

If `out_dir` is provided, both files are placed under that directory.

## Notes and troubleshooting

- The reader was implemented for template `LECROY_2_3`. A warning is printed if the file template differs — the code may still work for many LeCroy files but verify results.
- The program reads the whole file into memory; ensure enough RAM for very large traces.
- Requires a C++17 capable compiler for `<filesystem>` and other features.
- Common runtime errors reported by the program include inability to open files, invalid header (WAVEDESC not found), or unsupported COMM_ORDER/endian detection failures.

## Contact
licheng.zhang@cern.ch
