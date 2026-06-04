- [Pipeline.py](#pipelinepy)
  - [Note](#note)
  - [Input](#input)
  - [Output](#output)
- [Processing.py](#processingpy)
  - [Note](#note-1)
  - [Input](#input-1)
  - [Output](#output-1)
- [WavSplit.py](#wavsplitpy)
  - [Note](#note-2)
  - [Input](#input-2)
  - [Output](#output-2)
- [Graphs.py](#graphspy)
- [IceDetector.py](#icedetectorpy)
- [Utils.py](#utilspy)

## Pipeline.py
### Note
Runs the splitter then the processor.

### Input

| flag |    full flag     | type  | default  | notes                                                                                  |
| :--- | :--------------: | :---: | :------: | :------------------------------------------------------------------------------------- |
| -i   |   --input-dir    |  str  | REQUIRED | Exact filepath to raw audio dir                                                        |
| -m   |    --mat-file    |  str  | REQUIRED | Exact filepath to .mat file                                                            |
| -img |    --img-dir     |  str  | REQUIRED | Exact filepath to outermost dir with images                                            |
| -ol  |  --output-level  |  int  |    1     | Output level: 0 debug, 1 standard, 2 error only                                        |
| -sl  | --segment-length |  int  |    20    | Length of each segment                                                                 |
| -lc  |    --low-cut     | float |  500.0   | The lowcut applied in Hz                                                               |
| -hc  |    --high-cut    | float | 20000.0  | The highcut applied in Hz                                                              |
| -np  |    --nperseg     |  int  |   2048   | Size of chunk send to FFT                                                              |
| -fs  | --fs-calibration | float |  170.0   | Calibration level                                                                      |
| -fo  |  --filter-order  |  int  |    4     | Order of the bandpass filter                                                           |
| -so  |  --start-offset  |  int  |    0     | The number of seconds after the start to start the analysis                            |
| -eo  |   --end-offset   |  int  |    0     | The number of seconds before the end to end the analysis                               |
| -sc  |  --sample-count  |  int  | 1000000  | The number of samples to take during analysis. Values < 0 collect as many as possible. |

### Output
A folder titled "output_charts" placed inside the input dir where all graphs are placed. A folder titled "output_segments" containing processed audio if processing needs to be run again.

## Processing.py

### Note
Input .wav files must all have the same sample rate.

### Input
| var  |    secondary     | type  | default  | notes                                                                                  |
| :--- | :--------------: | :---: | :------: | :------------------------------------------------------------------------------------- |
| -i   |   --input-dir    |  str  | REQUIRED | Exact filepath to audio dir                                                            |
| -m   |    --mat-file    |  str  | REQUIRED | Exact filepath to .mat file                                                            |
| -img |    --img-dir     |  str  | REQUIRED | Exact filepath to outermost dir with images                                            |
| -ol  |  --output-level  |  int  |    1     | Output level: 0 debug, 1 standard, 2 error only                                        |
| -lc  |    --low-cut     | float |  500.0   | The lowcut applied in Hz                                                               |
| -hc  |    --high-cut    | float | 20000.0  | The highcut applied in Hz                                                              |
| -np  |    --nperseg     |  int  |   2048   | Size of chunk send to FFT                                                              |
| -fs  | --fs-calibration | float |  170.0   | Calibration level                                                                      |
| -fo  |  --filter-order  |  int  |    4     | Order of the bandpass filter                                                           |
| -so  |  --start-offset  |  int  |    0     | The number of seconds after the start to start the analysis                            |
| -eo  |   --end-offset   |  int  |    0     | The number of seconds before the end to end the analysis                               |
| -sc  |  --sample-count  |  int  | 1000000  | The number of samples to take during analysis. Values < 0 collect as many as possible. |

### Output
A folder titled "output_charts" placed next to the input dir where all graphs are placed.

## WavSplit.py

### Note
Should not be called directly unless only splitting audio is desired.

### Input
| var  |    secondary     | type  | default  | notes                                           |
| :--- | :--------------: | :---: | :------: | :---------------------------------------------- |
| -i   |   --input-dir    |  str  | REQUIRED | Exact filepath to raw audio dir                 |
| -ol  |  --output-level  |  int  |    1     | Output level: 0 debug, 1 standard, 2 error only |
| -sl  | --segment-length |  int  |    20    | Length of each segment                          |

### Output
A folder titled "output_segments" placed inside the input dir where split audio is placed.

## Graphs.py
Contains all graphs to be built.

## IceDetector.py
Contains the code to modify the neural network used when detecting ice in images.

## Utils.py
Contains utility functions and commonly used types.